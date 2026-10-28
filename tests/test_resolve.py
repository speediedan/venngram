# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
from conftest import adapter_roles, TinyDecoder
from torch import nn

from venngram import AddressUnresolvable, Location, ParamSpace, Refusal, select
from venngram.physical import block_lists, PhysicalStructure
from venngram.resolve import module_matches


def space_of(model: nn.Module, **kwargs) -> ParamSpace:
    return ParamSpace([n for n, _ in model.named_parameters()], roles=adapter_roles(model),
                      structure=PhysicalStructure(model), **kwargs)


def test_role_and_layers_select_a_subset_of_a_role(model):
    space = space_of(model)
    top = select("role:adapter & layers:-2:", space)
    assert top and all(n.startswith(("layers.4.", "layers.5.")) and ".adapter." in n for n in top)
    assert len(top) == 2 * 2 * 2  # two blocks, q and k adapters, lora_A and lora_B
    assert top == [n for n in space.names if n in top]  # reported in the space's order


def test_overlapping_sets_may_both_be_selected(model):
    space = space_of(model)
    top = set(select("role:adapter & layers:-2:", space))
    q = set(select("role:adapter & module:q_proj", space))
    assert top & q and top - q and q - top  # overlap, which one role per tensor cannot express


def test_minus_or_and_references(model):
    space = space_of(model)
    sets = {"top": "role:adapter & layers:-2:", "rest": {"role": "adapter", "exclude": "@top"}}
    rest = select("@rest", space, sets)
    assert set(rest) == set(select("role:adapter", space)) - set(select("@top", space, sets))
    both = select("@top | @rest", space, sets)
    assert set(both) == set(select("role:adapter", space))


def test_module_leaf_and_name(model):
    space = space_of(model)
    assert select("module:layers.*.mlp", space) == [f"layers.{i}.mlp.{p}" for i in range(6) for p in ("weight", "bias")]
    assert all(n.endswith("lora_A.weight") for n in select("leaf:lora_A.weight", space))
    assert select('name:"head\\..*"', space) == ["head.weight", "head.bias"]
    assert not module_matches("q_proj", "layers.0.attn.xq_proj.weight")


def test_aliases_match_name_and_module(model):
    aliases = {"layers.0.attn.q_proj.weight": ("blocks.0.attn.W_Q",)}
    space = space_of(model, aliases=aliases)
    assert select('name:"blocks\\.0\\.attn\\.W_Q"', space) == ["layers.0.attn.q_proj.weight"]


def test_tensors_outside_the_block_list_have_no_layer(model):
    space = space_of(model)
    selected = select("layers:0:", space)
    assert "embed.weight" not in selected and "head.weight" not in selected


def test_guards_optional_and_expect(model):
    space = space_of(model)
    assert select({"role": "intervention", "optional": True}, space) == []
    with pytest.raises(AddressUnresolvable, match="matches no tensor"):
        select("role:intervention", space)
    assert len(select({"role": "adapter", "layers": -1, "expect": 4}, space)) == 4
    with pytest.raises(AddressUnresolvable, match="where `expect` requires 5"):
        select({"role": "adapter", "layers": -1, "expect": 5}, space)


@pytest.mark.parametrize("address, sets, match", [
    ("@nope", {}, "no set is named `@nope`"),
    ("@a", {"a": "@b", "b": "@a"}, "cycle"),
    ("@empty", {"empty": "role:intervention"}, "`@empty` matches nothing"),
    ("point:resid_post & layers:3", {}, "site resolver"),
    ("features:sae & point:resid_post", {}, "latent resolver"),
    ("role:adapter & units:[0, 1]", {}, "unit adapter"),
    ("component:attn.q", {}, "vocabulary structure provider"),
    ("layers:[9]", {}, "outside a stack of 6"),
    ("stack:vision & layers:0", {}, "only the primary stack"),
])
def test_refusals_say_why(model, address, sets, match):
    with pytest.raises(AddressUnresolvable, match=match):
        select(address, space_of(model), sets)


def test_layers_without_a_structure_provider_are_refused(model):
    with pytest.raises(AddressUnresolvable, match="needs a structure provider"):
        select("layers:0", ParamSpace([n for n, _ in model.named_parameters()]))


def test_every_refusal_is_reported(model):
    with pytest.raises(AddressUnresolvable) as excinfo:
        select("@nope | point:resid_post", space_of(model))
    assert len(excinfo.value.refusals) == 2 and all(isinstance(r, Refusal) for r in excinfo.value.refusals)


def test_physical_structure_ignores_nested_expert_lists():
    model = TinyDecoder(depth=3, experts=8)
    assert block_lists(model) == [("layers", 3)]
    structure = PhysicalStructure(model)
    assert structure.depth() == 3 and structure.locate("layers.2.experts.7.weight") == Location(layer=2)


def test_physical_structure_refuses_ambiguity_and_takes_a_pattern():
    class EncDec(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.ModuleList(nn.Linear(2, 2) for _ in range(3))
            self.decoder = nn.ModuleList(nn.Linear(2, 2) for _ in range(5))

    model = EncDec()
    ambiguous = PhysicalStructure(model).depth()
    assert isinstance(ambiguous, Refusal) and ambiguous.alternatives == ("encoder", "decoder")
    assert PhysicalStructure(model, layers_pattern="decoder").depth() == 5
    space = ParamSpace([n for n, _ in model.named_parameters()], structure=PhysicalStructure(model))
    with pytest.raises(AddressUnresolvable, match="several top-level block lists"):
        select("layers:0", space)


def test_wrapper_segments_are_ignored():
    model = TinyDecoder(depth=2)
    names = [f"_orig_mod.{n}" for n, _ in model.named_parameters()]
    structure = PhysicalStructure(model)
    assert structure.locate("_orig_mod.layers.1.mlp.weight").layer == 1
    space = ParamSpace(names, structure=structure)
    assert select("module:mlp & layers:1", space) == ["_orig_mod.layers.1.mlp.weight", "_orig_mod.layers.1.mlp.bias"]
