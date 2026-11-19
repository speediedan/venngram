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
from torch import nn

import venngram
from venngram import AddressBook, ParamSpace
from venngram.maps import from_document
from venngram.maps.view import MappedView
from venngram.tuning.conformance import ConformanceError, role_address_case
from venngram.tuning.errors import TunableViewError
from venngram.tuning.views import CanonicalView


def lora_model(depth=3):
    """Blocks with an attention projection and an MLP, each carrying LoRA factors, and a buffer per block."""
    root = nn.Module()
    root.layers = nn.ModuleList()
    for _ in range(depth):
        block = nn.Module()
        for part in ("attn", "mlp"):
            m = nn.Linear(2, 2, bias=False, device="meta")
            m.lora_A = nn.Parameter(m.weight.new_empty((1, 2)))
            m.lora_B = nn.Parameter(m.weight.new_empty((2, 1)))
            setattr(block, part, m)
        block.register_buffer("scale", block.attn.weight.new_empty((1,)))
        root.layers.append(block)
    return root


MAP = {"schema_version": 5, "architecture": "LoraProbe", "components": {
    "blocks.{i}": {"module": "layers.{i}", "kind": "block"},
    "blocks.{i}.attn.q": {"module": "layers.{i}.attn", "kind": "linear", "layout": "out_in"},
    "blocks.{i}.mlp.in": {"module": "layers.{i}.mlp", "kind": "linear", "layout": "out_in"}}}
LORA = {f"layers.{i}.{p}.lora_{f}" for i in range(3) for p in ("attn", "mlp") for f in "AB"}


def mapped(roles, **kwargs):
    return MappedView(lora_model(), [from_document(MAP)], roles=roles, **kwargs)


def test_a_role_address_assigns_the_role_by_structure():
    view = mapped({"adapter:lora": "component:attn.* & leaf:lora_*"})
    assert {n for n, r in view.roles().items() if r == "adapter:lora"} == {
        f"layers.{i}.attn.lora_{f}" for i in range(3) for f in "AB"}
    assert "adapter" in view.support().declared_roles
    space = ParamSpace([n for n, _ in view.root.named_parameters()], structure=view.structure(), roles=view.roles())
    assert venngram.select("role:adapter & layers:-1:", space) == ["layers.2.attn.lora_A", "layers.2.attn.lora_B"]


def test_name_entries_and_addresses_mix_and_names_win():
    view = mapped({"adapter:lora": "leaf:lora_*", "layers.0.mlp.lora_A": "artifact"})
    roles = view.roles()
    assert roles["layers.0.mlp.lora_A"] == "artifact" and roles["layers.0.mlp.lora_B"] == "adapter:lora"
    assert {n for n, r in roles.items() if r == "adapter:lora"} == LORA - {"layers.0.mlp.lora_A"}


def test_a_physical_view_resolves_layers_and_buffers():
    view = CanonicalView(lora_model(), roles={"artifact": "layers:-1: & leaf:scale"})
    assert view.roles() == {"layers.2.scale": "artifact"}


def test_two_addresses_giving_one_tensor_two_roles_are_refused_unless_a_name_settles_it():
    roles = {"adapter:lora": "leaf:lora_*", "adapter:top": "layers:-1: & leaf:lora_*"}
    with pytest.raises(TunableViewError, match="different roles: 'layers.2.attn.lora_A' \\(adapter:lora and "
                                               "adapter:top\\)"):
        mapped(roles)
    settled = {**roles, **{n: "adapter:top" for n in LORA if n.startswith("layers.2.")}}
    assert mapped(settled).roles()["layers.2.mlp.lora_B"] == "adapter:top"


def test_an_address_may_reference_a_books_sets():
    book = AddressBook.from_mapping({"schema_version": 1, "name": "b", "sets": {
        "attn_adapters": {"component": "attn.*", "leaf": "lora_*"}}})
    view = venngram.view(lora_model(), [from_document(MAP)], roles={"adapter:lora": "@attn_adapters"}, sets=book)
    assert len([r for r in view.roles().values() if r == "adapter:lora"]) == 6


@pytest.mark.parametrize("roles, match", [
    ({"adapter:lora": "role:base & leaf:lora_*"}, "states a role"),
    ({"adapter:lora": "leaf:nothing_here"}, "does not resolve"),
    ({"adapter:lora": "@missing"}, "does not resolve"),
    ({"layers.9.attn.weight": "adapter:x"}, "neither a parameter nor a buffer"),
    ({"frobnicate:x": "leaf:lora_*"}, "not a role either"),
])
def test_malformed_role_addresses_are_refused_by_name(roles, match):
    with pytest.raises(TunableViewError, match=match):
        mapped(roles)


def test_the_conformance_case_compares_address_roles_with_the_names_they_expand_to():
    case = role_address_case(lambda roles: mapped(roles), {"adapter:lora": "component:attn.* & leaf:lora_*"})
    case.run()
    empty = role_address_case(lambda roles: mapped({}), {"adapter:lora": "leaf:lora_*"})
    with pytest.raises(ConformanceError, match="select no tensor"):
        empty.run()
