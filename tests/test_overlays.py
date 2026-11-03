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
import torch
from torch import nn

from venngram import view
from venngram.maps import (component_map_for, known_architectures, known_overlays, map_from_document, overlay_for,
                           overlay_from_document, overlay_roles, OverlayMap, overlays_for, register, unregister)
from venngram.maps.overlay import rule_roles
from venngram.tuning import TunableViewError

PEFT_LORA = {name: "adapter:{key}" for name in ("lora_A", "lora_B", "lora_E", "lora_embedding_A", "lora_embedding_B",
                                                "lora_magnitude_vector")}


class Scaler(nn.Module):
    """An instrument library's module, recognized by its class rather than by its tensors' names."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d))


class SubScaler(Scaler):
    pass


SCALER = f"{Scaler.__module__}.{Scaler.__qualname__}"


@pytest.fixture
def scaler_overlay():
    overlay = OverlayMap(name="test.scaler", roles={"gain": "intervention:scale"}, module_classes=(SCALER,))
    register(overlay)
    yield overlay
    unregister("test.scaler")


def test_the_seed_registry_serves_peft_lora_as_an_overlay():
    overlay = overlay_for("peft.lora")
    assert dict(overlay.roles) == PEFT_LORA and "peft.tuners.lora.layer.LoraLayer" in overlay.module_classes
    assert "peft.lora" in known_overlays() and not any(a.startswith("overlay:") for a in known_architectures())
    with pytest.raises(KeyError, match="no overlay map 'nope.dora'"):
        overlay_for("nope.dora")


def test_overlays_are_detected_from_the_modules_a_model_carries(model):
    assert [o.name for o in overlays_for(model)] == ["peft.lora"]
    roles = overlay_roles(model, overlays_for(model))
    assert roles and set(roles.values()) == {"adapter"}
    assert all(".lora_A." in n or ".lora_B." in n for n in roles)
    assert overlays_for(nn.Linear(2, 2)) == []


def test_an_overlay_applies_to_a_subclass_of_a_class_it_names(model, scaler_overlay):
    model.layers[0].add_module("scale", SubScaler(4))
    assert {o.name for o in overlays_for(model)} == {"peft.lora", "test.scaler"}
    assert overlay_roles(model, overlays_for(model))["layers.0.scale.gain"] == "intervention:scale"
    assert not scaler_overlay.applies_to(nn.Sequential(nn.Linear(2, 2)))


def test_view_composes_the_base_map_and_every_overlay_that_applies():
    from transformers import LlamaConfig, LlamaForCausalLM

    model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=2,
                                         num_attention_heads=2, num_key_value_heads=2))
    q = model.model.layers[1].self_attn.q_proj
    q.add_module("lora_A", nn.Linear(8, 2, bias=False))
    v = view(model)
    assert v.cmap.architecture == "LlamaForCausalLM" and [o.name for o in v.overlays] == ["peft.lora"]
    assert v.roles()["model.layers.1.self_attn.q_proj.lora_A.weight"] == "adapter"
    loc = v.structure().locate("model.layers.1.self_attn.q_proj.lora_A.weight")
    assert (loc.layer, loc.component) == (1, "attn.q")  # an overlay's tensor keeps its host's layer and component
    assert view(model, component_map_for("LlamaForCausalLM")).overlays == ()  # named maps replace detection
    assert view(model, roles={"model.norm.weight": "fitted"}).roles()["model.norm.weight"] == "fitted"


def test_named_maps_replace_detection(model, scaler_overlay):
    model.layers[0].add_module("scale", Scaler(4))
    assert set(overlay_roles(model, [scaler_overlay])) == {"layers.0.scale.gain"}
    assert overlay_roles(model, []) == {}


def test_a_registration_shadows_an_indexed_overlay_of_the_same_name(model):
    register(OverlayMap(name="peft.lora", roles={"lora_A": "adapter:down"}, module_names=("lora_A",)))
    try:
        assert set(overlay_roles(model, overlays_for(model)).values()) == {"adapter:down"}
    finally:
        unregister("peft.lora")
    assert set(overlay_roles(model, overlays_for(model)).values()) == {"adapter"}


def test_overlays_that_disagree_are_refused_by_name(model):
    register(OverlayMap(name="other.lora", roles={"lora_B": "intervention"}, module_names=("lora_B",)))
    try:
        with pytest.raises(TunableViewError, match="'other.lora' and 'peft.lora' both apply"):
            overlay_roles(model, overlays_for(model))
    finally:
        unregister("other.lora")


def test_an_overlay_document_round_trips_and_reads_as_a_map():
    overlay = overlay_for("peft.lora")
    assert overlay_from_document(overlay.to_mapping()) == overlay
    assert map_from_document(overlay.to_mapping()) == overlay


class Keyed(nn.Module):
    """A library's layer keeping what it adds in containers keyed by name, as PEFT keys adapters by adapter name."""

    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(4, 4)
        self.lora_A = nn.ModuleDict({"a": nn.Linear(4, 2, bias=False), "b": nn.Linear(4, 2, bias=False)})
        self.ia3_l = nn.ParameterDict({"a": nn.Parameter(torch.ones(4))})


def test_a_templated_role_takes_the_keyed_containers_entry():
    rules = {"lora_A": "adapter:{key}", "ia3_l": "adapter:{key}"}
    roles = rule_roles(nn.ModuleDict({"q": Keyed()}), rules)
    assert roles == {"q.lora_A.a.weight": "adapter:a", "q.lora_A.b.weight": "adapter:b", "q.ia3_l.a": "adapter:a"}
    plain = nn.Module()
    plain.lora_A = nn.Linear(4, 2)  # not a keyed container: no entry to take, so the qualifier is dropped
    assert rule_roles(plain, rules) == {"lora_A.weight": "adapter", "lora_A.bias": "adapter"}


def test_every_peft_tuner_family_takes_its_adapter_names():
    """The sanity check behind the seed overlays: against PEFT's own account of its adapter tensors (each tuner
    layer's ``adapter_layer_names``), the overlays give the same tensors, each named by its adapter."""
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    llama = transformers.LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2)
    targets = ["q_proj", "v_proj"]
    configs = [
        peft.LoraConfig(r=2, target_modules=targets),
        peft.LoraConfig(r=2, target_modules=targets, use_dora=True),
        peft.AdaLoraConfig(r=2, target_modules=targets, total_step=10),
        peft.IA3Config(target_modules=targets, feedforward_modules=[]),
        peft.LoHaConfig(r=2, target_modules=targets),
        peft.LoKrConfig(r=2, target_modules=targets),
        peft.OFTConfig(r=2, oft_block_size=0, target_modules=targets),
    ]
    for config in configs:
        model = peft.get_peft_model(transformers.LlamaForCausalLM(llama), config, adapter_name="first")
        names = {id(p): n for n, p in model.named_parameters()}
        expected = {}
        for module in model.modules():
            for layer_name in getattr(module, "adapter_layer_names", ()):
                for adapter, entry in getattr(module, layer_name).items():
                    for p in (entry.parameters() if isinstance(entry, nn.Module) else (entry,)):
                        expected[names[id(p)]] = f"adapter:{adapter}"
        # Parameters only: PEFT's account names parameters, and an overlay also gives roles to the buffers in an
        # adapter's entry (OFT keeps index buffers there).
        params = set(names.values())
        roles = {n: r for n, r in overlay_roles(model, overlays_for(model)).items() if n in params}
        assert roles == expected, type(config).__name__


@pytest.mark.parametrize("change, match", [
    ({"schema_version": 3}, "needs schema_version 4"),
    ({"schema_version": 4}, "`{key}` needs schema_version 5"),
    ({"roles": {"lora_A": "{key}:adapter"}}, "only as its last qualifier"),
    ({"overlay": ""}, "non-empty `overlay`"),
    ({"architecture": "X"}, "`architecture` belongs to a base map"),
    ({"roles": {}}, "must not be empty"),
    ({"roles": {"lora_A": "adaptor"}}, "role pattern 'lora_A': Unknown role 'adaptor'"),
    ({"applies_when": {"module_types": ["x"]}}, r"unknown `applies_when` condition\(s\) \['module_types'\]"),
    ({"applies_when": {"module_names": "lora_A"}}, "is a list of non-empty strings"),
])
def test_malformed_overlays_are_refused_by_name(change, match):
    doc = {**overlay_for("peft.lora").to_mapping(), **change}
    with pytest.raises(ValueError, match=match):
        overlay_from_document(doc, "doc.yaml")
