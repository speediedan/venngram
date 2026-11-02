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
from venngram.tuning import TunableViewError

PEFT_LORA = {"lora_A": "adapter", "lora_B": "adapter", "lora_embedding_A": "adapter", "lora_embedding_B": "adapter",
             "lora_magnitude_vector": "adapter"}


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


@pytest.mark.parametrize("change, match", [
    ({"schema_version": 3}, "needs schema_version 4"),
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
