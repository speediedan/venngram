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

from venngram.maps import (detect_role_rules, known_role_rules, register_role_rules, resolve_roles, role_rules_for,
                           role_rules_from_document, RoleRules, unregister_role_rules)
from venngram.tuning import TunableViewError

PEFT_LORA = {"lora_A": "adapter", "lora_B": "adapter", "lora_embedding_A": "adapter", "lora_embedding_B": "adapter",
             "lora_magnitude_vector": "adapter"}


class Scaler(nn.Module):
    """An adapter library's module, recognized by its class rather than by its tensors' names."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d))


class SubScaler(Scaler):
    pass


SCALER = f"{Scaler.__module__}.{Scaler.__qualname__}"


@pytest.fixture
def scaler_rules():
    rules = RoleRules(name="test.scaler", rules={"gain": "intervention:scale"}, module_classes=(SCALER,))
    register_role_rules(rules)
    yield rules
    unregister_role_rules("test.scaler")


def test_the_seed_registry_serves_peft_lora_by_name():
    rules = role_rules_for("peft.lora")
    assert dict(rules.rules) == PEFT_LORA and "peft.tuners.lora.layer.LoraLayer" in rules.module_classes
    assert "peft.lora" in known_role_rules()
    with pytest.raises(KeyError, match="no role rules 'nope.dora'"):
        role_rules_for("nope.dora")


def test_rules_are_detected_from_the_adapters_a_model_carries(model):
    assert [r.name for r in detect_role_rules(model)] == ["peft.lora"]
    roles = resolve_roles(model)
    assert roles and set(roles.values()) == {"adapter"}
    assert all(".lora_A." in n or ".lora_B." in n for n in roles)
    assert detect_role_rules(nn.Linear(2, 2)) == [] and resolve_roles(nn.Linear(2, 2)) == {}


def test_a_rule_set_applies_to_a_subclass_of_a_class_it_names(model, scaler_rules):
    model.layers[0].add_module("scale", SubScaler(4))
    assert {r.name for r in detect_role_rules(model)} == {"peft.lora", "test.scaler"}
    assert resolve_roles(model)["layers.0.scale.gain"] == "intervention:scale"
    assert not scaler_rules.applies_to(nn.Sequential(nn.Linear(2, 2)))


def test_a_named_or_inline_rule_set_replaces_detection(model, scaler_rules):
    model.layers[0].add_module("scale", Scaler(4))
    assert set(resolve_roles(model, "test.scaler")) == {"layers.0.scale.gain"}
    assert resolve_roles(model, scaler_rules) == resolve_roles(model, "test.scaler")
    assert set(resolve_roles(model, {"lora_B": "adapter:up"}).values()) == {"adapter:up"}
    assert resolve_roles(model, {}) == {}


def test_a_registration_shadows_an_indexed_rule_set_of_the_same_name(model):
    register_role_rules(RoleRules(name="peft.lora", rules={"lora_A": "adapter:down"}, module_names=("lora_A",)))
    try:
        assert set(resolve_roles(model).values()) == {"adapter:down"}
    finally:
        unregister_role_rules("peft.lora")
    assert set(resolve_roles(model).values()) == {"adapter"}


def test_detected_rule_sets_that_disagree_are_refused_by_name(model):
    register_role_rules(RoleRules(name="other.lora", rules={"lora_B": "intervention"}, module_names=("lora_B",)))
    try:
        with pytest.raises(TunableViewError, match="'other.lora' and 'peft.lora' both apply"):
            resolve_roles(model)
        assert set(resolve_roles(model, "peft.lora").values()) == {"adapter"}
    finally:
        unregister_role_rules("other.lora")


def test_a_document_round_trips():
    rules = role_rules_for("peft.lora")
    assert role_rules_from_document(rules.to_mapping()) == rules


@pytest.mark.parametrize("change, match", [
    ({"kind": "component_map"}, "expected `kind: role_rules`"),
    ({"schema_version": 2}, "schema_version 2 is not supported"),
    ({"name": ""}, "needs a `name`"),
    ({"rules": {}}, "must not be empty"),
    ({"rules": {"lora_A": "adaptor"}}, "rule 'lora_A': Unknown role 'adaptor'"),
    ({"applies_when": {"module_types": ["x"]}}, r"unknown `applies_when` condition\(s\) \['module_types'\]"),
    ({"applies_when": {"module_names": "lora_A"}}, "is a list of non-empty strings"),
])
def test_malformed_documents_are_refused_by_name(change, match):
    doc = {**role_rules_for("peft.lora").to_mapping(), **change}
    with pytest.raises(ValueError, match=match):
        role_rules_from_document(doc, "doc.yaml")
