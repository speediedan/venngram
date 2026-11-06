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
"""Contract tests describe arbitrary module trees and libraries, independently of registry membership."""
import pytest
import torch
from torch import nn
from map_contracts import base_map_contract, overlay_contract

from venngram import Location
from venngram.maps import from_document, overlay_from_document


class LibraryLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4)
        self.delta = nn.ModuleDict({k: nn.Linear(4, 2, bias=False) for k in ("alpha", "beta")})
        self.delta["alpha"].register_buffer("index", torch.ones(1))
        self.gain = nn.ParameterDict({k: nn.Parameter(torch.ones(4)) for k in ("alpha", "beta")})
        self.register_buffer("scratch", torch.ones(1))


def document():
    return {"schema_version": 2, "architecture": "Synthetic", "components": {
        "trunk": {"module": "base", "kind": "linear"}, "head": {"module": "head", "kind": "linear"}}}


def model():
    return nn.ModuleDict({"base": nn.Linear(4, 4), "head": nn.Linear(4, 2)})


def test_base_map_contract():
    base_map_contract(from_document(document()), model(), expected={
        "base.weight": Location(component="trunk"), "head.bias": Location(component="head")})


@pytest.mark.parametrize("defect", ["missing", "uncovered", "duplicate", "swapped"])
def test_base_contract_rejects_a_broken_document(defect):
    doc = document()
    if defect == "missing":
        doc["components"]["trunk"]["module"] = "missing"
    elif defect == "uncovered":
        del doc["components"]["head"]
    elif defect == "duplicate":
        doc["components"]["duplicate"] = dict(doc["components"]["trunk"])
    else:
        doc["components"]["trunk"]["module"], doc["components"]["head"]["module"] = "head", "base"
    with pytest.raises(AssertionError):
        base_map_contract(from_document(doc), model(), expected={"base.weight": Location(component="trunk")})


@pytest.mark.parametrize("condition", ["module_names", "module_classes"])
def test_overlay_contract_for_a_library_with_keyed_parameters_and_modules(condition):
    layer = LibraryLayer()
    applies = ["delta"] if condition == "module_names" else [f"{LibraryLayer.__module__}.LibraryLayer"]
    overlay = overlay_from_document({"schema_version": 5, "overlay": "synthetic", "applies_when": {condition: applies},
                                     "roles": {"delta": "adapter:{key}", "gain": "intervention:{key}"}})
    expected = {**{f"delta.{k}.weight": f"adapter:{k}" for k in ("alpha", "beta")},
                **{f"gain.{k}": f"intervention:{k}" for k in ("alpha", "beta")}}
    expected["delta.alpha.index"] = "adapter:alpha"
    overlay_contract(overlay, layer, applies=True, expected=expected)
    overlay_contract(overlay, nn.Linear(4, 4), applies=False, expected={})
    with pytest.raises(AssertionError):
        overlay_contract(overlay, layer, applies=True, expected={**expected, "base.weight": "adapter:alpha"})
    # Every keyed entry must preserve its own qualifier, not merely the correct set of tensors.
    with pytest.raises(AssertionError):
        overlay_contract(overlay, layer, applies=True, expected={n: "adapter:wrong" for n in expected})
