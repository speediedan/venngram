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
"""The venngram-addressing skill's examples resolve as the skill says they do."""
import importlib.util
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[1] / ".agents" / "skills" / "venngram-addressing" / "references" / "examples.py"


@pytest.fixture(scope="module")
def results():
    spec = importlib.util.spec_from_file_location("venngram_skill_examples", EXAMPLES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.examples()


def adapters(*layers):
    return [f"layers.{i}.attn.q_proj.lora_{ab}.weight" for i in layers for ab in "AB"]


def test_layer_slices_and_set_algebra(results):
    assert results["top adapters"] == adapters(2, 3)
    assert results["adapters minus the top two blocks"] == adapters(0, 1)
    assert results["book @rest"] == adapters(0, 1)
    assert results["base weights of the last block"] == ["layers.3.attn.q_proj.base.weight",
                                                         "layers.3.mlp.up_proj.weight"]


def test_module_leaf_and_name_fields(results):
    assert len(results["every q_proj tensor"]) == 16
    assert results["MLP or head"][-2:] == ["head.weight", "head.bias"]
    assert results["odd blocks' MLPs"] == ["layers.1.mlp.up_proj.weight", "layers.1.mlp.up_proj.bias",
                                           "layers.3.mlp.up_proj.weight", "layers.3.mlp.up_proj.bias"]
    assert results["a regex, as an address"] == ["layers.0.attn.q_proj.lora_A.weight"]


def test_string_and_mapping_forms_and_refusals(results):
    assert results["mapping to string"] == "role:adapter - @top"
    assert "Layer 9 is outside a stack of 4 blocks" in results["refusal"]
