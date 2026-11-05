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
"""A bounded sample of the documents served by the pinned registry, checked against actual libraries.

The snapshot is generated from indexed references, not maintained as an implementation fixture catalogue. Adding a
registry document does not require a new test. The sample changes only to cover a new structural behavior.
"""
import pytest
from map_contracts import base_map_contract, overlay_contract
from test_maps import SAMPLED_ARCHITECTURES, tiny
from torch import nn

from venngram import Location
from venngram.maps import component_map_for, overlay_for


LOCATIONS = {
    "GPT2LMHeadModel": {"transformer.wte.weight": Location(component="embed"),
                        "transformer.h.1.attn.c_attn.weight": Location(layer=1, component="attn.qkv")},
    "LlamaForCausalLM": {"model.embed_tokens.weight": Location(component="embed"),
                         "model.layers.1.mlp.down_proj.weight": Location(layer=1, component="mlp.out")},
    "Qwen3_5ForConditionalGeneration": {
        "model.language_model.layers.1.self_attn.q_proj.weight": Location(layer=1, component="attn.q"),
        "model.language_model.layers.0.linear_attn.out_proj.weight": Location(layer=0, component="linear_attn.out")},
}


@pytest.mark.parametrize("architecture", SAMPLED_ARCHITECTURES)
def test_sampled_base_map_contract(architecture):
    base_map_contract(component_map_for(architecture), tiny(architecture), expected=LOCATIONS[architecture])


@pytest.mark.parametrize("family", ["lora", "ia3", "oft"])
def test_sampled_overlay_contract(family):
    """LoRA (including magnitude vectors), scaling vectors and orthogonal adapters exercise different containers.

    PEFT's own adapter_layer_names supplies the oracle, independently of the overlay's patterns. Two adapter names
    ensure role qualifiers cannot collapse to a constant. Buffers belong to the contract alongside parameters.
    """
    import peft
    import transformers

    config = transformers.LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2)
    targets = ["q_proj", "v_proj"]
    adapter = {
        "lora": lambda: peft.LoraConfig(r=2, target_modules=targets, use_dora=True),
        "ia3": lambda: peft.IA3Config(target_modules=targets, feedforward_modules=[]),
        "oft": lambda: peft.OFTConfig(r=2, oft_block_size=0, target_modules=targets),
    }[family]()
    model = peft.get_peft_model(transformers.LlamaForCausalLM(config), adapter, adapter_name="first")
    model.add_adapter("second", adapter)
    names = {id(t): n for n, t in [*model.named_parameters(), *model.named_buffers()]}
    expected = {}
    for module in model.modules():
        for layer_name in getattr(module, "adapter_layer_names", ()):
            for key, entry in getattr(module, layer_name).items():
                tensors = [*entry.parameters(), *entry.buffers()] if isinstance(entry, nn.Module) else (entry,)
                for tensor in tensors:
                    expected[names[id(tensor)]] = f"adapter:{key}"
    assert set(expected.values()) == {"adapter:first", "adapter:second"}
    overlay_contract(overlay_for(f"peft.{family}"), model, applies=True, expected=expected)
