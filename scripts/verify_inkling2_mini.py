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
"""Generate the Inkling 2 Mini candidate map and verify its forward coordinates."""

import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from venngram.maps import MapStructure, from_document, verify
from venngram.maps.drafter import map_document
from venngram.maps.verification import check_order, with_layouts


ARCHITECTURE = "Inkling2LoopForConditionalGeneration"
TEXT = "model.language_model"
LAYER = TEXT + ".layers.{i}"


def generate(model):
    """Place the checkpoint's text components and model-owned vision weights."""
    cfg = model.config.text_config
    depth, start, stop = cfg.num_hidden_layers, cfg.loop_layer_index_start, cfg.loop_layer_index_end
    passes = cfg.loop_num_passes
    components = {}
    sites = {}

    def row(component, module, kind, *, layers=None, optional=False):
        entry = {"module": module, "kind": kind}
        if layers is not None:
            entry["layers"] = list(layers)
        if optional:
            entry["optional"] = True
        components[component] = entry

    def layer(component, module, kind, **kwargs):
        row(f"blocks.{{i}}.{component}", f"{LAYER}.{module}", kind, **kwargs)

    def site(point, module, *, hook="output", path=(), layers=None, first=1):
        entry = {"module": module, hook: list(path)}
        if layers is not None:
            entry["layers"] = f"{layers.start}:{layers.stop}" if isinstance(layers, range) else list(layers)
        if first > 1:
            entry["first_pass"] = first
        sites[point] = entry

    full = [i for i, kind in enumerate(cfg.layer_types) if kind != "hybrid_sliding"]
    sliding = [i for i in range(depth) if i not in full]
    core = range(start, stop)
    row("blocks.{i}", LAYER, "activation")
    row("embed", f"{TEXT}.embed_tokens", "embed")
    row("ln_final", f"{TEXT}.norm", "norm")
    row("state.final_mix", f"{TEXT}.final_input", "tuple_activation")
    row("message.bank", f"{TEXT}.message_input", "activation")
    row("unembed", "lm_head", "unembed")
    row("logit_softcap", "logit_softcap", "activation")
    row("vision", "model.vision_tower", "activation")
    layer("attn", "self_attn", "tuple_activation")
    layer("mlp", "mlp", "activation")
    layer("mlp.router", "mlp.gate", "tuple_activation")
    layer("mlp.experts", "mlp.experts", "activation")
    layer("ln1", "input_layernorm", "norm", layers=sliding)
    for name in ("q", "k", "v"):
        layer(f"attn.{name}.input_norm", f"{name}_input_layernorm", "norm", layers=full)
        layer(f"attn.{name}", f"self_attn.{name}_proj.linear", "linear")
    for name in ("k", "v"):
        layer(f"attn.{name}.conv", f"self_attn.{name}_proj.sconv", "activation")
    layer("attn.o", "self_attn.o_proj.linear", "linear")
    layer("attn.output_gate", "self_attn.o_proj.gate_proj", "linear")
    layer("attn.sink", "self_attn.sink_proj", "linear")
    layer("attn.relative", "self_attn.relative_position_bias", "activation")
    layer("ln2", "post_attention_layernorm", "norm")
    layer("attn.conv", "post_attention_convolution", "activation")
    layer("mlp.conv", "post_mlp_convolution", "activation")
    layer("attn.mix", "attention_input", "tuple_activation", optional=True)
    layer("mlp.mix", "mlp_input", "tuple_activation")

    all_layers = range(depth)
    site("state.enter", LAYER, hook="input", path=[0], layers=all_layers)
    site("state.active", LAYER, layers=all_layers)
    site("state.completed", LAYER, hook="input", path=[1], layers=all_layers)
    for index, boundary in enumerate(range(0, depth, cfg.n_layers_per_attn_res_block)):
        if boundary + 1 < depth:
            site(f"state.completed.{index}", LAYER, hook="input", path=[1, {"axis": -2, "index": index}],
                 layers=range(boundary + 1, depth))
    site("attn.mix", f"{LAYER}.attention_input", path=[0], layers=range(1, depth))
    for name, index in (("k", 1), ("v", 2)):
        if layers := [i for i in full if i != 0]:
            site(f"attn.mix.{name}", f"{LAYER}.attention_input", path=[index], layers=layers)
    site("mlp.mix", f"{LAYER}.mlp_input", path=[0], layers=all_layers)
    site("state.final_mix", f"{TEXT}.final_input", path=[0])
    site("logit_softcap", "logit_softcap")
    for index, name in enumerate(("q", "k", "v", "mlp")):
        layer(f"message.gate.{name}", f"{name}_message_gate", "activation", layers=core)
        site(f"message.gated.{name}", f"{LAYER}.{name}_message_gate", layers=core, first=2)
        site(f"message.raw.{name}", f"{TEXT}.message_input", path=[{"axis": -2, "index": index}],
             layers=core, first=2)
    for index in range((stop - start) // cfg.n_layers_per_attn_res_block - 1):
        site(f"message.bank.{index}", f"{TEXT}.message_input", hook="input",
             path=[0, {"axis": -2, "index": index}], layers=core, first=2)
    last = (stop - start) // cfg.n_layers_per_attn_res_block - 1
    site(f"message.bank.{last}", f"{TEXT}.message_input", hook="input", path=[1], layers=core, first=2)

    doc = {
        "schema_version": 7,
        "architecture": ARCHITECTURE,
        "properties": {"sublayers": []},
        "kinds": {"activation": {"output": "tensor"}, "tuple_activation": {"output": "tuple0"}},
        "components": components,
        "execution": {"depth": depth, "loops": [{"start": start, "stop": stop, "passes": passes}], "sites": sites},
        "verification": {
            "config": {"unpadded_vocab": ["unpadded_vocab_size"]},
            "widths": {"unembed": {"out": "unpadded_vocab"}},
            "order": {"norm_then_attention": [["ln1", "attn.q.input_norm", "attn.k.input_norm",
                                               "attn.v.input_norm"], ["attn.q", "attn.k", "attn.v"]]},
        },
    }
    return with_layouts(doc, model)


def reduced_config(config):
    """Reduce tensor widths and expert count while preserving all 48 text visits and their module classes."""
    cfg = deepcopy(config)
    text, vision = cfg.text_config, cfg.vision_config
    text.hidden_size = vision.text_hidden_size = 16
    text.head_dim = text.swa_head_dim = 4
    text.num_attention_heads = text.swa_num_attention_heads = 4
    text.num_key_value_heads = text.swa_num_key_value_heads = 2
    text.vocab_size, text.unpadded_vocab_size = 64, 61
    text.pad_token_id = 0
    text.d_rel, text.rel_extent, text.sliding_window_size = 4, 16, 16
    text.moe_intermediate_size, text.n_routed_experts = 16, 4
    text.num_experts_per_tok, text.loop_gate_rank = 2, 4
    vision.hidden_size, vision.intermediate_size, vision.num_attention_heads = 16, 32, 4
    vision.num_hidden_layers, vision.projection_hidden_dim = 1, 32
    cfg.image_token_id = 63
    return cfg


def verify_forward(model, cmap):
    """Run the real model implementation at reduced widths and compare every planned hook with the trace."""
    execution = cmap.execution
    assert execution is not None
    grouped = defaultdict(list)
    for site in execution.sites:
        grouped[site.module].append(site)
    layer_calls = []
    params = defaultdict(list)
    observed = defaultdict(int)

    def capture(path, sites):
        def hook(module, args, output):
            i = observed[path]
            observed[path] += 1
            for site in sites:
                if site.call_index == i:
                    value = site.read(args if site.hook == "input" else output, token=-1)
                    assert isinstance(value, torch.Tensor) and torch.isfinite(value).all(), (site, value)
                    if site.point == "state.active" and site.visit is not None:
                        layer_calls.append((site.visit.layer, i + 1))
                        params[site.visit.layer].append(id(module.self_attn.q_proj.linear.weight))

        return hook

    hooks = [model.get_submodule(path).register_forward_hook(capture(path, sites)) for path, sites in grouped.items()]
    try:
        def forward(module):
            with torch.inference_mode():
                logits = module(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=False).logits
            assert tuple(logits.shape) == (1, 4, module.config.text_config.unpadded_vocab_size), logits.shape

        order = check_order(cmap, model, forward)
        assert order.status == "passed", order.problems
    finally:
        for handle in hooks:
            handle.remove()
    assert layer_calls == [(v.layer, v.pass_number) for v in execution.visits], layer_calls
    assert all(len(set(ids)) == 1 for ids in params.values()), params
    assert len({ids[0] for ids in params.values()}) == execution.depth
    for path, sites in grouped.items():
        assert observed[path] == 1 + max(s.call_index for s in sites), (path, observed[path])
    print(f"{len(layer_calls)} layer visits, {len(params)} distinct query weights, "
          f"{len(execution.sites)} activation site calls checked across {len(observed)} modules")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="local directory holding the pinned configuration")
    parser.add_argument("--out", type=Path, help="write a checked candidate map")
    args = parser.parse_args()
    from transformers import AutoConfig, Inkling2LoopForConditionalGeneration

    torch.set_num_threads(1)
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    with torch.device("meta"):
        full = Inkling2LoopForConditionalGeneration(config)
    yaml_text = yaml.safe_dump(generate(full), sort_keys=False)
    cmap = from_document(yaml.safe_load(yaml_text))
    report = verify(full, cmap, order=False)
    print(report)
    assert report.passed
    reloaded = from_document(map_document(cmap))
    assert reloaded.components == cmap.components and reloaded.execution == cmap.execution
    assert MapStructure(full, cmap).unit_block("model.language_model.layers.6.self_attn.q_proj.linear.weight") == 128
    small = Inkling2LoopForConditionalGeneration(reduced_config(config)).eval()
    verify_forward(small, cmap)
    if args.out is not None:
        args.out.write_text(yaml_text, encoding="utf-8")


if __name__ == "__main__":
    main()
