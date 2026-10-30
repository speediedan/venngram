import copy
import dataclasses

import pytest
import torch
import yaml
from test_maps import CONFIGS, tiny
from test_maps_any_architecture import HYBRID, Hybrid

from venngram.maps import component_map_for, from_document
from venngram.maps.schema import ComponentEntry
from venngram.maps.verification import main, tiny_config, verify

pytest.importorskip("transformers")


@pytest.mark.parametrize("architecture", sorted(CONFIGS))
def test_every_seed_map_verifies_from_its_configuration(architecture):
    report = verify(CONFIGS[architecture](), architecture=architecture)
    assert report.status == "verified", str(report)
    record = report.record()
    assert record["status"] == "verified" and set(record["checks"]) == {"exists", "partition", "shapes", "order"}
    assert record["config"]["source"] == "config" and len(record["config"]["sha256"]) == 64


def mutated(architecture: str, **rows: str | None):
    """The seed map with some rows pointed at other modules (or removed, for ``None``)."""
    cmap = component_map_for(architecture)
    components = dict(cmap.components)
    for key, module in rows.items():
        key = key.replace("__", ".").replace("blocks.", "blocks.{i}.", 1)
        if module is None:
            components.pop(key)
        else:
            components[key] = ComponentEntry(module, components[key].kind)
    return dataclasses.replace(cmap, components=components)


LLAMA_LAYER = "model.layers.{i}"


def test_swapped_mlp_in_and_out_fail_shapes_and_order():
    cmap = mutated("LlamaForCausalLM", blocks__mlp__in=f"{LLAMA_LAYER}.mlp.down_proj",
                   blocks__mlp__out=f"{LLAMA_LAYER}.mlp.up_proj")
    report = verify(CONFIGS["LlamaForCausalLM"](), cmap)
    assert report.check("exists").status == "passed"  # both modules exist, which is why existence is not enough
    assert report.check("shapes").status == "failed" and report.check("order").status == "failed"


def test_swapped_norms_pass_shapes_and_fail_order():
    cmap = mutated("LlamaForCausalLM", blocks__ln1=f"{LLAMA_LAYER}.post_attention_layernorm",
                   blocks__ln2=f"{LLAMA_LAYER}.input_layernorm")
    report = verify(CONFIGS["LlamaForCausalLM"](), cmap)
    assert report.check("shapes").status == "passed"  # both norms are hidden-wide
    order = report.check("order")
    assert order.status == "failed" and any("blocks.{i}.ln1" in str(p) for p in order.problems)


def test_swapped_q_and_k_fail_shapes_under_grouped_query_attention():
    cmap = mutated("Qwen3ForCausalLM", blocks__attn__q=f"{LLAMA_LAYER}.self_attn.k_proj",
                   blocks__attn__k=f"{LLAMA_LAYER}.self_attn.q_proj")
    shapes = verify(CONFIGS["Qwen3ForCausalLM"](), cmap, order=False).check("shapes")
    assert shapes.status == "failed"
    assert {p.row for p in shapes.problems} == {"blocks.{i}.attn.q", "blocks.{i}.attn.k"}


def test_a_dropped_row_leaves_parameters_unclaimed_and_a_duplicate_is_named():
    report = verify(CONFIGS["LlamaForCausalLM"](), mutated("LlamaForCausalLM", blocks__ln2=None), order=False)
    partition = report.check("partition")
    assert partition.status == "failed"
    assert {p.row for p in partition.problems} == {f"model.layers.{i}.post_attention_layernorm.weight"
                                                   for i in range(3)}
    duplicate = mutated("LlamaForCausalLM", blocks__attn__k=f"{LLAMA_LAYER}.self_attn.q_proj")
    dup = verify(CONFIGS["LlamaForCausalLM"](), duplicate, order=False).check("partition")
    assert dup.status == "failed" and any("already names" in p.reason for p in dup.problems)


def test_a_model_is_verified_as_it_is_and_a_custom_map_is_partial_where_rules_do_not_reach():
    model, cmap = Hybrid(), from_document(yaml.safe_load(HYBRID))
    report = verify(model, cmap)  # no configuration: no widths to read and no small model to build
    assert report.status == "partial", str(report)
    assert [c.status for c in report.checks] == ["passed", "passed", "skipped", "skipped"]

    def run(m):
        x = m.tok(torch.tensor([[1, 2, 3]]))
        for i, layer in enumerate(m.trunk):
            x = layer.norm(x)
            if i % 2:
                a = layer.attention
                x = a.wo(a.wq(x) + a.wk(x) + a.wv(x))
            else:
                x = layer.mixer.gate(layer.mixer.conv(x.transpose(1, 2)).transpose(1, 2))
        return m.readout(x)
    order = verify(model, cmap, inputs=run).check("order")
    assert order.status == "passed" and order.note.startswith("4 ordered pairs")  # two per attention layer


def test_tiny_config_keeps_grouped_queries_and_shrinks_nested_sections():
    small = tiny_config(CONFIGS["Gemma3ForConditionalGeneration"]())
    text = small.text_config
    assert (text.num_attention_heads, text.num_key_value_heads, text.head_dim, text.hidden_size) == (4, 2, 8, 32)
    assert small.vision_config.hidden_size == 32
    assert tiny_config(CONFIGS["LlamaForCausalLM"](), layers=2).num_hidden_layers == 2


def test_the_command_line_writes_a_record(tmp_path, monkeypatch):
    import transformers

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda _id: CONFIGS["OPTForCausalLM"]())
    record = tmp_path / "record.yaml"
    assert main(["facebook/opt-tiny", "--architecture", "OPTForCausalLM", "--record", str(record)]) == 0
    written = yaml.safe_load(record.read_text())
    assert written["status"] == "verified" and written["config"]["source"] == "hf:facebook/opt-tiny"
    assert main(["facebook/opt-tiny", "--architecture", "NoSuchForCausalLM"]) == 2


def test_a_model_on_the_meta_device_and_no_config_skips_order_without_failing():
    report = verify(tiny("LlamaForCausalLM"), order=True)
    assert report.check("order").status in ("passed", "skipped")
    assert report.passed


def test_a_run_is_compared_against_a_stored_record():
    config = CONFIGS["LlamaForCausalLM"]()
    record = verify(config, architecture="LlamaForCausalLM").record()
    assert verify(config, architecture="LlamaForCausalLM", against=record).check("against_record").status == "passed"
    reordered = copy.deepcopy(record)
    reordered["checks"]["order"]["observed_order"]["blocks"].reverse()
    check = verify(config, architecture="LlamaForCausalLM", against=reordered).check("against_record")
    assert check.status == "failed" and "stack 'blocks'" in str(check.problems[0])
    other = verify(config, architecture="LlamaForCausalLM", against={**record, "architecture": "OPTForCausalLM"})
    assert other.check("against_record").status == "failed"
