import copy
import dataclasses

import pytest
import torch
import yaml
from torch import nn
from test_maps import CONFIGS, SAMPLED_ARCHITECTURES, tiny
from test_maps_any_architecture import HYBRID, Hybrid

from venngram import AddressBook, ParamSpace
from venngram.address import parse, to_mapping, to_string
from venngram.errors import AddressError, AddressUnresolvable
from venngram.maps import component_map_for, from_document
from venngram.maps.schema import ComponentEntry
from venngram.maps.drafter import map_document
from venngram.maps.structure import MapStructure
from venngram.maps.verification import check_unit_blocks, main, tiny_config, verify, with_layouts
from venngram.resolve import gradient_masks, gradient_selections, GradientSelection

pytest.importorskip("transformers")


@pytest.mark.parametrize("architecture", SAMPLED_ARCHITECTURES)
def test_sampled_registry_maps_verify_from_their_configuration(architecture):
    report = verify(CONFIGS[architecture](), architecture=architecture)
    assert report.status == "verified", str(report)
    record = report.record()
    assert record["status"] == "verified" and set(record["checks"]) == {"exists", "partition", "shapes", "layout",
                                                                                "order"}
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
    assert [c.status for c in report.checks] == ["passed", "passed", "skipped", "passed", "skipped"]

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


def seed_document(architecture):
    from pathlib import Path

    return yaml.safe_load(Path(component_map_for(architecture).source).read_text(encoding="utf-8"))


@pytest.mark.parametrize("architecture, neurons", [
    # (mlp.in weight's unit dimension, mlp.out weight's): GPT-2's Conv1D stores [in, out], nn.Linear [out, in]
    ("GPT2LMHeadModel", (1, 0)),
    ("LlamaForCausalLM", (0, 1)),
])
def test_the_builder_records_measured_layouts_and_a_tensor_names_its_unit_dimension(architecture, neurons):
    model = tiny(architecture)
    doc = with_layouts(seed_document(architecture), model)
    assert doc["schema_version"] == 6 and doc["components"]["embed"]["layout"] == "in_out"
    assert "layout" not in doc["components"]["blocks.{i}.ln1"]  # a vector has one axis, so nothing to record
    cmap = from_document(doc)
    written = map_document(cmap)
    assert written["schema_version"] == 6 and from_document(written).components == cmap.components
    layout = verify(model, cmap, order=False).check("layout")
    assert layout.status == "passed" and layout.note.endswith("layouts checked")
    structure = MapStructure(model, cmap)
    params = dict(model.named_parameters())
    up, down = (next(n for n in params if cmap.module_for(c, 1).replace("{i}", "1") + ".weight" == n)
                for c in ("mlp.in", "mlp.out"))
    assert (structure.unit_dim(up, 2), structure.unit_dim(down, 2)) == neurons
    assert params[up].shape[neurons[0]] == params[down].shape[neurons[1]]  # both run along the MLP's hidden width
    embed = next(n for n in params if n.startswith(cmap.module_for("embed", None)))
    assert structure.unit_dim(embed, 2) == 0  # a row per token
    bias = down.replace(".weight", ".bias")
    if bias in params:  # mlp.out's bias lies on the residual side, not along its neurons
        assert "whose units are its `in` side" in str(structure.unit_dim(bias, 1))


def test_a_unit_dimension_is_refused_by_name_where_it_is_not_defined():
    model = tiny("LlamaForCausalLM")
    doc = seed_document("LlamaForCausalLM")
    for row in doc["components"].values():  # a map that records no layout (the published one now does)
        row.pop("layout", None)
    unrecorded = MapStructure(model, from_document(doc))
    assert "records no `layout` for `mlp.out`" in str(unrecorded.unit_dim("model.layers.0.mlp.down_proj.weight", 2))
    structure = MapStructure(model, from_document(with_layouts(seed_document("LlamaForCausalLM"), model)))
    assert "only a row's own weight and bias" in str(
        structure.unit_dim("model.layers.0.mlp.down_proj.lora_A.default.weight", 2))
    assert "has 3 dimensions" in str(structure.unit_dim("model.layers.0.mlp.down_proj.weight", 3))
    assert "located at no component" in str(structure.unit_dim("elsewhere.weight", 2))


def test_a_recorded_layout_is_checked_and_needs_schema_5():
    model = tiny("LlamaForCausalLM")
    doc = with_layouts(seed_document("LlamaForCausalLM"), model)
    doc["components"]["blocks.{i}.mlp.out"]["layout"] = "in_out"
    doc["components"]["blocks.{i}.ln1"]["layout"] = "out_in"
    layout = verify(model, from_document(doc), order=False).check("layout")
    assert layout.status == "failed"
    assert any("records layout in_out, its weight is out_in" in str(p) for p in layout.problems)
    assert any("no weight whose sides can be told apart" in str(p) for p in layout.problems)
    with pytest.raises(ValueError, match="`layout` is a schema-5 key"):
        from_document({**doc, "schema_version": 4})
    with pytest.raises(ValueError, match="one of \\['out_in', 'in_out'\\]"):
        from_document({**doc, "components": {"embed": {"module": "model.embed_tokens", "kind": "embed",
                                                       "layout": "rows"}}})


@pytest.mark.parametrize("architecture, axis", [("LlamaForCausalLM", 1), ("GPT2LMHeadModel", 0)])
def test_head_addresses_expand_along_measured_projection_axes(architecture, axis):
    model = tiny(architecture)
    seed = seed_document(architecture)
    cmap = from_document(with_layouts(seed, model))
    shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
    space = ParamSpace(list(shapes), structure=MapStructure(model, cmap))
    projection = f"{cmap.module_for('attn.o', 1)}.weight"
    target = "component:attn.o & layers:1 & leaf:weight & of:grad"
    heads = f"{target} & units:{{heads:[1,3]}}"
    coords = f"{target} & axis:unit & units:[4,5,6,7,12,13,14,15]"
    expected = {projection: GradientSelection(axis, (4, 5, 6, 7, 12, 13, 14, 15))}
    assert gradient_selections(heads, space, shapes=shapes) == expected
    assert gradient_selections(coords, space, shapes=shapes) == expected
    assert gradient_selections(f"{target} & axis:unit & units:6:10", space, shapes=shapes) == {
        projection: GradientSelection(axis, (6, 7, 8, 9))}
    with pytest.raises(AddressUnresolvable, match="4 heads"):
        gradient_selections(f"{target} & units:{{heads:[4]}}", space, shapes=shapes)
    with pytest.raises(AddressUnresolvable, match="gradient_selections"):
        gradient_masks(heads, space, widths={n: s[0] for n, s in shapes.items()})
    legacy = ParamSpace(list(shapes), structure=MapStructure(model, from_document(seed)))
    with pytest.raises(AddressUnresolvable, match="unit_block"):
        gradient_selections(heads, legacy, shapes=shapes)
    assert gradient_selections(coords, legacy, shapes=shapes) == expected
    assert verify(model, cmap, order=False).check("unit_block").status == "passed"
    if architecture == "GPT2LMHeadModel":
        assert cmap.unit_block_of("attn.qkv", 1) is None
    else:
        assert cmap.unit_block_of("attn.k", 1) == cmap.unit_block_of("attn.v", 1) == 4
        key = f"{cmap.module_for('attn.k', 1)}.weight"
        assert gradient_selections(heads.replace("attn.o", "attn.k").replace("[1,3]", "[1]"), space,
                                   shapes=shapes) == {key: GradientSelection(0, (4, 5, 6, 7))}


def test_unit_blocks_validate_width_and_grouping_independently():
    model = tiny("LlamaForCausalLM")
    doc = with_layouts(seed_document("LlamaForCausalLM"), model)
    with pytest.raises(ValueError, match="unit_block.*schema-6"):
        from_document({**doc, "schema_version": 5})
    key_row = doc["components"]["blocks.{i}.attn.k"]
    for invalid in (False, 0, -2, 4.0, None):
        key_row["unit_block"] = invalid
        with pytest.raises(ValueError, match="unit_block.*positive integer"):
            from_document(doc)
    key_row["unit_block"] = 3
    check = check_unit_blocks(from_document(doc), model)
    assert check.status == "failed" and any("width 8 cannot be divided" in str(p) for p in check.problems)
    key_row["unit_block"] = 4
    model.model.layers[0].self_attn.k_proj = nn.Linear(16, 12, bias=False, device="meta")
    model.model.layers[0].self_attn.v_proj = nn.Linear(16, 12, bias=False, device="meta")
    check = check_unit_blocks(from_document(doc), model)
    assert check.status == "failed" and any("cannot be grouped" in str(p) for p in check.problems)


def test_builder_can_measure_head_sizes_from_projection_widths_and_head_counts():
    model = tiny("LlamaForCausalLM")
    model.config.head_dim = None
    for layer in model.model.layers:
        del layer.self_attn.head_dim
    cmap = from_document(with_layouts(seed_document("LlamaForCausalLM"), model))
    assert [cmap.unit_block_of(f"attn.{p}", 0) for p in ("q", "k", "v", "o")] == [4, 4, 4, 4]


@pytest.mark.parametrize("selection", [[1, 3], "-2:"])
def test_head_qualifier_round_trips_and_requires_book_schema_11(selection):
    expr = parse({"component": "attn.o", "units": {"heads": selection}, "of": "grad"})
    assert parse(to_string(expr)) == parse(to_mapping(expr)) == expr
    doc = {"schema_version": 10, "sets": {"heads": to_mapping(expr)}}
    with pytest.raises(AddressError, match="schema 11"):
        AddressBook.from_mapping(doc)
    assert AddressBook.from_mapping({**doc, "schema_version": 11}).sets() == {"heads": expr}


def test_a_recorded_head_size_this_checkpoint_does_not_share_is_refused_not_misread():
    import transformers

    from test_maps import TEXT

    recorded = from_document(with_layouts(seed_document("LlamaForCausalLM"), tiny("LlamaForCausalLM")))
    assert recorded.unit_block_of("attn.o", 1) == 4
    with torch.device("meta"):  # the same architecture with heads of 2: its widths still divide into 4
        other = transformers.LlamaForCausalLM(transformers.LlamaConfig(**{**TEXT, "head_dim": 2,
                                                                          "num_attention_heads": 8}))
    shapes = {n: tuple(p.shape) for n, p in other.named_parameters()}
    heads = "component:attn.o & layers:1 & leaf:weight & of:grad & units:{heads:[1]}"
    space = ParamSpace(list(shapes), structure=MapStructure(other, recorded))
    with pytest.raises(AddressUnresolvable, match="records `unit_block` 4 for `attn.o`.*measures heads of 2"):
        gradient_selections(heads, space, shapes=shapes)
    same = ParamSpace(list(shapes), structure=MapStructure(other, from_document(with_layouts(
        seed_document("LlamaForCausalLM"), other))))  # measured on this checkpoint, it resolves
    projection = f"{recorded.module_for('attn.o', 1)}.weight"
    assert gradient_selections(heads, same, shapes=shapes) == {projection: GradientSelection(1, (2, 3))}


def test_a_head_size_measured_through_a_wrapper_is_checked_too():
    class Wrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.model = inner

    import transformers

    from test_maps import TEXT

    recorded = from_document(with_layouts(seed_document("LlamaForCausalLM"), tiny("LlamaForCausalLM")))
    with torch.device("meta"):
        other = Wrapper(transformers.LlamaForCausalLM(transformers.LlamaConfig(**{**TEXT, "head_dim": 2,
                                                                                  "num_attention_heads": 8})))
    structure = MapStructure(other, recorded, mount="model")
    name = f"model.{recorded.module_for('attn.o', 1)}.weight"
    with pytest.raises(AddressError, match="measures heads of 2"):
        structure.unit_block(name)
    assert MapStructure(tiny("LlamaForCausalLM"), recorded).unit_block(name.removeprefix("model.")) == 4


def test_a_gated_query_projection_measures_whole_heads_of_query_and_gate():
    model = tiny("LlamaForCausalLM")
    for layer in model.model.layers:  # each query head carries its gate beside it, as a gated attention's does
        layer.self_attn.q_proj = nn.Linear(16, 2 * 4 * 4, bias=False, device="meta")
    cmap = from_document(with_layouts(seed_document("LlamaForCausalLM"), model))
    assert [cmap.unit_block_of(f"attn.{p}", 0) for p in ("q", "k", "v", "o")] == [8, 4, 4, 4]
    assert check_unit_blocks(cmap, model).status == "passed"  # 4 query heads of 8 over 2 key/value heads of 4


def test_the_view_measures_a_head_size_from_the_sources_the_builder_reads():
    model = tiny("LlamaForCausalLM")
    for layer in model.model.layers:  # the attention module states only head_dim; a distant config states a count
        del layer.self_attn.config
    model.config.num_attention_heads = 2
    cmap = from_document(with_layouts(seed_document("LlamaForCausalLM"), model))
    name = f"{cmap.module_for('attn.o', 1)}.weight"
    assert cmap.unit_block_of("attn.o", 1) == 4 == MapStructure(model, cmap).unit_block(name)


def test_a_packed_quantized_weight_is_not_measured_so_the_recorded_size_stands():
    model = tiny("LlamaForCausalLM")
    cmap = from_document(with_layouts(seed_document("LlamaForCausalLM"), model))
    o_proj = model.model.layers[1].self_attn.o_proj
    o_proj.weight = nn.Parameter(torch.empty(o_proj.in_features * o_proj.out_features // 2, 1, dtype=torch.uint8,
                                             device="meta"), requires_grad=False)  # a 4-bit weight, two per byte
    assert MapStructure(model, cmap).unit_block(f"{cmap.module_for('attn.o', 1)}.weight") == 4


def test_output_heads_read_value_heads_of_the_same_width():
    model = tiny("LlamaForCausalLM")
    doc = with_layouts(seed_document("LlamaForCausalLM"), model)
    for p in ("q", "o"):
        doc["components"][f"blocks.{{i}}.attn.{p}"]["unit_block"] = 8  # 2 heads of 8 over 2 key/value heads of 4
    check = check_unit_blocks(from_document(doc), model)
    assert check.status == "failed" and any("which it reads" in str(p) for p in check.problems)
