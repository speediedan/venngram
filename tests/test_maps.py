import types

import pytest
import torch
from torch import nn

from venngram import Location, ParamSpace, Refusal, select
from venngram.maps import (check_map_against_model, component_map_for, ComponentMapModelMismatch, from_document,
                           known_architectures, map_for, MappedView, MapStructure, PEFT_LORA_RULES, register,
                           validate_map_against_model, VOCABULARY)
from venngram.maps.schema import _REGISTRY, ComponentKey
from venngram.maps.transformer_lens import from_transformer_lens
from venngram.tuning import BoundView

transformers = pytest.importorskip("transformers")

TEXT = dict(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=2, head_dim=4)


def _gemma3_multimodal():
    return transformers.Gemma3Config(
        text_config=dict(TEXT), vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                                   num_attention_heads=4, image_size=28, patch_size=14,
                                                   vision_use_head=False),  # as the released checkpoints
        mm_tokens_per_image=4)


CONFIGS = {
    "LlamaForCausalLM": lambda: transformers.LlamaConfig(**TEXT),
    "Qwen3ForCausalLM": lambda: transformers.Qwen3Config(**TEXT),
    "Gemma2ForCausalLM": lambda: transformers.Gemma2Config(**TEXT),
    "Gemma3ForCausalLM": lambda: transformers.Gemma3TextConfig(**TEXT),
    "Gemma3ForConditionalGeneration": _gemma3_multimodal,
    "GPT2LMHeadModel": lambda: transformers.GPT2Config(vocab_size=64, n_embd=16, n_layer=3, n_head=4, n_positions=32,
                                                       bos_token_id=0, eos_token_id=0),
    "OPTForCausalLM": lambda: transformers.OPTConfig(vocab_size=64, hidden_size=16, ffn_dim=32, num_hidden_layers=3,
                                                     num_attention_heads=4, word_embed_proj_dim=16,
                                                     max_position_embeddings=32),
}


def tiny(architecture: str, device: str = "meta") -> nn.Module:
    with torch.device(device):
        return getattr(transformers, architecture)(CONFIGS[architecture]())


def test_every_bundled_architecture_has_a_model_here():
    assert set(known_architectures()) == set(CONFIGS)


@pytest.mark.parametrize("architecture", sorted(CONFIGS))
def test_bundled_map_describes_its_model(architecture):
    model = tiny(architecture)
    cmap = map_for(model)
    assert cmap.architecture == architecture
    assert check_map_against_model(cmap, model) == []


@pytest.mark.parametrize("architecture", sorted(CONFIGS))
def test_every_parameter_is_located_at_a_component(architecture):
    model = tiny(architecture)
    structure = MapStructure(model, map_for(model))
    unlocated = [n for n, _ in model.named_parameters() if structure.locate(n).component is None]
    assert unlocated == []
    assert structure.depth() == 3


@pytest.mark.parametrize("architecture", sorted(CONFIGS))
def test_bundled_components_are_in_the_vocabulary(architecture):
    cmap = component_map_for(architecture)
    # a convention the bundled seeds follow so addresses port across them; a map may name any component it has
    outside = {cmap.key(k).component for k in cmap.components} - set(VOCABULARY) - {""}
    assert outside <= {"projector"}


def test_a_wrong_map_is_refused_naming_every_problem():
    model = tiny("LlamaForCausalLM")
    wrong = component_map_for("OPTForCausalLM")
    problems = check_map_against_model(wrong, model)
    assert any("addresses no layer" in str(p) for p in problems)
    with pytest.raises(ComponentMapModelMismatch, match="OPTForCausalLM"):
        validate_map_against_model(wrong, model)


def test_a_layer_predicate_is_checked_both_ways():
    model = tiny("LlamaForCausalLM")
    doc = {"schema_version": 2, "architecture": "X", "components": {
        "blocks.{i}": {"module": "model.layers.{i}", "kind": "block"},
        "blocks.{i}.attn": {"module": "model.layers.{i}.self_attn", "kind": "attn", "layers": "odd"}}}
    problems = check_map_against_model(from_document(doc), model)
    assert [(p.row, p.layer) for p in problems] == [("blocks.{i}.attn", 0), ("blocks.{i}.attn", 2)]


@pytest.mark.parametrize("doc, match", [
    ({"architecture": "X", "components": {}}, "integer `schema_version`"),
    ({"schema_version": 99, "architecture": "X", "components": {}}, "newer than this venngram"),
    ({"schema_version": 1, "architecture": "X", "components": {"embed": {"module": "e", "kind": "nope"}}},
     "unknown component kind"),
    ({"schema_version": 1, "architecture": "X", "components": {"embed": {"module": "e", "kind": "embed",
                                                                          "layers": "odd"}}}, "schema-2 key"),
    ({"schema_version": 1, "architecture": "X", "stacks": {"v": {"module": "v.{i}"}}, "components": {}},
     "schema-2 key"),
    ({"schema_version": 2, "architecture": "X", "components": {"vision.blocks.{i}": {"module": "v.{i}",
                                                                                     "kind": "block"}}},
     "does not declare"),
    ({"schema_version": 2, "architecture": "X", "facts": {}, "components": {}}, "renamed to `properties`"),
    ({"schema_version": 2, "architecture": "X", "properties": {"sandwich_norms": "yes"}, "components": {}},
     "must be bool"),
])
def test_malformed_documents_are_refused_by_name(doc, match):
    with pytest.raises(ValueError, match=match):
        from_document(doc)


def test_keys_split_into_stack_and_component():
    assert ComponentKey.parse("blocks.{i}", ()) == ComponentKey(None, True, "")
    assert ComponentKey.parse("vision.embed", {"vision"}) == ComponentKey("vision", False, "embed")
    assert ComponentKey.parse("pos_embed", {"vision"}) == ComponentKey(None, False, "pos_embed")


def test_a_base_model_inside_a_task_head_is_located_without_its_prefix():
    model = tiny("LlamaForCausalLM").model  # the LlamaModel, whose own paths lack `model.`
    structure = MapStructure(model, component_map_for("LlamaForCausalLM"))
    assert structure.prefix == "model"
    assert structure.locate("layers.2.self_attn.q_proj.weight") == Location(layer=2, component="attn.q")
    assert structure.locate("norm.weight") == Location(component="ln_final")
    with pytest.raises(KeyError, match="Pass `architecture`"):
        map_for(nn.Linear(2, 2))


def test_a_model_without_the_maps_blocks_is_refused_at_depth():
    structure = MapStructure(nn.Sequential(nn.Linear(2, 2)), component_map_for("LlamaForCausalLM"))
    assert isinstance(structure.depth(), Refusal) and "does not describe it" in structure.depth().reason
    assert structure.locate("0.weight") == Location()


def lora(model: nn.Module, targets=("q_proj", "v_proj", "down_proj", "fc2")) -> nn.Module:
    """PEFT-shaped adapters: `lora_A` / `lora_B` children under each target module."""
    for module in list(model.modules()):
        for target in targets:
            child = getattr(module, target, None)
            if isinstance(child, nn.Linear):
                child.add_module("lora_A", nn.ModuleDict({"default": nn.Linear(child.in_features, 2, bias=False)}))
                child.add_module("lora_B", nn.ModuleDict({"default": nn.Linear(2, child.out_features, bias=False)}))
    return model


def space(view) -> ParamSpace:
    bound = BoundView(view)
    return ParamSpace([n for n, _ in bound.root.named_parameters()], roles=bound.roles(), structure=bound.structure())


def test_components_mean_the_same_tensors_across_architectures():
    for architecture, mlp_out in (("LlamaForCausalLM", "mlp.down_proj"), ("OPTForCausalLM", "fc2")):
        model = lora(tiny(architecture, "cpu"))
        view = MappedView(model, role_rules=PEFT_LORA_RULES)
        s = space(view)
        top_q = select("role:adapter & component:attn.q & layers:-1:", s)
        assert len(top_q) == 2 and all(".2.self_attn.q_proj.lora_" in n for n in top_q)
        out = select("role:base & component:mlp.out", s)
        assert len(out) == 3 * (2 if architecture == "OPTForCausalLM" else 1)  # OPT's linears carry a bias
        assert all(mlp_out in n for n in out)
        assert set(select("component:attn.*", s)) >= set(select("role:adapter & component:attn.*", s))


def test_a_second_stack_is_addressed_only_by_name():
    model = tiny("Gemma3ForConditionalGeneration")
    s = ParamSpace([n for n, _ in model.named_parameters()], structure=MapStructure(model, map_for(model)))
    vision_q = select("stack:vision & component:attn.q", s)
    text_q = select("component:attn.q", s)
    assert vision_q and all("vision_tower" in n for n in vision_q)
    assert text_q and not set(text_q) & set(vision_q)
    assert len(select("stack:vision & layers:-1: & component:attn.q", s)) == 2  # weight and bias
    with pytest.raises(Exception, match="no block stack `audio`"):
        select("stack:audio & layers:0", s)


def test_a_registration_takes_precedence_over_a_bundled_map():
    bundled = component_map_for("LlamaForCausalLM")
    mine = from_document({"schema_version": 1, "architecture": "LlamaForCausalLM",
                          "components": {"embed": {"module": "model.embed_tokens", "kind": "embed"}}}, source="mine")
    register(mine)
    try:
        assert component_map_for("LlamaForCausalLM") is mine
    finally:
        _REGISTRY.pop("LlamaForCausalLM")
    assert component_map_for("LlamaForCausalLM") is bundled


def test_a_map_derives_from_a_transformer_lens_adapter():
    bridge = lambda cls, name=None, **subs: type(cls, (), {})() if name is None else _named(cls, name, subs)
    adapter = types.SimpleNamespace(component_mapping={
        "embed": bridge("EmbeddingBridge", "model.embed_tokens"),
        "rotary_emb": bridge("RotaryEmbeddingBridge", "model.rotary_emb"),
        "blocks": bridge("BlockBridge", "model.layers", ln1=bridge("RMSNormalizationBridge", "input_layernorm"),
                         attn=bridge("AttentionBridge", "self_attn", q=bridge("LinearBridge", "q_proj"))),
        "unembed": bridge("UnembeddingBridge", "lm_head"),
    })
    cmap = from_transformer_lens(adapter, "LlamaForCausalLM")
    assert {k: e.module for k, e in cmap.components.items()} == {
        "embed": "model.embed_tokens", "blocks.{i}": "model.layers.{i}",
        "blocks.{i}.ln1": "model.layers.{i}.input_layernorm", "blocks.{i}.attn": "model.layers.{i}.self_attn",
        "blocks.{i}.attn.q": "model.layers.{i}.self_attn.q_proj", "unembed": "lm_head"}
    bundled = component_map_for("LlamaForCausalLM").components
    assert all(bundled[k].module == e.module and bundled[k].kind == e.kind for k, e in cmap.components.items())
    with pytest.raises(ValueError, match="does not know"):
        from_transformer_lens(types.SimpleNamespace(component_mapping={"x": bridge("MysteryBridge", "x")}), "X")


def _named(cls: str, name: str, subs: dict):
    return type(cls, (), {"name": name, "submodules": subs})()
