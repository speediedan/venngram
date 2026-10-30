"""The module-tree drafter: rule sources as data, rules mined from the registry, and model-assisted drafting."""
import pytest
import yaml

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from test_maps import CONFIGS, tiny  # noqa: E402

from venngram.maps import component_map_for, known_architectures  # noqa: E402
from venngram.maps.drafter import (bundled_drafting_rules, draft_map, draft_with_model, DraftingRules,  # noqa: E402
                                   map_document, mine_rules, ModuleTreeSource)
from venngram.sources import derive_map, register_map_source, SourceError, unregister  # noqa: E402

TEXT = ["GPT2LMHeadModel", "Gemma2ForCausalLM", "LlamaForCausalLM", "OPTForCausalLM", "Qwen3ForCausalLM"]


def leave_one_out(architecture):
    """Rules mined from every registered map except ``architecture``'s, then the bundled documents."""
    others = [component_map_for(a) for a in known_architectures() if a != architecture]
    return [mine_rules(others), *bundled_drafting_rules()]


@pytest.mark.parametrize("architecture", TEXT)
def test_a_draft_from_other_architectures_rules_agrees_with_the_bundled_map(architecture):
    if not hasattr(transformers, architecture):
        pytest.skip(f"this transformers has no {architecture}")
    bundled = component_map_for(architecture).components
    draft = draft_map(tiny(architecture), architecture, leave_one_out(architecture))
    named = {k: v for k, v in draft.components.items() if k not in draft.properties["unconfirmed"]}
    shared = set(named) & set(bundled)
    assert {k: (named[k].module, bundled[k].module) for k in shared if named[k].module != bundled[k].module} == {}
    assert {"blocks.{i}", "blocks.{i}.attn", "embed"} <= shared and len(shared) >= 0.8 * len(bundled)


def test_sandwich_norms_are_learned_from_the_registry_alone():
    rules = mine_rules()
    conditional = [r for r in rules.leaves["block"] if r.match == "post_attention_layernorm"]
    assert [(r.component, r.when_sibling is not None) for r in conditional] == [("ln1_post", True), ("ln2", False)]
    llama = draft_map(tiny("LlamaForCausalLM"), rules=[rules])
    gemma = draft_map(tiny("Gemma2ForCausalLM"), rules=[rules])
    assert llama.components["blocks.{i}.ln2"].module.endswith("post_attention_layernorm")
    assert gemma.components["blocks.{i}.ln1_post"].module.endswith("post_attention_layernorm")


def test_a_hybrid_stack_drafts_optional_rows():
    draft = draft_map(tiny("Qwen3_5ForConditionalGeneration"), rules=leave_one_out("Qwen3_5ForConditionalGeneration"))
    assert draft.components["blocks.{i}.attn"].optional and draft.components["blocks.{i}.mlp"].optional is False
    assert draft.properties["unmapped_stacks"] == ["model.visual.blocks"]  # listed, not guessed


def test_a_state_space_model_drafts_from_its_own_rule_source():
    config = transformers.MambaConfig(vocab_size=64, hidden_size=16, state_size=4, num_hidden_layers=2, expand=2,
                                      conv_kernel=2)
    with torch.device("meta"):
        model = transformers.MambaForCausalLM(config)
    draft = draft_map(model, rules=bundled_drafting_rules())
    assert {"blocks.{i}.mixer", "blocks.{i}.mixer.in", "blocks.{i}.mixer.out", "blocks.{i}.mixer.conv",
            "blocks.{i}.ln1", "embed", "ln_final", "unembed"} <= set(draft.components)
    assert draft.components["blocks.{i}.mixer"].kind == "ssm" and "ssm" in draft.kinds
    assert draft.properties["unconfirmed"] == []


def test_rule_documents_are_refused_by_name_when_malformed():
    with pytest.raises(SourceError, match="kind: drafting_rules"):
        DraftingRules.from_mapping({"kind": "rules", "schema_version": 1})
    with pytest.raises(SourceError, match="malformed drafting rule"):
        DraftingRules.from_mapping({"kind": "drafting_rules", "schema_version": 1,
                                    "leaves": {"block": [{"match": "(", "component": "x"}]}})


def test_model_assisted_drafting_is_checked_deterministically():
    model = tiny("LlamaForCausalLM")
    prompts = []

    def complete(prompt):
        prompts.append(prompt)
        doc = map_document(draft_map(model))
        doc["components"]["blocks.{i}.mystery"] = {"module": "model.layers.{i}.nowhere", "kind": "linear"}
        return "```yaml\n" + yaml.safe_dump(doc) + "```"

    cmap, problems = draft_with_model(model, complete)
    assert "Registry digest" in prompts[0] and "q_proj" in prompts[0] and cmap.properties["drafted_by"] == "model"
    assert problems and all(p.row == "blocks.{i}.mystery" for p in problems)  # one problem per layer
    with pytest.raises(SourceError, match="not a readable component map"):
        draft_with_model(model, lambda prompt: "components: [not, a, map]")


def test_the_drafter_is_a_map_source():
    register_map_source(ModuleTreeSource())
    try:
        cmap, provenance = derive_map("module_tree", "LlamaForCausalLM", model=tiny("LlamaForCausalLM"))
        assert provenance.source == "module_tree" and cmap.properties["draft"] is True
        with pytest.raises(SourceError, match="needs the model"):
            derive_map("module_tree", "LlamaForCausalLM")
    finally:
        unregister("module_tree")


def test_every_bundled_architecture_drafts():
    for architecture in CONFIGS:
        if hasattr(transformers, architecture):
            assert draft_map(tiny(architecture), architecture).components["blocks.{i}"].kind == "block"
