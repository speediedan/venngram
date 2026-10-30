"""The module-tree drafter against the bundled maps: where it names a row, it must name the bundled module."""
import pytest

transformers = pytest.importorskip("transformers")

from test_maps import CONFIGS, tiny  # noqa: E402

from venngram.maps import component_map_for  # noqa: E402
from venngram.maps.drafter import draft_map, ModuleTreeSource  # noqa: E402
from venngram.sources import derive_map, register_map_source, SourceError, unregister  # noqa: E402

TEXT_ONLY = [a for a in sorted(CONFIGS) if a in ("LlamaForCausalLM", "Qwen3ForCausalLM", "Gemma2ForCausalLM",
                                                  "GPT2LMHeadModel", "OPTForCausalLM")]


@pytest.mark.parametrize("architecture", TEXT_ONLY)
def test_drafted_rows_agree_with_the_bundled_map(architecture):
    if not hasattr(transformers, architecture):
        pytest.skip(f"this transformers has no {architecture}")
    bundled = component_map_for(architecture).components
    draft = draft_map(tiny(architecture), architecture)
    named = {k: v for k, v in draft.components.items() if ".draft." not in k and not k.startswith("draft.")}
    shared = set(named) & set(bundled)
    assert {k: (named[k].module, bundled[k].module) for k in shared if named[k].module != bundled[k].module} == {}
    assert {"blocks.{i}", "blocks.{i}.attn", "embed"} <= shared and len(shared) >= 0.6 * len(bundled)
    assert draft.properties["draft"] is True
    assert set(draft.properties["unconfirmed"]) == set(draft.components) - set(named)


def test_the_drafter_is_a_map_source_and_names_its_unknowns():
    register_map_source(ModuleTreeSource())
    try:
        cmap, provenance = derive_map("module_tree", "Gemma3ForConditionalGeneration",
                                      model=tiny("Gemma3ForConditionalGeneration"))
        assert provenance.source == "module_tree" and cmap.properties["draft"] is True
        assert cmap.properties.get("unmapped_stacks")  # the vision tower's layers are listed, not guessed
        with pytest.raises(SourceError, match="needs the model"):
            derive_map("module_tree", "LlamaForCausalLM")
    finally:
        unregister("module_tree")
