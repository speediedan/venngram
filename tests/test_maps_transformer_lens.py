"""A cross-check of the bundled seed maps against TransformerLens's own per-architecture component mapping.

TransformerLens is one independent source for these particular architectures, not a constraint on what a map may
express (``test_maps_any_architecture.py`` maps models no library adapts).

A map checked only against the model it describes can still name the wrong module for a component (a swapped `in` and
`out`, say) as long as both modules exist. TransformerLens maintains its adapters independently, so every bundled row it
also derives must agree with it, and a bundled row it has no counterpart for is allowed only where the table below says
why. Skipped when TransformerLens is not installed: venngram never depends on it.
"""
import pytest
from test_maps import CONFIGS, tiny

from venngram.maps import component_map_for
from venngram.maps.transformer_lens import from_transformer_lens

pytest.importorskip("transformer_lens")

def _without_counterpart(architecture: str) -> set[str]:
    """TransformerLens bridges a vision tower as one opaque component, and Qwen3.5's linear attention as one
    component without parts, so those rows are venngram's alone."""
    return {k for k in component_map_for(architecture).components
            if k.startswith("vision.") or k == "projector" or ".linear_attn." in k}


def _adapter(architecture: str):
    """TransformerLens's adapter for ``architecture``, selected from the tiny config alone (no weights, no Hub)."""
    from transformer_lens.config import TransformerBridgeConfig
    from transformer_lens.factories.architecture_adapter_factory import ArchitectureAdapterFactory
    from transformer_lens.model_bridge.sources.transformers import map_default_transformer_lens_config

    hf_config = tiny(architecture).config
    cfg = TransformerBridgeConfig.from_dict(dict(map_default_transformer_lens_config(hf_config).__dict__))
    cfg.architecture = architecture
    for nested in ("text_config", "vision_config"):
        if hasattr(hf_config, nested):
            setattr(cfg, nested, getattr(hf_config, nested))
    return ArchitectureAdapterFactory.select_architecture_adapter(cfg)


@pytest.mark.parametrize("architecture", sorted(CONFIGS))
def test_every_bundled_row_agrees_with_transformer_lens(architecture):
    import transformers

    if not hasattr(transformers, architecture):
        pytest.skip(f"this transformers ({transformers.__version__}) has no {architecture}")
    bundled = component_map_for(architecture).components
    derived = from_transformer_lens(_adapter(architecture), architecture).components
    mismatches = {k: (bundled[k], derived[k]) for k in set(bundled) & set(derived)
                  if (bundled[k].module, bundled[k].kind) != (derived[k].module, derived[k].kind)}
    assert not mismatches
    assert set(bundled) - set(derived) == _without_counterpart(architecture)
