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
r"""
Maps From TransformerLens
^^^^^^^^^^^^^^^^^^^^^^^^^

Derive a component map from a TransformerLens bridge ``ArchitectureAdapter``'s ``component_mapping``. TransformerLens
keeps one adapter per architecture, each naming the module every bridge component wraps, so this is an independent
source for the facts a bundled document carries, and the seed of the registry's maps. The adapter is read duck-typed:
venngram never imports TransformerLens.

.. warning:: This is an experimental API and may change in a future release.
"""
from typing import Any

from venngram.maps.schema import ComponentEntry, ComponentMap

#: Bridge classes -> component kinds, for the classes that wrap a module a map row names. A component without a module
#: of its own (a virtual q/k/v split) has no ``name`` and is skipped: a row must name a real module.
TL_KINDS: dict[str, str] = {
    "EmbeddingBridge": "embed",
    "PosEmbedBridge": "embed",
    "BlockBridge": "block",
    "NormalizationBridge": "norm",
    "RMSNormalizationBridge": "norm",
    "AttentionBridge": "attn",
    "JointQKVAttentionBridge": "attn",
    "PositionEmbeddingsAttentionBridge": "attn",
    "JointGateUpMLPBridge": "mlp",
    "MLPBridge": "mlp",
    "GatedMLPBridge": "mlp",
    "LinearBridge": "linear",
    "UnembeddingBridge": "unembed",
}

#: Bridge classes deliberately left out of a derived map: a rotary table holds no tensor a map addresses, and
#: TransformerLens bridges a vision tower as one opaque component. A class in neither table is refused by name, never
#: skipped, because a silently skipped class narrows every comparison built on the result.
TL_UNADDRESSED: frozenset[str] = frozenset({
    "RotaryEmbeddingBridge", "SiglipVisionEncoderBridge", "SiglipVisionEncoderLayerBridge", "VisionProjectionBridge",
    "GeneralizedComponent",
})


#: Bridge classes that group components without a module of their own (OPT's MLP, whose ``fc1`` and ``fc2`` sit on the
#: layer): no row, and their children resolve against the enclosing module.
TL_SYMBOLIC: frozenset[str] = frozenset({"SymbolicBridge"})


def from_transformer_lens(adapter: Any, architecture: str) -> ComponentMap:
    """A map from ``adapter.component_mapping``: the block list becomes ``blocks.{i}``, its children
    ``blocks.{i}.<name>``, and each row's kind follows its bridge class.

    Raises:
        ValueError: If a bridge class is in none of :data:`TL_KINDS`, :data:`TL_SYMBOLIC` and :data:`TL_UNADDRESSED`.
    """
    rows: dict[str, ComponentEntry] = {}

    def walk(path: str, comp: Any, module_prefix: str) -> None:
        cls_name = type(comp).__name__
        if cls_name in TL_UNADDRESSED:
            return
        if cls_name in TL_SYMBOLIC:
            for sub_name, sub in (getattr(comp, "submodules", None) or {}).items():
                walk(f"{path}.{sub_name}", sub, module_prefix)
            return
        kind = TL_KINDS.get(cls_name)
        if kind is None:
            raise ValueError(f"from_transformer_lens: {cls_name!r} at {path!r} is a bridge class this derivation does "
                             f"not know; add it to TL_KINDS with its kind, or to TL_UNADDRESSED with a reason")
        name = getattr(comp, "name", None)
        if not name:
            return
        if path == "blocks":
            module, key = f"{name}.{{i}}", "blocks.{i}"
        else:
            module = f"{module_prefix}.{name}" if module_prefix else name
            key = path.replace("blocks.", "blocks.{i}.", 1) if path.startswith("blocks.") else path
        rows[key] = ComponentEntry(module=module, kind=kind)
        for sub_name, sub in (getattr(comp, "submodules", None) or {}).items():
            walk(f"{path}.{sub_name}", sub, module)

    for top, comp in adapter.component_mapping.items():
        walk(top, comp, "")
    # no schema-2 key (declared kinds, stacks or layer predicates) is used, so this is a schema-1 document
    return ComponentMap(architecture=architecture, components=rows, source="transformer_lens", schema_version=1,
                        properties={"sandwich_norms": "blocks.{i}.ln2_post" in rows})


__all__ = ["from_transformer_lens", "TL_KINDS", "TL_SYMBOLIC", "TL_UNADDRESSED"]
