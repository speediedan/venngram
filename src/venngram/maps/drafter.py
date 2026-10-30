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
Drafting a Map From a Module Tree
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A map source that needs no library's knowledge of the architecture: :class:`ModuleTreeSource` drafts a component map
from a model's own module tree. It finds the block stack (the longest ``ModuleList`` of one block class), names the
modules of one block and those outside it with the vocabulary where common module names make the part clear
(``q_proj`` -> ``attn.q``, ``down_proj`` -> ``mlp.out``, ``lm_head`` -> ``unembed``), and drafts every other module
with parameters as a ``draft.<path>`` row of the kind its class suggests.

A draft is a starting point, not a verified map: its ``properties`` say ``draft: true`` and list the ``unconfirmed``
rows, which a person or an agent renames (or removes) before the map is verified (:func:`venngram.maps.verify`) and
published. Other ``ModuleList`` stacks (a vision tower, say) are listed under ``unmapped_stacks`` rather than guessed.

.. warning:: This is an experimental API and may change in a future release.
"""
import re
from typing import Any

from torch import nn

from venngram.maps.schema import ComponentEntry, ComponentMap
from venngram.sources import SourceError

_ATTN = {"self_attn", "attn", "attention", "self_attention"}
_MLP = {"mlp", "feed_forward", "ffn", "mlp_block"}

#: (context, module-name pattern, component). Context is where the module sits: ``attn``, ``mlp``, ``block`` (directly
#: in a block) or ``outer`` (outside the block stack). The first match wins.
NAME_RULES: tuple[tuple[str, str, str], ...] = (
    ("attn", r"q_proj|query|wq", "attn.q"),
    ("attn", r"k_proj|key|wk", "attn.k"),
    ("attn", r"v_proj|value|wv", "attn.v"),
    ("attn", r"c_attn|qkv|query_key_value|wqkv", "attn.qkv"),
    ("attn", r"o_proj|out_proj|dense|wo|c_proj", "attn.o"),
    ("attn", r"q_norm", "attn.q_norm"),
    ("attn", r"k_norm", "attn.k_norm"),
    ("mlp", r"gate_proj|w1|gate", "mlp.gate"),
    ("mlp", r"up_proj|fc1|c_fc|w3|dense_h_to_4h|fc_in", "mlp.in"),
    ("mlp", r"down_proj|fc2|c_proj|w2|dense_4h_to_h|fc_out", "mlp.out"),
    ("block", r"input_layernorm|ln_1|self_attn_layer_norm|attention_norm|norm1", "ln1"),
    ("block", r"pre_feedforward_layernorm", "ln2"),
    ("block", r"post_feedforward_layernorm", "ln2_post"),
    ("block", r"post_attention_layernorm|ln_2|final_layer_norm|ffn_norm|norm2", "ln2"),
    ("block", r"fc1", "mlp.in"),
    ("block", r"fc2", "mlp.out"),
    ("outer", r"embed_tokens|wte|word_embeddings|embed_in", "embed"),
    ("outer", r"wpe|embed_positions|position_embeddings", "pos_embed"),
    ("outer", r"norm|ln_f|final_layernorm|final_layer_norm", "ln_final"),
    ("outer", r"lm_head|embed_out", "unembed"),
)


def _kind(module: nn.Module, component: str) -> str:
    if component == "unembed":
        return "unembed"
    name = type(module).__name__
    if "Norm" in name:
        return "norm"
    if "Embedding" in name:
        return "embed"
    return "linear"


def _block_stack(model: nn.Module) -> tuple[str, nn.ModuleList]:
    stacks = [(path, m) for path, m in model.named_modules() if isinstance(m, nn.ModuleList) and len(m) >= 2
              and len({type(c) for c in m}) == 1]
    if not stacks:
        raise SourceError(f"{type(model).__name__}: no ModuleList of one block class to draft blocks from.")
    return max(stacks, key=lambda s: (len(s[1]), -s[0].count(".")))


def _name(context: str, leaf: str, taken: set[str]) -> str | None:
    for ctx, pattern, component in NAME_RULES:
        if ctx == context and re.fullmatch(pattern, leaf) and component not in taken:
            return component
    return None


def draft_map(model: nn.Module, architecture: str | None = None) -> ComponentMap:
    """A draft component map from ``model``'s module tree (see the module docstring)."""
    architecture = architecture or type(model).__name__
    blocks_path, blocks = _block_stack(model)
    rows: dict[str, ComponentEntry] = {"blocks.{i}": ComponentEntry(f"{blocks_path}.{{i}}", "block")}
    unconfirmed: list[str] = []
    taken: set[str] = set()
    # sandwich norms (a norm before and after each sublayer): the post-attention norm is then ln1_post, not ln2
    sandwich = any(rel.split(".")[-1] == "pre_feedforward_layernorm" for rel, _ in blocks[0].named_modules())
    for rel, module in blocks[0].named_modules():
        if not rel:
            continue
        parts = rel.split(".")
        leaf = parts[-1]
        if leaf in _ATTN and "attn" not in taken and any(True for _ in module.children()):
            rows["blocks.{i}.attn"] = ComponentEntry(f"{blocks_path}.{{i}}.{rel}", "attn")
            taken.add("attn")
            continue
        if leaf in _MLP and "mlp" not in taken and any(True for _ in module.children()):
            rows["blocks.{i}.mlp"] = ComponentEntry(f"{blocks_path}.{{i}}.{rel}", "mlp")
            taken.add("mlp")
            continue
        if not any(True for _ in module.parameters(recurse=False)):
            continue
        context = "attn" if set(parts[:-1]) & _ATTN else ("mlp" if set(parts[:-1]) & _MLP else "block")
        component = "ln1_post" if sandwich and leaf == "post_attention_layernorm" else _name(context, leaf, taken)
        if component is None:
            component = f"draft.{rel}"
            unconfirmed.append(f"blocks.{{i}}.{component}")
        taken.add(component)
        rows[f"blocks.{{i}}.{component}"] = ComponentEntry(f"{blocks_path}.{{i}}.{rel}", _kind(module, component))
    others = [p for p, m in model.named_modules() if isinstance(m, nn.ModuleList) and p != blocks_path]
    outer_taken: set[str] = set()
    for path, module in model.named_modules():
        if not path or path == blocks_path or path.startswith(f"{blocks_path}.") \
                or any(path == o or path.startswith(f"{o}.") for o in others):
            continue
        if not any(True for _ in module.parameters(recurse=False)):
            continue
        component = _name("outer", path.split(".")[-1], outer_taken)
        if component is None:
            component = f"draft.{path}"
            unconfirmed.append(component)
        outer_taken.add(component)
        rows[component] = ComponentEntry(path, _kind(module, component))
    unmapped = sorted(o for o in others if not any(o.startswith(f"{p}.") for p in others if p != o))
    properties: dict[str, Any] = {"draft": True, "unconfirmed": unconfirmed}
    if unmapped:
        properties["unmapped_stacks"] = unmapped
    return ComponentMap(architecture=architecture, components=rows, source="module_tree", schema_version=1,
                        properties=properties)


class ModuleTreeSource:
    """:func:`draft_map` as a :class:`~venngram.sources.MapSource`. It needs the model (a meta-device one will do)."""

    name = "module_tree"
    version = "1"
    unexpressed = ("named stacks other than the block stack", "the vocabulary names of unconventionally named modules")

    def architectures(self) -> tuple[str, ...]:
        return ()  # any: it reads whatever model it is given

    def derive(self, architecture: str, *, config: Any = None, model: Any = None) -> ComponentMap:
        if model is None:
            raise SourceError("module_tree: drafting needs the model (`model=`); a meta-device instance is enough.")
        return draft_map(model, architecture)


__all__ = ["draft_map", "ModuleTreeSource", "NAME_RULES"]
