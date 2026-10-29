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
Map-Backed Structure
^^^^^^^^^^^^^^^^^^^^

A vocabulary structure provider built from a component map: each tensor is placed at the
:class:`~venngram.resolve.Location` (stack, layer, component) of the deepest map row whose module owns it. A tensor
under a row's module belongs to that row whatever lies between, so an adapter PEFT attaches under ``q_proj``
(``...q_proj.lora_A.default.weight``) is located as ``attn.q``, exactly as the weight it adapts.

The map's module paths are written from the architecture class (``LlamaForCausalLM``). A provider over a submodule of
it (the ``LlamaModel`` inside) drops the leading segments that root does not have, found from where the blocks are;
rows outside that root (``lm_head``) then locate nothing.

.. warning:: This is an experimental API and may change in a future release.
"""
from typing import Any

from torch import nn

from venngram.errors import Refusal
from venngram.maps.check import stack_length, submodule
from venngram.maps.schema import component_map_for, ComponentMap, known_architectures, PRIMARY_STACK
from venngram.resolve import Location, strip_wrappers


def architecture_of(model: nn.Module) -> list[str]:
    """The architecture names a model may be mapped under: its class name, then its ``config.architectures``."""
    names = [type(model).__name__]
    config = getattr(model, "config", None)
    names += [a for a in (getattr(config, "architectures", None) or ()) if isinstance(a, str) and a not in names]
    return names


def map_for(model: nn.Module, architecture: str | None = None, *, strict: bool = False) -> ComponentMap:
    """The component map for ``model``: the one for ``architecture`` when given, else the first of
    :func:`architecture_of` that has one.

    Raises:
        KeyError: If none has a map, naming the names tried and the known architectures.
    """
    if architecture is not None:
        return component_map_for(architecture, strict=strict)
    tried = architecture_of(model)
    for name in tried:
        if name in known_architectures():
            return component_map_for(name, strict=strict)
    raise KeyError(f"no component map for {tried}; known: {known_architectures()}. Pass `architecture` to name the map "
                   "meant (a base model inside a task head is mapped under the head's class), or register one.")


def detect_prefix(root: nn.Module, cmap: ComponentMap) -> str | None:
    """The leading module path ``root`` lacks relative to the map (``""`` when ``root`` is the mapped class itself,
    ``"model"`` for the ``LlamaModel`` inside a ``LlamaForCausalLM``), found from the primary stack's blocks; ``None``
    when ``root`` has them under no suffix of the template.

    A suffix counts only when it keeps the block list's own name (a bare index matches any ``nn.Sequential``) and the
    first block it finds holds a module some block-relative row names.
    """
    template = cmap.block_template()
    if template is None:
        return None
    segments = template.split(".")
    inner = [e.module for k, e in cmap.components.items() if k.startswith("blocks.{i}.") and e.module.startswith(
        f"{template}.")]
    for k in range(len(segments) - 1):
        prefix, block = ".".join(segments[:k]), ".".join(segments[k:])
        if submodule(root, block.replace("{i}", "0")) is None:
            continue
        if not inner or any(submodule(root, m[len(prefix) + 1 if prefix else 0:].replace("{i}", "0")) is not None
                            for m in inner):
            return prefix
    return None


class MapStructure:
    """Places ``root``'s tensors from ``cmap``.

    Args:
        root: The module whose tensor names are located.
        cmap: Its component map.
        prefix: The leading module path ``root`` lacks relative to the map; found from the blocks when ``None``.
    """

    vocabulary = True

    def __init__(self, root: nn.Module, cmap: ComponentMap, prefix: str | None = None) -> None:
        self.cmap = cmap
        self.prefix = detect_prefix(root, cmap) if prefix is None else prefix
        self.refusal: Refusal | None = None
        self._depths: dict[str, int] = {}
        self._table: dict[str, Location] = {}
        if self.prefix is None:
            self.refusal = Refusal(f"{type(root).__name__} has none of the blocks the {cmap.architecture} component "
                                   f"map names ({cmap.block_template()!r}), so the map does not describe it")
            return
        for stack in cmap.stack_names():
            template = self._relative(cmap.block_template(stack))
            if template is not None:
                self._depths[stack] = stack_length(root, template)
        for key, entry in cmap.components.items():
            path, k = self._relative(entry.module), cmap.key(key)
            if path is None:
                continue
            if "{i}" not in path:
                self._table[path] = Location(stack=k.stack, component=k.component)
                continue
            for layer in range(self._depths.get(k.stack or PRIMARY_STACK, 0)):
                if entry.layers is None or entry.layers.covers(layer):
                    self._table[path.replace("{i}", str(layer))] = Location(k.stack, layer, k.component or None)

    def _relative(self, path: str | None) -> str | None:
        if path is None or not self.prefix:
            return path
        return path[len(self.prefix) + 1:] if path.startswith(f"{self.prefix}.") else None

    def locate(self, name: str) -> Location:
        """The location of the deepest map row owning the tensor ``name`` (an empty one when no row does)."""
        owner = strip_wrappers(name).rpartition(".")[0]
        while True:
            found = self._table.get(owner)
            if found is not None:
                return found
            if not owner:
                return Location()
            owner = owner.rpartition(".")[0]

    def depth(self, stack: str | None = None) -> int | Refusal:
        """The number of blocks in ``stack`` (the primary one when ``None``), or why it is unknown."""
        if self.refusal is not None:
            return self.refusal
        depth = self._depths.get(stack or PRIMARY_STACK)
        if depth is None:
            return Refusal(f"the {self.cmap.architecture} component map has no block stack `{stack}`",
                           tuple(s for s in self._depths if s != PRIMARY_STACK))
        return depth

    def components(self) -> list[str]:
        """The components this structure can locate, in map order."""
        return list(dict.fromkeys(loc.component for loc in self._table.values() if loc.component))


def map_structure(model: nn.Module, architecture: str | None = None, **kwargs: Any) -> MapStructure:
    """A :class:`MapStructure` over ``model`` from its component map (:func:`map_for`)."""
    return MapStructure(model, map_for(model, architecture), **kwargs)


__all__ = ["architecture_of", "detect_prefix", "map_for", "map_structure", "MapStructure"]
