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

A root may also *add* leading path: a wrapper module carrying the mapped model as a submodule (a framework's module
holding a Hugging Face model under ``model.``, with steering vectors or spliced dictionaries registered beside it). The
map is then **mounted** at that submodule, named with ``mount`` or found as the one outermost descendant the map
describes (:func:`find_mounts`). The wrapper's own tensors stay outside every structure field.

.. warning:: This is an experimental API and may change in a future release.
"""
from collections.abc import Collection
from typing import Any

from torch import nn

from venngram.errors import Refusal
from venngram.maps.check import stack_length, submodule
from venngram.maps.schema import component_map_for, ComponentMap, PRIMARY_STACK
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
        try:
            return component_map_for(name, strict=strict)
        except KeyError:
            continue
    raise KeyError(f"no component map for {tried} registered in this process or indexed by the registries. Pass "
                   "`architecture` to name the map meant (a base model inside a task head is mapped under the head's "
                   "class), register one, or draft one (`venngram.maps.drafter.draft_map`).")


def _block_prefix(root: nn.Module, template: str, inner: list[str]) -> str | None:
    segments = template.split(".")
    for k in range(len(segments) - 1):
        prefix, block = ".".join(segments[:k]), ".".join(segments[k:])
        if submodule(root, block.replace("{i}", "0")) is None:
            continue
        if not inner or any(submodule(root, m[len(prefix) + 1 if prefix else 0:].replace("{i}", "0")) is not None
                            for m in inner):
            return prefix
    return None


def detect_prefix(root: nn.Module, cmap: ComponentMap) -> str | None:
    """The leading module path ``root`` lacks relative to the map (``""`` when ``root`` is the mapped class itself,
    ``"model"`` for the ``LlamaModel`` inside a ``LlamaForCausalLM``); ``None`` when the map does not describe ``root``.

    It is found from the first block stack that resolves (the primary one, then the declared ones in name order). A
    suffix of a stack's template counts only when it keeps the block list's own name (a bare index matches any
    ``nn.Sequential``) and the first block it finds holds a module some row of that stack names. A map with no block
    stack at all (a model without a block list) is placed from its other rows.
    """
    templates = [(stack, cmap.block_template(stack)) for stack in cmap.stack_names()]
    templates = [(stack, t) for stack, t in templates if t is not None]
    for stack, template in templates:
        head = "blocks.{i}." if stack == PRIMARY_STACK else f"{stack}.blocks.{{i}}."
        inner = [e.module for k, e in cmap.components.items() if k.startswith(head) and e.module.startswith(
            f"{template}.")]
        prefix = _block_prefix(root, template, inner)
        if prefix is not None:
            return prefix
    if templates:
        return None
    for entry in cmap.components.values():
        segments = entry.module.split(".")
        for k in range(len(segments)):
            if submodule(root, ".".join(segments[k:])) is not None:
                return ".".join(segments[:k])
    return None


def find_mounts(root: nn.Module, architectures: Collection[str]) -> list[str]:
    """The paths of ``root``'s outermost descendants mapped under one of ``architectures`` (by
    :func:`architecture_of`), shallowest first; a mapped module inside another is not listed.

    >>> wrapper = nn.Module()
    >>> wrapper.add_module("model", nn.Sequential(nn.Linear(2, 2)))
    >>> find_mounts(wrapper, {"Sequential"})
    ['model']
    """
    wanted, found = set(architectures), []
    for path, module in root.named_modules():
        if not path or any(path.startswith(f"{m}.") for m in found):
            continue
        if wanted.intersection(architecture_of(module)):
            found.append(path)
    return sorted(found, key=lambda p: (p.count("."), p))


def _mount_of(root: nn.Module, cmap: ComponentMap) -> tuple[str | None, Refusal | None]:
    """The one outermost descendant ``cmap`` describes (by structure, as :func:`detect_prefix` reads it, so a bare
    ``LlamaModel`` under a wrapper is found as well as a ``LlamaForCausalLM``), or why there is none to mount it at."""
    mounts: list[str] = []
    for path, module in root.named_modules():
        if path and not any(path.startswith(f"{m}.") for m in mounts) and detect_prefix(module, cmap) is not None:
            mounts.append(path)
    if len(mounts) == 1:
        return mounts[0], None
    if mounts:
        return None, Refusal(f"{type(root).__name__} carries {len(mounts)} modules the {cmap.architecture} component "
                             f"map describes ({', '.join(map(repr, mounts))}); name the one meant with `mount=`")
    return None, None


class MapStructure:
    """Places ``root``'s tensors from ``cmap``.

    Args:
        root: The module whose tensor names are located.
        cmap: Its component map.
        prefix: The leading module path the mapped model lacks relative to the map; found from the blocks when
            ``None``.
        mount: The path of the submodule of ``root`` the map describes, for a root that wraps the mapped model
            (``"model"``). ``None`` places the map at ``root`` when it describes ``root``, and otherwise at the one
            outermost descendant it describes (:func:`find_mounts`); ``""`` places it at ``root`` only. Names stay
            relative to ``root``, and ``root``'s tensors outside the mount are located nowhere.
    """

    vocabulary = True

    def __init__(self, root: nn.Module, cmap: ComponentMap, prefix: str | None = None,
                 mount: str | None = None) -> None:
        self.cmap = cmap
        self.refusal: Refusal | None = None
        self._depths: dict[str, int] = {}
        self._table: dict[str, Location] = {}
        self.mount = mount
        target = root if not mount else submodule(root, mount)
        if target is None:
            self.prefix = None
            self.refusal = Refusal(f"{type(root).__name__} has no module {mount!r} to mount the {cmap.architecture} "
                                   "component map at")
            return
        self.prefix = detect_prefix(target, cmap) if prefix is None else prefix
        if self.prefix is None and mount is None:
            found, ambiguous = _mount_of(root, cmap)
            if ambiguous is not None:
                self.refusal = ambiguous
                return
            if found is not None:
                self.mount, target = found, root.get_submodule(found)
                self.prefix = detect_prefix(target, cmap)
        if self.prefix is None:
            names = [t for t in map(cmap.block_template, cmap.stack_names()) if t] or list(cmap.components)
            self.refusal = Refusal(f"{type(root).__name__} has none of the modules the {cmap.architecture} component "
                                   f"map places from ({', '.join(map(repr, names))}), so the map does not describe it")
            return
        self._build(target, cmap)
        if self.mount:
            # Located names are matched after wrapper segments are stripped, so the mount path is stripped too.
            lead = strip_wrappers(f"{self.mount}.")
            if lead:
                self._table = {f"{lead}{path}" if path else lead[:-1]: loc for path, loc in self._table.items()}

    def _build(self, root: nn.Module, cmap: ComponentMap) -> None:
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
                concrete = path.replace("{i}", str(layer))
                if entry.optional and submodule(root, concrete) is None:
                    continue
                if entry.layers is None or entry.layers.covers(layer):
                    self._table[concrete] = Location(k.stack, layer, k.component or None)

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
            named = tuple(f"stack:{s}" for s in self._depths if s != PRIMARY_STACK)
            what = f"block stack `{stack}`" if stack else "primary block stack (`blocks.{i}`)"
            return Refusal(f"the {self.cmap.architecture} component map has no {what}", named)
        return depth

    def placed(self) -> dict[str, Location]:
        """Every module path (relative to the root) this structure places, with its location."""
        return dict(self._table)

    def components(self) -> list[str]:
        """The components this structure can locate, in map order."""
        return list(dict.fromkeys(loc.component for loc in self._table.values() if loc.component))


def map_structure(model: nn.Module, architecture: str | None = None, **kwargs: Any) -> MapStructure:
    """A :class:`MapStructure` over ``model`` from its component map (:func:`map_for`)."""
    return MapStructure(model, map_for(model, architecture), **kwargs)


__all__ = ["architecture_of", "detect_prefix", "find_mounts", "map_for", "map_structure", "MapStructure"]
