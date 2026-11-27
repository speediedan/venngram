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

A tensor's **unit axis** (:meth:`MapStructure.unit_dim`) is the dimension of a row's own weight or bias that runs
along its component's units: the side :data:`~venngram.maps.schema.UNIT_SIDES` names, through the row's recorded
``layout``. Only a row's own tensors have one; an adapter under the row's module does not.

The map's module paths are written from the architecture class (``LlamaForCausalLM``). A provider over a submodule of
it (the ``LlamaModel`` inside) drops the leading segments that root does not have, found from where the blocks are;
rows outside that root (``lm_head``) then locate nothing.

A root may also *add* leading path: a wrapper module carrying the mapped model as a submodule (a framework's module
holding a Hugging Face model under ``model.``, with steering vectors or spliced dictionaries registered beside it). The
map is then **mounted** at that submodule, named with ``mount`` or found as the one outermost descendant the map
describes (:func:`find_mounts`). The wrapper's own tensors stay outside every structure field, unless they are
**attached**: a module registered beside the mapped model and spliced into it by a hook (a latent model, a trainable
intervention) is placed with ``attached={"latent_models.sae3": "model.transformer.h.3"}`` at the module it is attached
to, exactly as if it were nested under that module. Only the caller knows where a hook splices a module, so the
placement is an argument of the structure rather than a row of a published map.

.. warning:: This is an experimental API and may change in a future release.
"""
from collections.abc import Collection, Mapping, Sequence
from typing import Any

import torch
from torch import nn

from venngram.errors import AddressError, Refusal
from venngram.maps.check import stack_length, submodule
from venngram.maps.schema import component_map_for, ComponentMap, PRIMARY_STACK, UNIT_SIDES
from venngram.resolve import Location, strip_wrappers
from venngram.tuning.errors import TunableViewError


def architecture_of(model: nn.Module) -> list[str]:
    """The architecture names a model may be mapped under: its class name, then its ``config.architectures``."""
    names = [type(model).__name__]
    config = getattr(model, "config", None)
    names += [a for a in (getattr(config, "architectures", None) or ()) if isinstance(a, str) and a not in names]
    return names


def unit_width(module: Any, component: str, layout: str | None) -> int | None:
    """The width of ``module``'s weight along ``component``'s unit side, given its recorded ``layout``."""
    side, weight = UNIT_SIDES.get(component), getattr(module, "weight", None)
    if side is None or layout is None or not isinstance(weight, torch.Tensor) or weight.ndim != 2 \
            or not weight.is_floating_point():  # a packed quantized weight's shape is not the projection's
        return None
    return int(weight.shape[0 if layout.startswith(side) else 1])


def head_size(component: str, sources: Sequence[Any], width: int) -> int | None:
    """The width of one whole head of a projection of ``width`` units: ``width`` over the first head count among
    ``sources`` (the projection, its attention module, their configurations; key and value heads for ``attn.k`` and
    ``attn.v``), else the first ``head_dim`` among them, or ``None`` when neither is stated.

    The head count comes first because a head's block on a projection need not be ``head_dim`` wide: a gated
    attention's query projection carries each head's query and gate side by side, so its whole head is twice
    ``head_dim``.
    """
    names = (("num_key_value_heads", "num_kv_heads") if component in ("attn.k", "attn.v")
             else ("num_heads", "num_attention_heads", "n_head"))
    counts = (getattr(obj, name, None) for obj in sources for name in names)
    count = next((c for c in counts if type(c) is int and c > 0), None)
    if count is not None and width > 0 and width % count == 0:
        return width // count
    for obj in sources:
        size = getattr(obj, "head_dim", None)
        if type(size) is int and size > 0:
            return size
    return None


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
        attached: Modules of ``root`` outside the mapped model, each placed at the module of ``root`` it is attached
            to (``{"latent_models.sae3": "model.transformer.h.3"}``): a tensor under a key's module takes the location
            of the value module, as if it were nested under it, and has no unit axis.

    Raises:
        TunableViewError: If an attached key is not a module of ``root``, is or holds part of the mapped model, or lies
            inside the mount; or if its value is not a module of ``root`` the map places.
    """

    vocabulary = True

    def __init__(self, root: nn.Module, cmap: ComponentMap, prefix: str | None = None,
                 mount: str | None = None, attached: Mapping[str, str] | None = None) -> None:
        self.cmap = cmap
        self._root = root
        self._modules: dict[str, nn.Module] | None = None
        self._blocks: dict[str, int | None] = {}
        self.refusal: Refusal | None = None
        self._depths: dict[str, int] = {}
        self._table: dict[str, Location] = {}
        self._attached: dict[str, Location] = {}
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
        for key, at in (attached or {}).items():
            self._attach(root, key, at)

    def _attach(self, root: nn.Module, key: str, at: str) -> None:
        if not key or submodule(root, key) is None:
            raise TunableViewError(f"{type(root).__name__} has no module {key!r} to attach.")
        path = strip_wrappers(key)
        mapped = list(self._table)
        if self.mount and strip_wrappers(f"{self.mount}."):
            mapped.append(strip_wrappers(f"{self.mount}.")[:-1])
        overlap = next((p for p in mapped if p == path or p.startswith(f"{path}.") or path.startswith(f"{p}.")), None)
        if overlap is not None:
            raise TunableViewError(f"{key!r} is or holds part of the mapped model ({overlap!r}); only a module outside "
                                   "it is attached, since the map places the ones inside it already.")
        if submodule(root, at) is None:
            raise TunableViewError(f"{type(root).__name__} has no module {at!r} to attach {key!r} to.")
        loc = self._row_location(strip_wrappers(at))
        if loc is None:
            raise TunableViewError(f"the {self.cmap.architecture} component map places no module at or above {at!r}, "
                                   f"so {key!r} cannot be attached there.")
        self._attached[path] = loc

    def _row_location(self, owner: str) -> Location | None:
        """The location of the deepest map row at or above the module path ``owner``, if any."""
        while True:
            if owner in self._table:
                return self._table[owner]
            if not owner:
                return None
            owner = owner.rpartition(".")[0]

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
            found = self._table.get(owner, self._attached.get(owner))
            if found is not None:
                return found
            if not owner:
                return Location()
            owner = owner.rpartition(".")[0]

    def unit_dim(self, name: str, ndim: int) -> int | Refusal:
        """The dimension of the tensor ``name`` (of ``ndim`` dimensions) that runs along its component's units, or why
        it has none: it is not a row's own weight or bias, its component has no unit side, its row records no
        ``layout``, or (a vector) it lies along the side that is not its units."""
        if self.refusal is not None:
            return self.refusal
        owner = strip_wrappers(name).rpartition(".")[0]
        loc = self._table.get(owner)
        if loc is None:
            found = self.locate(name)
            if any(owner == p or owner.startswith(f"{p}.") for p in self._attached):
                return Refusal(f"`{name}` belongs to a module attached at `{found.component or 'blocks'}`, not to the "
                               "row's own module: only a row's own weight and bias have a unit axis")
            if found.component is not None:
                return Refusal(f"`{name}` lies under the `{found.component}` row's module, not on it: only a row's own "
                               "weight and bias have a unit axis")
            return Refusal(f"`{name}` is located at no component")
        component = loc.component or ""
        side = UNIT_SIDES.get(component)
        if side is None:
            return Refusal(f"component `{component or 'blocks'}` has no unit side (venngram.maps.schema.UNIT_SIDES "
                           "names the vocabulary's)")
        if ndim == 1:
            if side == "out":  # a bias, or a norm's elementwise weight
                return 0
            return Refusal(f"`{name}` lies along the `out` side of `{component}`, whose units are its `in` side")
        if ndim != 2:
            return Refusal(f"`{name}` has {ndim} dimensions; a unit axis is defined for a vector or a matrix")
        layout = self.cmap.layout_of(component, loc.layer, loc.stack)
        if layout is None:
            return Refusal(f"the {self.cmap.architecture} component map records no `layout` for `{component}` "
                           "(component map schema 5, written by `venngram.maps.verification.with_layouts`)")
        return (0 if side == "out" else 1) if layout == "out_in" else (0 if side == "in" else 1)

    def unit_block(self, name: str) -> int | None:
        """The recorded head block size for a tensor on a map row's own module, if defined.

        A component map describes every checkpoint of its architecture, and head sizes can differ between them, so a
        recorded size is checked against the head size this model measures (its attention module's or configuration's
        ``head_dim``, or its width over its head count) wherever that can be measured.

        Raises:
            AddressError: If the recorded size differs from the measured one: whole heads are then refused rather than
                cut at the wrong boundaries.
        """
        owner, _, leaf = strip_wrappers(name).rpartition(".")
        loc = self._table.get(owner) if leaf in ("weight", "bias") else None
        if not (loc and loc.component):
            return None
        recorded = self.cmap.unit_block_of(loc.component, loc.layer, loc.stack)
        if recorded is None:
            return None
        if owner not in self._blocks:
            self._blocks[owner] = self._measured_block(owner, loc)
        measured = self._blocks[owner]
        if measured is not None and measured != recorded:
            raise AddressError(f"the {self.cmap.architecture} component map records `unit_block` {recorded} for "
                               f"`{loc.component}`, but this model's `{owner}` measures heads of {measured}: the map "
                               "was recorded on a checkpoint with another head size, so whole heads are refused rather "
                               "than cut at the wrong boundaries.")
        return recorded

    def _measured_block(self, owner: str, loc: Location) -> int | None:
        """The head size ``owner`` (a row's module) measures, from the sources the map builder reads: the module, its
        attention module (the ``attn`` row's, else its parent) and that module's configuration."""
        if self._modules is None:
            self._modules = {strip_wrappers(n): m for n, m in self._root.named_modules()}
        module = self._modules.get(owner)
        component = str(loc.component)
        width = unit_width(module, component, self.cmap.layout_of(component, loc.layer, loc.stack))
        if module is None or width is None:
            return None
        path = self._root_path(self.cmap.module_for("attn", loc.layer, loc.stack))
        attention = self._modules.get(path if path is not None else owner.rpartition(".")[0])
        return head_size(component, (module, attention, getattr(attention, "config", None)), width)

    def _root_path(self, path: str | None) -> str | None:
        """A map module path as the (wrapper-stripped) path of the module under this structure's root."""
        rel = self._relative(path)
        if rel is None:
            return None
        lead = strip_wrappers(f"{self.mount}.") if self.mount else ""
        return f"{lead}{rel}" if rel else lead[:-1]

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
        """Every module path (relative to the root) this structure places from its map's rows, with its location."""
        return dict(self._table)

    def attached(self) -> dict[str, Location]:
        """Every attached module path (relative to the root, wrapper segments stripped), with the location it takes."""
        return dict(self._attached)

    def components(self) -> list[str]:
        """The components this structure can locate, in map order."""
        return list(dict.fromkeys(loc.component for loc in self._table.values() if loc.component))


def map_structure(model: nn.Module, architecture: str | None = None, **kwargs: Any) -> MapStructure:
    """A :class:`MapStructure` over ``model`` from its component map (:func:`map_for`)."""
    return MapStructure(model, map_for(model, architecture), **kwargs)


__all__ = ["architecture_of", "detect_prefix", "find_mounts", "map_for", "map_structure", "MapStructure"]
