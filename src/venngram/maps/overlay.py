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
Overlay Maps
^^^^^^^^^^^^

A map comes in two shapes. A **base map** (:class:`~venngram.maps.schema.ComponentMap`) describes an architecture: the
stack, layer and component of each of its modules. An **overlay map** (:class:`OverlayMap`) describes what a library
adds to a model's module tree (PEFT's LoRA layers, a spliced SAE or transcoder, a steering module) and the role each of
its tensors takes. A tensor under an overlay's module keeps the layer and component of the module it is attached to,
so one overlay serves every architecture, and venngram's code names no adapter library.

.. code-block:: yaml

    schema_version: 4
    overlay: peft.lora
    applies_when:            # any listed condition holding means the overlay applies
      module_classes: [peft.tuners.lora.layer.LoraLayer]   # matched against each module's class hierarchy
      module_names: [lora_A, lora_B]                       # matched against the segments of tensor names
    roles:                   # module pattern -> role; the first matching pattern wins
      lora_A: adapter
      lora_B: adapter

Overlays are documents of the ``component_map`` kind, found the way base maps are: registered in this process
(:func:`~venngram.maps.schema.register`), else the best one the configured registries index (as subject
``overlay:<name>``). :func:`overlays_for` returns every known overlay whose ``applies_when`` holds for a model, which is
how :class:`~venngram.maps.view.MappedView` composes a model's view without being told what it carries.

Loading an overlay runs no code.

.. warning:: This is an experimental API and may change in a future release.
"""
import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from torch import nn

from venngram.resolve import leaf_matches, module_matches
from venngram.roles import parse_role

OVERLAY_SCHEMA_VERSION = 4
APPLIES_WHEN_KEYS = ("module_classes", "module_names")
#: The registry subject of an overlay is its name under this prefix, so architecture lookups never see overlays.
OVERLAY_SUBJECT_PREFIX = "overlay:"


def rule_roles(root: nn.Module, rules: Mapping[str, str]) -> dict[str, str]:
    """The role of each of ``root``'s parameters and buffers that a pattern matches (the first matching one wins).

    A pattern matches the tensors under a module of that name and, as a leaf, a tensor registered under that name
    itself (DoRA's ``lora_magnitude_vector`` is a parameter, not a module, in some PEFT versions).
    """
    roles: dict[str, str] = {}
    for name, _ in itertools.chain(root.named_parameters(), root.named_buffers()):
        for pattern, role in rules.items():
            if module_matches(pattern, name) or leaf_matches(pattern, name):
                roles[name] = role
                break
    return roles


def _qualified_names(module: nn.Module) -> set[str]:
    return {f"{cls.__module__}.{cls.__qualname__}" for cls in type(module).__mro__}


@dataclass(frozen=True)
class OverlayMap:
    """What one library adds to a model's module tree, the roles its tensors take, and when it applies."""

    name: str
    roles: Mapping[str, str]
    """Module pattern -> role, in order (the first matching pattern wins)."""
    module_classes: tuple[str, ...] = ()
    """Qualified class names (``package.module.Class``); a model with a module of one, or of a subclass, matches."""
    module_names: tuple[str, ...] = ()
    """Name segments; a model with a parameter or buffer whose dotted name has one of these segments matches."""
    source: str = field(default="inline", compare=False)

    @property
    def subject(self) -> str:
        """The overlay's registry subject (``overlay:<name>``)."""
        return f"{OVERLAY_SUBJECT_PREFIX}{self.name}"

    def applies_to(self, root: nn.Module) -> bool:
        """Whether any of ``applies_when``'s conditions holds for ``root`` (an overlay with none never applies)."""
        if self.module_classes:
            wanted = set(self.module_classes)
            if any(wanted & _qualified_names(m) for m in root.modules()):
                return True
        if self.module_names:
            wanted = set(self.module_names)
            for name, _ in itertools.chain(root.named_parameters(), root.named_buffers()):
                if wanted.intersection(name.split(".")):
                    return True
        return False

    def roles_for(self, root: nn.Module) -> dict[str, str]:
        """The role of each of ``root``'s tensors that a pattern matches."""
        return rule_roles(root, self.roles)

    def to_mapping(self) -> dict[str, Any]:
        applies = {k: list(getattr(self, k)) for k in APPLIES_WHEN_KEYS if getattr(self, k)}
        return {"schema_version": OVERLAY_SCHEMA_VERSION, "overlay": self.name,
                **({"applies_when": applies} if applies else {}), "roles": dict(self.roles)}


def _strings(raw: Any, source: str, key: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(x, str) and x for x in raw):
        raise ValueError(f"{source}: `applies_when.{key}` is a list of non-empty strings, got {raw!r}.")
    return tuple(raw)


def overlay_from_document(doc: Any, source: str = "<document>") -> OverlayMap:
    """Read a parsed overlay map document, refusing a malformed one by name."""
    if not isinstance(doc, Mapping):
        raise ValueError(f"{source}: an overlay map document must be a mapping, got {type(doc).__name__}.")
    version = doc.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < OVERLAY_SCHEMA_VERSION:
        raise ValueError(f"{source}: an overlay map needs schema_version {OVERLAY_SCHEMA_VERSION} or later, got "
                         f"{version!r}.")
    name = doc.get("overlay")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{source}: an overlay map names itself with a non-empty `overlay`.")
    for key in ("architecture", "components"):
        if key in doc:
            raise ValueError(f"{source}: `{key}` belongs to a base map; an overlay map has `overlay`, `applies_when` "
                             "and `roles`.")
    roles = doc.get("roles")
    if not isinstance(roles, Mapping) or not roles:
        raise ValueError(f"{source}: `roles` maps module patterns to roles and must not be empty.")
    for pattern, role in roles.items():
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(f"{source}: a role pattern must be a non-empty string, got {pattern!r}.")
        try:
            parse_role(role)
        except ValueError as err:
            raise ValueError(f"{source}: role pattern {pattern!r}: {err}") from None
    applies = doc.get("applies_when") or {}
    if not isinstance(applies, Mapping):
        raise ValueError(f"{source}: `applies_when` maps {' and '.join(APPLIES_WHEN_KEYS)} to lists.")
    unknown = set(applies) - set(APPLIES_WHEN_KEYS)
    if unknown:
        raise ValueError(f"{source}: unknown `applies_when` condition(s) {sorted(unknown)} (this venngram knows "
                         f"{', '.join(APPLIES_WHEN_KEYS)}); a reader that ignored one would apply the overlay wrongly.")
    return OverlayMap(name=name, roles=dict(roles), source=source,
                      **{k: _strings(applies.get(k), source, k) for k in APPLIES_WHEN_KEYS})


_OVERLAYS: dict[str, OverlayMap] = {}


def overlay_for(name: str) -> OverlayMap:
    """The overlay called ``name``: one registered in this process, else the best one the registries index.

    Raises:
        KeyError: If no overlay of that name is registered or indexed.
        venngram.registry.RegistryError: If a registry cannot be read and was never fetched before.
    """
    from venngram.registry import lookup_overlay, registries

    overlay = _OVERLAYS.get(name) or lookup_overlay(name)
    if overlay is None:
        raise KeyError(f"no overlay map {name!r} registered in this process ({sorted(_OVERLAYS)}) or indexed by the "
                       f"registries {registries()}. Register one with `venngram.maps.register`, or publish one and "
                       "add its location to a registry.")
    return overlay


def known_overlays() -> list[str]:
    """Every overlay registered in this process or indexed by a configured registry."""
    from venngram.registry import subjects

    indexed = (s[len(OVERLAY_SUBJECT_PREFIX):] for s in subjects("component_map")
               if s.startswith(OVERLAY_SUBJECT_PREFIX))
    return sorted({*_OVERLAYS, *indexed})


def overlays_for(root: nn.Module) -> list[OverlayMap]:
    """Every known overlay whose ``applies_when`` holds for ``root``, by name (a registration shadows an indexed overlay
    of the same name)."""
    from venngram.registry import lookup_overlay

    found = []
    for name in known_overlays():
        overlay = _OVERLAYS.get(name) or lookup_overlay(name)
        if overlay is not None and overlay.applies_to(root):
            found.append(overlay)
    return found


__all__ = [
    "APPLIES_WHEN_KEYS",
    "known_overlays",
    "OVERLAY_SCHEMA_VERSION",
    "OVERLAY_SUBJECT_PREFIX",
    "overlay_for",
    "overlay_from_document",
    "OverlayMap",
    "overlays_for",
    "rule_roles",
]
