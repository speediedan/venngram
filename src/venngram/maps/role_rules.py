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
Role Rules
^^^^^^^^^^

Which tensors an adapter library adds to a model, and the role each one takes, as a data document: ``kind:
role_rules``. A component map describes an architecture; a rule set describes an adapter convention (PEFT's LoRA, for
example), so one rule set serves every architecture and venngram's code names no adapter library.

.. code-block:: yaml

    kind: role_rules
    schema_version: 1
    name: peft.lora
    applies_when:            # any listed condition holding means the rules apply
      module_classes: [peft.tuners.lora.layer.LoraLayer]   # matched against each module's class hierarchy
      module_names: [lora_A, lora_B]                       # matched against the segments of tensor names
    rules:                   # module pattern -> role; the first matching rule wins
      lora_A: adapter
      lora_B: adapter

Rule sets are found the way maps are: one registered in this process (:func:`register_role_rules`), else the best one
the configured registries index (:mod:`venngram.registry`, subject: the rule set's name). :func:`detect_role_rules`
returns every known rule set whose ``applies_when`` holds for a model, which is how
:class:`~venngram.maps.view.MappedView` finds a model's roles without being told its adapter library.

Loading a rule set runs no code.

.. warning:: This is an experimental API and may change in a future release.
"""
import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from torch import nn

from venngram.resolve import leaf_matches, module_matches
from venngram.roles import parse_role

ROLE_RULES_SCHEMA_VERSION = 1
APPLIES_WHEN_KEYS = ("module_classes", "module_names")


def rule_roles(root: nn.Module, rules: Mapping[str, str]) -> dict[str, str]:
    """The role of each of ``root``'s parameters and buffers that a rule matches (the first matching rule wins).

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
class RoleRules:
    """One adapter convention's role rules, and the conditions under which they apply to a model."""

    name: str
    rules: Mapping[str, str]
    """Module pattern -> role, in order (the first matching rule wins)."""
    module_classes: tuple[str, ...] = ()
    """Qualified class names (``package.module.Class``); a model with a module of one, or of a subclass, matches."""
    module_names: tuple[str, ...] = ()
    """Name segments; a model with a parameter or buffer whose dotted name has one of these segments matches."""
    source: str = field(default="<document>", compare=False)

    def applies_to(self, root: nn.Module) -> bool:
        """Whether any of ``applies_when``'s conditions holds for ``root`` (a rule set with none never applies)."""
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

    def roles(self, root: nn.Module) -> dict[str, str]:
        """The role of each of ``root``'s tensors that a rule matches."""
        return rule_roles(root, self.rules)

    def to_mapping(self) -> dict[str, Any]:
        applies = {k: list(getattr(self, k)) for k in APPLIES_WHEN_KEYS if getattr(self, k)}
        return {"kind": "role_rules", "schema_version": ROLE_RULES_SCHEMA_VERSION, "name": self.name,
                **({"applies_when": applies} if applies else {}), "rules": dict(self.rules)}


def _strings(raw: Any, source: str, key: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(x, str) and x for x in raw):
        raise ValueError(f"{source}: `applies_when.{key}` is a list of non-empty strings, got {raw!r}.")
    return tuple(raw)


def role_rules_from_document(doc: Any, source: str = "<document>") -> RoleRules:
    """Read a parsed ``role_rules`` document, refusing a malformed one by name."""
    if not isinstance(doc, Mapping):
        raise ValueError(f"{source}: a role rules document must be a mapping, got {type(doc).__name__}.")
    if doc.get("kind", "role_rules") != "role_rules":
        raise ValueError(f"{source}: expected `kind: role_rules`, got {doc.get('kind')!r}.")
    version = doc.get("schema_version")
    if version != ROLE_RULES_SCHEMA_VERSION:
        raise ValueError(f"{source}: role rules schema_version {version!r} is not supported (this venngram reads "
                         f"{ROLE_RULES_SCHEMA_VERSION}).")
    name = doc.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{source}: a role rules document needs a `name`.")
    rules = doc.get("rules")
    if not isinstance(rules, Mapping) or not rules:
        raise ValueError(f"{source}: `rules` maps module patterns to roles and must not be empty.")
    for pattern, role in rules.items():
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(f"{source}: a rule pattern must be a non-empty string, got {pattern!r}.")
        try:
            parse_role(role)
        except ValueError as err:
            raise ValueError(f"{source}: rule {pattern!r}: {err}") from None
    applies = doc.get("applies_when") or {}
    if not isinstance(applies, Mapping):
        raise ValueError(f"{source}: `applies_when` maps {' and '.join(APPLIES_WHEN_KEYS)} to lists.")
    unknown = set(applies) - set(APPLIES_WHEN_KEYS)
    if unknown:
        raise ValueError(f"{source}: unknown `applies_when` condition(s) {sorted(unknown)} (this venngram knows "
                         f"{', '.join(APPLIES_WHEN_KEYS)}); a reader that ignored one would apply the rules wrongly.")
    return RoleRules(name=name, rules=dict(rules), source=source,
                     **{k: _strings(applies.get(k), source, k) for k in APPLIES_WHEN_KEYS})


def load_role_rules(path: str | Path) -> RoleRules:
    """Read one YAML role rules document. Loading runs no code (``yaml.safe_load``)."""
    with open(path, encoding="utf-8") as fh:
        return role_rules_from_document(yaml.safe_load(fh), source=str(path))


_REGISTRY: dict[str, RoleRules] = {}


def register_role_rules(rules: RoleRules) -> None:
    """Register (or replace) a rule set in this process; it takes precedence over the registries."""
    _REGISTRY[rules.name] = rules


def unregister_role_rules(name: str) -> None:
    """Remove an in-process registration (for tests and replacements)."""
    _REGISTRY.pop(name, None)


def registered_role_rules() -> list[str]:
    """Every rule set registered in this process."""
    return sorted(_REGISTRY)


def role_rules_for(name: str) -> RoleRules:
    """The rule set called ``name``: one registered in this process, else the best one the registries index.

    Raises:
        KeyError: If no rule set of that name is registered or indexed.
        venngram.registry.RegistryError: If a registry cannot be read and was never fetched before.
    """
    from venngram.registry import lookup_role_rules, registries

    rules = _REGISTRY.get(name) or lookup_role_rules(name)
    if rules is None:
        raise KeyError(f"no role rules {name!r} registered in this process ({registered_role_rules()}) or indexed by "
                       f"the registries {registries()}. Register a rule set with "
                       "`venngram.maps.register_role_rules`, or publish one and add its location to a registry.")
    return rules


def known_role_rules() -> list[str]:
    """Every rule set registered in this process or indexed by a configured registry."""
    from venngram.registry import subjects

    return sorted({*_REGISTRY, *subjects("role_rules")})


def detect_role_rules(root: nn.Module) -> list[RoleRules]:
    """Every known rule set whose ``applies_when`` holds for ``root``, by name (a registration shadows an indexed rule
    set of the same name)."""
    from venngram.registry import lookup_role_rules, subjects

    found = []
    for name in sorted({*_REGISTRY, *subjects("role_rules")}):
        rules = _REGISTRY.get(name) or lookup_role_rules(name)
        if rules is not None and rules.applies_to(root):
            found.append(rules)
    return found


__all__ = [
    "APPLIES_WHEN_KEYS",
    "detect_role_rules",
    "known_role_rules",
    "load_role_rules",
    "register_role_rules",
    "registered_role_rules",
    "role_rules_for",
    "role_rules_from_document",
    "ROLE_RULES_SCHEMA_VERSION",
    "RoleRules",
    "rule_roles",
    "unregister_role_rules",
]
