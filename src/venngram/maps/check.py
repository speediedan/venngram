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
Checking a Map Against a Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A map that is wrong about its model fails here, named, rather than later as a plausible module path that does not
exist. The assertions are cheap and need no forward pass, so they run on a meta-device model:

- every row's module exists for every layer it covers, and does not exist for a layer its ``layers`` predicate excludes
  (a row declared for all layers of a heterogeneous stack fails on the first layer that lacks it);
- every derived property agrees with the model (a document may state one as a cross-check; the model decides);
- every declared stack's template addresses at least one layer.

.. warning:: This is an experimental API and may change in a future release.
"""
import inspect
from dataclasses import dataclass
from typing import Any

from venngram.maps.schema import ComponentMap, PRIMARY_STACK


@dataclass(frozen=True)
class MapProblem:
    """One disagreement between a map and its model."""

    row: str
    layer: int | None
    reason: str

    def __str__(self) -> str:
        where = self.row if self.layer is None else f"{self.row} at layer {self.layer}"
        return f"{where}: {self.reason}"


class ComponentMapModelMismatch(ValueError):
    """The map does not describe this model; every problem is listed."""


def submodule(model: Any, path: str) -> Any | None:
    """The module at ``path`` (``""`` is ``model`` itself), or ``None`` if it does not exist."""
    node = model
    for part in path.split(".") if path else ():
        if part.isdigit():
            try:
                node = node[int(part)]
            except (TypeError, IndexError, KeyError):
                return None
            continue
        node = getattr(node, part, None)
        if node is None:
            return None
    return node


def stack_length(model: Any, template: str) -> int:
    """How many layers a ``{i}`` template addresses on ``model`` (the first index that does not resolve)."""
    n = 0
    while n <= 4096 and submodule(model, template.replace("{i}", str(n))) is not None:  # a template that always
        n += 1                                                                             # resolves is not a stack
    return n


def derive_rmsnorm_offset(model: Any, cmap: ComponentMap) -> bool | None:
    """Whether the final norm scales by ``(1 + weight)`` (Gemma, Gemma 2, Gemma 3) rather than by ``weight``, read from
    the module's own forward rather than a family name; ``None`` when there is no final norm or its source cannot be
    read."""
    path = cmap.module_for("ln_final", None)
    norm = submodule(model, path) if path else None
    if norm is None:
        return None
    try:
        source = inspect.getsource(type(norm).forward)
    except (OSError, TypeError):
        return None
    return any(s in source for s in ("1.0 + self.weight", "1 + self.weight"))


def check_map_against_model(cmap: ComponentMap, model: Any) -> list[MapProblem]:
    """Every disagreement between ``cmap`` and ``model``; empty when the map describes it."""
    problems: list[MapProblem] = []
    lengths: dict[str, int] = {}
    for stack in cmap.stack_names():
        template = cmap.block_template(stack)
        if template is None:
            continue
        lengths[stack] = stack_length(model, template)
        if not lengths[stack]:
            problems.append(MapProblem(f"stack {stack!r}", None, f"template {template!r} addresses no layer on this "
                                       "model"))
    for name, entry in cmap.components.items():
        if "{i}" not in entry.module:
            if submodule(model, entry.module) is None:
                problems.append(MapProblem(name, None, f"module {entry.module!r} does not exist"))
            continue
        stack = cmap.key(name).stack or PRIMARY_STACK
        for layer in range(lengths.get(stack, 0)):
            exists = submodule(model, entry.module.replace("{i}", str(layer))) is not None
            covered = entry.layers is None or entry.layers.covers(layer)
            if covered and not exists:
                problems.append(MapProblem(name, layer, f"row covers this layer but {entry.module!r} does not exist "
                                           "here"))
            elif exists and not covered:
                problems.append(MapProblem(name, layer, f"row excludes this layer ({entry.layers.describe()}) but "  # type: ignore[union-attr]
                                           "the module exists"))
    declared = cmap.properties.get("rmsnorm_offset")
    derived = derive_rmsnorm_offset(model, cmap) if declared is not None else None
    if derived is not None and derived != declared:
        problems.append(MapProblem("properties.rmsnorm_offset", None, f"document says {declared}, the model's norm "
                                   f"says {derived}"))
    return problems


def validate_map_against_model(cmap: ComponentMap, model: Any) -> None:
    """Return when ``cmap`` describes ``model``.

    Raises:
        ComponentMapModelMismatch: Listing every problem otherwise.
    """
    problems = check_map_against_model(cmap, model)
    if problems:
        listing = "\n  ".join(str(p) for p in problems)
        raise ComponentMapModelMismatch(f"the component map for {cmap.architecture!r} does not describe this model "
                                        f"({len(problems)} problems):\n  {listing}")


__all__ = [
    "check_map_against_model",
    "ComponentMapModelMismatch",
    "derive_rmsnorm_offset",
    "MapProblem",
    "stack_length",
    "submodule",
    "validate_map_against_model",
]
