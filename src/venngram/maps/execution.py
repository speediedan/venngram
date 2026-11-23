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
"""Resolve activation sites in a component map's forward execution.

A visit identifies a weight layer, a one-based pass and its position in the forward. Several visits can use
the same parameters. Each site describes a module, its zero-based call index, and a path into its positional
arguments or return value. Consumers choose how to capture or change that value.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from venngram.address import And, Expr, Guarded, LayerSet, Minus, Or, parse, Ref, Term, to_string
from venngram.errors import AddressError, AddressUnresolvable, Refusal


@dataclass(frozen=True)
class LayerVisit:
    """A weight layer visit, in chronological order within one forward."""

    layer: int
    pass_number: int
    step: int


@dataclass(frozen=True)
class TensorIndex:
    """A position along a tensor axis. Plain integer path steps index a positional argument or sequence item."""

    axis: int
    index: int


@dataclass(frozen=True)
class ExecutionSite:
    """One named input or output at a module call, optionally associated with a weight layer visit."""

    point: str
    module: str
    call_index: int
    hook: str
    path: tuple[int | TensorIndex, ...]
    visit: LayerVisit | None

    def read(self, value: Any, *, token: int | None = None) -> Any:
        """Follow the path in a hook's positional arguments or return value. Read a token on axis 1 if requested."""
        for step in self.path:
            value = value[step] if isinstance(step, int) else value.select(step.axis, step.index)
        return value if token is None else value.select(1, token)


@dataclass(frozen=True)
class ExecutionMap:
    """One forward's ordered visits and the activation sites available at those visits."""

    depth: int
    visits: tuple[LayerVisit, ...]
    sites: tuple[ExecutionSite, ...]
    _document: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: Any, source: str) -> "ExecutionMap":
        if (not isinstance(raw, dict) or set(raw) - {"depth", "loops", "sites"}
                or type(raw.get("depth")) is not int or raw["depth"] < 1):
            raise ValueError(f"{source}: execution needs a positive integer depth, optional loops and named sites")
        depth = raw["depth"]
        visits = _visits(depth, raw.get("loops", []), source)
        points = raw.get("sites")
        if not isinstance(points, dict) or not points:
            raise ValueError(f"{source}: execution sites must be a non-empty mapping")
        sites = []
        placements: dict[str, tuple[LayerVisit | None, ...]] = {}
        for point, row in points.items():
            where = f"{source}: execution site {point!r}"
            if not isinstance(point, str) or not point or not isinstance(row, dict):
                raise ValueError(f"{where}: a named site must be a mapping")
            sites.extend(_site_rows(point, row, visits, depth, placements, where))
        sites.sort(key=lambda site: site.visit.step if site.visit else len(visits))
        return cls(depth, visits, tuple(sites), deepcopy(raw))

    def to_mapping(self) -> dict[str, Any]:
        """Return the document form used to read this execution map."""
        return deepcopy(self._document)

    def resolve(self, address: Any, sets: Mapping[str, Any] | None = None) -> tuple[ExecutionSite, ...]:
        """Resolve a point address to concrete calls, in forward order.

        ``layers`` selects weight layers. ``passes`` selects a one-based pass. A point without a pass selects all
        its calls; a missing requested visit is refused. Run this against the map verified for the forward in use.
        """
        expr = parse(address)
        try:
            found = self._resolve(expr, sets or {}, ())
            if not found and not (isinstance(expr, Guarded) and expr.optional):
                raise AddressError("it matches no execution site or requested visit")
        except AddressError as err:
            raise AddressUnresolvable(to_string(expr), [Refusal(str(err))]) from err
        return tuple(site for site in self.sites if site in found)

    def _resolve(self, expr: Expr, sets: Mapping[str, Any], seen: tuple[str, ...]) -> set[ExecutionSite]:
        if isinstance(expr, Term):
            return self._term(expr)
        if isinstance(expr, Guarded):
            result = self._resolve(expr.expr, sets, seen)
            if expr.expect is not None and len(result) != expr.expect:
                raise AddressError(f"it selects {len(result)} execution sites, expected {expr.expect}")
            return result
        if isinstance(expr, Ref):
            if expr.name in seen or expr.name not in sets:
                raise AddressError(f"execution set @{expr.name} is cyclic or missing")
            return self._resolve(parse(sets[expr.name]), sets, (*seen, expr.name))
        if isinstance(expr, Minus):
            return self._resolve(expr.left, sets, seen) - self._resolve(expr.right, sets, seen)
        if isinstance(expr, (And, Or)):
            items = [self._resolve(item, sets, seen) for item in expr.items]
            return set.intersection(*items) if isinstance(expr, And) else set.union(*items)
        raise AddressError("execution sites select point calls")

    def _term(self, term: Term) -> set[ExecutionSite]:
        bad = term.set_fields().keys() - {"point", "layers", "passes"}
        if bad:
            raise AddressError(f"execution sites do not support fields {sorted(bad)}")
        if term.point is not None and not any(s.point == term.point for s in self.sites):
            raise AddressError(f"execution map has no point {term.point!r}")
        layers = set(term.layers.select(self.depth)) if term.layers is not None else None
        found = {s for s in self.sites if term.point in (None, s.point)
                 and (layers is None or s.visit is not None and s.visit.layer in layers)
                 and (term.passes is None or s.visit is not None and s.visit.pass_number == term.passes)}
        if term.point is not None and layers is not None and term.passes is not None:
            missing = layers - {s.visit.layer for s in found if s.visit is not None}
            if missing:
                raise AddressError(f"point {term.point!r} has no pass {term.passes} at weight layers {sorted(missing)}")
        if term.point is not None and term.passes is not None and not found:
            raise AddressError(f"point {term.point!r} has no pass {term.passes}")
        return found


def _visits(depth: int, loops: Any, source: str) -> tuple[LayerVisit, ...]:
    if not isinstance(loops, list):
        raise ValueError(f"{source}: execution loops must be a list")
    pairs: list[tuple[int, int]] = []
    start_at = 0
    for loop in loops:
        if (not isinstance(loop, dict) or set(loop) != {"start", "stop", "passes"}
                or any(type(v) is not int for v in loop.values())):
            raise ValueError(f"{source}: each execution loop needs integer start, stop and passes")
        start, stop, count = loop["start"], loop["stop"], loop["passes"]
        if not start_at <= start < stop <= depth or count < 2:
            raise ValueError(f"{source}: execution loop ranges must be ordered and disjoint, with at least two passes")
        pairs.extend((layer, 1) for layer in range(start_at, start))
        pairs.extend((layer, p) for p in range(1, count + 1) for layer in range(start, stop))
        start_at = stop
    pairs.extend((layer, 1) for layer in range(start_at, depth))
    return tuple(LayerVisit(layer, p, step) for step, (layer, p) in enumerate(pairs))


def _value_path(raw: Any, where: str) -> tuple[int | TensorIndex, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{where}: input or output path must be a list")
    path: list[int | TensorIndex] = []
    for part in raw:
        if type(part) is int and part >= 0:
            path.append(part)
        elif (isinstance(part, dict) and set(part) == {"axis", "index"}
              and type(part["axis"]) is int and type(part["index"]) is int and part["index"] >= 0):
            path.append(TensorIndex(part["axis"], part["index"]))
        else:
            raise ValueError(f"{where}: path steps are non-negative indices or tensor selections {{axis, index}}")
    return tuple(path)


def _site_rows(point: str, row: dict[str, Any], visits: tuple[LayerVisit, ...], depth: int,
               placements: dict[str, tuple[LayerVisit | None, ...]], where: str) -> list[ExecutionSite]:
    hook = row.keys() & {"input", "output"}
    template = row.get("module")
    start = row.get("first_pass", 1)
    if (len(hook) != 1 or row.keys() - {"module", "input", "output", "layers", "first_pass"}
            or not isinstance(template, str) or not template or type(start) is not int or start < 1):
        raise ValueError(f"{where}: expected a module and one of input or output, with optional layers and first_pass")
    mode = hook.pop()
    path = _value_path(row[mode], where)
    if mode == "input" and (not path or not isinstance(path[0], int)):
        raise ValueError(f"{where}: inputs start with the positional argument index")
    if "layers" in row:
        layers = LayerSet.parse(row["layers"]).select(depth)
        selected = [v for v in visits if v.layer in layers and v.pass_number >= start]
        if not selected:
            raise ValueError(f"{where}: no execution visits match")
        by_module: dict[str, list[LayerVisit | None]] = {}
        for v in selected:
            by_module.setdefault(template.replace("{i}", str(v.layer)), []).append(v)
    else:
        if "{i}" in template or "first_pass" in row:
            raise ValueError(f"{where}: layer templates and pass restrictions need layers")
        by_module = {template: [None]}
    out = []
    for module, calls in by_module.items():
        timeline = tuple(calls)
        if placements.setdefault(module, timeline) != timeline:
            raise ValueError(f"{where}: inconsistent call timeline for {module!r}")
        out.extend(ExecutionSite(point, module, i, mode, path, visit) for i, visit in enumerate(calls))
    return out
