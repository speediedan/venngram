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
Gradient Paths
^^^^^^^^^^^^^^

A **path** names the routes a gradient takes between ordered waypoints (:class:`venngram.address.Path`): a source, a
target and any stops between them, each an address or a selection of a graph subset's nodes, optionally anchored to
token positions. Every route between two consecutive waypoints belongs to the path, parallel routes included.

.. code-block:: yaml

    sets:
      heads: {component: "attn.*", layers: "0.5d:0.75d"}
      premise_to_answer:
        path:
          - {component: embed, anchor: {kind: token_match, token_ids: [5, 9]}}
          - "@heads"
          - {component: unembed, anchor: {kind: from_end, offset: 0}}
    treatments:
      cut:
        model: {id: example/model, revision: pinned}
        gradients:
          - {op: stop, target: {set: premise_to_answer}}                    # stop the gradient along it
          - {op: route, target: {set: premise_to_answer}, params: {reduce: sum}}

**What venngram declares and what a consumer does.** venngram reads and checks a path: its waypoints in order, every
reference resolved, a graph waypoint's nodes present in the book's graph subset (:func:`paths`). Turning waypoints into
the edges of a forward graph is a consumer's **route resolver** (:class:`RouteResolver`), as a site resolver places an
activation point, and executing an op along the routes is the consumer's too. The path ops are ``stop`` (stop the
gradient along the path, keeping every other route) and ``route`` (keep only the path), in a treatment's ``gradients``
section; their ``reduce`` parameter says how parallel routes between consecutive waypoints combine (``each``, the
default, acts on every route; ``sum`` and ``mean`` on their combination).

**Portability.** A path of address waypoints names components, roles and layers, so it applies on any architecture its
addresses resolve on. A graph waypoint names the nodes of one checkpoint's attribution graph, so a book with one is a
``checkpoint`` book.

**Set algebra.** Paths unite (``any: [@a, @b]`` is both paths). An intersection or a difference of paths, and a union of
a path with other addresses, are refused by name until their meaning is defined.

.. warning:: This is an experimental API and may change in a future release.
"""
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from venngram.address import And, Expr, Guarded, Minus, Or, parse, Path, Ref, to_string, Waypoint
from venngram.errors import AddressUnresolvable, Refusal

#: The kinds of waypoint a route resolver may honour (see :class:`venngram.ops.Support`'s ``routes``): an address, a
#: graph subset's nodes, and an anchor placing either at token positions.
WAYPOINT_KINDS = ("address", "graph_nodes", "anchor")


def paths(address: Any, sets: Mapping[str, Any] | None = None) -> tuple[Path, ...]:
    """The paths ``address`` names: a path, a reference to one, or a union of them, following references into
    ``sets``.

    >>> [len(p.waypoints) for p in paths({"path": ["component:embed", "component:unembed"]})]
    [2]

    Raises:
        AddressUnresolvable: If a referenced set is missing or references form a cycle, or the address intersects or
            subtracts paths, unites a path with another address, or names no path.
    """
    expr = parse(address)
    sets = sets or {}
    out: list[Path] = []
    refusals: list[Refusal] = []

    def walk(e: Expr, stack: tuple[str, ...]) -> None:
        if isinstance(e, Path):
            out.append(e)
        elif isinstance(e, Ref):
            if e.name in stack:
                refusals.append(Refusal(f"the set references form a cycle: "
                                        f"{' -> '.join('@' + s for s in (*stack, e.name))}"))
            elif e.name not in sets:
                refusals.append(Refusal(f"no set is named `@{e.name}`", tuple(sorted(f"@{s}" for s in sets))))
            else:
                walk(parse(sets[e.name]), (*stack, e.name))
        elif isinstance(e, Guarded):
            walk(e.expr, stack)
        elif isinstance(e, Or):
            for i in e.items:
                walk(i, stack)
        elif isinstance(e, (And, Minus)):
            refusals.append(Refusal(f"`{to_string(e)}` intersects or subtracts paths, which is not defined yet: unite "
                                    "paths, or name the waypoints they share"))
        else:
            refusals.append(Refusal(f"`{to_string(e)}` is not a path; a path set unites only paths"))

    walk(expr, ())
    if not out and not refusals:
        refusals.append(Refusal("it names no path"))
    if refusals:
        raise AddressUnresolvable(to_string(expr), refusals)
    return tuple(out)


def is_path(address: Any, sets: Mapping[str, Any] | None = None) -> bool:
    """Whether ``address`` names paths (a path, a reference to one, or a union of them), without checking it."""
    expr = parse(address)
    sets = sets or {}
    seen: set[str] = set()
    while True:
        if isinstance(expr, Path):
            return True
        if isinstance(expr, Guarded):
            expr = expr.expr
        elif isinstance(expr, Or):
            return any(is_path(i, sets) for i in expr.items)
        elif isinstance(expr, Ref) and expr.name in sets and expr.name not in seen:
            seen.add(expr.name)
            expr = parse(sets[expr.name])
        else:
            return False


def waypoint_kinds(path: Path) -> frozenset[str]:
    """The :data:`WAYPOINT_KINDS` a path uses."""
    out = {"graph_nodes" if w.graph is not None else "address" for w in path.waypoints}
    if any(w.anchor is not None for w in path.waypoints):
        out.add("anchor")
    return frozenset(out)


@runtime_checkable
class RouteResolver(Protocol):
    """A consumer's placement of paths in its forward graph (venngram never builds one).

    ``routes`` returns the consumer's own representation of every route between consecutive waypoints, or a
    :class:`venngram.errors.Refusal` naming the waypoint it cannot route through (a component its backend does not
    expose, an anchor that does not resolve in the prompt), never a nearby route instead.
    """

    def routes(self, path: Path, *, graphs: Mapping[str, Any] | None = None) -> Any:
        """The routes of ``path``; ``graphs`` are the book's graph subsets, for graph waypoints."""
        ...


__all__ = ["is_path", "Path", "paths", "RouteResolver", "Waypoint", "WAYPOINT_KINDS", "waypoint_kinds"]
