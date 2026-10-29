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
Verification Rules
^^^^^^^^^^^^^^^^^^

What the verifier (:mod:`~venngram.maps.verification`) expects of an architecture, as data rather than code, so a map
for an architecture nobody anticipated can say what holds for it. A rule set is a document with three sections:

.. code-block:: yaml

    schema_version: 1
    name: my-rules
    config:            # a meaning -> where a configuration states it, first that resolves wins
      hidden: [hidden_size, n_embd]
      head_dim: [head_dim, "hidden // heads"]
    widths:            # a component pattern -> the widths its module must have (expressions over the meanings)
      attn.q: {out: "heads * head_dim"}
      "mixer.*": {in: hidden}
    kinds:             # a component kind -> the same, for every row of that kind
      norm: {ndim: 1}
    order:             # a named chain: within one block, every component of a group starts before any of the next
      attention: [ln1, [attn.q, attn.k, attn.v], attn.o]

A name in a ``config`` source is a configuration field if the configuration has it (as an integer), else another
meaning; so ``kv_heads: [num_key_value_heads, heads]`` falls back to the head count. Patterns are globs over component
names (``*`` one segment, ``**`` several), as an address's ``component`` field
takes. An expression uses integers, meanings, ``+ - * // /`` (``/`` must divide exactly) and parentheses; a rule
whose expression names a meaning the configuration does not state is skipped, not failed.

Rule sets layer: the bundled ``core`` set (:func:`core_rules`, the conventional decoder parts), then a map's own
``verification:`` section, then any the caller passes. A later layer replaces an entry by its key, and a ``null`` value
removes one, so a parallel-residual architecture drops the ``attn_then_mlp`` chain and GPT-2 says its intermediate
size defaults to ``4 * hidden``.

.. warning:: This is an experimental API and may change in a future release.
"""
import ast
import operator
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from venngram.resolve import component_matches

#: The schema of a rule set document.
RULES_SCHEMA_VERSION = 1

_DATA = Path(__file__).parent / "data" / "rules"
_WIDTHS = ("in", "out", "rows", "numel", "ndim")
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv}


class Unstated(LookupError):
    """An expression names a meaning the configuration does not state."""


def evaluate(expr: Any, value: Any) -> int:
    """Evaluate an integer or an arithmetic expression, looking each name up with ``value(name)``.

    >>> evaluate("(heads + 2 * kv_heads) * head_dim", {"heads": 4, "kv_heads": 2, "head_dim": 8}.__getitem__)
    64

    Raises:
        Unstated: If ``value`` raises it for a name.
        ValueError: If the expression is malformed, uses another construct, or ``/`` does not divide exactly.
    """
    if isinstance(expr, int) and not isinstance(expr, bool):
        return expr
    if not isinstance(expr, str):
        raise ValueError(f"an expression must be an integer or a string, got {expr!r}")
    try:
        tree = ast.parse(expr, mode="eval").body
    except SyntaxError as err:
        raise ValueError(f"malformed expression {expr!r}: {err.msg}") from None

    def walk(node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Name):
            return value(node.id)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            num, den = walk(node.left), walk(node.right)
            if den == 0 or num % den:
                raise ValueError(f"{expr!r}: {num} / {den} does not divide exactly; use // to round down")
            return num // den
        raise ValueError(f"{expr!r}: only integers, names, + - * // / and parentheses are allowed")

    return walk(tree)


@dataclass(frozen=True)
class RuleSet:
    """The verifier's expectations, merged from one or more documents (see the module docstring)."""

    config: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    widths: dict[str, dict[str, Any]] = field(default_factory=dict)
    kinds: dict[str, dict[str, Any]] = field(default_factory=dict)
    order: dict[str, tuple[tuple[str, ...], ...]] = field(default_factory=dict)
    sources: tuple[str, ...] = ()
    """The documents merged, in order (``core``, then a map's section, then the caller's)."""

    def then(self, other: "RuleSet") -> "RuleSet":
        """This set with ``other`` layered over it."""
        return RuleSet(*(_layer(getattr(self, k), getattr(other, k)) for k in ("config", "widths", "kinds", "order")),
                       sources=self.sources + other.sources)  # type: ignore[arg-type]

    def widths_for(self, component: str, kind: str) -> dict[str, Any]:
        """The width expectations of a row: its kind's, then every matching component pattern's, in document order."""
        spec = dict(self.kinds.get(kind) or {})
        for pattern, rules in self.widths.items():
            if rules and component_matches(pattern, component):
                spec.update(rules)
        return spec

    def resolver(self, config: Any):
        """A ``value(name)`` for :func:`evaluate` over ``config``: each meaning from its first source that resolves."""
        cache: dict[str, int] = {}
        active: set[str] = set()

        def value(name: str) -> int:
            if name in cache:
                return cache[name]
            if name not in self.config or name in active:
                raise Unstated(name)
            active.add(name)
            try:
                for source in self.config[name]:
                    found = _source(config, source, value)
                    if found is not None:
                        cache[name] = found
                        return found
            finally:
                active.discard(name)
            raise Unstated(name)

        return value


def config_field(config: Any, name: str) -> int | None:
    """An integer field of ``config`` (an object or a mapping), or ``None`` if it has none."""
    found = config.get(name) if isinstance(config, Mapping) else getattr(config, name, None)
    return found if isinstance(found, int) and not isinstance(found, bool) else None


def _source(config: Any, source: Any, value) -> int | None:
    def name(n: str) -> int:  # a configuration field, else another meaning
        found = config_field(config, n)
        return found if found is not None else value(n)
    try:
        return evaluate(source, name)
    except Unstated:
        return None


def _layer(base: dict, top: dict | None) -> dict:
    merged = dict(base)
    for key, val in (top or {}).items():
        if val is None:
            merged.pop(key, None)
        else:
            merged[key] = val
    return merged


def _group(entry: Any, where: str) -> tuple[str, ...]:
    group = (entry,) if isinstance(entry, str) else tuple(entry) if isinstance(entry, list) else None
    if not group or not all(isinstance(p, str) and p for p in group):
        raise ValueError(f"{where}: an order entry is a component pattern or a list of them, got {entry!r}")
    return group


def rules_from_document(doc: Any, source: str = "<rules>", *, section: bool = False) -> RuleSet:
    """Read a rule set document (or, with ``section``, a map's ``verification:`` section, which has no header),
    refusing a malformed one by name. Entries are validated here; ``null`` entries are kept for layering."""
    if not isinstance(doc, dict):
        raise ValueError(f"{source}: a rule set must be a mapping, got {type(doc).__name__}")
    if not section:
        version = doc.get("schema_version")
        if version != RULES_SCHEMA_VERSION:
            raise ValueError(f"{source}: rule set `schema_version` must be {RULES_SCHEMA_VERSION}, got {version!r}")
    unknown = set(doc) - {"schema_version", "name", "config", "widths", "kinds", "order"}
    if unknown:
        raise ValueError(f"{source}: unknown rule set keys {sorted(unknown)}")
    config: dict[str, Any] = {}
    for meaning, sources in (doc.get("config") or {}).items():
        if sources is None:
            config[meaning] = None
            continue
        sources = [sources] if isinstance(sources, (str, int)) else sources
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"{source}: config {meaning!r} must be a field name, an expression or a list of them")
        for s in sources:
            if not (isinstance(s, str) and s.isidentifier()):
                evaluate(s, lambda _n: 1)  # a malformed expression is refused at load, not at verification
        config[meaning] = tuple(sources)
    widths, kinds = (_widths(doc.get(k), source, k) for k in ("widths", "kinds"))
    order: dict[str, Any] = {}
    for name, chain in (doc.get("order") or {}).items():
        if chain is None:
            order[name] = None
            continue
        if not isinstance(chain, list) or len(chain) < 2:
            raise ValueError(f"{source}: order chain {name!r} must list at least two groups")
        order[name] = tuple(_group(g, f"{source}: order chain {name!r}") for g in chain)
    return RuleSet(config, widths, kinds, order, (doc.get("name") or source,))  # type: ignore[arg-type]


def _widths(raw: Any, source: str, key: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pattern, spec in (raw or {}).items():
        if spec is None:
            out[pattern] = None
            continue
        if not isinstance(spec, dict) or not spec or set(spec) - set(_WIDTHS):
            raise ValueError(f"{source}: {key} {pattern!r} must map some of {list(_WIDTHS)} to expressions")
        for expr in spec.values():
            evaluate(expr, lambda _n: 1)
        out[pattern] = dict(spec)
    return out


def load_rules(path: str | Path) -> RuleSet:
    """Read one YAML rule set document. Loading runs no code."""
    with open(path, encoding="utf-8") as fh:
        return rules_from_document(yaml.safe_load(fh), source=str(path))


_CORE: list[RuleSet] = []


def core_rules() -> RuleSet:
    """The bundled ``core`` rule set: the conventional parts of a decoder block, keyed by the shared vocabulary."""
    if not _CORE:
        _CORE.append(load_rules(_DATA / "core.yaml"))
    return _CORE[0]


def layered(*sets: RuleSet | Mapping[str, Any] | str | Path | None) -> RuleSet:
    """Rule sets layered in order; each is a :class:`RuleSet`, a document, or a path to one (``None`` is skipped)."""
    merged = RuleSet()
    for s in _as_sets(sets):
        merged = merged.then(s)
    return merged


def _as_sets(sets: Iterable[Any]) -> Iterable[RuleSet]:
    for s in sets:
        if s is None:
            continue
        if isinstance(s, RuleSet):
            yield s
        elif isinstance(s, Mapping):
            yield rules_from_document(dict(s))
        else:
            yield load_rules(s)


__all__ = [
    "config_field",
    "core_rules",
    "evaluate",
    "layered",
    "load_rules",
    "RuleSet",
    "rules_from_document",
    "RULES_SCHEMA_VERSION",
    "Unstated",
]
