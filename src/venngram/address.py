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
Addresses
^^^^^^^^^

An **address** selects a slice of a model: tensors (for tuning), activation sites (for steering and capture), or units
inside either (neurons, latent features, featurizer blocks, subspaces). Its canonical form is a mapping whose fields
intersect; its string form is shorthand for the same record:

=========  ======================================================================  ===============================
field      value                                                                   example
=========  ======================================================================  ===============================
role       a role selector (:mod:`venngram.roles`)                                 ``adapter``, ``adapter:top``
name       a regex over tensor names, matched in full (the only regex field)       ``"model\.layers\.2[4-7]\..*"``
module     a module path pattern, matched on whole segments; ``*`` is one          ``q_proj``, ``self_attn.q_proj``
           segment, ``**`` any number; a tensor matches when it is under a
           matching module
layers     an index, a list, a half-open slice (negative indices count from the    ``-4:``, ``8:-4``, ``[3, 7]``
           depth), ``odd`` or ``even``
stack      a block stack name (the primary stack when unset)                       ``vision``
component  an architecture-neutral component (needs a vocabulary provider)         ``attn.q``, ``mlp.*``
point      an activation point: makes the target a site                            ``resid_post``
features   a latent model or an external feature id: makes the target its          ``sae``
           feature space
subspace   a named basis at a site: makes the target a subspace                    ``entailment_dir``
space      a registered linear map the site is read through                        ``jlens``
leaf       a tensor-name suffix, matched on whole segments                         ``weight``, ``lora_A.*``
units      an index set along the target's unit axis: a list, a slice, or per      ``[431, 1007]``
           layer
=========  ======================================================================  ===============================

The string form:

- a term is ``<field>:<value>``, or a reference ``@<set>`` (``@<import>.<set>``) to a named set of an
  :class:`~venngram.book.AddressBook`;
- ``&`` intersects, ``-`` subtracts and ``|`` unites; ``&`` binds tighter than ``-``, which binds tighter than ``|``;
  parentheses group; operators are delimited by whitespace, so values keep their own ``-`` and ``:``;
- a ``name`` value is quoted (``name:"layers\.2[4-7]\."``); list and mapping values are bracketed.

In the mapping form, the fields of one mapping intersect, ``any: [...]`` unites, ``all: [...]`` intersects,
``exclude: ...`` subtracts, ``optional: true`` allows an empty match and ``expect: N`` requires exactly ``N`` members.

>>> parse("role:adapter & layers:-4:")
Term(role='adapter', layers=LayerSet('-4:'))
>>> to_string(parse({"role": "adapter", "exclude": "@top"}))
'role:adapter - @top'
>>> is_address("model.layers.0.weight"), is_address("(q|k)_proj"), is_address("(role:adapter | @top)")
(False, False, True)

.. warning:: This is an experimental API and may change in a future release.
"""
import ast
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Union

from venngram.errors import AddressError
from venngram.roles import parse_selector

FIELDS = ("role", "name", "module", "layers", "stack", "component", "point", "features", "subspace", "space", "leaf",
          "units")
COMBINATORS = ("any", "all", "exclude")
MODIFIERS = ("optional", "expect")
TARGETS = ("param", "site", "feature", "subspace")
_PREFIX = re.compile(r"^(%s):" % "|".join(FIELDS))
_SET_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*(\.[A-Za-z_][A-Za-z0-9_-]*)*")
_TOKEN = re.compile(
    r"\s*(?:(?P<op>[&|()]|-(?=\s))|(?P<ref>@[A-Za-z_][A-Za-z0-9_.-]*)"
    r"|(?P<term>(?:%s):(?:\"(?:[^\"\\]|\\.)*\"|\[[^\]]*\]|\{[^{}]*\}|[^\s&|()]+)))" % "|".join(FIELDS))
_PRECEDENCE = {"|": 1, "-": 2, "&": 3}
_SLICE = re.compile(r"^(-?\d+)?:(-?\d+)?$")


def _literal(raw: str, what: str) -> Any:
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError) as err:
        raise AddressError(f"Cannot read the {what} value {raw!r}: {err}.") from err


def _slice_of(text: str) -> tuple[int | None, int | None] | None:
    m = _SLICE.match(text.strip())
    if not m:
        return None
    return (int(m.group(1)) if m.group(1) else None, int(m.group(2)) if m.group(2) else None)


@dataclass(frozen=True)
class LayerSet:
    """A set of block indices: explicit indices, a half-open slice, or a parity.

    Negative indices and slice ends count from the stack's depth, so ``-4:`` is the top four blocks of any model.

    >>> LayerSet.parse("8:-4").select(28)
    (8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)
    >>> LayerSet.parse([3, -1]).select(12), LayerSet.parse("odd").select(6)
    ((3, 11), (1, 3, 5))
    """

    spec: str
    indices: tuple[int, ...] | None = None
    bounds: tuple[int | None, int | None] | None = None
    parity: str | None = None

    def __repr__(self) -> str:
        return f"LayerSet({self.spec!r})"

    @classmethod
    def parse(cls, raw: Any) -> "LayerSet":
        """Read a layer set from an int, a list of ints, or a string (``13``, ``[3, 7]``, ``-4:``, ``odd``).

        Raises:
            AddressError: If ``raw`` is none of these.
        """
        if isinstance(raw, LayerSet):
            return raw
        if isinstance(raw, bool):
            raise AddressError(f"A layer set cannot be a boolean, got {raw!r}.")
        if isinstance(raw, int):
            return cls(spec=str(raw), indices=(raw,))
        if isinstance(raw, (list, tuple)):
            if not raw or not all(isinstance(i, int) and not isinstance(i, bool) for i in raw):
                raise AddressError(f"A layer list must be a non-empty list of integers, got {raw!r}.")
            return cls(spec=f"[{', '.join(str(i) for i in raw)}]", indices=tuple(raw))
        if not isinstance(raw, str):
            raise AddressError(f"A layer set is an index, a list, a slice or a parity, got {raw!r}.")
        text = raw.strip()
        if text in ("odd", "even"):
            return cls(spec=text, parity=text)
        if text.startswith("["):
            return cls.parse(_literal(text, "layers"))
        if re.fullmatch(r"-?\d+", text):
            return cls.parse(int(text))
        bounds = _slice_of(text)
        if bounds is None:
            raise AddressError(
                f"Cannot read the layer set {raw!r}: use an index (`13`), a list (`[3, 7]`), a half-open slice (`-4:`,"
                " `8:-4`) or a parity (`odd`, `even`). A dash range (`4-11`) is not a layer set: slices are half-open.")
        return cls(spec=text, bounds=bounds)

    def select(self, depth: int) -> tuple[int, ...]:
        """The selected block indices of a stack ``depth`` blocks deep, ascending.

        Raises:
            AddressError: If an explicit index lies outside the stack.
        """
        if self.parity is not None:
            return tuple(i for i in range(depth) if i % 2 == (1 if self.parity == "odd" else 0))
        if self.bounds is not None:
            return tuple(range(depth)[slice(*self.bounds)])
        out = set()
        for i in self.indices or ():
            if not -depth <= i < depth:
                raise AddressError(f"Layer {i} is outside a stack of {depth} blocks.")
            out.add(i % depth)
        return tuple(sorted(out))


@dataclass(frozen=True)
class UnitSet:
    """Indices along a target's unit axis: a list or a slice, for every selected layer or per layer.

    >>> UnitSet.parse("[431, 1007]").for_layer(12, 4096)
    (431, 1007)
    >>> UnitSet.parse({12: [431], 14: "8:10"}).for_layer(14, 4096)
    (8, 9)
    """

    spec: str
    flat: LayerSet | None = None
    per_layer: tuple[tuple[int, LayerSet], ...] | None = None

    def __repr__(self) -> str:
        return f"UnitSet({self.spec!r})"

    @classmethod
    def parse(cls, raw: Any) -> "UnitSet":
        """Read a unit set from a list, a slice string, or a mapping from layer to either.

        Raises:
            AddressError: If ``raw`` is none of these.
        """
        if isinstance(raw, UnitSet):
            return raw
        if isinstance(raw, str) and raw.strip().startswith("{"):
            raw = _literal(raw.strip(), "units")
        if isinstance(raw, Mapping):
            if not raw:
                raise AddressError("A per-layer unit set must name at least one layer.")
            per = []
            for layer, units in raw.items():
                if not isinstance(layer, int) or isinstance(layer, bool):
                    raise AddressError(f"A per-layer unit set is keyed by layer index, got {layer!r}.")
                per.append((layer, cls._indices(units)))
            spec = "{" + ", ".join(f"{k}: {v.spec}" for k, v in per) + "}"
            return cls(spec=spec, per_layer=tuple(per))
        flat = cls._indices(raw)
        return cls(spec=flat.spec, flat=flat)

    @staticmethod
    def _indices(raw: Any) -> LayerSet:
        # unit indices share the layer set's grammar (indices, half-open slices), but never a parity
        try:
            parsed = LayerSet.parse(raw)
        except AddressError as err:
            raise AddressError(f"Cannot read the unit set {raw!r}: use a list (`[431, 1007]`) or a half-open slice"
                               " (`0:64`).") from err
        if parsed.parity is not None:
            raise AddressError(f"A unit set takes indices or a slice, not a parity ({raw!r}).")
        return parsed

    @property
    def layers(self) -> tuple[int, ...] | None:
        """The layers a per-layer unit set names, or ``None`` for a flat one."""
        return None if self.per_layer is None else tuple(k for k, _ in self.per_layer)

    def for_layer(self, layer: int | None, width: int) -> tuple[int, ...]:
        """The unit indices at ``layer`` along an axis of ``width`` units (empty for a layer the set does not name)."""
        if self.flat is not None:
            return self.flat.select(width)
        for k, units in self.per_layer or ():
            if k == layer:
                return units.select(width)
        return ()


def _check_field(name: str, value: Any) -> Any:
    if name == "layers":
        return LayerSet.parse(value)
    if name == "units":
        return UnitSet.parse(value)
    if not isinstance(value, str) or not value:
        raise AddressError(f"The `{name}` field takes a non-empty string, got {value!r}.")
    if name == "role":
        parse_selector(f"role:{value}")
    if name == "name":
        try:
            re.compile(value)
        except re.error as err:
            raise AddressError(f"The `name` field {value!r} is not a valid regex: {err}.") from err
    return value


@dataclass(frozen=True, repr=False)
class Term:
    """One record: the intersection of the fields it sets (unset fields do not filter).

    Raises:
        AddressError: If a field's value is invalid, no field is set, or two target fields conflict.
    """

    role: str | None = None
    name: str | None = None
    module: str | None = None
    layers: LayerSet | None = None
    stack: str | None = None
    component: str | None = None
    point: str | None = None
    features: str | None = None
    subspace: str | None = None
    space: str | None = None
    leaf: str | None = None
    units: UnitSet | None = None

    def __post_init__(self) -> None:
        for f in FIELDS:
            value = getattr(self, f)
            if value is not None:
                object.__setattr__(self, f, _check_field(f, value))
        if not any(getattr(self, f) is not None for f in FIELDS):
            raise AddressError("An address term must set at least one field.")

    def check_target(self) -> None:
        """Check the fields that only make sense together: run on a complete conjunction, since the string form writes
        each field as its own term.

        Raises:
            AddressError: If the term names both ``features`` and a ``subspace``, or a ``space`` without a ``point``.
        """
        if self.features is not None and self.subspace is not None:
            raise AddressError("An address names `features` or a `subspace`, not both: a featurizer block's span is "
                               "reached from its `features` address.")
        if self.space is not None and self.point is None:
            raise AddressError(f"`space:{self.space}` reads a site through a linear map, so it needs a `point`.")

    @property
    def target(self) -> str:
        """What the term addresses: ``param``, ``site``, ``feature`` or ``subspace``."""
        if self.features is not None:
            return "feature"
        if self.subspace is not None:
            return "subspace"
        return "site" if self.point is not None else "param"

    def set_fields(self) -> dict[str, Any]:
        """The fields this term sets, in canonical order."""
        return {f: getattr(self, f) for f in FIELDS if getattr(self, f) is not None}

    def __repr__(self) -> str:
        return "Term(" + ", ".join(f"{k}={v!r}" for k, v in self.set_fields().items()) + ")"


@dataclass(frozen=True)
class Ref:
    """A reference to a named set (``@top``, ``@lora.top``)."""

    name: str

    def __post_init__(self) -> None:
        if not _SET_NAME.fullmatch(self.name):
            raise AddressError(f"`@{self.name}` is not a valid set reference (letters, digits, `_` and `-`, with `.`"
                               " between an import and its set).")


@dataclass(frozen=True)
class And:
    """The intersection of its items."""

    items: tuple["Expr", ...]


@dataclass(frozen=True)
class Or:
    """The union of its items."""

    items: tuple["Expr", ...]


@dataclass(frozen=True)
class Minus:
    """``left`` without ``right``."""

    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Guarded:
    """An expression with the ``optional`` or ``expect`` modifiers of the mapping that wrote it."""

    expr: "Expr"
    optional: bool = False
    expect: int | None = None


Expr = Union[Term, Ref, And, Or, Minus, Guarded]


def is_address(entry: Any) -> bool:
    """Whether ``entry`` is an address rather than a bare name or regex.

    A mapping is always an address. A string is one only when, after any leading ``(``, it starts with a field prefix or
    ``@``, so every bare name and regex (including one that starts with a group, ``(q|k)_proj``) keeps its meaning.
    """
    if isinstance(entry, Mapping):
        return True
    if not isinstance(entry, str):
        return False
    body = entry.lstrip("( \t")
    return bool(_PREFIX.match(body)) or body.startswith("@")


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens, pos = [], 0
    while pos < len(text):
        if not text[pos:].strip():
            break
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise AddressError(f"Cannot parse the address {text!r} at {text[pos:].strip()!r}: a term is "
                               "`<field>:<value>` or `@<set>`, and operators (`&`, `|`, `-`) are delimited by spaces.")
        kind = m.lastgroup or ""
        tokens.append((kind, m.group(kind)))
        pos = m.end()
    return tokens


def _term_from_token(token: str) -> Term:
    field_name, _, value = token.partition(":")
    if field_name == "name":
        if not (len(value) >= 2 and value[0] == value[-1] == '"'):
            raise AddressError(f'A `name` value is quoted in the string form: name:"{value}".')
        value = value[1:-1].replace('\\"', '"')
    return Term(**{field_name: _check_field(field_name, value)})


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = _tokenize(text)
        self.pos = 0

    def parse(self) -> Expr:
        if not self.tokens:
            raise AddressError("An empty string is not an address.")
        expr = self._expr(1)
        if self.pos != len(self.tokens):
            raise AddressError(f"Unexpected {self.tokens[self.pos][1]!r} in the address {self.text!r}.")
        return expr

    def _atom(self) -> Expr:
        if self.pos >= len(self.tokens):
            raise AddressError(f"The address {self.text!r} ends where a term was expected.")
        kind, value = self.tokens[self.pos]
        self.pos += 1
        if kind == "op" and value == "(":
            expr = self._expr(1)
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ("op", ")"):
                raise AddressError(f"Unbalanced parentheses in the address {self.text!r}.")
            self.pos += 1
            return expr
        if kind == "ref":
            return Ref(value[1:])
        if kind == "term":
            return _term_from_token(value)
        raise AddressError(f"Unexpected {value!r} in the address {self.text!r}.")

    def _expr(self, min_prec: int) -> Expr:
        lhs = self._atom()
        while self.pos < len(self.tokens):
            kind, op = self.tokens[self.pos]
            if kind != "op" or op not in _PRECEDENCE or _PRECEDENCE[op] < min_prec:
                break
            self.pos += 1
            rhs = self._expr(_PRECEDENCE[op] + 1)
            lhs = _combine(op, lhs, rhs)
        return lhs


def _merge_terms(items: Sequence[Expr]) -> tuple[Expr, ...]:
    # a conjunction's terms merge into one record while their fields are disjoint (a CSS compound selector), so
    # `role:adapter & layers:-4:` is the record {role: adapter, layers: -4:}; a repeated field stays a separate term
    merged: list[Expr] = []
    current: dict[str, Any] = {}
    for item in items:
        if isinstance(item, Term) and not set(item.set_fields()) & set(current):
            current.update(item.set_fields())
            continue
        merged.append(item)
    if current:
        merged.insert(0, Term(**current))
    return tuple(merged)


def _combine(op: str, lhs: Expr, rhs: Expr) -> Expr:
    if op == "-":
        return Minus(lhs, rhs)
    cls = And if op == "&" else Or
    items = (lhs.items if isinstance(lhs, cls) else (lhs,)) + (rhs.items if isinstance(rhs, cls) else (rhs,))
    if cls is And:
        items = _merge_terms(items)
        if len(items) == 1:
            return items[0]
    return cls(items)


def _entries(raw: Any, key: str) -> tuple[Expr, ...]:
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    if not items:
        raise AddressError(f"`{key}` needs at least one entry.")
    return tuple(parse(i) for i in items)


def _from_mapping(raw: Mapping[str, Any]) -> Expr:
    unknown = [k for k in raw if k not in FIELDS + COMBINATORS + MODIFIERS]
    if unknown:
        raise AddressError(f"Unknown address keys {unknown}: the fields are {', '.join(FIELDS)}, the combinators "
                           f"{', '.join(COMBINATORS)} and the modifiers {', '.join(MODIFIERS)}.")
    parts: list[Expr] = []
    field_values = {k: raw[k] for k in FIELDS if k in raw}
    if field_values:
        parts.append(Term(**field_values))
    if "all" in raw:
        parts.extend(_entries(raw["all"], "all"))
    if "any" in raw:
        alts = _entries(raw["any"], "any")
        parts.append(alts[0] if len(alts) == 1 else Or(alts))
    if not parts:
        raise AddressError(f"The address {dict(raw)!r} selects nothing: set a field, `all` or `any`.")
    merged = _merge_terms(parts)
    expr: Expr = merged[0] if len(merged) == 1 else And(merged)
    if "exclude" in raw:
        excluded = _entries(raw["exclude"], "exclude")
        expr = Minus(expr, excluded[0] if len(excluded) == 1 else Or(excluded))
    optional, expect = bool(raw.get("optional", False)), raw.get("expect")
    if optional or expect is not None:
        if expect is not None and (not isinstance(expect, int) or isinstance(expect, bool) or expect < 0):
            raise AddressError(f"`expect` is a member count, got {expect!r}.")
        expr = Guarded(expr, optional=optional, expect=expect)
    return expr


def parse(entry: Any) -> Expr:
    """Parse an address from its string or mapping form (an already parsed address is returned unchanged).

    Raises:
        AddressError: If ``entry`` is not a valid address. A bare name or regex is not an address: in an address, a
            regex is written ``name:"..."``.
    """
    if isinstance(entry, (Term, Ref, And, Or, Minus, Guarded)):
        return entry
    if isinstance(entry, Mapping):
        return _from_mapping(entry)
    if not isinstance(entry, str):
        raise AddressError(f"An address is a string or a mapping, got {type(entry).__name__} ({entry!r}).")
    if not is_address(entry):
        raise AddressError(f"{entry!r} is not an address: a term is `<field>:<value>` or `@<set>`, and a regex over "
                           f'tensor names is written name:"{entry}".')
    return _Parser(entry).parse()


def _value_string(name: str, value: Any) -> str:
    if name == "name":
        return '"' + value.replace('"', '\\"') + '"'
    if isinstance(value, (LayerSet, UnitSet)):
        return value.spec.replace(" ", "")
    return str(value)


def to_string(expr: Expr, _parent: int = 0) -> str:
    """The string form of ``expr`` (``optional`` and ``expect`` have no string form and are dropped)."""
    if isinstance(expr, Guarded):
        return to_string(expr.expr, _parent)
    if isinstance(expr, Term):
        terms = [f"{k}:{_value_string(k, v)}" for k, v in expr.set_fields().items()]
        out, prec = " & ".join(terms), (_PRECEDENCE["&"] if len(terms) > 1 else 4)
    elif isinstance(expr, Ref):
        return f"@{expr.name}"
    elif isinstance(expr, Minus):
        prec = _PRECEDENCE["-"]
        out = f"{to_string(expr.left, prec)} - {to_string(expr.right, prec + 1)}"
    else:
        op = "&" if isinstance(expr, And) else "|"
        prec = _PRECEDENCE[op]
        out = f" {op} ".join(to_string(i, prec) for i in expr.items)  # both are associative
    return f"({out})" if prec < _parent else out


def _mapping_value(name: str, value: Any) -> Any:
    if isinstance(value, LayerSet):
        indices: tuple[int, ...] | None = value.indices
        if indices is None:
            return value.spec
        return indices[0] if len(indices) == 1 else list(indices)
    if isinstance(value, UnitSet):
        if value.per_layer is not None:
            return {k: _mapping_value("layers", v) for k, v in value.per_layer}
        return _mapping_value("layers", value.flat)
    return value


def to_mapping(expr: Expr) -> Any:
    """The canonical (mapping) form of ``expr``: what an address book stores. A reference stays a ``@`` string."""
    if isinstance(expr, Ref):
        return f"@{expr.name}"
    if isinstance(expr, Term):
        return {k: _mapping_value(k, v) for k, v in expr.set_fields().items()}
    if isinstance(expr, Guarded):
        inner = to_mapping(expr.expr)
        out: dict[str, Any] = dict(inner) if isinstance(inner, dict) else {"all": [inner]}
        if expr.optional:
            out["optional"] = True
        if expr.expect is not None:
            out["expect"] = expr.expect
        return out
    if isinstance(expr, Minus):
        excluded = [expr.right]
        left = expr.left
        while isinstance(left, Minus):  # a - b - c is a without (b or c)
            excluded.insert(0, left.right)
            left = left.left
        base = to_mapping(left)
        res: dict[str, Any] = dict(base) if isinstance(left, Term) else {"all": [base]}
        res["exclude"] = to_mapping(excluded[0]) if len(excluded) == 1 else {"any": [to_mapping(e) for e in excluded]}
        return res
    key = "all" if isinstance(expr, And) else "any"
    return {key: [to_mapping(i) for i in expr.items]}


def references(expr: Expr) -> list[str]:
    """Every set name ``expr`` references, in order of first appearance."""
    out: list[str] = []

    def walk(e: Expr) -> None:
        if isinstance(e, Ref):
            if e.name not in out:
                out.append(e.name)
        elif isinstance(e, Guarded):
            walk(e.expr)
        elif isinstance(e, Minus):
            walk(e.left)
            walk(e.right)
        elif isinstance(e, (And, Or)):
            for i in e.items:
                walk(i)

    walk(expr)
    return out


def rename_refs(expr: Expr, rename: Callable[[str], str]) -> Expr:
    """``expr`` with every reference renamed by ``rename`` (an imported book's sets are namespaced this way)."""
    if isinstance(expr, Ref):
        return Ref(rename(expr.name))
    if isinstance(expr, Guarded):
        return Guarded(rename_refs(expr.expr, rename), optional=expr.optional, expect=expr.expect)
    if isinstance(expr, Minus):
        return Minus(rename_refs(expr.left, rename), rename_refs(expr.right, rename))
    if isinstance(expr, (And, Or)):
        return type(expr)(tuple(rename_refs(i, rename) for i in expr.items))
    return expr


def terms(expr: Expr) -> list[Term]:
    """Every term in ``expr`` (references are not followed)."""
    if isinstance(expr, Term):
        return [expr]
    if isinstance(expr, Guarded):
        return terms(expr.expr)
    if isinstance(expr, Minus):
        return terms(expr.left) + terms(expr.right)
    if isinstance(expr, (And, Or)):
        return [t for i in expr.items for t in terms(i)]
    return []


def field_names() -> Sequence[str]:
    """The address fields, in canonical order."""
    return tuple(f.name for f in fields(Term) if f.name in FIELDS)


__all__ = [
    "And",
    "COMBINATORS",
    "Expr",
    "FIELDS",
    "field_names",
    "Guarded",
    "is_address",
    "LayerSet",
    "Minus",
    "MODIFIERS",
    "Or",
    "parse",
    "Ref",
    "references",
    "rename_refs",
    "TARGETS",
    "Term",
    "terms",
    "to_mapping",
    "to_string",
    "UnitSet",
]
