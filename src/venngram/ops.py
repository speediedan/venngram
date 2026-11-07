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
Recipe Ops
^^^^^^^^^^

What a recipe op may be, as declared specs: venngram validates and composes ops, and never executes one (applying an op
is the consumer's). An :class:`OpSpec` declares an op's operand kinds, its parameters, the declarations it requires (a
``basis``), the targets it may act on, whether its value is required, whether it may override its stage's token scope,
and how two of it compose on one target (``sum``, ``product``, ``max``, ``min``, or ``ordered``: never merged).

The core ops (:data:`OPS`): ``replace``, ``add`` (sums), ``scale`` (multiplies), ``project`` and ``reject`` (keep or
remove the component in a span, a ``fraction`` of it), ``patch`` (swap coordinates along a pair of directions) and
``clamp`` (bound coordinates in a basis). :func:`register_op` adds one.

Values pass through a **transform pipeline** from a small declared vocabulary (:data:`TRANSFORMS`), applied in order
to each member of the op: each unit or feature its target set names, or the whole activation, one member, for an op
without a target. A member's value starts as the op's ``dose``, its own entry in ``doses``, or its own value in the
``from`` column of the evidence table, and each transform maps it to the next:

====================================  =============================================================================
transform                             value for a member, given ``v``
====================================  =============================================================================
``scale {factor: f}``                 ``v * f``
``abs``                               ``|v|``
``normalize {by: max_abs}``           ``v / max(|v_m|)`` over the op's members ``m`` (not the whole evidence table);
                                      zero for every member when every value is zero
``normalize {by: <statistic>}``       ``v / t``: expresses ``v`` in units of the named statistic ``t``
``statistic {name: <statistic>}``     ``v * t``: ``v`` is a dose in units of the named statistic ``t``, made absolute
``sign_of {column: c}``               ``sign(e_c) * |v|``, for the member's value ``e_c`` in column ``c`` of the
                                      ``from`` evidence table
``sign_aware_scale {factor: f}``      ``sign(s) * |v| * f``, for the member's score ``s`` in the ``from`` column:
                                      ``sign_of`` that column, then ``scale``
====================================  =============================================================================

``sign(0)`` is ``0``. ``sign_of`` and ``sign_aware_scale`` read the evidence table, so they apply only to a ``from``
value (a recipe giving them a ``dose`` or ``doses`` is refused when read). A statistic is computed from the model, not
recorded: the consumer supplies ``t`` for each member (a value per layer, or per unit, such as a layer's median
residual norm over a unit's output column norm), declares ``statistic`` in its support record, and refuses a
statistic it cannot compute, or one that is zero for ``normalize``, by name. ``statistic`` and ``normalize`` by the
same statistic are inverses. :func:`register_transform` adds a transform, with a stated meaning and conformance
vectors of its own.

:func:`transform_cases` returns conformance vectors (each a pipeline, its members' inputs and the expected values), and
:func:`check_transforms` runs a consumer's evaluator against them, so two consumers that honour a recipe are tested to
give it the same values.

A consumer declares what it honours as a :class:`Support` record, in code or as the ``steering`` section of a
``support_record`` document (:mod:`venngram.support`); :func:`check_support` lists every refusal by name before anything
runs.

.. warning:: This is an experimental API and may change in a future release.
"""
import builtins
import fnmatch
import math
import re
from collections.abc import Callable, Mapping, Sequence
from importlib import resources
from dataclasses import dataclass, field
from typing import Any

import yaml

from venngram.address import OF

OPERAND_KINDS = ("none", "vector", "pair", "subspace")
TARGETS = ("activation", "units", "features")
TOKEN_SCOPES = ("all", "last", "prompt", "generated", "positions", "slice")
VALUE_SOURCES = ("dose", "doses", "from")
COMPOSITION = ("sum", "product", "max", "min", "ordered")
_COMBINE: dict[str, Callable[[float, float], float]] = {"sum": lambda a, b: a + b, "product": lambda a, b: a * b,
                                                        "max": max, "min": min}


@dataclass(frozen=True)
class ParamSpec:
    """One op parameter: its type, whether it is required, and an optional inclusive range."""

    type: builtins.type = float
    required: bool = False
    range: tuple[float, float] | None = None


@dataclass(frozen=True)
class OpSpec:
    """An op as data (see the module docstring)."""

    name: str
    operand: tuple[str, ...] = ("none",)
    params: Mapping[str, ParamSpec] = field(default_factory=dict)
    requires: tuple[str, ...] = ()
    targets: tuple[str, ...] = TARGETS
    value: str = "optional"
    """``required``, ``optional`` or ``none``."""
    token_override: bool = False
    composition: str = "ordered"
    at_least_one_of: tuple[str, ...] = ()
    """Parameters of which at least one must be given (a clamp needs a bound)."""

    def combine(self, a: float, b: float) -> float:
        """Two values of this op on one target, combined (only for a ``sum``/``product``/``max``/``min`` op)."""
        return _COMBINE[self.composition](a, b)


_FRACTION = ParamSpec(float, False, (0.0, 1.0))

#: The registered ops, by name.
OPS: dict[str, OpSpec] = {spec.name: spec for spec in (
    OpSpec("replace", ("none", "vector"), targets=TARGETS),
    OpSpec("add", ("none", "vector"), composition="sum"),
    OpSpec("scale", ("none",), value="required", composition="product"),
    OpSpec("project", ("vector", "subspace"), {"fraction": _FRACTION}, ("basis",), ("activation",), "none"),
    OpSpec("reject", ("vector", "subspace"), {"fraction": _FRACTION}, ("basis",), ("activation",), "none"),
    OpSpec("patch", ("pair",), {}, ("basis",), ("activation",), "none"),
    OpSpec("clamp", ("vector", "subspace"), {"min": ParamSpec(), "max": ParamSpec()}, ("basis",), ("activation",),
           "none", at_least_one_of=("min", "max")),
)}

#: Schema 1 op names read by :func:`venngram.recipe.migrate`.
ALIASES = {"mul": "scale"}

#: Value transforms, by name, with their parameters (what each computes: the module docstring).
TRANSFORMS: dict[str, Mapping[str, ParamSpec]] = {
    "scale": {"factor": ParamSpec(float, True)},
    "abs": {},
    "normalize": {"by": ParamSpec(str, True)},          # max_abs over the op's members, or v / a named statistic
    "sign_of": {"column": ParamSpec(str, True)},
    "sign_aware_scale": {"factor": ParamSpec(float, True)},
    "statistic": {"name": ParamSpec(str, True)},        # v * a statistic the consumer computes, e.g. a layer median
}

#: What each transform computes, by name (the table above for the core ones; :func:`register_transform` adds one).
TRANSFORM_MEANINGS: dict[str, str] = {
    "scale": "v * factor",
    "abs": "|v|",
    "normalize": "by max_abs: v / max |v_m| over the op's members (zero throughout when every value is zero); by a "
                 "statistic t: v / t",
    "sign_of": "sign(e_c) * |v|, for the member's value e_c in `column` of the `from` evidence table",
    "sign_aware_scale": "sign(s) * |v| * factor, for the member's score s in the `from` column",
    "statistic": "v * t, for the named statistic t the consumer supplies",
}

#: Transforms that read the ``from`` evidence table, so apply only to a ``from`` value.
EVIDENCE_TRANSFORMS: set[str] = {"sign_of", "sign_aware_scale"}

_REGISTERED_CASES: dict[str, tuple[Any, ...]] = {}


@dataclass(frozen=True)
class Transform:
    """One step of a value's transform pipeline: a :data:`TRANSFORMS` name and its parameters."""

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransformMember:
    """One member's inputs to a conformance vector: its starting value, its row of the ``from`` evidence table, and
    the statistics the consumer would supply for it."""

    start: float
    columns: Mapping[str, float] = field(default_factory=dict)
    statistics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TransformCase:
    """A conformance vector: a pipeline over an op's members, and the value each member must end with.

    ``source`` is ``dose``, ``doses`` or ``from``; ``column`` is the ``from`` column (``None`` otherwise), whose value
    in each member's ``columns`` is that member's ``start``.
    """

    name: str
    source: str
    column: str | None
    transform: tuple[Transform, ...]
    members: tuple[TransformMember, ...]
    expect: tuple[float, ...]


def transform_cases() -> tuple[TransformCase, ...]:
    """The transform conformance vectors: venngram's own (``data/transform_vectors.yaml``, covering every core
    transform), then those each :func:`register_transform` supplied."""
    doc = yaml.safe_load((resources.files("venngram") / "data" / "transform_vectors.yaml").read_text("utf-8"))
    return tuple(_case(raw) for raw in doc["cases"]) + tuple(c for cs in _REGISTERED_CASES.values() for c in cs)


def register_transform(name: str, params: Mapping[str, ParamSpec], *, meaning: str,
                       cases: Sequence[Mapping[str, Any]], reads_evidence: bool = False, replace: bool = False) -> None:
    """Register a value transform, so recipes may name it and consumers may declare it.

    A transform is shared only with what it computes: ``meaning`` states it, and ``cases`` are conformance vectors in
    the ``data/transform_vectors.yaml`` form (each a pipeline using it, its members' inputs and the expected values),
    which :func:`check_transforms` then runs against every consumer's evaluator.

    Raises:
        ValueError: If the name is taken and ``replace`` is not set, ``meaning`` is empty, no case uses the transform,
            a case is malformed, or a transform that reads the evidence table has a case whose value is not ``from``.
    """
    if name in TRANSFORMS and not replace:
        raise ValueError(f"transform {name!r} is already registered; pass replace=True to replace it")
    if not isinstance(meaning, str) or not meaning.strip():
        raise ValueError(f"transform {name!r}: state what it computes (`meaning`)")
    known = {**TRANSFORMS, name: params}
    try:
        parsed = tuple(_case(c) for c in cases)
    except (KeyError, StopIteration, TypeError, ValueError) as err:
        raise ValueError(f"transform {name!r}: a conformance case is malformed ({type(err).__name__}: {err})") from err
    if not any(t.name == name for c in parsed for t in c.transform):
        raise ValueError(f"transform {name!r}: no conformance case uses it")
    for c in parsed:
        unknown = [t.name for t in c.transform if t.name not in known]
        if unknown:
            raise ValueError(f"transform {name!r}: case {c.name!r} uses unknown transforms {unknown}")
        if len(c.expect) != len(c.members):
            raise ValueError(f"transform {name!r}: case {c.name!r} expects {len(c.expect)} values for "
                             f"{len(c.members)} members")
        if reads_evidence and c.source != "from":
            raise ValueError(f"transform {name!r} reads the evidence table, so case {c.name!r} needs a `from` value")
    TRANSFORMS[name] = params
    TRANSFORM_MEANINGS[name] = meaning
    _REGISTERED_CASES[name] = parsed
    if reads_evidence:
        EVIDENCE_TRANSFORMS.add(name)
    else:
        EVIDENCE_TRANSFORMS.discard(name)


def _case(raw: Mapping[str, Any]) -> TransformCase:
    value = raw["value"]
    source = next(iter(value))
    column = value["from"] if source == "from" else None
    members = []
    for i, m in enumerate(raw["members"]):
        columns = {str(k): float(v) for k, v in (m.get("columns") or {}).items()}
        start = {"dose": lambda: value["dose"], "doses": lambda: value["doses"][i],
                 "from": lambda: columns[str(column)]}[source]()
        members.append(TransformMember(float(start), columns,
                                       {str(k): float(v) for k, v in (m.get("statistics") or {}).items()}))
    transform = tuple(Transform(item, {}) if isinstance(item, str) else Transform(*next(iter(item.items())))
                      for item in raw["transform"])
    return TransformCase(raw["name"], source, column, transform, tuple(members), tuple(float(e) for e in raw["expect"]))


def check_transforms(evaluate: Callable[[TransformCase], Sequence[float]], *, rel_tol: float = 1e-9,
                     abs_tol: float = 1e-12) -> list[str]:
    """Every conformance vector ``evaluate`` (a consumer's evaluator, adapted to take a :class:`TransformCase` and
    return one value per member) gets wrong, by name; empty when it agrees with venngram's meanings."""
    out: list[str] = []
    for case in transform_cases():
        try:
            got = [float(v) for v in evaluate(case)]
        except Exception as err:  # a consumer's refusal or bug, reported by case rather than raised
            out.append(f"{case.name}: raised {type(err).__name__}: {err}")
            continue
        ok = len(got) == len(case.expect) and all(
            math.isclose(g, e, rel_tol=rel_tol, abs_tol=abs_tol) for g, e in zip(got, case.expect))
        if not ok:
            out.append(f"{case.name}: expected {list(case.expect)}, got {got}")
    return out


def register_op(spec: OpSpec, *, replace: bool = False) -> None:
    """Register an op spec.

    Raises:
        ValueError: If the name is taken and ``replace`` is not set, or the spec names an unknown operand kind, target
            or composition.
    """
    if spec.name in OPS and not replace:
        raise ValueError(f"op {spec.name!r} is already registered; pass replace=True to replace it")
    bad = ([k for k in spec.operand if k not in OPERAND_KINDS] + [t for t in spec.targets if t not in TARGETS]
           + ([spec.composition] if spec.composition not in COMPOSITION else [])
           + ([spec.value] if spec.value not in ("required", "optional", "none") else []))
    if bad:
        raise ValueError(f"op {spec.name!r}: unknown {bad}")
    OPS[spec.name] = spec


@dataclass(frozen=True)
class Support:
    """What a consumer honours. ``sites`` is a set of site names, or a predicate over them; ``None`` fields mean any.

    A consumer may publish its record as a ``support_record`` document (:meth:`from_mapping`), so a recipe can be
    checked against it without the consumer installed:

    .. code-block:: yaml

        schema_version: 1
        kind: support_record
        name: example-steerer
        ops: [add, scale]
        targets: [units]                  # default: every target
        tokens: [all, prompt]             # default: [all]
        values: [dose, doses]             # default: [dose, doses]
        transforms: [statistic]           # default: none
        sites: [mlp.out.hook_in, "blocks.*"]   # names or fnmatch patterns; default: any
        bases: [unfolded]                 # default: any

    ``of`` lists what its ops may act on: ``value`` (default) and, for a consumer that intervenes on the backward pass,
    ``grad`` (a target set with ``of: grad``). It is declared in the ``steering`` section of a schema 3 record.
    """

    ops: frozenset[str]
    targets: frozenset[str] = frozenset(TARGETS)
    tokens: frozenset[str] = frozenset({"all"})
    values: frozenset[str] = frozenset({"dose", "doses"})
    transforms: frozenset[str] = frozenset()
    sites: frozenset[str] | Callable[[str], bool] | None = None
    bases: frozenset[str] | None = None
    name: str | None = None
    of: frozenset[str] = frozenset({"value"})

    def site_ok(self, site: str) -> bool:
        if self.sites is None:
            return True
        if callable(self.sites):
            return self.sites(site)
        return site in self.sites or any(fnmatch.fnmatchcase(site, p) for p in self.sites if _GLOB.search(p))

    @classmethod
    def from_mapping(cls, doc: Any, *, source: str = "<inline>") -> "Support":
        """Read a ``support_record`` document, refusing a malformed one by name.

        Raises:
            ValueError: If the document is not a schema 1 ``support_record``, has unknown keys, or names a target,
                token scope or value source venngram does not define.
        """
        if not isinstance(doc, Mapping) or doc.get("kind") != "support_record" or doc.get("schema_version") != 1:
            raise ValueError(f"{source}: a support record is `kind: support_record`, `schema_version: 1`.")
        unknown = set(doc) - {"schema_version", "kind", "name", "ops", *_SUPPORT_FIELDS}
        if unknown:
            raise ValueError(f"{source}: unknown support record keys {sorted(unknown)}.")
        if "ops" not in doc:
            raise ValueError(f"{source}: a support record lists the `ops` its consumer honours.")
        fields: dict[str, Any] = {}
        for key in ("ops", *_SUPPORT_FIELDS):
            if key not in doc:
                continue
            raw = doc[key]
            if not isinstance(raw, list) or not all(isinstance(v, str) for v in raw):
                raise ValueError(f"{source}: `{key}` is a list of names, got {raw!r}.")
            allowed = _SUPPORT_FIELDS.get(key)
            bad = sorted(set(raw) - set(allowed)) if allowed else []
            if bad:
                raise ValueError(f"{source}: `{key}` names {bad}, which venngram does not define (known: "
                                 f"{list(allowed or ())}).")
            fields[key] = frozenset(raw)
        name = doc.get("name")
        return cls(**fields, name=str(name) if name is not None else None)

    @classmethod
    def from_file(cls, path: Any) -> "Support":
        """Read a ``support_record`` document from YAML (loading runs no code)."""
        with open(path, encoding="utf-8") as fh:
            return cls.from_mapping(yaml.safe_load(fh), source=str(path))


#: Support record fields beyond ``ops``, with the names each may take (``None``: any name).
_SUPPORT_FIELDS: dict[str, tuple[str, ...] | None] = {"targets": TARGETS, "tokens": TOKEN_SCOPES,
                                                      "values": VALUE_SOURCES, "transforms": None, "sites": None,
                                                      "bases": None, "of": OF}
_GLOB = re.compile(r"[*?\[]")


def check_support(recipe: Any, support: Support) -> list[str]:
    """Every way ``recipe`` (a :class:`venngram.recipe.Recipe`) asks for something ``support`` does not honour."""
    out: list[str] = []
    for i, stage in enumerate(recipe.stages):
        where = f"stage {i} ({stage.site})"
        if not support.site_ok(stage.site):
            out.append(f"{where}: the site `{stage.site}` is not one this consumer reaches")
        for j, op in enumerate(stage.ops):
            at = f"{where} op {j} (`{op.op}`)"
            tokens = (op.tokens or stage.tokens).kind
            if op.op not in support.ops:
                out.append(f"{at}: the op is not supported (supported: {sorted(support.ops)})")
            if op.target_kind(recipe.book) not in support.targets:
                out.append(f"{at}: target `{op.target_kind(recipe.book)}` is not supported")
            of = (recipe.book.sets()[op.target].of or "value") if op.target is not None else "value"
            if of not in support.of:
                out.append(f"{at}: target set `{op.target}` addresses `of: {of}`, which this consumer does not act on "
                           f"(it declares {sorted(support.of)})")
            if tokens not in support.tokens:
                out.append(f"{at}: token scope `{tokens}` is not supported")
            if op.value is not None:
                if op.value.source not in support.values:
                    out.append(f"{at}: value source `{op.value.source}` is not supported")
                for t in op.value.transform:
                    if t.name not in support.transforms:
                        out.append(f"{at}: transform `{t.name}` is not supported")
            if op.operand is not None and support.bases is not None:
                basis = recipe.operands[op.operand].basis
                if basis not in support.bases:
                    out.append(f"{at}: operand `{op.operand}` is in basis `{basis}`, which is not supported")
    return out


__all__ = [
    "ALIASES",
    "check_support",
    "check_transforms",
    "COMPOSITION",
    "EVIDENCE_TRANSFORMS",
    "OPERAND_KINDS",
    "OPS",
    "OpSpec",
    "ParamSpec",
    "register_op",
    "register_transform",
    "Support",
    "TARGETS",
    "TOKEN_SCOPES",
    "Transform",
    "TransformCase",
    "TRANSFORM_MEANINGS",
    "TransformMember",
    "transform_cases",
    "TRANSFORMS",
    "VALUE_SOURCES",
]
