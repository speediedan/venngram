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

Values derived from evidence pass through a **transform pipeline** from a small declared vocabulary
(:data:`TRANSFORMS`); a transform a consumer computes from the model (a dose normalized by a layer statistic) is a
``statistic`` the consumer names and declares it supports.

A consumer declares what it honours as a :class:`Support` record, in code or as a ``support_record`` document;
:func:`check_support` lists every refusal by name before anything runs.

.. warning:: This is an experimental API and may change in a future release.
"""
import builtins
import fnmatch
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import yaml

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

#: Value transforms, by name, with their parameters.
TRANSFORMS: dict[str, Mapping[str, ParamSpec]] = {
    "scale": {"factor": ParamSpec(float, True)},
    "abs": {},
    "normalize": {"by": ParamSpec(str, True)},          # max_abs, or a named statistic
    "sign_of": {"column": ParamSpec(str, True)},
    "sign_aware_scale": {"factor": ParamSpec(float, True)},
    "statistic": {"name": ParamSpec(str, True)},        # a consumer-computed statistic, e.g. a layer residual median
}


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
    """

    ops: frozenset[str]
    targets: frozenset[str] = frozenset(TARGETS)
    tokens: frozenset[str] = frozenset({"all"})
    values: frozenset[str] = frozenset({"dose", "doses"})
    transforms: frozenset[str] = frozenset()
    sites: frozenset[str] | Callable[[str], bool] | None = None
    bases: frozenset[str] | None = None
    name: str | None = None

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
                                                      "bases": None}
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
    "COMPOSITION",
    "OPERAND_KINDS",
    "OPS",
    "OpSpec",
    "ParamSpec",
    "register_op",
    "Support",
    "TARGETS",
    "TOKEN_SCOPES",
    "TRANSFORMS",
    "VALUE_SOURCES",
]
