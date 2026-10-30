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
Recipes
^^^^^^^

A recipe is a *treatment* over an address book's sets: what is done at which sites, to which units or features, in
which order. It is kept apart from the selection (the book) and from the evidence (generations, graphs, scores, which it
names), so one selection can carry several treatments and a treatment can be re-run against a revised selection.

.. code-block:: yaml

    schema_version: 2
    kind: recipe
    name: chess-then-reject
    model: {id: Qwen/Qwen3.5-4B, revision: 3f1e..., revision_source: pinned}
    book: ./chess.yaml                       # a path relative to the recipe, or the book inline
    operands:                                # named tensors, each in a declared basis
      concept: {file: dirs.safetensors, key: concept, sha256: ..., shape: [2560], basis: unfolded}
    stages:                                  # in order; a stage applies at one site
      - site: mlp.out.hook_in
        tokens: all                          # all | last | prompt | generated | {positions: [...]} | {slice: "-4:"}
        ops:                                 # in order; each sees the previous op's output
          - {op: add, target: {set: chess}, value: {dose: 0.8}}
          - {op: add, target: {set: openings}, value: {doses: {15: {4041: 0.5}}}}
          - {op: reject, operand: concept, params: {fraction: 1.0}}
    sampling: {temperature: 0.7}
    evidence: {examples: [...]}

**Ops are declared specs** (:mod:`venngram.ops`): this module validates a recipe against them and composes recipes; it
never executes an op. Applying a recipe belongs to the consumer (e.g. interpretune steers, finetuning-scheduler tunes a
unit adapter), which declares what it honours as a :class:`~venngram.ops.Support` record and checks a recipe with
:func:`~venngram.ops.check_support` before running it. A format another tool writes is read by an adapter outside this
module (:mod:`venngram.sources`).

**Targets**: an op acts on the whole activation (no ``target``), the units a book set names, or the features of a
latent space a book set names (its ``space`` and ``features``; the consumer supplies the space's encoder). **Values**:
a ``dose``, a per-unit ``doses`` table, or ``from`` a column of recorded evidence through a ``transform`` pipeline of
declared transforms (:data:`~venngram.ops.TRANSFORMS`). **Tokens** are set per stage; an op may override them only if
its spec allows. ``prompt`` is every position of the input before any generated token; ``generated`` is every
position the consumer generates, at each decode step (a consumer that does not generate refuses it by name).

**Composition** (:func:`compose`): recipes on one model revision compose stage by stage. Without an order, ops from
different recipes at one site must commute (the same summing or multiplying op, or disjoint unit sets); otherwise the
composition must state ``order: listed``.

**Versions**: schema 2 is written; schema 1 (one ``method`` and its ``treatments``) is read through :func:`migrate`
with a deprecation warning.

.. warning:: This is an experimental API and may change in a future release.
"""
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from venngram.address import Term
from venngram.book import AddressBook
from venngram.errors import AddressError
from venngram.operands import INLINE_LIMIT, read_operand
from venngram.ops import ALIASES, OPS, OpSpec, TRANSFORMS

#: The recipe document schema this code writes.
RECIPE_SCHEMA_VERSION = 2

#: The recipe schemas this code reads (schema 1 through :func:`migrate`).
RECIPE_WINDOW = (1, 2)

#: Token scopes written as a bare string.
_BARE_SCOPES = ("all", "last", "prompt", "generated")
_APPLY_TO = {"all_tokens": "all", "last_token": "last", "prompt_tokens": "prompt", "generated_tokens": "generated"}


class RecipeSchemaWarning(DeprecationWarning):
    """A recipe was read in an older schema and migrated; re-write it in the current one."""


@dataclass(frozen=True)
class Operand:
    """A named tensor an op acts with: inline ``values`` or a sidecar ``{file, key, sha256}``, a shape and a basis."""

    name: str
    shape: tuple[int, ...]
    basis: str | None = None
    space: str | None = None
    file: str | None = None
    key: str | None = None
    sha256: str | None = None
    values: tuple[float, ...] | None = None

    @property
    def kind(self) -> str:
        return "vector" if len(self.shape) == 1 else ("pair" if self.shape[0] == 2 else "subspace")

    def load(self, base_dir: str | Path = ".") -> tuple[tuple[float, ...], tuple[int, ...]]:
        """The operand's values (row-major) and shape, reading its sidecar relative to ``base_dir``."""
        if self.values is not None:
            return self.values, self.shape
        values, shape = read_operand(Path(base_dir) / str(self.file), str(self.key), sha256=self.sha256)
        if shape != self.shape:
            raise AddressError(f"operand `{self.name}` declares shape {list(self.shape)}, its sidecar holds "
                               f"{list(shape)}.")
        return values, shape

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.values is not None:
            out["values"] = list(self.values)
        else:
            out.update(file=self.file, key=self.key, sha256=self.sha256)
        out["shape"] = list(self.shape)
        for k in ("basis", "space"):
            if getattr(self, k) is not None:
                out[k] = getattr(self, k)
        return out


@dataclass(frozen=True)
class Tokens:
    """A token scope: ``all``, ``last``, ``prompt``, ``generated``, explicit ``positions``, or a ``slice``."""

    kind: str = "all"
    positions: tuple[int, ...] | None = None
    slice: str | None = None

    def to_doc(self) -> Any:
        if self.kind in _BARE_SCOPES:
            return self.kind
        return {"positions": list(self.positions or ())} if self.kind == "positions" else {"slice": self.slice}


@dataclass(frozen=True)
class Transform:
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Value:
    """An op's value: a ``dose``, per-unit ``doses``, or ``from`` evidence through a transform pipeline."""

    source: str
    dose: float | None = None
    doses: Mapping[int, Mapping[int, float]] | None = None
    evidence: str | None = None
    column: str | None = None
    transform: tuple[Transform, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.source == "dose":
            out["dose"] = self.dose
        elif self.source == "doses":
            out["doses"] = {k: dict(v) for k, v in (self.doses or {}).items()}
        else:
            out["from"] = {"evidence": self.evidence, "column": self.column}
        if self.transform:
            out["transform"] = [{t.name: dict(t.params)} if t.params else t.name for t in self.transform]
        return out


@dataclass(frozen=True)
class Op:
    """One op application in a stage."""

    op: str
    target: str | None = None
    operand: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    value: Value | None = None
    tokens: Tokens | None = None

    @property
    def spec(self) -> OpSpec:
        return OPS[self.op]

    def target_kind(self, book: AddressBook) -> str:
        """``activation`` (no target), ``units`` or ``features``, from the target set."""
        if self.target is None:
            return "activation"
        term = book.sets()[self.target]
        return "units" if isinstance(term, Term) and term.units is not None else "features"

    def unit_doses(self, book: AddressBook, source: str = "<recipe>") -> dict[tuple[int, int], float]:
        """``(layer, unit) -> dose`` for a unit-targeted op whose value is a ``dose`` or ``doses``."""
        if self.target is None or self.value is None or self.value.source not in ("dose", "doses"):
            raise AddressError(f"{source}: `{self.op}` has no unit doses (it needs a unit set target and a dose).")
        out = {}
        for layer, unit in _units(book.sets()[self.target], self.target, source):
            dose = self.value.dose if self.value.source == "dose" else (self.value.doses or {}).get(layer, {}).get(unit)
            if dose is not None:
                out[(layer, unit)] = dose
        return out

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {"op": self.op}
        if self.target is not None:
            out["target"] = {"set": self.target}
        if self.operand is not None:
            out["operand"] = self.operand
        if self.params:
            out["params"] = dict(self.params)
        if self.value is not None:
            out["value"] = self.value.to_mapping()
        if self.tokens is not None:
            out["tokens"] = self.tokens.to_doc()
        return out


@dataclass(frozen=True)
class Stage:
    """Ops applied in order at one site, under one token scope."""

    site: str
    ops: tuple[Op, ...]
    tokens: Tokens = Tokens()

    def to_mapping(self) -> dict[str, Any]:
        return {"site": self.site, "tokens": self.tokens.to_doc(), "ops": [o.to_mapping() for o in self.ops]}


@dataclass(frozen=True)
class Recipe:
    """A treatment document (see the module docstring). Use :meth:`from_mapping` or :meth:`from_file`."""

    name: str
    model_id: str
    revision: str
    book: AddressBook
    stages: tuple[Stage, ...]
    operands: Mapping[str, Operand] = field(default_factory=dict)
    book_ref: Any = None
    """What the document says for ``book`` (a path, or the inline mapping), kept so it round-trips."""
    revision_source: str = "pinned"
    sampling: Mapping[str, Any] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source: str = "<inline>"
    base_dir: str | None = None

    @classmethod
    def from_mapping(cls, doc: Mapping[str, Any], *, source: str = "<inline>",
                     base_dir: str | Path | None = None) -> "Recipe":
        """Read a recipe document (schema 2, or schema 1 through :func:`migrate`), refusing a malformed one by name.

        Raises:
            AddressError: If a key is missing or malformed, an op is unknown or misused (an operand of the wrong kind, a
                missing basis, an unknown parameter, a target it cannot act on, a missing or forbidden value), the book
                cannot load or pins another revision, or a target names a set the book lacks.
        """
        version = _check_header(doc, source)
        if version == 1:
            warnings.warn(f"{source}: recipe schema 1 is read through `venngram.recipe.migrate`; re-write it in "
                          f"schema {RECIPE_SCHEMA_VERSION}.", RecipeSchemaWarning, stacklevel=2)
            doc = migrate(doc, source=source)
        model = doc.get("model")
        if not isinstance(model, Mapping) or not model.get("id") or not model.get("revision"):
            raise AddressError(f"{source}: `model` is {{id, revision}}: a recipe's units mean something in one "
                               "revision only.")
        raw_book = doc.get("book")
        if isinstance(raw_book, str):
            book = AddressBook.from_file(Path(base_dir or ".") / raw_book)
        elif isinstance(raw_book, Mapping):
            book = AddressBook.from_mapping(raw_book, source=f"{source}: book", base_dir=base_dir, inline=True)
        else:
            raise AddressError(f"{source}: `book` is a path to an address book or the book inline.")
        pinned = [m for m in (book.applies_to.get("models") or []) if isinstance(m, Mapping)]
        if pinned and not any(m.get("id") == model["id"] and m.get("revision") == model["revision"] for m in pinned):
            raise AddressError(f"{source}: the book pins {[(m.get('id'), m.get('revision')) for m in pinned]}, not the "
                               f"recipe's {(model['id'], model['revision'])}.")
        operands = {name: _operand(name, raw, source) for name, raw in (doc.get("operands") or {}).items()}
        raw_stages = doc.get("stages")
        if not isinstance(raw_stages, list) or not raw_stages:
            raise AddressError(f"{source}: `stages` lists at least one {{site, ops}}.")
        stages = tuple(_stage(raw, i, book, operands, source) for i, raw in enumerate(raw_stages))
        return cls(name=str(doc.get("name") or source), model_id=model["id"], revision=model["revision"], book=book,
                   stages=stages, operands=operands, book_ref=raw_book,
                   revision_source=model.get("revision_source", "pinned"), sampling=dict(doc.get("sampling") or {}),
                   evidence=dict(doc.get("evidence") or {}), metadata=dict(doc.get("metadata") or {}), source=source,
                   base_dir=str(base_dir) if base_dir is not None else None)

    @classmethod
    def from_file(cls, path: str | Path) -> "Recipe":
        """Read a recipe document from YAML (loading runs no code)."""
        path = Path(path)
        with open(path, encoding="utf-8") as fh:
            return cls.from_mapping(yaml.safe_load(fh), source=str(path), base_dir=path.parent)

    def to_mapping(self) -> dict[str, Any]:
        """The recipe as a schema 2 document."""
        doc: dict[str, Any] = {
            "schema_version": RECIPE_SCHEMA_VERSION, "kind": "recipe", "name": self.name,
            "model": {"id": self.model_id, "revision": self.revision, "revision_source": self.revision_source},
            "book": self.book_ref}
        if self.operands:
            doc["operands"] = {k: o.to_mapping() for k, o in self.operands.items()}
        doc["stages"] = [s.to_mapping() for s in self.stages]
        for key in ("sampling", "evidence", "metadata"):
            if getattr(self, key):
                doc[key] = dict(getattr(self, key))
        return doc


def _check_header(doc: Any, source: str) -> int:
    if not isinstance(doc, Mapping):
        raise AddressError(f"{source}: a recipe is a mapping, got {type(doc).__name__}.")
    version = doc.get("schema_version")
    lo, hi = RECIPE_WINDOW
    if not isinstance(version, int) or not lo <= version <= hi:
        raise AddressError(f"{source}: recipe `schema_version` must be in {lo}..{hi} for this venngram, got "
                           f"{version!r}.")
    if doc.get("kind", "recipe") != "recipe":
        raise AddressError(f"{source}: `kind` is `recipe`, got {doc.get('kind')!r}.")
    return version


def migrate(doc: Mapping[str, Any], *, source: str = "<inline>") -> dict[str, Any]:
    """A schema 1 recipe rewritten as schema 2: one stage at ``method.site``, one op per treatment.

    ``mul`` becomes ``scale``; ``apply_to`` becomes the stage's ``tokens``; a ``normalization`` other than ``raw``
    becomes a ``statistic`` transform on each value, naming it.

    Raises:
        AddressError: If ``method`` is malformed or ``apply_to`` names a scope schema 2 has no spelling for.
    """
    method = doc.get("method")
    if not isinstance(method, Mapping) or not method.get("op") or not method.get("site"):
        raise AddressError(f"{source}: a schema 1 `method` names at least an `op` and a `site`.")
    unknown = set(method) - {"op", "site", "normalization", "apply_to"}
    if unknown:
        raise AddressError(f"{source}: unknown `method` keys {sorted(unknown)}.")
    apply_to = method.get("apply_to", "all_tokens")
    if apply_to not in _APPLY_TO:
        raise AddressError(f"{source}: `apply_to: {apply_to}` has no schema 2 token scope; known: {sorted(_APPLY_TO)}.")
    normalization = method.get("normalization", "raw")
    ops = []
    for t in doc.get("treatments") or ():
        if not isinstance(t, Mapping):
            raise AddressError(f"{source}: a treatment is {{set, dose}} or {{set, doses}}, got {t!r}.")
        value = {k: t[k] for k in ("dose", "doses") if k in t}
        if normalization != "raw":
            value["transform"] = [{"statistic": {"name": normalization}}]
        ops.append({"op": ALIASES.get(method["op"], method["op"]), "target": {"set": t.get("set")}, "value": value})
    out = {k: v for k, v in doc.items() if k not in ("method", "treatments")}
    out["schema_version"] = RECIPE_SCHEMA_VERSION
    out["stages"] = [{"site": method["site"], "tokens": _APPLY_TO[apply_to], "ops": ops}]
    return out


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _operand(name: str, raw: Any, source: str) -> Operand:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("shape"), list) or not raw["shape"] \
            or not all(isinstance(d, int) and d > 0 for d in raw["shape"]) or len(raw["shape"]) > 2:
        raise AddressError(f"{source}: operand `{name}` gives a `shape` of one or two positive dimensions.")
    shape = tuple(raw["shape"])
    if "values" in raw:
        values = raw["values"]
        if not isinstance(values, list) or not all(_number(v) for v in values) or len(values) > INLINE_LIMIT:
            raise AddressError(f"{source}: operand `{name}`: inline `values` are at most {INLINE_LIMIT} numbers.")
        if len(values) != (shape[0] * (shape[1] if len(shape) == 2 else 1)):
            raise AddressError(f"{source}: operand `{name}` has {len(values)} values for shape {list(shape)}.")
        return Operand(name, shape, raw.get("basis"), raw.get("space"), values=tuple(float(v) for v in values))
    if not raw.get("file") or not raw.get("key") or not raw.get("sha256"):
        raise AddressError(f"{source}: operand `{name}` is inline `values` or a sidecar {{file, key, sha256}}.")
    return Operand(name, shape, raw.get("basis"), raw.get("space"), str(raw["file"]), str(raw["key"]),
                   str(raw["sha256"]))


def _tokens(raw: Any, where: str) -> Tokens:
    if raw in (None, "all"):
        return Tokens("all")
    if raw in _BARE_SCOPES:
        return Tokens(raw)
    if isinstance(raw, Mapping) and isinstance(raw.get("positions"), list) \
            and all(isinstance(p, int) for p in raw["positions"]):
        return Tokens("positions", positions=tuple(raw["positions"]))
    if isinstance(raw, Mapping) and isinstance(raw.get("slice"), str):
        return Tokens("slice", slice=raw["slice"])
    raise AddressError(f"{where}: `tokens` is all, last, prompt, generated, {{positions: [...]}} or "
                       f"{{slice: \"a:b\"}}, got {raw!r}.")


def _params(raw: Any, specs: Mapping[str, Any], where: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise AddressError(f"{where}: `params` is a mapping.")
    unknown = set(raw) - set(specs)
    if unknown:
        raise AddressError(f"{where}: unknown params {sorted(unknown)} (known: {sorted(specs)}).")
    out = {}
    for name, ps in specs.items():
        if name not in raw:
            if ps.required:
                raise AddressError(f"{where}: param `{name}` is required.")
            continue
        value = raw[name]
        if ps.type is float and not _number(value) or ps.type is str and not isinstance(value, str):
            raise AddressError(f"{where}: param `{name}` is a {ps.type.__name__}, got {value!r}.")
        if ps.range is not None and not ps.range[0] <= value <= ps.range[1]:
            raise AddressError(f"{where}: param `{name}` lies in [{ps.range[0]}, {ps.range[1]}], got {value!r}.")
        out[name] = float(value) if ps.type is float else value
    return out


def _value(raw: Any, where: str) -> Value:
    if not isinstance(raw, Mapping):
        raise AddressError(f"{where}: `value` is {{dose}}, {{doses}} or {{from}}.")
    sources = [k for k in ("dose", "doses", "from") if k in raw]
    if len(sources) != 1:
        raise AddressError(f"{where}: `value` gives exactly one of `dose`, `doses` and `from`.")
    transform = []
    for item in raw.get("transform") or ():
        name, params = (item, {}) if isinstance(item, str) else (next(iter(item)), next(iter(item.values())) or {}) \
            if isinstance(item, Mapping) and len(item) == 1 else (None, None)
        if name not in TRANSFORMS:
            raise AddressError(f"{where}: unknown transform {item!r} (known: {sorted(TRANSFORMS)}).")
        transform.append(Transform(name, _params(params, TRANSFORMS[name], f"{where} transform `{name}`")))
    src = sources[0]
    if src == "dose":
        if not _number(raw["dose"]):
            raise AddressError(f"{where}: `dose` is a number, got {raw['dose']!r}.")
        return Value("dose", dose=float(raw["dose"]), transform=tuple(transform))
    if src == "doses":
        doses = raw["doses"]
        ok = isinstance(doses, Mapping) and all(
            isinstance(layer, int) and isinstance(units, Mapping)
            and all(isinstance(u, int) and _number(d) for u, d in units.items()) for layer, units in doses.items())
        if not ok:
            raise AddressError(f"{where}: `doses` is {{layer: {{unit: dose}}}}.")
        return Value("doses", doses={int(k): {int(u): float(d) for u, d in v.items()} for k, v in doses.items()},
                     transform=tuple(transform))
    frm = raw["from"]
    if not isinstance(frm, Mapping) or not all(isinstance(frm.get(k), str) for k in ("evidence", "column")):
        raise AddressError(f"{where}: `from` is {{evidence: <name>, column: <column>}}.")
    return Value("from", evidence=frm["evidence"], column=frm["column"], transform=tuple(transform))


def _op(raw: Any, where: str, book: AddressBook, operands: Mapping[str, Operand]) -> Op:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("op"), str):
        raise AddressError(f"{where}: an op is {{op, ...}}, got {raw!r}.")
    name = raw["op"]
    if name not in OPS:
        renamed = f" (`{name}` is now `{ALIASES[name]}`)" if name in ALIASES else ""
        raise AddressError(f"{where}: unknown op {name!r}{renamed}; known: {sorted(OPS)} (register one with "
                           "`venngram.ops.register_op`).")
    spec = OPS[name]
    where = f"{where} (`{name}`)"
    unknown = set(raw) - {"op", "target", "operand", "params", "value", "tokens"}
    if unknown:
        raise AddressError(f"{where}: unknown keys {sorted(unknown)}.")
    target = None
    if "target" in raw:
        t = raw["target"]
        sets = book.sets()
        if not isinstance(t, Mapping) or t.get("set") not in sets:
            raise AddressError(f"{where}: `target` names a set of the book ({sorted(sets)}), got {t!r}.")
        term = sets[t["set"]]
        if not isinstance(term, Term) or (term.units is None and term.features is None):
            raise AddressError(f"{where}: target set `{t['set']}` must be one address with `units` or `features`.")
        target = t["set"]
    operand = raw.get("operand")
    if operand is not None:
        if operand not in operands:
            raise AddressError(f"{where}: operand `{operand}` is not declared (operands: {sorted(operands)}).")
        kinds = [k for k in spec.operand if k != "none"]
        o = operands[operand]
        fits = (o.kind == "vector" and "vector" in kinds) or (len(o.shape) == 2 and (
            ("pair" in kinds and o.shape[0] == 2) or "subspace" in kinds))
        if not fits:
            raise AddressError(f"{where}: operand `{operand}` of shape {list(o.shape)} is not a "
                               f"{' or '.join(kinds) or 'valid'} operand for this op.")
        if "basis" in spec.requires and not o.basis:
            raise AddressError(f"{where}: operand `{operand}` must declare its `basis`: two bases are not "
                               "interchangeable, so none is assumed.")
    elif "none" not in spec.operand:
        raise AddressError(f"{where}: needs an operand ({' or '.join(spec.operand)}).")
    params = _params(raw.get("params") or {}, spec.params, where)
    if spec.at_least_one_of and not set(spec.at_least_one_of) & set(params):
        raise AddressError(f"{where}: give at least one of {list(spec.at_least_one_of)}.")
    value = _value(raw["value"], where) if "value" in raw else None
    if value is not None and spec.value == "none":
        raise AddressError(f"{where}: takes no `value`.")
    if value is None and (spec.value == "required" or (operand is None and spec.value != "none")):
        raise AddressError(f"{where}: needs a `value`{'' if spec.value == 'required' else ' or an operand'}.")
    kind = "activation" if target is None else Op(name, target).target_kind(book)
    if kind not in spec.targets:
        raise AddressError(f"{where}: acts on {list(spec.targets)}, not `{kind}`.")
    if value is not None and value.source == "doses" and kind != "units":
        raise AddressError(f"{where}: per-unit `doses` need a target set with `units`.")
    tokens = None
    if "tokens" in raw:
        if not spec.token_override:
            raise AddressError(f"{where}: cannot override its stage's `tokens`.")
        tokens = _tokens(raw["tokens"], where)
    return Op(name, target, operand, params, value, tokens)


def _stage(raw: Any, i: int, book: AddressBook, operands: Mapping[str, Operand], source: str) -> Stage:
    where = f"{source}: stage {i}"
    if not isinstance(raw, Mapping) or not isinstance(raw.get("site"), str) or not isinstance(raw.get("ops"), list) \
            or not raw["ops"]:
        raise AddressError(f"{where}: a stage is {{site, tokens, ops: [...]}} with at least one op.")
    unknown = set(raw) - {"site", "tokens", "ops"}
    if unknown:
        raise AddressError(f"{where}: unknown keys {sorted(unknown)}.")
    ops = tuple(_op(o, f"{where} op {j}", book, operands) for j, o in enumerate(raw["ops"]))
    return Stage(raw["site"], ops, _tokens(raw.get("tokens"), where))


def _units(expr: Any, name: str, source: str) -> list[tuple[int, int]]:
    """The ``(layer, unit)`` pairs a set names, for a set whose layers and units are explicit."""
    if not isinstance(expr, Term) or expr.units is None:
        raise AddressError(f"{source}: set `{name}` must be one address with `units` to carry doses.")
    units = expr.units
    if units.per_layer is not None:
        layers = [k for k, _ in units.per_layer]
    elif expr.layers is not None and expr.layers.indices is not None:
        layers = list(expr.layers.indices)
    else:
        raise AddressError(f"{source}: set `{name}` names its units for every layer of a slice or parity, so its "
                           "doses depend on the model's depth; name the layers explicitly.")
    out = []
    for layer in layers:
        idx = (dict(units.per_layer)[layer] if units.per_layer is not None else units.flat).indices  # type: ignore
        if idx is None:
            raise AddressError(f"{source}: set `{name}` names a slice of units at layer {layer}; a recipe needs "
                               "explicit units.")
        out += [(layer, u) for u in idx]
    return out


@dataclass(frozen=True)
class BoundOp:
    """An op with the recipe it came from (its sets and operands are that recipe's)."""

    op: Op
    recipe: Recipe


@dataclass(frozen=True)
class ComposedStage:
    """A composition's ops at one site and token scope, in order."""

    site: str
    tokens: Tokens
    ops: tuple[BoundOp, ...]

    def unit_doses(self, op: str) -> dict[tuple[int, int], float]:
        """Every unit dose of ``op`` (a summing or multiplying op) in this stage, combined per unit by its law."""
        spec = OPS[op]
        if spec.composition == "ordered":
            raise AddressError(f"`{op}` composes only in order; its doses do not combine.")
        out: dict[tuple[int, int], float] = {}
        for b in self.ops:
            if b.op.op == op and b.op.target is not None and b.op.value is not None \
                    and b.op.value.source in ("dose", "doses"):
                for unit, dose in b.op.unit_doses(b.recipe.book, b.recipe.source).items():
                    out[unit] = spec.combine(out[unit], dose) if unit in out else dose
        return out


def _commute(a: BoundOp, b: BoundOp) -> bool:
    if a.op.op == b.op.op and OPS[a.op.op].composition != "ordered":
        return True
    if a.op.target is not None and b.op.target is not None and a.op.value is not None and b.op.value is not None:
        try:
            ua = set(_units(a.recipe.book.sets()[a.op.target], a.op.target, a.recipe.source))
            ub = set(_units(b.recipe.book.sets()[b.op.target], b.op.target, b.recipe.source))
        except AddressError:
            return False
        return not ua & ub
    return False


def compose(recipes: Sequence[Recipe], *, ordered: bool = False) -> list[ComposedStage]:
    """The stages several recipes on one model revision amount to.

    Stages at the same site and token scope are joined, in recipe order. Without ``ordered``, ops from different
    recipes at one site must commute: the same summing or multiplying op (doses then combine per unit, see
    :meth:`ComposedStage.unit_doses`), or ops on disjoint unit sets.

    Raises:
        AddressError: If the recipes pin different models or revisions, or two ops at one site do not commute and the
            composition gives no order.
    """
    if not recipes:
        return []
    first = recipes[0]
    for r in recipes[1:]:
        if (r.model_id, r.revision) != (first.model_id, first.revision):
            raise AddressError(f"{r.source}: composes {r.model_id}@{r.revision} with {first.model_id}@"
                               f"{first.revision}; a composition is over one revision.")
    out: list[ComposedStage] = []
    for r in recipes:
        for s in r.stages:
            bound = tuple(BoundOp(o, r) for o in s.ops)
            match = next((i for i, c in enumerate(out) if (c.site, c.tokens) == (s.site, s.tokens)), None)
            if ordered:
                if out and (out[-1].site, out[-1].tokens) == (s.site, s.tokens):
                    out[-1] = ComposedStage(s.site, s.tokens, out[-1].ops + bound)
                else:
                    out.append(ComposedStage(s.site, s.tokens, bound))
                continue
            if match is None:
                out.append(ComposedStage(s.site, s.tokens, bound))
                continue
            for b in bound:
                clash = next((a for a in out[match].ops if a.recipe is not r and not _commute(a, b)), None)
                if clash is not None:
                    raise AddressError(f"{r.source}: `{b.op.op}` at {s.site} does not commute with `{clash.op.op}` "
                                       f"from {clash.recipe.source}; compose with an order.")
            out[match] = ComposedStage(s.site, s.tokens, out[match].ops + bound)
    return out


@dataclass(frozen=True)
class Composition:
    """Several recipes applied together, in order when ``ordered``."""

    recipes: tuple[Recipe, ...]
    ordered: bool = False

    def stages(self) -> list[ComposedStage]:
        """The stages to apply, in order (see :func:`compose`)."""
        return compose(self.recipes, ordered=self.ordered)


def load_recipe(path: str | Path) -> Recipe | Composition:
    """A recipe document, or a composition (``compose: [a.yaml, b.yaml]``, optional ``order: listed``)."""
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if isinstance(doc, Mapping) and "compose" in doc:
        _check_header({**doc, "kind": doc.get("kind", "recipe")}, str(path))
        parts = doc["compose"]
        if not isinstance(parts, list) or not parts or not all(isinstance(p, str) for p in parts):
            raise AddressError(f"{path}: `compose` lists recipe paths.")
        order = doc.get("order")
        if order not in (None, "listed"):
            raise AddressError(f"{path}: `order` is `listed` or absent, got {order!r}.")
        return Composition(tuple(Recipe.from_file(path.parent / p) for p in parts), ordered=order == "listed")
    return Recipe.from_mapping(doc, source=str(path), base_dir=path.parent)


__all__ = [
    "BoundOp",
    "compose",
    "ComposedStage",
    "Composition",
    "load_recipe",
    "migrate",
    "Op",
    "Operand",
    "Recipe",
    "RECIPE_SCHEMA_VERSION",
    "RECIPE_WINDOW",
    "RecipeSchemaWarning",
    "Stage",
    "Tokens",
    "Transform",
    "Value",
]
