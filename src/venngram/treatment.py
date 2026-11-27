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
Treatments
^^^^^^^^^^

A treatment is what is done to an address book's sets: at which sites, to which units or features, in which order. It
is kept apart from the values and findings it reads (unit tables and graph subsets the book names). A treatment lives
in the book it treats, by name, under its ``treatments`` section (:mod:`venngram.book`), so one selection carries
several treatments and a consumer reads one kind of document; :meth:`Treatment.from_mapping` reads one, and
:meth:`Treatment.to_mapping` writes one. :class:`Treatment` is a treatment in memory: its book, and the body below.

.. code-block:: yaml

    treatments:
      chess-then-reject:
        model: {id: Qwen/Qwen3.5-4B, revision: 3f1e..., revision_source: pinned}
        operands:                              # named tensors, each in a declared basis
          concept: {file: dirs.safetensors, key: concept, sha256: ..., shape: [2560], basis: unfolded}
        stages:                                # in order; a stage applies at one site
          - site: mlp.out.hook_in
            tokens: all                        # all | last | prompt | generated | {positions: [...]} | {slice: "-4:"}
            ops:                               # in order; each sees the previous op's output
              - {op: add, target: {set: chess}, value: {dose: 0.8}}
              - {op: add, target: {set: openings}, value: {doses: {15: {4041: 0.5}}}}
              - {op: reject, operand: concept, params: {fraction: 1.0}}
        sampling: {temperature: 0.7}
        metadata: {examples: [...]}      # free-form, carried and never read

**Ops are declared specs** (:mod:`venngram.ops`): this module validates a treatment against them and composes
treatments; it never executes an op. Applying a treatment belongs to the consumer (e.g. interpretune steers,
finetuning-scheduler tunes a unit adapter), which declares what it honours as a :class:`~venngram.ops.Support` record
and checks a treatment with :func:`~venngram.ops.check_support` before running it. A format another tool writes is
read by an adapter outside this module (:mod:`venngram.sources`).

**Targets**: an op acts on the whole activation (no ``target``), the units a book set names, or the features of a
latent space a book set names (its ``space`` and ``features``; the consumer supplies the space's encoder). A unit
target may be one address with explicit layers and ``units``, or a composition of such sets at one site with the book's
set algebra (``@a & @b``, ``@a | @b``, ``@a - @control``), evaluated over ``(layer, unit)`` pairs; a ``layers``-only
term in it filters by layer (``(@a & @b & layers:12) - @control``). A composed target is resolved when the treatment is
read, and refused by name if it is empty, mixes sites, composes feature sets or names units only by slice. **Values**:
a ``dose``, a per-unit ``doses`` table, or ``from`` a column of a unit table through a ``transform`` pipeline of
declared transforms (:data:`~venngram.ops.TRANSFORMS`); the table is one the book names
(:mod:`venngram.tables`), or the caller's when the book names none. **Tokens** are set per stage; an op may
override them only if its spec allows. ``prompt`` is every position of the input before any generated token;
``generated`` is every position the consumer generates, at each decode step (a consumer that does not generate
refuses it by name).

**Weights**: a treatment's ``weights`` section (treatment schema 3) lists ops on parameters, in order, applied to the
tensors before any stage runs and kept in them (a consumer materializes them, reversibly or for export). Each names a
target set that addresses tensors (no ``point``, ``features`` or ``subspace``): a whole tensor is a ``param`` target,
a set with ``units`` a ``param_units`` one, along each tensor's unit axis
(:meth:`venngram.maps.structure.MapStructure.unit_dim`). A weight op acts on every token, so its token scope is
``all``, and on the tensor's value, never ``of: grad``. An ``add`` or ``replace`` may take a ``delta`` operand, a
weight patch (``operands: {edit: {patch: edit.yaml}}``, :mod:`venngram.patches`). A treatment may have ``weights``
and no ``stages``.

.. code-block:: yaml

    weights:
      - {op: scale, target: {set: chess_columns}, value: {dose: 0.0}}       # zero neurons' write columns
      - {op: add, target: {set: down_13}, operand: edit}                     # a low-rank patch from a document

**Gradients**: a treatment's ``gradients`` section (treatment schema 4) lists ops on parameter gradients, in order, each
acting on a step's accumulated gradient of the tensors it targets and never materialized into them. Its target sets
address tensors as a ``weights`` op's do, and say ``of: grad``; a ``param_units`` target acts along each tensor's unit
axis. An op belongs there when its spec declares the target and ``grad`` in its ``of``: among the core ops ``scale``,
``replace`` (a zero ``value`` is the declared form of a gradient mask), ``add`` (never with a weight patch), and
``project`` and ``reject`` on a whole tensor. When an op runs relative to accumulation, clipping, the optimizer step or
sharding, and any state it keeps, are the consumer's.

.. code-block:: yaml

    gradients:
      - {op: replace, target: {set: neuron_grads}, value: {dose: 0.0}}     # keep these rows' gradients out of a step
      - {op: reject, target: {set: down_grad}, operand: concept}            # train orthogonally to a direction

**Readouts**: a stage op on an ``of: grad`` target acts on the gradient of a scalar, and the treatment says which with a
``readout`` (:mod:`venngram.readouts`): the weighted sum of terms, each a registered reduction (``sum``, ``project``
with a vector operand, ``norm``) of the activations a book set addresses at a token scope. A stage may state its own,
replacing the treatment's for its ops. From treatment schema 4 a stage with an ``of: grad`` op needs one of the two,
and a readout where no stage acts on a gradient is refused; an earlier document without one leaves the scalar to the
consumer. Parameter gradients (the ``gradients`` section) are the training objective's, which is the tuning tool's.

.. code-block:: yaml

    readout:                                                              # a logit difference at the last token
      terms:
        - {reduce: sum, set: answer_logit, tokens: last, weight: 1.0}
        - {reduce: sum, set: foil_logit, tokens: last, weight: -1.0}

**Graphs**: an op may target a graph subset the book names (``target: {graph: <name>}``, treatment schema 5,
:mod:`venngram.graphs`): it acts on the graph's feature nodes at the stage's site, each placed by its anchor, and reads
its value ``from`` a node column, optionally as a delta from a baseline column. ``replace`` and ``add`` keep their
meanings. ``target: {graph: <name>, outside: {set: <universe>, tokens: prompt}}`` targets everything else in a finite
feature set of the book (a restriction, with ``replace`` and a ``dose``). Anchors fix a graph target's positions, so its
stage gives no ``tokens`` and the op no ``tokens`` or operand; every site of the graph's feature nodes must be reached
by a stage, and the graph must pin the treatment's model. Graph values pass only through transforms that act on each
value alone (:data:`~venngram.graphs.GRAPH_TRANSFORMS`). A consumer declares what it honours in its support record's
``circuits.subsets`` section (:class:`venngram.support.SubsetSupport`).

.. code-block:: yaml

    stages:
      - site: mlp.out.hook_out
        ops: [{op: replace, target: {graph: reasoning}, value: {from: {graph: reasoning, column: activation}}}]

**Composition** (:func:`compose`): treatments on one model revision compose stage by stage. Without an order, ops
from different treatments at one site must commute (the same summing or multiplying op, or disjoint unit sets);
otherwise the composition must state ``order: listed``. ``weights`` and ``gradients`` ops compose the same way,
section by section (:func:`compose_weights`).

**Versions**: a treatment is read at the schema its sections need: 2, 3 when it has ``weights``, 4 when it
has ``gradients`` or a ``readout``, or 5 when an op targets a graph (the book's own schema says the same, see
:mod:`venngram.book`).

.. warning:: This is an experimental API and may change in a future release.
"""
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from venngram.address import And, Expr, Guarded, Minus, Or, Ref, references, Term, terms, to_string
from venngram.address import Path as PathExpr
from venngram.book import _states_readout, _uses_passes, AddressBook
from venngram.errors import AddressError, AddressUnresolvable, VenngramError
from venngram.graphs import GRAPH_TRANSFORMS
from venngram.operands import INLINE_LIMIT, read_operand
from venngram.ops import ALIASES, TABLE_TRANSFORMS, OPS, OpSpec, Transform, TRANSFORMS
from venngram.paths import is_path, paths
from venngram.readouts import READOUTS
from venngram.resolve import gradient_of
from venngram.units import _sha256 as _sha256_file

#: The schema of a treatment without ``weights``, ``gradients``, a ``readout`` or a graph target.
TREATMENT_SCHEMA_VERSION = 2

#: The first treatment schema with a ``weights`` section (a key an older reader would pass over silently).
WEIGHTS_SCHEMA = 3
#: The first treatment schema with a ``gradients`` section (ops on parameter gradients) or a ``readout`` (what a
#: stage's gradients are taken of); an older reader would pass over either silently.
GRADIENTS_SCHEMA = 4
#: The first treatment schema whose ops may target a graph subset (``target: {graph: ...}``).
GRAPHS_SCHEMA = 5
#: The ops a graph target admits, by target kind.
GRAPH_OPS = {"graph_nodes": ("replace", "add"), "graph_complement": ("replace",)}

#: Token scopes written as a bare string.
_BARE_SCOPES = ("all", "last", "prompt", "generated")


@dataclass(frozen=True)
class Operand:
    """A named tensor an op acts with: inline ``values`` or a sidecar ``{file, key, sha256}``, a shape and a basis; or a
    weight patch (``{patch: <path>}``, a ``delta`` operand), whose shape is the tensor's."""

    name: str
    shape: tuple[int, ...]
    basis: str | None = None
    space: str | None = None
    file: str | None = None
    key: str | None = None
    sha256: str | None = None
    values: tuple[float, ...] | None = None
    patch: str | None = None
    """The path of a ``weight_patch`` document, for a ``delta`` operand (``sha256`` is then the document's)."""

    @property
    def kind(self) -> str:
        if self.patch is not None:
            return "delta"
        return "vector" if len(self.shape) == 1 else ("pair" if self.shape[0] == 2 else "subspace")

    def weight_patch(self, base_dir: str | Path = ".") -> Any:
        """The :class:`~venngram.patches.WeightPatch` a ``delta`` operand names, read relative to ``base_dir``."""
        from venngram.patches import read_weight_patch

        if self.patch is None:
            raise AddressError(f"operand `{self.name}` is not a weight patch.")
        path = Path(base_dir) / self.patch
        if self.sha256 is not None and path.is_file() and _sha256_file(path) != self.sha256.lower():
            raise AddressError(f"operand `{self.name}`: {path} differs from the sha256 the treatment names.")
        return read_weight_patch(path)

    def load(self, base_dir: str | Path = ".") -> tuple[tuple[float, ...], tuple[int, ...]]:
        """The operand's values (row-major) and shape, reading its sidecar relative to ``base_dir`` (a weight patch's
        dense delta, for a ``delta`` operand)."""
        if self.patch is not None:
            return self.weight_patch(base_dir).delta(), self.shape
        if self.values is not None:
            return self.values, self.shape
        values, shape = read_operand(Path(base_dir) / str(self.file), str(self.key), sha256=self.sha256)
        if shape != self.shape:
            raise AddressError(f"operand `{self.name}` declares shape {list(self.shape)}, its sidecar holds "
                               f"{list(shape)}.")
        return values, shape

    def to_mapping(self) -> dict[str, Any]:
        if self.patch is not None:
            return {"patch": self.patch, **({"sha256": self.sha256} if self.sha256 is not None else {})}
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
class Value:
    """An op's value: a ``dose``, per-unit ``doses``, ``from`` a unit table's column, or ``from`` a graph's node column
    (source ``graph``, optionally less a ``baseline`` column), through a transform pipeline."""

    source: str
    dose: float | None = None
    doses: Mapping[int, Mapping[int, float]] | None = None
    table: str | None = None
    column: str | None = None
    transform: tuple[Transform, ...] = ()
    graph: str | None = None
    baseline: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.source == "dose":
            out["dose"] = self.dose
        elif self.source == "doses":
            out["doses"] = {k: dict(v) for k, v in (self.doses or {}).items()}
        elif self.source == "graph":
            out["from"] = {"graph": self.graph, "column": self.column,
                           **({"baseline": self.baseline} if self.baseline is not None else {})}
        else:
            out["from"] = {"table": self.table, "column": self.column}
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
    weights: bool = False
    """An op of the treatment's ``weights`` section: its target is a parameter."""
    gradients: bool = False
    """An op of the treatment's ``gradients`` section: its target is a parameter's gradient."""
    graph: str | None = None
    """The graph subset of the book the op targets (``target: {graph: ...}``)."""
    outside: str | None = None
    """For a restriction, the finite feature set of the book whose members outside the graph the op targets."""

    @property
    def spec(self) -> OpSpec:
        return OPS[self.op]

    def target_kind(self, book: AddressBook) -> str:
        """``activation`` (no target), ``units`` or ``features``, from the target set (a composed target is a unit
        set: a treatment refuses any other composition when it is read); for a ``weights`` or ``gradients`` op,
        ``param`` or ``param_units``; for a graph target, ``graph_nodes`` or (with ``outside``) ``graph_complement``."""
        if self.graph is not None:
            return "graph_complement" if self.outside is not None else "graph_nodes"
        if self.gradients and self.target is not None and is_path(book.sets()[self.target], book.sets()):
            return "path"
        if self.weights or self.gradients:
            return "param_units" if _names_units(book.sets(), self.target) else "param"
        if self.target is None:
            return "activation"
        term = book.sets()[self.target]
        if not isinstance(term, Term):
            return "units"
        return "units" if term.units is not None else "features"

    def target_of(self, book: AddressBook) -> str:
        """What the op acts on at its target: ``value`` or ``grad`` (``value`` without a target)."""
        if self.gradients:
            return "grad"
        if self.target is None:
            return "value"  # a graph target acts on activations
        term = book.sets()[self.target]
        if isinstance(term, Term):
            return term.of or "value"
        return _UnitSets(book.sets(), self.target, "<treatment>").resolve()[1]

    def unit_members(self, book: AddressBook, source: str = "<treatment>") -> list[tuple[int, int]]:
        """The ``(layer, unit)`` pairs a unit target names, in order of first appearance, whatever the op's value is: a
        ``doses`` table may dose only some of them, and a consumer looks up a ``from`` column's value for each."""
        if self.target is None:
            raise AddressError(f"{source}: `{self.op}` has no unit members (it needs a unit set target).")
        return list(dict.fromkeys(_units(book.sets(), self.target, source)))

    def unit_doses(self, book: AddressBook, source: str = "<treatment>") -> dict[tuple[int, int], float]:
        """``(layer, unit) -> dose`` for a unit-targeted op whose value is a ``dose`` or ``doses``."""
        if self.target is None or self.value is None or self.value.source not in ("dose", "doses"):
            raise AddressError(f"{source}: `{self.op}` has no unit doses (it needs a unit set target and a dose).")
        out = {}
        for layer, unit in self.unit_members(book, source):  # a param_units target's units, likewise
            dose = self.value.dose if self.value.source == "dose" else (self.value.doses or {}).get(layer, {}).get(unit)
            if dose is not None:
                out[(layer, unit)] = dose
        return out

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {"op": self.op}
        if self.target is not None:
            out["target"] = {"set": self.target}
        if self.graph is not None:
            out["target"] = {"graph": self.graph, **({"outside": {"set": self.outside, "tokens": "prompt"}}
                                                     if self.outside is not None else {})}
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
class ReadoutTerm:
    """One term of a readout: a registered reduction (:data:`~venngram.readouts.READOUTS`) of the activations a book
    set addresses, at a token scope, with a weight and, for a reduction that takes one, an operand."""

    reduce: str
    set: str
    tokens: Tokens = Tokens()
    weight: float = 1.0
    operand: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {"reduce": self.reduce, "set": self.set, "tokens": self.tokens.to_doc(),
                               "weight": self.weight}
        if self.operand is not None:
            out["operand"] = self.operand
        if self.params:
            out["params"] = dict(self.params)
        return out


@dataclass(frozen=True)
class Readout:
    """What a stage's gradients are taken of: the weighted sum of its terms (:mod:`venngram.readouts`)."""

    terms: tuple[ReadoutTerm, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {"terms": [t.to_mapping() for t in self.terms]}


@dataclass(frozen=True)
class Stage:
    """Ops applied in order at one site, under one token scope."""

    site: str
    ops: tuple[Op, ...]
    tokens: Tokens = Tokens()
    readout: Readout | None = None
    """What this stage's ``of: grad`` ops act on the gradients of, replacing the treatment's readout."""

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {"site": self.site}
        if not (any(o.graph is not None for o in self.ops) and self.tokens == Tokens()):  # anchors place graph nodes
            out["tokens"] = self.tokens.to_doc()
        out["ops"] = [o.to_mapping() for o in self.ops]
        if self.readout is not None:
            out["readout"] = self.readout.to_mapping()
        return out


@dataclass(frozen=True)
class Treatment:
    """A treatment over its book (see the module docstring). A book's loader reads one with :meth:`from_mapping`."""

    name: str
    model_id: str
    revision: str
    book: AddressBook
    stages: tuple[Stage, ...]
    operands: Mapping[str, Operand] = field(default_factory=dict)
    weights: tuple[Op, ...] = ()
    """The ``weights`` section: ops on parameters, in order, applied before any stage."""
    gradients: tuple[Op, ...] = ()
    """The ``gradients`` section: ops on parameter gradients, in order, applied to each step's accumulated gradient."""
    readout: Readout | None = None
    """What its stages' ``of: grad`` ops act on the gradients of, unless a stage states its own."""
    revision_source: str = "pinned"
    sampling: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Free-form: carried with the treatment and never read by venngram."""
    source: str = "<inline>"
    base_dir: str | None = None

    @classmethod
    def from_mapping(cls, body: Any, book: AddressBook, *, name: str | None = None, source: str = "<inline>",
                     base_dir: str | Path | None = None) -> "Treatment":
        """Read one entry of a book's ``treatments`` section: a treatment's body (``model``, ``operands``,
        ``stages``, ``weights``, ``gradients``, ``readout``, ``sampling``, ``metadata``) over ``book``'s sets, refusing
        a malformed one by name. ``name`` is its key in the section (``source`` when not given), and operand files
        resolve against ``base_dir``. It is read at the treatment schema its sections need (see the module docstring).

        Raises:
            AddressError: If the body is not a mapping or carries a header (``schema_version``, ``kind``, ``name``) or
                its own ``book``, a key is missing or malformed, an op is unknown or misused (an operand of the wrong
                kind, a missing basis, an unknown parameter, a target it cannot act on, a missing or forbidden value),
                the book pins another revision, or a target names a set the book lacks.
        """
        if not isinstance(body, Mapping):
            raise AddressError(f"{source}: a treatment is a mapping (model, stages, ...), got {type(body).__name__}.")
        header = sorted({"schema_version", "kind", "name", "book"} & set(body))
        if header:
            raise AddressError(f"{source}: a treatment is written in the book it treats, so it has no {header}.")
        version = _schema(weights="weights" in body, gradients="gradients" in body, readout=_states_readout(body),
                          graphs=_uses_graphs(body))
        model = body.get("model")
        if not isinstance(model, Mapping) or not model.get("id") or not model.get("revision"):
            raise AddressError(f"{source}: `model` is {{id, revision}}: a treatment's units mean something in one "
                               "revision only.")
        pinned = [m for m in (book.applies_to.get("models") or []) if isinstance(m, Mapping)]
        if pinned and not any(m.get("id") == model["id"] and m.get("revision") == model["revision"] for m in pinned):
            raise AddressError(f"{source}: the book pins {[(m.get('id'), m.get('revision')) for m in pinned]}, not the "
                               f"treatment's {(model['id'], model['revision'])}.")
        operands = {k: _operand(k, raw, source, base_dir) for k, raw in (body.get("operands") or {}).items()}
        raw_weights = body.get("weights")
        if raw_weights is not None and (not isinstance(raw_weights, list) or not raw_weights):
            raise AddressError(f"{source}: `weights` lists at least one op on a parameter.")
        weights = tuple(_op(o, f"{source}: weights op {j}", book, operands, weights=True)
                        for j, o in enumerate(raw_weights or ()))
        raw_gradients = body.get("gradients")
        if raw_gradients is not None and (not isinstance(raw_gradients, list) or not raw_gradients):
            raise AddressError(f"{source}: `gradients` lists at least one op on a parameter's gradient.")
        gradients = tuple(_op(o, f"{source}: gradients op {j}", book, operands, gradients=True)
                          for j, o in enumerate(raw_gradients or ()))
        raw_stages = body.get("stages")
        if raw_stages is None and (weights or gradients):
            raw_stages = []
        if not isinstance(raw_stages, list) or not (raw_stages or weights or gradients):
            raise AddressError(f"{source}: `stages` lists at least one {{site, ops}} (or the treatment has `weights` "
                               "or `gradients`).")
        stages = tuple(_stage(raw, i, book, operands, source) for i, raw in enumerate(raw_stages))
        if any(o.graph is not None for st in stages for o in st.ops):
            _check_graphs(stages, model, book, source)
        readout = None
        if body.get("readout") is not None:
            readout = _readout(body["readout"], f"{source}: readout", book, operands)
        _check_readouts(stages, readout, book, source, version)
        _check_tables(stages, weights, gradients, book, source)
        return cls(name=str(name or source), model_id=model["id"], revision=model["revision"], book=book,
                   stages=stages, operands=operands, weights=weights, gradients=gradients, readout=readout,
                   revision_source=model.get("revision_source", "pinned"), sampling=dict(body.get("sampling") or {}),
                   metadata=_metadata(body), source=source,
                   base_dir=str(base_dir) if base_dir is not None else None)

    @property
    def schema_version(self) -> int:
        """The treatment schema its sections need (see the module docstring); a book carrying it is at least that."""
        return _schema(weights=bool(self.weights), gradients=bool(self.gradients),
                       readout=self.readout is not None or any(s.readout for s in self.stages),
                       graphs=any(o.graph is not None for s in self.stages for o in s.ops))

    def to_mapping(self) -> dict[str, Any]:
        """This treatment as an entry of its book's ``treatments`` section (its body, without the name it is listed
        under)."""
        doc: dict[str, Any] = {
            "model": {"id": self.model_id, "revision": self.revision, "revision_source": self.revision_source}}
        if self.operands:
            doc["operands"] = {k: o.to_mapping() for k, o in self.operands.items()}
        if self.weights:
            doc["weights"] = [o.to_mapping() for o in self.weights]
        if self.gradients:
            doc["gradients"] = [o.to_mapping() for o in self.gradients]
        if self.readout is not None:
            doc["readout"] = self.readout.to_mapping()
        if self.stages or not (self.weights or self.gradients):
            doc["stages"] = [s.to_mapping() for s in self.stages]
        for key in ("sampling", "metadata"):
            if getattr(self, key):
                doc[key] = dict(getattr(self, key))
        return doc


def _metadata(doc: Mapping[str, Any]) -> dict[str, Any]:
    """A treatment's free-form ``metadata``; an ``evidence`` mapping an earlier document carried is kept under it."""
    out = dict(doc.get("metadata") or {})
    if doc.get("evidence"):
        out.setdefault("evidence", dict(doc["evidence"]))
    return out


def _check_tables(stages: Sequence[Stage], weights: Sequence[Op], gradients: Sequence[Op], book: AddressBook,
                    source: str) -> None:
    """In a book with a ``tables`` section, refuse a ``from`` value that names a table or a column the book does not
    have (its own, or one a transform such as ``sign_of`` reads), or whose target has a member without a row
    (:mod:`venngram.tables`). A book without the section leaves the tables to its caller."""
    if not book.tables:
        return
    from venngram.tables import check_members

    located = [(f"stage {i} ({st.site}) op {j}", op) for i, st in enumerate(stages) for j, op in enumerate(st.ops)]
    located += [(f"weights op {j}", op) for j, op in enumerate(weights)]
    located += [(f"gradients op {j}", op) for j, op in enumerate(gradients)]
    for at, op in located:
        value = op.value
        if value is None or value.source != "from" or value.table is None:
            continue
        where = f"{source}: {at} (`{op.op}`)"
        if value.table not in book.tables:
            raise AddressError(f"{where}: the value reads table `{value.table}`, which the book's `tables` "
                               f"does not name (tables: {sorted(book.tables)}).")
        table = book.tables[value.table]
        read = [str(value.column)] + [str(t.params["column"]) for t in value.transform
                                      if t.name in TABLE_TRANSFORMS and "column" in t.params]
        missing = [c for c in dict.fromkeys(read) if c not in table.columns]
        if missing:
            raise AddressError(f"{where}: table `{value.table}` has no column {missing} (columns: "
                               f"{sorted(table.columns)}).")
        try:
            members = op.unit_members(book, where) if op.target is not None else []
        except AddressError:
            members = []  # members known only on a model (a slice, a role): the consumer checks them there
        check_members(table, str(value.column), members, where)


def _schema(*, weights: bool, gradients: bool, readout: bool = False, graphs: bool = False) -> int:
    """The treatment schema a body's sections need."""
    if graphs:
        return GRAPHS_SCHEMA
    return GRADIENTS_SCHEMA if gradients or readout else WEIGHTS_SCHEMA if weights else TREATMENT_SCHEMA_VERSION


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _operand(name: str, raw: Any, source: str, base_dir: str | Path | None = None) -> Operand:
    if isinstance(raw, Mapping) and "patch" in raw:
        unknown = sorted(set(raw) - {"patch", "sha256"})
        if unknown or not isinstance(raw["patch"], str) or not raw["patch"]:
            raise AddressError(f"{source}: a weight patch operand `{name}` is {{patch: <path>, sha256}}, got {raw!r}.")
        o = Operand(name, (), patch=raw["patch"], sha256=raw.get("sha256"))
        return replace(o, shape=o.weight_patch(base_dir or ".").shape)
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
        if ps.choices is not None and value not in ps.choices:
            raise AddressError(f"{where}: param `{name}` is one of {list(ps.choices)}, got {value!r}.")
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
    reads_table = sorted({t.name for t in transform} & TABLE_TRANSFORMS)
    if reads_table and src != "from":
        raise AddressError(f"{where}: {reads_table} read the `from` table, so apply only to a `from` value, not "
                           f"`{src}`.")
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
    if isinstance(frm, Mapping) and "graph" in frm:
        if not set(frm) <= {"graph", "column", "baseline"} or not all(isinstance(frm.get(k), str) and frm[k]
                                                                       for k in frm) or "column" not in frm:
            raise AddressError(f"{where}: a graph value is `from: {{graph: <name>, column: <column>, baseline: "
                               f"<column>}}` (baseline optional), got {frm!r}.")
        elementwise = sorted({t.name for t in transform} - set(GRAPH_TRANSFORMS))
        if elementwise:
            raise AddressError(f"{where}: {elementwise} do not act on each value alone, so a graph value refuses "
                               f"them (it admits {list(GRAPH_TRANSFORMS)}): an `all` anchor expands a node to many "
                               "rows, and those transforms would mean different things on the two.")
        return Value("graph", graph=frm["graph"], column=frm["column"], baseline=frm.get("baseline"),
                     transform=tuple(transform))
    if not isinstance(frm, Mapping) or not all(isinstance(frm.get(k), str) for k in ("table", "column")):
        raise AddressError(f"{where}: `from` is {{table: <name>, column: <column>}}, or {{graph: <name>, "
                           "column: <column>}}.")
    return Value("from", table=frm["table"], column=frm["column"], transform=tuple(transform))


def _op(raw: Any, where: str, book: AddressBook, operands: Mapping[str, Operand], weights: bool = False,
        gradients: bool = False) -> Op:
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
    section = "weights" if weights else "gradients" if gradients else None
    if section is not None and "target" not in raw:
        raise AddressError(f"{where}: a `{section}` op names the tensors it acts on (`target: {{set: ...}}`).")
    if isinstance(raw.get("target"), Mapping) and "graph" in raw["target"] and "set" not in raw["target"] \
            and section is None:
        return _graph_op(raw, name, where, book)
    if "target" in raw:
        t = raw["target"]
        sets = book.sets()
        if not isinstance(t, Mapping) or set(t) != {"set"} or not isinstance(t["set"], str) or t["set"] not in sets:
            raise AddressError(f"{where}: `target` names a set of the book ({sorted(sets)}) and contains only `set`, "
                               f"got {t!r}.")
        _check_treatment_passes(sets, t["set"], where)
        term = sets[t["set"]]
        if is_path(term, sets):
            if section != "gradients":
                raise AddressError(f"{where}: target set `{t['set']}` is a path; a path is acted on in a treatment's "
                                   "`gradients` section (`stop`, `route`).")
            try:
                paths(term, sets)
            except VenngramError as err:
                raise AddressError(f"{where}: target set `{t['set']}`: {err}") from err
            return _path_op(raw, name, spec, where, t["set"], book)
        axes = [to_string(x) for x in _leaf_terms(sets, t["set"]) if x.axis is not None]
        if axes:
            raise AddressError(f"{where}: target set `{t['set']}` names an `axis` ({axes[0]}); a treatment's unit "
                               "targets run along their component's unit axis, and no treatment reads `axis` yet.")
        if section is not None:
            of = _check_param_set(sets, t["set"], where, section)
        elif isinstance(term, Term):
            if term.units is None and term.features is None:
                raise AddressError(f"{where}: target set `{t['set']}` must be one address with `units` or `features`, "
                                   "or a composition of unit sets.")
            of = term.of
        else:
            of = _UnitSets(sets, t["set"], where).resolve()[1]
        target = t["set"]
        if (of or "value") not in spec.of:
            raise AddressError(f"{where}: target set `{target}` addresses `of: {of or 'value'}`, and this op acts "
                               f"on {list(spec.of)}.")
    operand = raw.get("operand")
    if operand is not None:
        if operand not in operands:
            raise AddressError(f"{where}: operand `{operand}` is not declared (operands: {sorted(operands)}).")
        kinds = [k for k in spec.operand if k != "none"]
        o = operands[operand]
        if o.kind == "delta" and not weights:  # a weight patch applies to a value, never to a gradient
            raise AddressError(f"{where}: operand `{operand}` is a weight patch, which only a `weights` op applies.")
        fits = (o.kind == "delta" and "delta" in kinds) if o.kind == "delta" else \
            (o.kind == "vector" and "vector" in kinds) or (len(o.shape) == 2 and (
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
    kind = "activation" if target is None else Op(name, target, weights=weights, gradients=gradients).target_kind(book)
    if kind not in spec.targets:
        raise AddressError(f"{where}: acts on {list(spec.targets)}, not `{kind}`.")
    if value is not None and value.source == "doses" and kind not in ("units", "param_units"):
        raise AddressError(f"{where}: per-unit `doses` need a target set with `units`.")
    tokens = None
    if section is not None and "tokens" in raw:
        scope = _tokens(raw["tokens"], where)
        if scope.kind != "all":
            what = "changes the tensor itself" if weights else "acts on a step's accumulated gradient"
            raise AddressError(f"{where}: a `{section}` op {what}, so it acts on every token: its token scope is "
                               f"`all`, not `{scope.kind}`.")
    elif "tokens" in raw:
        if not spec.token_override:
            raise AddressError(f"{where}: cannot override its stage's `tokens`.")
        tokens = _tokens(raw["tokens"], where)
    return Op(name, target, operand, params, value, tokens, weights, gradients)


def _path_op(raw: Mapping[str, Any], name: str, spec: OpSpec, where: str, target: str, book: AddressBook) -> Op:
    """Read a ``gradients`` op whose target is a path set."""
    if "path" not in spec.targets:
        raise AddressError(f"{where}: acts on {list(spec.targets)}, not `path`.")
    for key in ("operand", "value"):
        if key in raw:
            raise AddressError(f"{where}: a path op takes no `{key}`; what it does along the routes is the op itself.")
    if "tokens" in raw and _tokens(raw["tokens"], where).kind != "all":
        raise AddressError(f"{where}: a `gradients` op acts on a step's accumulated gradient, so its token scope is "
                           "`all`; a path's token positions are its waypoints' anchors.")
    params = _params(raw.get("params") or {}, spec.params, where)
    return Op(name, target, None, params, None, None, False, True)


def _graph_op(raw: Mapping[str, Any], name: str, where: str, book: AddressBook) -> Op:
    """Read an op whose target is a graph subset of the book (``{graph}``, or ``{graph, outside}`` for a
    restriction)."""
    t = raw["target"]
    graph = t.get("graph")
    if not isinstance(graph, str) or graph not in book.graphs:
        raise AddressError(f"{where}: `target` names a graph of the book ({sorted(book.graphs)}), got {t!r}.")
    if not set(t) <= {"graph", "outside"}:
        raise AddressError(f"{where}: a graph target contains only `graph` and `outside`, got {t!r}: anchors already "
                           "place its nodes, so a set or token filter on it could drop some silently.")
    outside = None
    if "outside" in t:
        o = t["outside"]
        sets = book.sets()
        if not isinstance(o, Mapping) or set(o) != {"set", "tokens"} or o.get("tokens") != "prompt" \
                or o.get("set") not in sets:
            raise AddressError(f"{where}: `outside` is {{set: <a feature set of the book>, tokens: prompt}}, the "
                               f"finite universe a restriction acts in, got {o!r}.")
        _check_treatment_passes(sets, o["set"], where)
        spaces = book.graphs[graph].spaces
        for term in _leaf_terms(sets, o["set"]):
            if term.target != "feature" or term.features not in spaces:
                raise AddressError(f"{where}: the universe `{o['set']}` addresses `{to_string(term)}`; a restriction's "
                                   f"universe is the features of the graph's spaces ({sorted(spaces)}).")
        outside = o["set"]
    kind = "graph_complement" if outside is not None else "graph_nodes"
    if name not in GRAPH_OPS[kind]:
        raise AddressError(f"{where}: a `{kind}` target admits {list(GRAPH_OPS[kind])}, not `{name}`.")
    unknown = sorted(set(raw) & {"operand", "tokens"})
    if unknown:
        raise AddressError(f"{where}: a graph target takes no {unknown}: its nodes' anchors fix the positions, and "
                           "its values come from the graph.")
    spec = OPS[name]
    params = _params(raw.get("params") or {}, spec.params, where)
    if "value" not in raw:
        raise AddressError(f"{where}: a graph target needs a `value`.")
    value = _value(raw["value"], where)
    if outside is not None and value.source != "dose":
        raise AddressError(f"{where}: a restriction replaces the universe's other members with a `dose` (zero, or a "
                           f"statistic through a transform), not a `{value.source}` value.")
    if outside is None and value.source not in ("dose", "graph"):
        raise AddressError(f"{where}: a graph target's value is a `dose` or `from` a column of the graph, not a "
                           f"`{value.source}` value.")
    if value.source == "graph":
        if value.graph != graph:
            raise AddressError(f"{where}: the value reads graph `{value.graph}`, and the target is graph `{graph}`; "
                               "values come from the nodes they are applied to.")
        columns = book.graphs[graph].columns
        missing = [c for c in (value.column, value.baseline) if c is not None and c not in columns]
        if missing:
            raise AddressError(f"{where}: graph `{graph}` has no column {missing} on every node (columns: "
                               f"{list(columns)}).")
    return Op(name, params=params, value=value, graph=graph, outside=outside)


def _check_graphs(stages: Sequence[Stage], model: Mapping[str, Any], book: AddressBook, source: str) -> None:
    """Refuse graph ops whose graph pins another model, whose stage has no feature node of the graph or a token
    scope, or whose graph has feature nodes at a site no stage reaches."""
    reached: dict[str, set[str]] = {}
    for i, stage in enumerate(stages):
        for op in stage.ops:
            if op.graph is None:
                continue
            graph = book.graphs[op.graph]
            if (graph.model.get("id"), str(graph.model.get("revision"))) != (model["id"], str(model["revision"])):
                raise AddressError(f"{source}: graph `{op.graph}` pins {graph.model.get('id')}@"
                                   f"{graph.model.get('revision')}, not the treatment's "
                                   f"{model['id']}@{model['revision']}.")
            if stage.site not in graph.sites():
                raise AddressError(f"{source}: stage {i} ({stage.site}) targets graph `{op.graph}`, which has no "
                                   f"feature node there (its sites: {list(graph.sites())}).")
            reached.setdefault(op.graph, set()).add(stage.site)
    for name, sites in reached.items():
        unreached = [s for s in book.graphs[name].sites() if s not in sites]
        if unreached:
            raise AddressError(f"{source}: graph `{name}` has feature nodes at {unreached}, which no stage targeting "
                               "it reaches; select a smaller graph rather than drop its nodes.")


def _uses_graphs(body: Any) -> bool:
    """Whether a treatment mapping has an op targeting a graph."""
    stages = body.get("stages") if isinstance(body, Mapping) else None
    return isinstance(stages, list) and any(
        isinstance(st, Mapping) and isinstance(st.get("ops"), list)
        and any(isinstance(o, Mapping) and isinstance(o.get("target"), Mapping) and "graph" in o["target"]
                for o in st["ops"]) for st in stages)


def _reachable_sets(sets: Mapping[str, Expr], name: str) -> Iterator[Expr]:
    """The named set and each set it references, including references inside path waypoints."""
    seen: set[str] = set()
    todo = [name]
    while todo:
        n = todo.pop()
        if n in seen or n not in sets:
            continue
        seen.add(n)
        yield sets[n]
        todo += references(sets[n])


def _leaf_terms(sets: Mapping[str, Expr], name: str) -> list[Term]:
    """Every term the set ``name`` reaches, following references."""
    return [term for expr in _reachable_sets(sets, name) for term in terms(expr)]


def _check_treatment_passes(sets: Mapping[str, Expr], name: str, where: str) -> None:
    if any(_uses_passes(expr) for expr in _reachable_sets(sets, name)):
        raise AddressError(f"{where}: set `{name}` uses `passes`. Treatment consumers do not bind execution passes. "
                           "Resolve the site with its execution map outside the treatment.")


def _names_units(sets: Mapping[str, Expr], name: str | None) -> bool:
    return name is not None and any(t.units is not None for t in _leaf_terms(sets, name))


def _check_param_set(sets: Mapping[str, Expr], name: str, where: str, section: str = "weights") -> str:
    """Refuse a ``weights`` or ``gradients`` target set that does not address tensors, or addresses the other of their
    value and gradient; what the op acts on (``value`` or ``grad``)."""
    of = "value" if section == "weights" else "grad"
    for t in _leaf_terms(sets, name):
        if t.target != "param":
            raise AddressError(f"{where}: target set `{name}` addresses a {t.target} (`{to_string(t)}`); a "
                               f"`{section}` op acts on tensors, so its set names no `point`, `features` or "
                               "`subspace`.")
    stated = gradient_of(sets[name], sets) if any(t.of for t in _leaf_terms(sets, name)) else None
    if section == "weights" and stated not in (None, "value"):
        raise AddressError(f"{where}: target set `{name}` addresses `of: {stated}`; a `weights` op changes a "
                           "tensor's value, not its gradient.")
    if section == "gradients" and stated != "grad":
        raise AddressError(f"{where}: target set `{name}` addresses `of: {stated or 'value'}`; a `gradients` op acts "
                           "on a tensor's gradient, so its set says `of: grad`.")
    if _names_units(sets, name) and not isinstance(sets[name], Term):
        _UnitSets(sets, name, where).resolve()  # a composed unit set resolves when the treatment is read, as in a stage
    return of


def _stage(raw: Any, i: int, book: AddressBook, operands: Mapping[str, Operand], source: str) -> Stage:
    where = f"{source}: stage {i}"
    if not isinstance(raw, Mapping) or not isinstance(raw.get("site"), str) or not isinstance(raw.get("ops"), list) \
            or not raw["ops"]:
        raise AddressError(f"{where}: a stage is {{site, tokens, ops: [...]}} with at least one op.")
    unknown = set(raw) - {"site", "tokens", "ops", "readout"}
    if unknown:
        raise AddressError(f"{where}: unknown keys {sorted(unknown)}.")
    ops = tuple(_op(o, f"{where} op {j}", book, operands) for j, o in enumerate(raw["ops"]))
    if "tokens" in raw and any(o.graph is not None for o in ops):
        raise AddressError(f"{where}: a stage with a graph target gives no `tokens`: the graph's anchors place its "
                           "nodes, and a token scope could drop some silently.")
    readout = None
    if raw.get("readout") is not None:
        readout = _readout(raw["readout"], f"{where} readout", book, operands)
    return Stage(raw["site"], ops, _tokens(raw.get("tokens"), where), readout)


def _readout(raw: Any, where: str, book: AddressBook, operands: Mapping[str, Operand]) -> Readout:
    """Read a readout: a weighted sum of terms, each a registered reduction of a book set's activations."""
    if not isinstance(raw, Mapping) or set(raw) != {"terms"} or not isinstance(raw["terms"], list) or not raw["terms"]:
        raise AddressError(f"{where}: a readout is {{terms: [...]}} with at least one {{reduce, set, ...}} term; it is "
                           "the weighted sum of its terms, so it carries no expression.")
    sets = book.sets()
    terms_out = []
    for k, t in enumerate(raw["terms"]):
        at = f"{where} term {k}"
        if not isinstance(t, Mapping) or not isinstance(t.get("reduce"), str) or not isinstance(t.get("set"), str):
            raise AddressError(f"{at}: a term is {{reduce, set, tokens, weight, operand, params}}, got {t!r}.")
        unknown = set(t) - {"reduce", "set", "tokens", "weight", "operand", "params"}
        if unknown:
            raise AddressError(f"{at}: unknown keys {sorted(unknown)}.")
        spec = READOUTS.get(t["reduce"])
        if spec is None:
            raise AddressError(f"{at}: unknown reduction {t['reduce']!r}; known: {sorted(READOUTS)} (register one with "
                               "`venngram.readouts.register_readout`).")
        if t["set"] not in sets:
            raise AddressError(f"{at}: `set` names a set of the book ({sorted(sets)}), got {t['set']!r}.")
        _check_treatment_passes(sets, t["set"], at)
        for leaf in _leaf_terms(sets, t["set"]):
            if leaf.target == "param":
                raise AddressError(f"{at}: set `{t['set']}` addresses tensors (`{to_string(leaf)}`); a readout reads "
                                   "activations: a site's units or a latent space's features.")
            if leaf.of == "grad":
                raise AddressError(f"{at}: set `{t['set']}` addresses `of: grad`; a readout reads values, and is what "
                                   "the gradients are taken of.")
        weight = t.get("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight):
            raise AddressError(f"{at}: `weight` is a finite number, got {weight!r}.")
        operand = t.get("operand")
        if spec.operand:
            if not isinstance(operand, str) or operand not in operands:
                raise AddressError(f"{at}: `{spec.name}` takes an operand declared under `operands` "
                                   f"({sorted(operands)}), got {operand!r}.")
            o = operands[operand]
            if o.kind != "vector" or not o.basis:
                raise AddressError(f"{at}: operand `{operand}` must be a vector with a declared `basis`.")
        elif operand is not None:
            raise AddressError(f"{at}: `{spec.name}` takes no operand.")
        params = _params(t.get("params") or {}, spec.params, at)
        terms_out.append(ReadoutTerm(spec.name, t["set"], _tokens(t.get("tokens"), at), float(weight), operand, params))
    return Readout(tuple(terms_out))


def _check_readouts(stages: Sequence[Stage], readout: Readout | None, book: AddressBook, source: str,
                    version: int) -> None:
    """A treatment's readout, or a stage's, is stated exactly where a stage acts on gradients: refused where no stage
    (or not that stage) has an ``of: grad`` op, and required (from treatment schema 4) where one has and none
    applies."""
    grads = [any(o.target_of(book) == "grad" for o in s.ops) for s in stages]
    for i, (stage, has) in enumerate(zip(stages, grads)):
        if stage.readout is not None and not has:
            raise AddressError(f"{source}: stage {i} states a readout, but none of its ops acts on a gradient "
                               "(`of: grad`), so it takes no gradient of it.")
        if has and stage.readout is None and readout is None and version >= GRADIENTS_SCHEMA:
            raise AddressError(f"{source}: stage {i} acts on gradients (`of: grad`), so the treatment or the stage "
                               "states the `readout` they are gradients of.")
    if readout is not None and not any(has and s.readout is None for s, has in zip(stages, grads)):
        raise AddressError(f"{source}: the treatment states a readout, but no stage takes its gradients from it (no "
                           "stage acts on `of: grad` without a readout of its own).")


def _term_units(expr: Term, name: str, source: str) -> list[tuple[int, int]]:
    """The ``(layer, unit)`` pairs one address names, for an address whose layers and units are explicit."""
    if expr.passes is not None:
        raise AddressError(f"{source}: set `{name}` selects an execution pass, which needs a pass-aware consumer.")
    if expr.units is None:
        raise AddressError(f"{source}: set `{name}` must be one address with `units`, or a composition of unit sets, "
                           "to carry doses.")
    units = expr.units
    if units.heads is not None:
        raise AddressError(f"{source}: set `{name}` selects whole heads, which need a component map to resolve. "
                           "Doses need explicit coordinate indices.")
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
            raise AddressError(f"{source}: set `{name}` names a slice of units at layer {layer}; a treatment needs "
                               "explicit units.")
        out += [(layer, u) for u in idx]
    return out


def _layer_filter(expr: Expr) -> tuple[int, ...] | None:
    """The layers a ``layers``-only term names, or ``None`` for any other expression."""
    if isinstance(expr, Term) and set(expr.set_fields()) == {"layers"}:
        return expr.layers.indices  # type: ignore[union-attr]
    return None


def _of_only(expr: Expr) -> bool:
    # `of` does not filter (venngram.address), so an `of`-only term in an intersection says what is acted on and no more
    return isinstance(expr, Term) and set(expr.set_fields()) == {"of"}


class _UnitSets:
    """The ``(layer, unit)`` pairs a book set names when it composes unit sets with the set algebra.

    This is the treatment's counterpart of the tensor resolution in :mod:`venngram.resolve`: ``&`` intersects, ``|``
    unites and ``-`` subtracts, over pairs rather than tensor names. Every unit address it reaches must be at one site
    (the same fields besides ``layers``, ``units`` and ``of``), since a treatment's stage gives the site; ``of`` is read
    as :func:`~venngram.resolve.gradient_of` reads it.
    """

    def __init__(self, sets: Mapping[str, Expr], name: str, source: str) -> None:
        self.sets, self.name, self.source = sets, name, source
        self.sites: dict[tuple[tuple[str, str], ...], str] = {}
        self._stack: list[str] = []

    def refuse(self, why: str) -> AddressError:
        return AddressError(f"{self.source}: set `{self.name}` {why}.")

    def resolve(self) -> tuple[list[tuple[int, int]], str]:
        """The set's pairs in order of first appearance, and what is acted on there (``value`` or ``grad``)."""
        self._stack.append(self.name)
        out = self.run(self.sets[self.name])
        if len(self.sites) > 1:
            raise self.refuse(f"composes units at different sites ({', '.join(sorted(self.sites.values()))}); a "
                              "stage applies at one site, so compose sets of one site")
        if not out:
            raise self.refuse("names no units: its composition is empty")
        try:
            of = gradient_of(self.sets[self.name], self.sets)
        except AddressUnresolvable as err:
            raise self.refuse("; ".join(str(r) for r in err.refusals)) from err
        return out, of

    def run(self, expr: Expr) -> list[tuple[int, int]]:
        if isinstance(expr, Ref):
            if expr.name in self._stack:
                raise self.refuse(f"references form a cycle: {' -> '.join('@' + s for s in self._stack + [expr.name])}")
            if expr.name not in self.sets:
                raise self.refuse(f"references `@{expr.name}`, which the book does not name (sets: "
                                  f"{sorted(self.sets)})")
            self._stack.append(expr.name)
            out = self.run(self.sets[expr.name])
            self._stack.pop()
            return out
        if isinstance(expr, Guarded):
            out = self.run(expr.expr)
            if expr.expect is not None and len(out) != expr.expect:
                raise self.refuse(f"has `{to_string(expr.expr)}` name {len(out)} units where `expect` requires "
                                  f"{expr.expect}")
            return out
        if isinstance(expr, Term):
            return self.term(expr)
        if isinstance(expr, PathExpr):
            raise self.refuse("is a path, not a set of units")
        if isinstance(expr, Minus):
            layers = _layer_filter(expr.right)
            left = self.run(expr.left)
            if layers is not None:
                return [p for p in left if p[0] not in layers]
            drop = set(self.run(expr.right))
            return [p for p in left if p not in drop]
        items = tuple(i for i in expr.items if not (isinstance(expr, And) and _of_only(i)))
        filters = [_layer_filter(i) for i in items]
        if isinstance(expr, Or) and any(f is not None for f in filters):
            raise self.refuse("unites a `layers`-only term, which names every unit of its layers; a treatment needs "
                              "explicit units")
        results = [self.run(i) for i, f in zip(items, filters) if f is None]
        if not results:
            raise self.refuse("intersects only `layers` terms, which name no explicit units")
        if isinstance(expr, Or):
            return list(dict.fromkeys(p for r in results for p in r))
        keep = set.intersection(*(set(r) for r in results[1:])) if len(results) > 1 else None
        layers = [f for f in filters if f is not None]
        return [p for p in results[0] if (keep is None or p in keep) and all(p[0] in f for f in layers)]

    def term(self, t: Term) -> list[tuple[int, int]]:
        if t.features is not None:
            raise self.refuse(f"composes the feature set `{to_string(t)}`; a feature target is one address with "
                              "`features`, and feature sets do not compose in a treatment")
        if t.layers is not None and t.layers.indices is None and set(t.set_fields()) == {"layers"}:
            raise self.refuse(f"filters by `{to_string(t)}`, which depends on the model's depth; name the layers "
                              "explicitly")
        if t.units is None:
            raise self.refuse(f"composes `{to_string(t)}`, which names no explicit units")
        site = tuple((k, str(v)) for k, v in t.set_fields().items() if k not in ("layers", "units", "of"))
        self.sites.setdefault(site, to_string(Term(**{k: getattr(t, k) for k, _ in site})) if site
                              else "no site fields")
        return _term_units(t, self.name, self.source)


def _units(sets: Mapping[str, Expr], name: str, source: str) -> list[tuple[int, int]]:
    """The ``(layer, unit)`` pairs a book set names: one address with explicit layers and units, or a composition of
    such sets at one site (see :class:`_UnitSets`)."""
    expr = sets[name]
    if isinstance(expr, Term):
        return _term_units(expr, name, source)
    return _UnitSets(sets, name, source).resolve()[0]


@dataclass(frozen=True)
class BoundOp:
    """An op with the treatment it came from (its sets and operands are that treatment's)."""

    op: Op
    treatment: Treatment


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
                for unit, dose in b.op.unit_doses(b.treatment.book, b.treatment.source).items():
                    out[unit] = spec.combine(out[unit], dose) if unit in out else dose
        return out


def _commute(a: BoundOp, b: BoundOp) -> bool:
    if a.op.op == b.op.op and OPS[a.op.op].composition != "ordered":
        return True
    if a.op.target is not None and b.op.target is not None and a.op.value is not None and b.op.value is not None:
        try:
            ua = set(_units(a.treatment.book.sets(), a.op.target, a.treatment.source))
            ub = set(_units(b.treatment.book.sets(), b.op.target, b.treatment.source))
        except AddressError:
            return False
        return not ua & ub
    return False


def compose(treatments: Sequence[Treatment], *, ordered: bool = False) -> list[ComposedStage]:
    """The stages several treatments on one model revision amount to.

    Stages at the same site and token scope are joined, in treatment order. Without ``ordered``, ops from different
    treatments at one site must commute: the same summing or multiplying op (doses then combine per unit, see
    :meth:`ComposedStage.unit_doses`), or ops on disjoint unit sets.

    Raises:
        AddressError: If the treatments pin different models or revisions, or two ops at one site do not commute and
            the composition gives no order.
    """
    if not treatments:
        return []
    first = treatments[0]
    for r in treatments[1:]:
        if (r.model_id, r.revision) != (first.model_id, first.revision):
            raise AddressError(f"{r.source}: composes {r.model_id}@{r.revision} with {first.model_id}@"
                               f"{first.revision}; a composition is over one revision.")
    out: list[ComposedStage] = []
    for r in treatments:
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
                clash = next((a for a in out[match].ops if a.treatment is not r and not _commute(a, b)), None)
                if clash is not None:
                    raise AddressError(f"{r.source}: `{b.op.op}` at {s.site} does not commute with `{clash.op.op}` "
                                       f"from {clash.treatment.source}; compose with an order.")
            out[match] = ComposedStage(s.site, s.tokens, out[match].ops + bound)
    return out


def compose_weights(treatments: Sequence[Treatment], *, ordered: bool = False,
                    section: str = "weights") -> tuple[BoundOp, ...]:
    """The ``weights`` ops (or, with ``section="gradients"``, the ``gradients`` ops) several treatments on one model
    revision amount to, in treatment order.

    Without ``ordered``, ops from different treatments must commute as stage ops do: the same summing or multiplying op
    (additive deltas on one tensor sum), or ops on disjoint units.

    Raises:
        AddressError: If the treatments pin different revisions, or two ops do not commute and no order is given.
    """
    compose(treatments, ordered=True)  # the revision check; stages are composed by `compose` itself
    out: list[BoundOp] = []
    for r in treatments:
        for o in getattr(r, section):
            b = BoundOp(o, r)
            if not ordered:
                clash = next((a for a in out if a.treatment is not r and not _commute(a, b)), None)
                if clash is not None:
                    raise AddressError(f"{r.source}: {section} op `{o.op}` does not commute with `{clash.op.op}` from "
                                       f"{clash.treatment.source}; compose with an order.")
            out.append(b)
    return tuple(out)


@dataclass(frozen=True)
class Composition:
    """Several treatments applied together, in order when ``ordered``."""

    treatments: tuple[Treatment, ...]
    ordered: bool = False

    def stages(self) -> list[ComposedStage]:
        """The stages to apply, in order (see :func:`compose`)."""
        return compose(self.treatments, ordered=self.ordered)

    def weights(self) -> tuple[BoundOp, ...]:
        """The weight ops to apply before any stage, in order (see :func:`compose_weights`)."""
        return compose_weights(self.treatments, ordered=self.ordered)

    def gradients(self) -> tuple[BoundOp, ...]:
        """The ops to apply to parameter gradients, in order (see :func:`compose_weights`)."""
        return compose_weights(self.treatments, ordered=self.ordered, section="gradients")


__all__ = [
    "BoundOp",
    "compose",
    "compose_weights",
    "ComposedStage",
    "Composition",
    "Op",
    "Operand",
    "Readout",
    "ReadoutTerm",
    "GRADIENTS_SCHEMA",
    "GRAPH_OPS",
    "GRAPHS_SCHEMA",
    "WEIGHTS_SCHEMA",
    "Stage",
    "Tokens",
    "Transform",
    "Treatment",
    "TREATMENT_SCHEMA_VERSION",
    "Value",
]
