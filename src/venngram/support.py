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
Support Records
^^^^^^^^^^^^^^^

A **support record** declares what a consumer honours, so a document asking for anything else is refused by name
before anything runs, and so the check can run without the consumer installed. A record is a ``support_record``
document of **sections**, one per contract the consumer offers, and it opts in to each:

.. code-block:: yaml

    schema_version: 9
    kind: support_record
    name: finetuning-scheduler
    sections:
      tuning:                          # address books over parameters
        fields: [role, name, module, layers, stack, component, leaf, of]
        targets: [param]
        units: grad-mask               # none | unit-adapter | grad-mask
        naming_schemes: [physical]
      treatments: {ops: [add, scale]}  # treatments (schema 1's fields)
    extensions:                        # sections defined outside venngram
      acme.probes: {schema: "hub-model:acme/probe-support:schema.yaml", description: probe heads}

**Core sections** (:data:`CORE_SECTIONS`):

- ``treatments``: the treatments it applies, as :class:`venngram.ops.Support` (ops, targets, token scopes, value
  sources, transforms, sites, bases), whether it steers activations with them or applies their ``weights`` and
  ``gradients`` ops while tuning;
- ``tuning``: address books whose sets select tensors to tune (:class:`TuningSupport`);
- ``circuits``: traced or validated circuits, attribution graphs in general (:class:`CircuitSupport`), and, under
  ``subsets`` (schema 6), the treatments over graph subsets it applies (:class:`SubsetSupport`). This section is a
  prototype: its vocabulary is expected to change as circuit documents are defined;
- ``formats``: the foreign formats a package translates into venngram documents, and where to install it from
  (:class:`FormatSupport`), so a reader can say how to read a foreign document without running anything.

**Opt-in.** A section a record omits is a contract its consumer does not offer: :meth:`SupportRecord.check_book` and
:meth:`SupportRecord.check_treatment` refuse every document that needs it, naming the section.

**Unknown sections are skipped, not fatal.** A record written for a newer venngram, or carrying another project's
section, still loads: a section type this reader cannot parse is reported (:class:`UnknownSupportSectionWarning`,
naming it) and kept, unparsed, in :attr:`SupportRecord.skipped`. A section defined outside venngram is namespaced
(``<owner>.<name>``) and declared under ``extensions`` with the schema that defines it; code that understands it
registers a parser (:func:`register_support_section`), and the section is then parsed like a core one.

**Renamed in schema 9** (:data:`TREATMENTS_SCHEMA`): the ``steering`` section is ``treatments``, and the ``circuits``
section's ``evidence`` is ``levels``. A record of schema 1 to 8 still reads, its old names through an alias table with a
:class:`~venngram.deprecation.VenngramDeprecationWarning`; a schema 9 record using an old name is refused by name.
Schema 9 also adds the ``grad-mask`` unit pattern (:data:`UNIT_PATTERNS`).

**Schema 1** records (a ``steering`` contract written at the top level) still read, as a record with only a
``treatments`` section.

.. warning:: This is an experimental API and may change in a future release.
"""
import re
import warnings
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from venngram.address import FIELDS, references, TARGETS, terms
from venngram.book import AddressBook, PIN_KEYS, PORTABILITY, STRUCTURE_LEVELS
from venngram.deprecation import deprecated
from venngram.errors import NewerSchemaError
from venngram.ops import check_support, PARAM_TARGETS, Support
from venngram.resolve import _of_values, _split, GRADIENT_FORMS, gradient_forms

SCHEMA_VERSION = 2
SCHEMA_WINDOW = (1, 9)
#: The first schema whose ``treatments`` section (``steering`` before schema 9) may declare ``of`` (a key older
#: readers refuse).
OF_SCHEMA = 3
#: The first schema whose ``treatments`` section may declare a parameter target (``param``, ``param_units``).
PARAM_SCHEMA = 4
#: The first schema whose ``tuning`` section may declare ``gradients`` (the gradient readings it honours) and whose
#: ``treatments`` section may declare ``gradient_ops`` (the ops it applies to parameter gradients) and ``readouts`` (the
#: reductions it computes a readout with).
GRADIENTS_SCHEMA = 5
#: The first schema whose ``circuits`` section may declare ``subsets`` (the graph subset treatments it applies).
SUBSETS_SCHEMA = 6
#: The first schema whose ``tuning`` section may declare ``pins`` (that it checks a book's model pins).
PINS_SCHEMA = 7
#: The first schema whose ``treatments`` section may declare ``routes`` (the path waypoint kinds it places).
ROUTES_SCHEMA = 8
#: The first schema whose treatments section is named ``treatments`` (``steering`` before), whose ``circuits`` section
#: names its ``levels`` (``evidence`` before), and whose ``tuning`` section may declare ``units: grad-mask``.
TREATMENTS_SCHEMA = 9
CORE_SECTIONS = ("treatments", "tuning", "circuits", "formats")
#: The section names a schema 1 to 8 record may use, read as their schema 9 names with a deprecation warning.
SECTION_ALIASES = {"steering": "treatments"}
#: How a tuning consumer honours a unit set: ``none`` (it refuses them), ``unit-adapter`` (through a module whose
#: parameters are per-unit values) or ``grad-mask`` (only under ``of: grad``, masking each tensor's gradient).
UNIT_PATTERNS = ("none", "unit-adapter", "grad-mask")
_EXTENSION_NAME = re.compile(r"[a-z][a-z0-9-]*(\.[a-z][a-z0-9_-]*)+")
_TOP_KEYS = ("schema_version", "kind", "name", "description", "sections", "extensions")


class UnknownSupportSectionWarning(UserWarning):
    """A support record has a section type this reader cannot parse; the section is skipped."""


def _names(raw: Any, key: str, source: str, allowed: tuple[str, ...] | None = None) -> frozenset[str]:
    if not isinstance(raw, list) or not all(isinstance(v, str) for v in raw):
        raise ValueError(f"{source}: `{key}` is a list of names, got {raw!r}.")
    bad = sorted(set(raw) - set(allowed)) if allowed is not None else []
    if bad:
        raise ValueError(f"{source}: `{key}` names {bad}, which venngram does not define (known: "
                         f"{list(allowed or ())}).")
    return frozenset(raw)


def _read_fields(doc: Any, source: str, spec: Mapping[str, tuple[str, ...] | None]) -> dict[str, Any]:
    if not isinstance(doc, Mapping):
        raise ValueError(f"{source}: a section is a mapping, got {type(doc).__name__}.")
    unknown = sorted(set(doc) - set(spec))
    if unknown:
        raise ValueError(f"{source}: unknown keys {unknown} (known: {sorted(spec)}).")
    return {k: _names(v, k, source, spec[k]) for k, v in doc.items()}


@dataclass(frozen=True)
class TuningSupport:
    """What a tuning consumer honours in an address book. ``None`` fields mean any.

    Attributes:
        fields: The address fields its resolution honours (see :mod:`venngram.address`). Defaults exclude ``passes``,
            ``of`` and ``axis``. A consumer acts on gradients (``of: grad``) only when it lists ``of``, and masks them
            along a named dimension (``axis``, resolved by :func:`venngram.resolve.gradient_selections`) only when it
            lists ``axis`` and the forms (``unit-axis``, ``physical-axis``) under ``gradients``.
        targets: What its sets may select (``param`` for tensors; a tuning consumer reaches sites, features or
            subspaces only through an adapter or a latent model it owns).
        units: How it tunes a unit set: ``none`` (it refuses unit sets), ``unit-adapter`` (a module whose parameters
            are per-unit values, so the units are tuned through that module's tensors) or ``grad-mask`` (a unit set
            only under ``of: grad``, masking each selected tensor's gradient to those units; a unit set read as values
            is refused). ``grad-mask`` needs schema 9.
        naming_schemes: The naming schemes its views provide; a book whose ``requires.naming_scheme`` is another is
            refused.
        gradients: The gradient readings it honours beyond one ``of`` per set and one mask per conjunction
            (:data:`venngram.resolve.GRADIENT_FORMS`): ``mixed-sets`` (a union of a value part and a gradient part,
            which it splits with :func:`venngram.resolve.split_of`) and ``union-masks`` (a union of gradient
            conjunctions with unit sets). None by default.
        structure: The ``requires.structure`` levels it provides.
        portability: The book portabilities it accepts.
        pins: Whether it checks the models a book pins in its ``requires`` (``base_model``, ``latent_models``)
            against the ones it loaded, with :meth:`venngram.AddressBook.check_pins`. A book that pins them is refused
            by a consumer that does not (schema 7).
    """

    fields: frozenset[str] = frozenset(f for f in FIELDS if f not in ("of", "axis", "passes"))
    targets: frozenset[str] = frozenset({"param"})
    units: str = "none"
    naming_schemes: frozenset[str] | None = None
    structure: frozenset[str] = frozenset(STRUCTURE_LEVELS)
    portability: frozenset[str] = frozenset(PORTABILITY)
    gradients: frozenset[str] = frozenset()
    pins: bool = False

    @classmethod
    def from_mapping(cls, doc: Any, *, source: str = "<inline>") -> "TuningSupport":
        """Read a ``tuning`` section.

        Raises:
            ValueError: If the section has unknown keys or names a field, target, unit pattern, structure level or
                portability venngram does not define.
        """
        spec: dict[str, tuple[str, ...] | None] = {"fields": FIELDS, "targets": TARGETS, "naming_schemes": None,
                                                   "structure": STRUCTURE_LEVELS, "portability": PORTABILITY,
                                                   "gradients": GRADIENT_FORMS}
        units = doc.get("units", "none") if isinstance(doc, Mapping) else None
        if not isinstance(units, str) or units not in UNIT_PATTERNS:
            raise ValueError(f"{source}: `units` is one of {list(UNIT_PATTERNS)}, got {units!r}.")
        pins = doc.get("pins", False) if isinstance(doc, Mapping) else False
        if not isinstance(pins, bool):
            raise ValueError(f"{source}: `pins` is true or false, got {pins!r}.")
        rest = {k: v for k, v in doc.items() if k not in ("units", "pins")} if isinstance(doc, Mapping) else doc
        return cls(**_read_fields(rest, source, spec), units=units, pins=pins)

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {"fields": [f for f in FIELDS if f in self.fields],
                               "targets": [t for t in TARGETS if t in self.targets], "units": self.units}
        if self.naming_schemes is not None:
            out["naming_schemes"] = sorted(self.naming_schemes)
        out["structure"] = [s for s in STRUCTURE_LEVELS if s in self.structure]
        out["portability"] = [p for p in PORTABILITY if p in self.portability]
        if self.gradients:
            out["gradients"] = [g for g in GRADIENT_FORMS if g in self.gradients]
        if self.pins:
            out["pins"] = True
        return out

    def check_book(self, book: AddressBook, *, sets: Collection[str] | None = None) -> list[str]:
        """Every way ``book`` (and each book it imports) asks for something this section does not honour.

        Args:
            book: The book.
            sets: Check only these sets (as :meth:`venngram.AddressBook.sets` names them, ``<namespace>.<set>`` for an
                import's) and the sets they reference, so a book shared for several purposes is judged by the sets a
                consumer uses. The book-level requirements are always checked. ``None`` checks every set.
        """
        out: list[str] = []
        self._check(book, book.source, out, None if sets is None else _closure(book, sets))
        return out

    def _check(self, book: AddressBook, where: str, out: list[str], sets: set[str] | None) -> None:
        scheme = book.requires.get("naming_scheme")
        if scheme is not None and self.naming_schemes is not None and scheme not in self.naming_schemes:
            out.append(f"{where}: it is written in `{scheme}` names; this consumer's views provide "
                       f"{sorted(self.naming_schemes)}")
        structure = book.requires.get("structure", "physical")
        if structure not in self.structure:
            out.append(f"{where}: it requires `{structure}` structure, which this consumer does not provide")
        if book.portability not in self.portability:
            out.append(f"{where}: `{book.portability}` books are not accepted")
        pinned = [k for k in PIN_KEYS if k in book.requires]
        if pinned and not self.pins:
            named = ", ".join(f"`requires.{k}`" for k in pinned)
            out.append(f"{where}: it pins the models it was written against ({named}), which this consumer does not "
                       "check (`pins: false`)")
        every = book.sets()
        masked = _masked_sets(book.own_sets, every) if self.units == "grad-mask" else set()
        for set_name, expr in book.own_sets.items():
            if sets is not None and set_name not in sets:
                continue
            at = f"{where}: set `{set_name}`"
            needed = sorted(gradient_forms(expr, every) - self.gradients)
            if needed:
                out.append(f"{at}: it needs the gradient readings {needed}, which this consumer does not honour "
                           f"(`gradients`: {[g for g in GRADIENT_FORMS if g in self.gradients]})")
            for term in terms(expr):
                used = term.set_fields()
                bad = [f for f in used if f not in self.fields]
                if bad:
                    out.append(f"{at}: the fields {bad} are not honoured (honoured: "
                               f"{[f for f in FIELDS if f in self.fields]})")
                if term.target not in self.targets:
                    out.append(f"{at}: it selects a `{term.target}`, which this consumer does not tune "
                               f"(it tunes {sorted(self.targets)})")
                if "units" in used and self.units == "none":
                    out.append(f"{at}: it selects units, and this consumer tunes no unit sets (`units: none`)")
            if self.units == "grad-mask" and set_name not in masked:
                valued = [t for t in _value_unit_terms(expr, every)]
                if valued:
                    out.append(f"{at}: it selects units as values (`{valued[0]}`), and this consumer applies unit sets "
                               "only as gradient masks (`units: grad-mask`), under `of: grad`")
        for ns, imported in book.imports.items():
            imported_out: list[str] = []
            within = None if sets is None else {s[len(ns) + 1:] for s in sets if s.startswith(f"{ns}.")}
            self._check(imported, imported.source, imported_out, within)
            out.extend(f"import `{ns}`: {line}" for line in imported_out)


def _value_unit_terms(expr: Any, sets: Mapping[str, Any]) -> list[str]:
    """The terms with ``units`` in ``expr``'s value part (a set that cannot be split has none here; resolving it
    reports why)."""
    from venngram.address import to_string

    value, _, refusals = _split(expr, sets)
    if refusals:
        return []
    return [to_string(t) for part in value for t in terms(part) if t.units is not None]


def _masked_sets(own: Mapping[str, Any], sets: Mapping[str, Any]) -> set[str]:
    """The book's sets that state no ``of`` of their own and that another set's gradient part references
    (transitively): a unit set written once and masked by its users (``@heads & of:grad``), so it is judged there."""
    out: set[str] = set()
    todo = []
    for expr in own.values():
        _, grad, refusals = _split(expr, sets)
        if not refusals:
            todo.extend(r for part in grad for r in references(part))
    while todo:
        name = todo.pop()
        if name in out or name not in own or _of_values(own[name], sets)[0]:
            continue
        out.add(name)
        todo.extend(references(own[name]))
    return out


def _closure(book: AddressBook, names: Collection[str]) -> set[str]:
    """``names`` and every set they reference, transitively, as :meth:`venngram.AddressBook.sets` names them.

    Raises:
        ValueError: If a name is not a set of the book.
    """
    every = book.sets()
    missing = sorted(set(names) - set(every))
    if missing:
        raise ValueError(f"{book.source}: no set is named {missing} (sets: {sorted(every)}).")
    out: set[str] = set()
    todo = list(names)
    while todo:
        name = todo.pop()
        if name not in out:
            out.add(name)
            todo.extend(r for r in references(every[name]) if r in every)
    return out


#: The node kinds of an attribution graph this prototype names.
CIRCUIT_NODES = ("feature", "error", "embedding", "logit")
#: The processing levels of the graphs a consumer accepts: traced (from attribution), pruned (thresholded), validated
#: (by intervention).
CIRCUIT_LEVELS = ("traced", "pruned", "validated")


SUBSET_VALUES = ("scalar", "vector")


@dataclass(frozen=True)
class SubsetSupport:
    """What a consumer applies of a treatment over a graph subset (:mod:`venngram.graphs`). The defaults apply nothing:
    a consumer lists what it honours.

    .. code-block:: yaml

        circuits:
          subsets:
            ops: [replace, add]
            anchors: [absolute, from_end, token_match]
            featurizers: [transcoder]        # the space kinds whose nodes it places; default any
            values: [scalar]                 # scalar | vector (a block's intra-block state); default scalar
            columns: [activation, baseline]  # default any
            transforms: [scale]              # default none
            complement: true                 # restriction (`outside:`); default false
            sites: [mlp.out.hook_out]        # names or fnmatch patterns; default any

    Attributes:
        ops: The ops it applies to graph targets.
        anchors: The anchor kinds it resolves (:data:`venngram.graphs.ANCHORS`).
        featurizers: The space kinds whose nodes it places (``None``: any).
        values: Whether it applies scalar values, vector values (``value_shape: [d]``), or both.
        columns: The value columns it reads (``None``: any).
        transforms: The transforms it applies to graph values.
        complement: Whether it applies a restriction over a finite universe.
        sites: The sites it reaches (``None``: any).
    """

    ops: frozenset[str] = frozenset()
    anchors: frozenset[str] = frozenset()
    featurizers: frozenset[str] | None = None
    values: frozenset[str] = frozenset({"scalar"})
    columns: frozenset[str] | None = None
    transforms: frozenset[str] = frozenset()
    complement: bool = False
    sites: frozenset[str] | None = None

    @classmethod
    def from_mapping(cls, doc: Any, *, source: str = "<inline>") -> "SubsetSupport":
        """Read a ``subsets`` declaration.

        Raises:
            ValueError: If it has unknown keys, names an anchor or value form venngram does not define, or
                ``complement`` is not a boolean.
        """
        from venngram.graphs import ANCHORS

        if not isinstance(doc, Mapping):
            raise ValueError(f"{source}: `subsets` is a mapping.")
        rest = {k: v for k, v in doc.items() if k != "complement"}
        complement = doc.get("complement", False)
        if not isinstance(complement, bool):
            raise ValueError(f"{source}: `complement` is true or false, got {complement!r}.")
        return cls(**_read_fields(rest, source, {"ops": None, "anchors": ANCHORS, "featurizers": None,
                                                 "values": SUBSET_VALUES, "columns": None, "transforms": None,
                                                 "sites": None}), complement=complement)

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {k: sorted(v) for k in ("ops", "anchors", "featurizers", "values", "columns",
                                                      "transforms", "sites")
                               if (v := getattr(self, k)) is not None}
        out["complement"] = self.complement
        return out

    def check_treatment(self, treatment: Any) -> list[str]:
        """Every way ``treatment``'s graph ops ask for something this declaration does not honour."""
        import fnmatch

        out: list[str] = []
        for i, stage in enumerate(treatment.stages):
            for j, op in enumerate(stage.ops):
                if op.graph is None:
                    continue
                at = f"stage {i} ({stage.site}) op {j} (`{op.op}`, graph `{op.graph}`)"
                graph = treatment.book.graphs[op.graph]
                if op.op not in self.ops:
                    out.append(f"{at}: the op is not applied to graph targets (`subsets.ops`: {sorted(self.ops)})")
                if self.sites is not None and stage.site not in self.sites and not any(
                        fnmatch.fnmatchcase(stage.site, p) for p in self.sites):
                    out.append(f"{at}: the site `{stage.site}` is not one this consumer reaches")
                if op.outside is not None and not self.complement:
                    out.append(f"{at}: a restriction (`outside`) is not applied (`subsets.complement` is false)")
                nodes = [n for n in graph.nodes if n.kind == "feature" and n.site == stage.site]
                for kind in sorted({n.anchor.kind for n in nodes if n.anchor is not None} - self.anchors):
                    out.append(f"{at}: anchor `{kind}` is not resolved (`subsets.anchors`: {sorted(self.anchors)})")
                spaces = {graph.spaces[str(n.space)] for n in nodes}
                for space in sorted(spaces, key=lambda s: s.name):
                    if self.featurizers is not None and space.kind not in self.featurizers:
                        out.append(f"{at}: space `{space.name}` is a `{space.kind}` featurizer, which is not placed "
                                   f"(`subsets.featurizers`: {sorted(self.featurizers)})")
                    form = "vector" if space.value_shape else "scalar"
                    if form not in self.values:
                        out.append(f"{at}: space `{space.name}` has {form} values, which are not applied "
                                   f"(`subsets.values`: {sorted(self.values)})")
                value = op.value
                if value is not None and value.source == "graph" and self.columns is not None:
                    for c in [value.column, value.baseline]:
                        if c is not None and c not in self.columns:
                            out.append(f"{at}: column `{c}` is not read (`subsets.columns`: {sorted(self.columns)})")
                for t in value.transform if value is not None else ():
                    if t.name not in self.transforms:
                        out.append(f"{at}: transform `{t.name}` is not applied to graph values")
        return out


@dataclass(frozen=True)
class CircuitSupport:
    """What a consumer of circuits honours (prototype). ``None`` fields mean any.

    A circuit is a subgraph of a model's computation with evidence for it: an attribution graph (circuit-tracer's is one
    producer), pruned to a threshold, perhaps validated by intervening on its nodes. Its nodes are addressed with
    venngram addresses (a feature, a site, a token position), so a circuit document composes with books and treatments.

    Attributes:
        graphs: The graph kinds it reads (``attribution``; others as they are defined).
        nodes: The node kinds it honours (see :data:`CIRCUIT_NODES`).
        features: The feature sources its feature nodes may come from (``transcoder``, ``clt``, ``sae``; any name).
        levels: The processing levels of the graphs it accepts (see :data:`CIRCUIT_LEVELS`); ``evidence`` before
            schema 9.
        subsets: The treatments over graph subsets it applies (``None``: none; reading graphs is not permission to
            intervene with them).
    """

    graphs: frozenset[str] = frozenset({"attribution"})
    nodes: frozenset[str] | None = None
    features: frozenset[str] | None = None
    levels: frozenset[str] | None = None
    subsets: SubsetSupport | None = None

    @classmethod
    def from_mapping(cls, doc: Any, *, source: str = "<inline>") -> "CircuitSupport":
        """Read a ``circuits`` section.

        Raises:
            ValueError: If the section has unknown keys or names a node kind or level venngram does not define.
        """
        if isinstance(doc, Mapping) and "evidence" in doc:
            raise ValueError(f"{source}: `evidence` is named `levels` from support record schema {TREATMENTS_SCHEMA}.")
        rest = {k: v for k, v in doc.items() if k != "subsets"} if isinstance(doc, Mapping) else doc
        subsets = SubsetSupport.from_mapping(doc["subsets"], source=f"{source}: subsets") \
            if isinstance(doc, Mapping) and "subsets" in doc else None
        return cls(**_read_fields(rest, source, {"graphs": None, "nodes": CIRCUIT_NODES, "features": None,
                                                 "levels": CIRCUIT_LEVELS}), subsets=subsets)

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {k: sorted(v) for k, v in (("graphs", self.graphs), ("nodes", self.nodes),
                                                         ("features", self.features), ("levels", self.levels))
                               if v is not None}
        if self.subsets is not None:
            out["subsets"] = self.subsets.to_mapping()
        return out

    def check_treatment(self, treatment: Any) -> list[str]:
        """Every way ``treatment``'s graph ops ask for something this section does not honour."""
        if self.subsets is None:
            return [f"treatment `{getattr(treatment, 'name', '?')}`: its graph targets need `circuits.subsets`, which "
                    "this consumer does not declare (reading graphs is not applying them)"]
        return self.subsets.check_treatment(treatment)


@dataclass(frozen=True)
class FormatSupport:
    """The foreign formats a package reads (each a :class:`venngram.sources.Format` it registers), its version, and
    where to install it from. A record with only this section describes a format adapter as data, so a manifest can
    list one a reader has not installed; installing stays an explicit act.

    .. code-block:: yaml

        formats:
          reads: [neuronary]
          version: 0.1.0
          install: {pip: venngram-neuronary}        # or {git: https://github.com/<owner>/<repo>, commit: <sha>}
    """

    reads: tuple[str, ...]
    install: Mapping[str, str]
    version: str | None = None

    @classmethod
    def from_mapping(cls, doc: Any, *, source: str = "<inline>") -> "FormatSupport":
        """Read a ``formats`` section.

        Raises:
            ValueError: If the section has unknown keys, lists no format, or its ``install`` names neither a ``pip``
                requirement nor a ``git`` URL pinned to a ``commit``.
        """
        if not isinstance(doc, Mapping):
            raise ValueError(f"{source}: a section is a mapping, got {type(doc).__name__}.")
        unknown = sorted(set(doc) - {"reads", "install", "version"})
        if unknown:
            raise ValueError(f"{source}: unknown keys {unknown}; a `formats` section has `reads`, `install` and "
                             "`version`.")
        reads = doc.get("reads")
        if not isinstance(reads, list) or not reads or not all(isinstance(f, str) for f in reads):
            raise ValueError(f"{source}: `reads` lists the format names the package reads.")
        install = doc.get("install")
        if not isinstance(install, Mapping) or not ({"pip", "git"} & set(install)):
            raise ValueError(f"{source}: `install` names a `pip` requirement or a `git` URL (with a `commit`).")
        if "git" in install and not install.get("commit"):
            raise ValueError(f"{source}: a `git` install pins a `commit`, so the record cites one package.")
        version = doc.get("version")
        return cls(tuple(reads), {str(k): str(v) for k, v in install.items()},
                   str(version) if version is not None else None)

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {"reads": list(self.reads)}
        if self.version is not None:
            out["version"] = self.version
        out["install"] = dict(self.install)
        return out


def _treatments(doc: Any, *, source: str) -> Support:
    if not isinstance(doc, Mapping):
        raise ValueError(f"{source}: a section is a mapping, got {type(doc).__name__}.")
    return Support.from_mapping({"schema_version": 1, "kind": "support_record", **doc}, source=source)


#: Parsers by section type: the section's mapping -> its parsed contract.
_SECTIONS: dict[str, Callable[..., Any]] = {"treatments": _treatments, "tuning": TuningSupport.from_mapping,
                                            "circuits": CircuitSupport.from_mapping,
                                            "formats": FormatSupport.from_mapping}


def register_support_section(name: str, parser: Callable[..., Any], *, replace: bool = False) -> None:
    """Parse sections of type ``name`` with ``parser(doc, source=...)``. An extension's name is namespaced
    (``<owner>.<name>``); the core names are venngram's.

    Raises:
        ValueError: If ``name`` is a core section or is not namespaced, or is registered and ``replace`` is false.
    """
    if name in CORE_SECTIONS or name in SECTION_ALIASES or not _EXTENSION_NAME.fullmatch(name):
        raise ValueError(f"`{name}`: an extension section is namespaced, `<owner>.<name>` in lowercase, and is not one"
                         f" of venngram's core sections {list(CORE_SECTIONS)}.")
    if name in _SECTIONS and not replace:
        raise ValueError(f"support section `{name}` is already registered; pass replace=True to replace it.")
    _SECTIONS[name] = parser


def support_sections() -> tuple[str, ...]:
    """The section types this process can parse."""
    return tuple(_SECTIONS)


@dataclass(frozen=True)
class SupportRecord:
    """A consumer's support record: its parsed sections, and the sections it carries that this reader skipped.

    Attributes:
        sections: Parsed contracts by section type.
        skipped: Sections this reader could not parse, by type, kept as written.
        extensions: The extension sections the record declares, with the schema that defines each.
    """

    name: str | None = None
    sections: Mapping[str, Any] = field(default_factory=dict)
    skipped: Mapping[str, Any] = field(default_factory=dict)
    extensions: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    description: str | None = None

    @property
    def treatments(self) -> Support | None:
        """The ``treatments`` section: the treatments the consumer applies."""
        return self.sections.get("treatments")

    @property
    def tuning(self) -> TuningSupport | None:
        return self.sections.get("tuning")

    @property
    def circuits(self) -> CircuitSupport | None:
        return self.sections.get("circuits")

    @property
    def formats(self) -> FormatSupport | None:
        return self.sections.get("formats")

    @classmethod
    def from_mapping(cls, doc: Any, *, source: str = "<inline>") -> "SupportRecord":
        """Read a ``support_record`` document (schema 1 to 9; see the module docs for the schema 9 renames).

        Raises:
            ValueError: If the document is not a ``support_record`` in the schema window, has unknown top-level keys,
                declares a malformed extension, or a section this reader parses is malformed.
        """
        if not isinstance(doc, Mapping) or doc.get("kind") != "support_record":
            raise ValueError(f"{source}: a support record is `kind: support_record`.")
        version = doc.get("schema_version")
        if isinstance(version, int) and not isinstance(version, bool) and version > SCHEMA_WINDOW[1]:
            raise NewerSchemaError(f"{source}: support record schema {version} is newer than this venngram reads "
                                   f"({SCHEMA_WINDOW[0]} to {SCHEMA_WINDOW[1]}); upgrade venngram.",
                                   kind="support_record", version=version, newest=SCHEMA_WINDOW[1])
        if not isinstance(version, int) or not SCHEMA_WINDOW[0] <= version <= SCHEMA_WINDOW[1]:
            raise ValueError(f"{source}: this venngram reads support record schemas {SCHEMA_WINDOW[0]} to "
                             f"{SCHEMA_WINDOW[1]}, got {version!r}.")
        name = doc.get("name")
        name = str(name) if name is not None else None
        if version == 1:
            if "of" in doc:
                raise ValueError(f"{source}: `of` is declared in the `treatments` section of a schema {OF_SCHEMA} "
                                 "record, not at the top level of a schema 1 one.")
            if _param_targets(doc):
                raise ValueError(f"{source}: a parameter target is declared in the `treatments` section of a schema "
                                 f"{PARAM_SCHEMA} record, not at the top level of a schema 1 one.")
            for key in ("gradient_ops", "readouts"):
                if key in doc:
                    raise ValueError(f"{source}: `{key}` is declared in the `treatments` section of a schema "
                                     f"{GRADIENTS_SCHEMA} record, not at the top level of a schema 1 one.")
            return cls(name=name, sections={"treatments": Support.from_mapping(doc, source=source)})
        unknown = sorted(set(doc) - set(_TOP_KEYS))
        if unknown:
            raise ValueError(f"{source}: unknown support record keys {unknown}; contracts go under `sections`.")
        extensions = cls._read_extensions(doc.get("extensions") or {}, source)
        raw = doc.get("sections")
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError(f"{source}: `sections` maps each contract the consumer offers (for example "
                             f"{list(CORE_SECTIONS)}) to its declaration.")
        sections: dict[str, Any] = {}
        skipped: dict[str, Any] = {}
        for kind, body in raw.items():
            where = f"{source}: section `{kind}`"
            kind, body = _renamed(kind, body, version, where)
            if kind in sections:
                raise ValueError(f"{where}: the record declares its `{kind}` section twice (once by its schema 1 to 8 "
                                 "name).")
            if kind in _SECTIONS:
                if kind == "treatments" and isinstance(body, Mapping) and "of" in body and version < OF_SCHEMA:
                    raise ValueError(f"{where}: `of` needs support record schema {OF_SCHEMA}, not {version}: an older "
                                     "venngram would refuse the record by name rather than pass over it.")
                if kind == "treatments" and isinstance(body, Mapping) and "routes" in body and \
                        version < ROUTES_SCHEMA:
                    raise ValueError(f"{where}: `routes` needs support record schema {ROUTES_SCHEMA}, not {version}: "
                                     "an older venngram would refuse the record by name rather than pass over it.")
                if kind == "tuning" and isinstance(body, Mapping) and "pins" in body and version < PINS_SCHEMA:
                    raise ValueError(f"{where}: `pins` needs support record schema {PINS_SCHEMA}, not {version}: an "
                                     "older venngram would refuse the record by name rather than pass over it.")
                keys = {"tuning": ("gradients",), "treatments": ("gradient_ops", "readouts")}.get(kind, ())
                for key in keys:
                    if isinstance(body, Mapping) and key in body and version < GRADIENTS_SCHEMA:
                        raise ValueError(f"{where}: `{key}` needs support record schema {GRADIENTS_SCHEMA}, not "
                                         f"{version}: an older venngram would refuse the record by name rather than "
                                         "pass over it.")
                if kind == "circuits" and isinstance(body, Mapping) and "subsets" in body and version < SUBSETS_SCHEMA:
                    raise ValueError(f"{where}: `subsets` needs support record schema {SUBSETS_SCHEMA}, not {version}: "
                                     "an older venngram would refuse the record by name rather than pass over it.")
                if kind == "treatments" and _param_targets(body) and version < PARAM_SCHEMA:
                    raise ValueError(f"{where}: the parameter targets {_param_targets(body)} need support record "
                                     f"schema {PARAM_SCHEMA}, not {version}: an older venngram would refuse the record "
                                     "by name rather than pass over it.")
                if kind == "tuning" and isinstance(body, Mapping) and body.get("units") == "grad-mask" and \
                        version < TREATMENTS_SCHEMA:
                    raise ValueError(f"{where}: `units: grad-mask` needs support record schema {TREATMENTS_SCHEMA}, "
                                     f"not {version}: an older venngram would refuse the record by name rather than "
                                     "pass over it.")
                sections[kind] = _SECTIONS[kind](body, source=where)
                continue
            if kind in extensions:
                why = (f"it is an extension (schema {extensions[kind].get('schema', 'not given')}) with no parser "
                       "registered in this process")
            else:
                why = "this venngram does not know the section type and the record declares no extension for it"
            warnings.warn(f"{where} is skipped: {why}.", UnknownSupportSectionWarning, stacklevel=2)
            skipped[kind] = body
        return cls(name=name, sections=sections, skipped=skipped, extensions=extensions,
                   description=doc.get("description"))

    @staticmethod
    def _read_extensions(raw: Any, source: str) -> dict[str, Mapping[str, Any]]:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{source}: `extensions` maps an extension section's name to its `schema`.")
        out = {}
        for name, decl in raw.items():
            if not isinstance(name, str) or name in CORE_SECTIONS or not _EXTENSION_NAME.fullmatch(name):
                raise ValueError(f"{source}: extension `{name}` is not namespaced (`<owner>.<name>`, lowercase, not a"
                                 " core section).")
            if not isinstance(decl, Mapping) or not isinstance(decl.get("schema"), str):
                raise ValueError(f"{source}: extension `{name}` gives the `schema` that defines it (a document "
                                 "reference or a URL).")
            out[name] = dict(decl)
        return out

    @classmethod
    def from_file(cls, path: str | Path) -> "SupportRecord":
        """Read a ``support_record`` document from YAML (loading runs no code)."""
        with open(path, encoding="utf-8") as fh:
            return cls.from_mapping(yaml.safe_load(fh), source=str(path))

    def to_mapping(self) -> dict[str, Any]:
        """The record as a document of the first schema its declarations need: 9 (:data:`TREATMENTS_SCHEMA`) when it
        has a ``treatments`` section, a ``circuits`` section declaring ``levels`` or a ``tuning`` section declaring
        ``units: grad-mask``, since those names are schema 9's; otherwise 2, or 5 when its tuning section declares
        ``gradients``, 6 when its circuits section declares ``subsets``, 7 when its tuning section declares ``pins``
        (skipped sections are written back as they were read)."""
        sections: dict[str, Any] = {}
        for kind, contract in self.sections.items():
            if isinstance(contract, Support):
                sections[kind] = _treatments_mapping(contract)
            elif hasattr(contract, "to_mapping"):
                sections[kind] = contract.to_mapping()
            else:
                raise ValueError(f"section `{kind}` ({type(contract).__name__}) has no `to_mapping`.")
        sections.update(self.skipped)
        contracts = list(self.sections.values())
        needs_names = any(isinstance(c, Support) for c in contracts) or any(
            isinstance(c, CircuitSupport) and c.levels is not None for c in contracts) or any(
            isinstance(c, TuningSupport) and c.units == "grad-mask" for c in contracts)
        needs_gradients = any(isinstance(c, TuningSupport) and c.gradients for c in contracts)
        needs_subsets = any(isinstance(c, CircuitSupport) and c.subsets is not None for c in contracts)
        needs_pins = any(isinstance(c, TuningSupport) and c.pins for c in contracts)
        version = TREATMENTS_SCHEMA if needs_names else PINS_SCHEMA if needs_pins else \
            SUBSETS_SCHEMA if needs_subsets else GRADIENTS_SCHEMA if needs_gradients else SCHEMA_VERSION
        out: dict[str, Any] = {"schema_version": version, "kind": "support_record"}
        if self.name is not None:
            out["name"] = self.name
        if self.description is not None:
            out["description"] = self.description
        out["sections"] = sections
        if self.extensions:
            out["extensions"] = {k: dict(v) for k, v in self.extensions.items()}
        return out

    def _refuse_missing(self, section: str, what: str) -> list[str]:
        who = f"`{self.name}`" if self.name else "this consumer"
        return [f"{what}: {who} declares no `{section}` section, so it offers no {section} contract"]

    def check_book(self, book: AddressBook, *, sets: Collection[str] | None = None) -> list[str]:
        """Every way ``book`` asks for something the ``tuning`` section does not honour (all of it, without one);
        ``sets`` limits the check to those sets and what they reference (see :meth:`TuningSupport.check_book`)."""
        if self.tuning is None:
            return self._refuse_missing("tuning", book.source)
        return self.tuning.check_book(book, sets=sets)

    def check_treatments(self, book: AddressBook, *, names: Collection[str] | None = None) -> list[str]:
        """Every way the book's treatments (those in ``names``, else all) ask for something the ``treatments`` section
        does not honour, each refusal prefixed with its treatment's name."""
        chosen = [n for n in book.treatments if names is None or n in names]
        return [f"treatment `{n}`: {r}" for n in chosen for r in self.check_treatment(book.treatments[n])]

    def check_treatment(self, treatment: Any) -> list[str]:
        """Every way ``treatment`` asks for something the ``treatments`` section does not honour (see
        :func:`venngram.ops.check_support`), and its graph ops anything ``circuits.subsets`` does not."""
        what = f"treatment `{getattr(treatment, 'name', '?')}`"
        stages = getattr(treatment, "stages", ())
        graph_ops = [o for s in stages for o in s.ops if getattr(o, "graph", None) is not None]
        other = len(graph_ops) < sum(len(s.ops) for s in stages) or getattr(treatment, "weights", ()) \
            or getattr(treatment, "gradients", ()) or getattr(treatment, "readout", None) is not None
        out: list[str] = []
        if other or not graph_ops:
            out += self._refuse_missing("treatments", what) if self.treatments is None else \
                check_support(treatment, self.treatments)
        if graph_ops:
            out += self._refuse_missing("circuits", what) if self.circuits is None else \
                self.circuits.check_treatment(treatment)
        return out


def _param_targets(body: Any) -> list[str]:
    targets = body.get("targets") if isinstance(body, Mapping) else None
    return sorted(set(targets) & set(PARAM_TARGETS)) if isinstance(targets, list) else []


def _renamed(kind: Any, body: Any, version: int, where: str) -> tuple[Any, Any]:
    """A section's schema 9 name and body: a schema 1 to 8 record's old names are read through
    :data:`SECTION_ALIASES` (and ``circuits.evidence`` as ``levels``) with a deprecation warning, and refused in a
    schema 9 one; a schema 9 name in an older record is refused, since an older reader would skip it."""
    if kind in SECTION_ALIASES:
        new = SECTION_ALIASES[kind]
        if version >= TREATMENTS_SCHEMA:
            raise ValueError(f"{where}: the section is named `{new}` from support record schema {TREATMENTS_SCHEMA}.")
        deprecated("support record section", kind, since="0.1", removed_in="0.2", replacement=f"`{new}`", stacklevel=4)
        kind = new
    elif kind in SECTION_ALIASES.values() and version < TREATMENTS_SCHEMA:
        old = next(k for k, v in SECTION_ALIASES.items() if v == kind)
        raise ValueError(f"{where}: `{kind}` needs support record schema {TREATMENTS_SCHEMA}, not {version} (a schema "
                         f"{version} record names it `{old}`).")
    if kind == "circuits" and isinstance(body, Mapping):
        if version < TREATMENTS_SCHEMA and "levels" in body:
            raise ValueError(f"{where}: `levels` needs support record schema {TREATMENTS_SCHEMA}, not {version} (a "
                             f"schema {version} record names it `evidence`).")
        if version < TREATMENTS_SCHEMA and "evidence" in body:
            deprecated("support record key", "circuits.evidence", since="0.1", removed_in="0.2",
                       replacement="`circuits.levels`", stacklevel=4)
            body = {("levels" if k == "evidence" else k): v for k, v in body.items()}
    return kind, body


def _treatments_mapping(s: Support) -> dict[str, Any]:
    out: dict[str, Any] = {"ops": sorted(s.ops)}
    for key in ("targets", "tokens", "values", "transforms", "bases"):
        value = getattr(s, key)
        if value is not None:
            out[key] = sorted(value)
    if s.sites is not None:
        if callable(s.sites):
            raise ValueError("a treatments section whose `sites` is a predicate cannot be written as a document.")
        out["sites"] = sorted(s.sites)
    if s.of != Support.of:
        out["of"] = sorted(s.of)
    if s.gradient_ops:
        out["gradient_ops"] = sorted(s.gradient_ops)
    if s.readouts:
        out["readouts"] = sorted(s.readouts)
    if s.routes:
        out["routes"] = sorted(s.routes)
    return out


def as_record(support: "SupportRecord | Support") -> SupportRecord:
    """A record for ``support``: a :class:`venngram.ops.Support` becomes a record with only a ``treatments`` section."""
    if isinstance(support, SupportRecord):
        return support
    return SupportRecord(name=support.name, sections={"treatments": support})


__all__ = [
    "as_record",
    "CIRCUIT_LEVELS",
    "CIRCUIT_NODES",
    "CircuitSupport",
    "CORE_SECTIONS",
    "FormatSupport",
    "PINS_SCHEMA",
    "ROUTES_SCHEMA",
    "register_support_section",
    "SCHEMA_VERSION",
    "SCHEMA_WINDOW",
    "SECTION_ALIASES",
    "SUBSET_VALUES",
    "SUBSETS_SCHEMA",
    "SubsetSupport",
    "support_sections",
    "SupportRecord",
    "TREATMENTS_SCHEMA",
    "TuningSupport",
    "UNIT_PATTERNS",
    "UnknownSupportSectionWarning",
]
