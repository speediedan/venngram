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

    schema_version: 2
    kind: support_record
    name: finetuning-scheduler
    sections:
      tuning:                          # address books over parameters
        fields: [role, name, module, layers, stack, component, leaf]
        targets: [param]
        units: none                    # none | unit-adapter
        naming_schemes: [physical]
      steering: {ops: [add, scale]}    # recipes (schema 1's fields)
    extensions:                        # sections defined outside venngram
      acme.probes: {schema: "hub-model:acme/probe-support:schema.yaml", description: probe heads}

**Core sections** (:data:`CORE_SECTIONS`):

- ``steering``: recipes, as :class:`venngram.ops.Support` (ops, targets, token scopes, value sources, transforms,
  sites, bases);
- ``tuning``: address books whose sets select tensors to tune (:class:`TuningSupport`);
- ``circuits``: traced or validated circuits, attribution graphs in general (:class:`CircuitSupport`). This section is
  a prototype: its vocabulary is expected to change as circuit documents are defined.

**Opt-in.** A section a record omits is a contract its consumer does not offer: :meth:`SupportRecord.check_book` and
:meth:`SupportRecord.check_recipe` refuse every document that needs it, naming the section.

**Unknown sections are skipped, not fatal.** A record written for a newer venngram, or carrying another project's
section, still loads: a section type this reader cannot parse is reported (:class:`UnknownSupportSectionWarning`,
naming it) and kept, unparsed, in :attr:`SupportRecord.skipped`. A section defined outside venngram is namespaced
(``<owner>.<name>``) and declared under ``extensions`` with the schema that defines it; code that understands it
registers a parser (:func:`register_support_section`), and the section is then parsed like a core one.

**Schema 1** records (a ``steering`` contract written at the top level) still read, as a record with only a
``steering`` section.

.. warning:: This is an experimental API and may change in a future release.
"""
import re
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from venngram.address import FIELDS, TARGETS, terms
from venngram.book import AddressBook, PORTABILITY, STRUCTURE_LEVELS
from venngram.ops import check_support, Support

SCHEMA_VERSION = 2
SCHEMA_WINDOW = (1, 2)
CORE_SECTIONS = ("steering", "tuning", "circuits")
UNIT_PATTERNS = ("none", "unit-adapter")
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
        fields: The address fields its resolution honours (see :mod:`venngram.address`).
        targets: What its sets may select (``param`` for tensors; a tuning consumer reaches sites, features or
            subspaces only through an adapter or a latent model it owns).
        units: How it tunes a unit set: ``none`` (it refuses unit sets) or ``unit-adapter`` (a module whose parameters
            are per-unit values, so the units are tuned through that module's tensors).
        naming_schemes: The naming schemes its views provide; a book whose ``requires.naming_scheme`` is another is
            refused.
        structure: The ``requires.structure`` levels it provides.
        portability: The book portabilities it accepts.
    """

    fields: frozenset[str] = frozenset(FIELDS)
    targets: frozenset[str] = frozenset({"param"})
    units: str = "none"
    naming_schemes: frozenset[str] | None = None
    structure: frozenset[str] = frozenset(STRUCTURE_LEVELS)
    portability: frozenset[str] = frozenset(PORTABILITY)

    @classmethod
    def from_mapping(cls, doc: Any, *, source: str = "<inline>") -> "TuningSupport":
        """Read a ``tuning`` section.

        Raises:
            ValueError: If the section has unknown keys or names a field, target, unit pattern, structure level or
                portability venngram does not define.
        """
        spec: dict[str, tuple[str, ...] | None] = {"fields": FIELDS, "targets": TARGETS, "naming_schemes": None,
                                                   "structure": STRUCTURE_LEVELS, "portability": PORTABILITY}
        units = doc.get("units", "none") if isinstance(doc, Mapping) else None
        if units not in UNIT_PATTERNS:
            raise ValueError(f"{source}: `units` is one of {list(UNIT_PATTERNS)}, got {units!r}.")
        rest = {k: v for k, v in doc.items() if k != "units"} if isinstance(doc, Mapping) else doc
        return cls(**_read_fields(rest, source, spec), units=units)

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {"fields": [f for f in FIELDS if f in self.fields],
                               "targets": [t for t in TARGETS if t in self.targets], "units": self.units}
        if self.naming_schemes is not None:
            out["naming_schemes"] = sorted(self.naming_schemes)
        out["structure"] = [s for s in STRUCTURE_LEVELS if s in self.structure]
        out["portability"] = [p for p in PORTABILITY if p in self.portability]
        return out

    def check_book(self, book: AddressBook) -> list[str]:
        """Every way ``book`` (and each book it imports) asks for something this section does not honour."""
        out: list[str] = []
        self._check(book, book.source, out)
        return out

    def _check(self, book: AddressBook, where: str, out: list[str]) -> None:
        scheme = book.requires.get("naming_scheme")
        if scheme is not None and self.naming_schemes is not None and scheme not in self.naming_schemes:
            out.append(f"{where}: it is written in `{scheme}` names; this consumer's views provide "
                       f"{sorted(self.naming_schemes)}")
        structure = book.requires.get("structure", "physical")
        if structure not in self.structure:
            out.append(f"{where}: it requires `{structure}` structure, which this consumer does not provide")
        if book.portability not in self.portability:
            out.append(f"{where}: `{book.portability}` books are not accepted")
        for set_name, expr in book.own_sets.items():
            at = f"{where}: set `{set_name}`"
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
        for ns, imported in book.imports.items():
            imported_out: list[str] = []
            self._check(imported, imported.source, imported_out)
            out.extend(f"import `{ns}`: {line}" for line in imported_out)


#: The node kinds of an attribution graph this prototype names.
CIRCUIT_NODES = ("feature", "error", "embedding", "logit")
#: The evidence a circuit document may carry: traced (from attribution), pruned (thresholded), validated (by
#: intervention).
CIRCUIT_EVIDENCE = ("traced", "pruned", "validated")


@dataclass(frozen=True)
class CircuitSupport:
    """What a consumer of circuits honours (prototype). ``None`` fields mean any.

    A circuit is a subgraph of a model's computation with evidence for it: an attribution graph (circuit-tracer's is one
    producer), pruned to a threshold, perhaps validated by intervening on its nodes. Its nodes are addressed with
    venngram addresses (a feature, a site, a token position), so a circuit document composes with books and recipes.

    Attributes:
        graphs: The graph kinds it reads (``attribution``; others as they are defined).
        nodes: The node kinds it honours (see :data:`CIRCUIT_NODES`).
        features: The feature sources its feature nodes may come from (``transcoder``, ``clt``, ``sae``; any name).
        evidence: The evidence it can use or produce (see :data:`CIRCUIT_EVIDENCE`).
    """

    graphs: frozenset[str] = frozenset({"attribution"})
    nodes: frozenset[str] | None = None
    features: frozenset[str] | None = None
    evidence: frozenset[str] | None = None

    @classmethod
    def from_mapping(cls, doc: Any, *, source: str = "<inline>") -> "CircuitSupport":
        """Read a ``circuits`` section.

        Raises:
            ValueError: If the section has unknown keys or names a node kind or evidence venngram does not define.
        """
        return cls(**_read_fields(doc, source, {"graphs": None, "nodes": CIRCUIT_NODES, "features": None,
                                                "evidence": CIRCUIT_EVIDENCE}))

    def to_mapping(self) -> dict[str, Any]:
        return {k: sorted(v) for k, v in (("graphs", self.graphs), ("nodes", self.nodes), ("features", self.features),
                                          ("evidence", self.evidence)) if v is not None}


def _steering(doc: Any, *, source: str) -> Support:
    if not isinstance(doc, Mapping):
        raise ValueError(f"{source}: a section is a mapping, got {type(doc).__name__}.")
    return Support.from_mapping({"schema_version": 1, "kind": "support_record", **doc}, source=source)


#: Parsers by section type: the section's mapping -> its parsed contract.
_SECTIONS: dict[str, Callable[..., Any]] = {"steering": _steering, "tuning": TuningSupport.from_mapping,
                                            "circuits": CircuitSupport.from_mapping}


def register_support_section(name: str, parser: Callable[..., Any], *, replace: bool = False) -> None:
    """Parse sections of type ``name`` with ``parser(doc, source=...)``. An extension's name is namespaced
    (``<owner>.<name>``); the core names are venngram's.

    Raises:
        ValueError: If ``name`` is a core section or is not namespaced, or is registered and ``replace`` is false.
    """
    if name in CORE_SECTIONS or not _EXTENSION_NAME.fullmatch(name):
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
    def steering(self) -> Support | None:
        return self.sections.get("steering")

    @property
    def tuning(self) -> TuningSupport | None:
        return self.sections.get("tuning")

    @property
    def circuits(self) -> CircuitSupport | None:
        return self.sections.get("circuits")

    @classmethod
    def from_mapping(cls, doc: Any, *, source: str = "<inline>") -> "SupportRecord":
        """Read a ``support_record`` document (schema 1 or 2).

        Raises:
            ValueError: If the document is not a ``support_record`` in the schema window, has unknown top-level keys,
                declares a malformed extension, or a section this reader parses is malformed.
        """
        if not isinstance(doc, Mapping) or doc.get("kind") != "support_record":
            raise ValueError(f"{source}: a support record is `kind: support_record`.")
        version = doc.get("schema_version")
        if not isinstance(version, int) or not SCHEMA_WINDOW[0] <= version <= SCHEMA_WINDOW[1]:
            raise ValueError(f"{source}: this venngram reads support record schemas {SCHEMA_WINDOW[0]} to "
                             f"{SCHEMA_WINDOW[1]}, got {version!r}.")
        name = doc.get("name")
        name = str(name) if name is not None else None
        if version == 1:
            return cls(name=name, sections={"steering": Support.from_mapping(doc, source=source)})
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
            if kind in _SECTIONS:
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
        """The record as a schema 2 document (skipped sections are written back as they were read)."""
        sections: dict[str, Any] = {}
        for kind, contract in self.sections.items():
            if isinstance(contract, Support):
                sections[kind] = _steering_mapping(contract)
            elif hasattr(contract, "to_mapping"):
                sections[kind] = contract.to_mapping()
            else:
                raise ValueError(f"section `{kind}` ({type(contract).__name__}) has no `to_mapping`.")
        sections.update(self.skipped)
        out: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "kind": "support_record"}
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

    def check_book(self, book: AddressBook) -> list[str]:
        """Every way ``book`` asks for something the ``tuning`` section does not honour (all of it, without one)."""
        if self.tuning is None:
            return self._refuse_missing("tuning", book.source)
        return self.tuning.check_book(book)

    def check_recipe(self, recipe: Any) -> list[str]:
        """Every way ``recipe`` asks for something the ``steering`` section does not honour (see
        :func:`venngram.ops.check_support`)."""
        if self.steering is None:
            return self._refuse_missing("steering", f"recipe `{getattr(recipe, 'name', '?')}`")
        return check_support(recipe, self.steering)


def _steering_mapping(s: Support) -> dict[str, Any]:
    out: dict[str, Any] = {"ops": sorted(s.ops)}
    for key in ("targets", "tokens", "values", "transforms", "bases"):
        value = getattr(s, key)
        if value is not None:
            out[key] = sorted(value)
    if s.sites is not None:
        if callable(s.sites):
            raise ValueError("a steering section whose `sites` is a predicate cannot be written as a document.")
        out["sites"] = sorted(s.sites)
    return out


def as_record(support: "SupportRecord | Support") -> SupportRecord:
    """A record for ``support``: a :class:`venngram.ops.Support` becomes a record with only a ``steering`` section."""
    if isinstance(support, SupportRecord):
        return support
    return SupportRecord(name=support.name, sections={"steering": support})


__all__ = [
    "as_record",
    "CIRCUIT_EVIDENCE",
    "CIRCUIT_NODES",
    "CircuitSupport",
    "CORE_SECTIONS",
    "register_support_section",
    "SCHEMA_VERSION",
    "SCHEMA_WINDOW",
    "support_sections",
    "SupportRecord",
    "TuningSupport",
    "UNIT_PATTERNS",
    "UnknownSupportSectionWarning",
]
