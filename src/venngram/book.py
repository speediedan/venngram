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
Address Books
^^^^^^^^^^^^^

An address book is a data document of named sets: shareable (it holds no code, so loading one needs no trust gate),
composable (books import books, by namespace) and versioned.

.. code-block:: yaml

    schema_version: 1
    name: qwen3-tuning-sets
    portability: family          # family: roles, layers, modules, components; checkpoint: units of one model revision
    applies_to: {architectures: [Qwen3ForCausalLM]}
    requires: {structure: physical, naming_scheme: physical}
    imports: {base: ./common.yaml}
    sets:
      top:  {role: adapter, layers: "-4:"}
      rest: {role: adapter, exclude: "@top"}

**Treatments** are optional: what is done to the book's sets for one model, by name, each the body of a recipe
(:mod:`venngram.recipe`: ``model``, ``operands``, ``stages``, ``sampling``, ``evidence``) whose targets name this book's
sets, so a selection and what is done to it travel together and a consumer reads one kind of document:

.. code-block:: yaml

    treatments:
      steer-top:
        model: {id: Qwen/Qwen3-0.6B, revision: c1899de}
        stages: [{site: mlp.out.hook_in, ops: [{op: add, target: {set: top}, value: {dose: 0.8}}]}]

A consumer that only selects (a tuning schedule) ignores them.

**Graphs** are optional too: graph subsets (:mod:`venngram.graphs`) the book's treatments apply, by name, each a
``graph_subset`` document beside the book, cited by hash (book schema 5). A book whose treatments apply only graphs may
have an empty ``sets``:

.. code-block:: yaml

    graphs:
      reasoning: {file: reasoning.yaml, sha256: 4be1...}

A standalone ``kind: recipe`` document, written before books and recipes merged, is retired:
``python -m venngram.migrate`` rewrites one as its book, carrying the recipe as a treatment (:mod:`venngram.migrate`).

**Versions.** ``schema_version`` is required and read within a window (:data:`SCHEMA_WINDOW`). A key an older reader
may ignore without misreading the book (``description``, ``charts``, any unknown top-level key) never changes the
version; a key that changes what a set selects does. Schema 2 adds the ``of`` address field (a set's gradients), so a
book whose sets use it is schema 2, and every other book stays readable as schema 1. Schema 3 lets a treatment carry a
``weights`` section (ops on parameters), and schema 4 a ``gradients`` section (ops on parameter gradients) or a
``readout`` (what a stage's gradients are taken of), each of which an older reader would pass over silently. Schema 5
adds ``graphs``, and treatments whose ops target a graph; schema 6 the ``axis`` address field, schema 7
layer sets relative to the depth (``0.25d:0.75d``), schema 8 the model pins in ``requires``, schema 9 element
masks (``units: {elements: ...}``), schema 10 paths (``path: [...]``, :mod:`venngram.paths`), and schema 11 whole
attention heads (``units: {heads: ...}``).

**Requirements are declared.** ``requires.structure`` is the structure a consumer must provide (``physical`` block
indices, or a ``vocabulary`` of components). ``requires.naming_scheme`` names the vocabulary a book's ``name``,
``module`` and ``leaf`` values are written in (a tunable view's naming scheme: ``physical`` for a model's own parameter
names, ``tl`` for TransformerLens names); a consumer calls :meth:`AddressBook.check_requires` with its view's scheme,
and a book written in another vocabulary is refused by name rather than matching the wrong tensors. A book that uses
only vocabulary-neutral fields (``role``, ``layers``, ``component``) need not declare one.

**What a book was written against is pinned.** A book that names latent models (``module:latent_models.sae``,
``role: latent:sae``) selects the same tensors wherever it loads, but it means the same thing only on the base model and
the dictionaries it was written for. Its ``requires`` records them (schema 8):

.. code-block:: yaml

    requires:
      base_model: {id: google/gemma-2-2b, revision: c5ebcd4}
      latent_models:
        sae:
          release: {sae_lens: gemma-scope-2b-pt-res, id: layer_12/width_16k/average_l0_82}
          frames: {weekday: "3"}           # the version of each frame the book's units are read in
        bsf_l17:
          release: {hub: org/bsf-l17, revision: 9ab1f0e}

A ``release`` is compared key by key, so any dictionary source can be pinned; a ``hub`` release gives its
``revision``, since a repository alone follows its latest snapshot. ``frames`` is given only where a frame is declared.
A consumer calls :meth:`AddressBook.check_pins` with what it loaded, and the book is refused with every entry that
differs named (:class:`venngram.errors.PinMismatchError`). Pins bind through imports, and a book whose imports pin the
same model differently is refused when it loads.

**Portability is declared.** A ``checkpoint`` book carries unit indices meaningful in one checkpoint only, so it must
pin ``applies_to.models`` with an ``id`` and a ``revision`` (and may give the ``basis`` its units were found in); it is
refused against any other model. An element mask keeps individual weights of one checkpoint's tensors, so a book whose
sets use one is a ``checkpoint`` book.

**References.** ``@<set>`` resolves within the book, ``@<import>.<set>`` within an import. Every reference is checked
when the book loads, and cycles are refused, naming the cycle.

**Imports** are paths relative to the importing book, document references (``<store>:<ref>[@<revision>]:<path>``,
:mod:`venngram.store`), fetched with the files they need, or URIs whose scheme has a registered resolver
(:func:`register_uri_resolver`).

.. warning:: This is an experimental API and may change in a future release.
"""
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from venngram.address import _contains_path, And, Expr, Guarded, Minus, Or, parse, references, rename_refs, terms
from venngram.address import Path as PathExpr
from venngram.errors import AddressError, NewerAddressSchemaError, PinMismatchError, VenngramError
from venngram.units import sidecar_root

SCHEMA_VERSION = 1
SCHEMA_WINDOW = (1, 11)
#: The first schema whose sets may use the ``of`` field (a key that changes what a consumer acts on).
OF_SCHEMA = 2
#: The first schema whose treatments may carry a ``weights`` section (ops on parameters).
WEIGHTS_SCHEMA = 3
#: The first schema whose treatments may carry a ``gradients`` section (ops on parameter gradients) or a ``readout``
#: (what a stage's gradients are taken of).
GRADIENTS_SCHEMA = 4
#: The first schema with a ``graphs`` section (graph subsets its treatments apply).
GRAPHS_SCHEMA = 5
#: The first schema whose sets may use the ``axis`` field (which dimension of a gradient a unit set indexes).
AXIS_SCHEMA = 6
#: The first schema whose layer sets may give a position as a fraction of the depth (``layers: 0.25d:0.75d``).
DEPTH_SCHEMA = 7
#: The first schema whose ``requires`` may pin the models a book was written against (``base_model``,
#: ``latent_models``): an older venngram would pass over them silently.
PINS_SCHEMA = 8
#: The first schema whose sets may use an element mask (``units: {elements: {file, key, sha256}}``).
ELEMENTS_SCHEMA = 9
#: The first schema whose sets may be paths (``path: [...]``, :mod:`venngram.paths`).
PATHS_SCHEMA = 10
#: The first schema whose sets may select whole heads (``units: {heads: ...}``).
HEADS_SCHEMA = 11
PORTABILITY = ("family", "checkpoint")
STRUCTURE_LEVELS = ("physical", "vocabulary")
_SET_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
_URI = re.compile(r"^[a-z][a-z0-9+.-]*://")

_URI_RESOLVERS: dict[str, Callable[[str], Mapping[str, Any]]] = {}


def register_uri_resolver(scheme: str, resolver: Callable[[str], Mapping[str, Any]], *, replace: bool = False) -> None:
    """Register ``resolver`` to load the book a ``<scheme>://...`` import names (it returns the book's mapping).

    Raises:
        AddressError: If ``scheme`` already has a resolver and ``replace`` is not set.
    """
    if scheme in _URI_RESOLVERS and not replace:
        raise AddressError(f"The URI scheme `{scheme}` already has a resolver; pass replace=True to replace it.")
    _URI_RESOLVERS[scheme] = resolver


def _load_yaml(path: Path) -> Mapping[str, Any]:
    import yaml

    with open(path) as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, Mapping):
        raise AddressError(f"{path}: an address book is a mapping, got {type(doc).__name__}.")
    return doc


def _version(doc: Mapping[str, Any], source: str) -> int:
    version = doc.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise AddressError(f"{source}: `schema_version` is required and is an integer (currently {SCHEMA_VERSION}).")
    lo, hi = SCHEMA_WINDOW
    if version > hi:
        raise NewerAddressSchemaError(f"{source}: schema_version {version} was written by a newer venngram (this one "
                                      f"reads {lo} to {hi}); upgrade venngram.", kind="address_book", version=version,
                                      newest=hi)
    if version < lo:
        raise AddressError(f"{source}: schema_version {version} is older than this venngram reads ({lo} to {hi}); "
                           "re-publish the book against a current schema.")
    return version


def _check_portability(portability: Any, applies_to: Mapping[str, Any], source: str) -> None:
    if portability not in PORTABILITY:
        raise AddressError(f"{source}: `portability` is one of {', '.join(PORTABILITY)}, got {portability!r}.")
    if portability != "checkpoint":
        return
    models = applies_to.get("models")
    pinned = isinstance(models, list) and models and all(
        isinstance(m, Mapping) and m.get("id") and m.get("revision") for m in models)
    if not pinned:
        raise AddressError(f"{source}: a checkpoint book's units mean something in one checkpoint only, so it pins "
                           "`applies_to.models` as a list of {id, revision}.")


@dataclass(frozen=True)
class AddressBook:
    """A loaded address book (see the module docs for the format).

    Use :meth:`from_mapping` or :meth:`from_file` rather than the constructor.
    """

    name: str
    schema_version: int
    portability: str
    applies_to: Mapping[str, Any]
    requires: Mapping[str, Any]
    own_sets: Mapping[str, Expr]
    imports: Mapping[str, "AddressBook"]
    source: str = "<inline>"
    description: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    treatments: Mapping[str, Any] = field(default_factory=dict)
    """Treatments by name (each a :class:`venngram.recipe.Recipe` over this book), from the ``treatments`` section."""
    graphs: Mapping[str, Any] = field(default_factory=dict)
    """Graph subsets by name (each a :class:`venngram.graphs.GraphSubset`), from the ``graphs`` section."""

    @classmethod
    def from_mapping(cls, doc: Mapping[str, Any], *, source: str = "<inline>", base_dir: str | Path | None = None,
                     inline: bool = False, _loading: tuple[str, ...] = ()) -> "AddressBook":
        """Load a book from its mapping.

        Args:
            doc: The book.
            source: Where it came from, for messages.
            base_dir: The directory relative imports resolve against (the current directory when ``None``).
            inline: A book written inside another document (a schedule's ``addresses`` key): ``schema_version`` and
                ``name`` default to the current version (the one its sets need) and ``source``.

        Raises:
            AddressError: If the book is malformed, an import cannot load or forms a cycle, a reference names no set,
                or the sets' references form a cycle.
        """
        if not isinstance(doc, Mapping):
            raise AddressError(f"{source}: an address book is a mapping, got {type(doc).__name__}.")
        version = doc.get("schema_version", SCHEMA_VERSION if inline else None)
        _version({**doc, "schema_version": version}, source)
        applies_to = dict(doc.get("applies_to") or {})
        portability = doc.get("portability", "family")
        _check_portability(portability, applies_to, source)
        requires = dict(doc.get("requires") or {})
        structure = requires.get("structure", "physical")
        if structure not in STRUCTURE_LEVELS:
            raise AddressError(f"{source}: `requires.structure` is one of {', '.join(STRUCTURE_LEVELS)}, got "
                               f"{structure!r}.")
        scheme = requires.get("naming_scheme")
        if scheme is not None and (not isinstance(scheme, str) or not scheme):
            raise AddressError(f"{source}: `requires.naming_scheme` is a view's naming scheme, a non-empty string"
                               f" such as `physical` or `tl`, got {scheme!r}.")
        requires.update(_pins(requires, source))
        pinned = sorted(k for k in PIN_KEYS if k in requires)
        if pinned and inline and "schema_version" not in doc:
            version = max(version, PINS_SCHEMA)
        if pinned and version < PINS_SCHEMA:
            raise AddressError(f"{source}: `requires` pins {pinned}, which needs address book schema {PINS_SCHEMA}, "
                               f"not {version}: an older venngram would pass over them silently.")
        if portability == "checkpoint" and "base_model" in requires:
            base = requires["base_model"]
            if not any(m["id"] == base["id"] and str(m["revision"]) == base["revision"]
                       for m in applies_to["models"]):
                raise AddressError(f"{source}: `requires.base_model` is {base['id']}@{base['revision']}, which is not "
                                   "among the checkpoint's pinned `applies_to.models`.")
        graphs = _graphs(doc.get("graphs"), source, base_dir) if doc.get("graphs") is not None else {}
        if graphs and inline and "schema_version" not in doc:
            version = GRAPHS_SCHEMA
        if graphs and version < GRAPHS_SCHEMA:
            raise AddressError(f"{source}: `graphs` needs address book schema {GRAPHS_SCHEMA}, not {version}: an older "
                               "venngram would pass over it silently.")
        raw_sets = doc.get("sets")
        if not isinstance(raw_sets, Mapping) or not (raw_sets or graphs):
            raise AddressError(f"{source}: `sets` is a non-empty mapping of set names to addresses (it may be empty "
                               "when the book has `graphs`).")
        own: dict[str, Expr] = {}
        with sidecar_root(base_dir):  # a sidecar unit set's path is relative to the book
            for set_name, address in raw_sets.items():
                if not isinstance(set_name, str) or not _SET_NAME.fullmatch(set_name):
                    raise AddressError(f"{source}: {set_name!r} is not a valid set name (letters, digits, `_` and "
                                       "`-`).")
                try:
                    own[set_name] = parse(address)
                except VenngramError as err:
                    raise AddressError(f"{source}: set `{set_name}`: {err}") from err
        uses_of = sorted(n for n, expr in own.items() if any(t.of is not None for t in terms(expr)))
        if uses_of and inline and "schema_version" not in doc:
            version = max(version, OF_SCHEMA)  # an inline book takes the version its sets need
        uses_axis = sorted(n for n, expr in own.items() if any(t.axis is not None for t in terms(expr)))
        if uses_axis and inline and "schema_version" not in doc:
            version = max(version, AXIS_SCHEMA)
        uses_depth = sorted(n for n, expr in own.items()
                            if any(t.layers is not None and t.layers.depth_relative for t in terms(expr)))
        if uses_depth and inline and "schema_version" not in doc:
            version = max(version, DEPTH_SCHEMA)
        uses_elements = sorted(n for n, expr in own.items()
                               if any(t.units is not None and t.units.is_elements for t in terms(expr)))
        if uses_elements and inline and "schema_version" not in doc:
            version = max(version, ELEMENTS_SCHEMA)
        if uses_elements and version < ELEMENTS_SCHEMA:
            raise AddressError(f"{source}: the sets {uses_elements} use an element mask, which needs address book "
                               f"schema {ELEMENTS_SCHEMA}, not {version}: an older venngram would refuse it by name "
                               "rather than pass over it.")
        if uses_elements and portability != "checkpoint":
            raise AddressError(f"{source}: the sets {uses_elements} use an element mask, which keeps individual "
                               "weights of one checkpoint's tensors, so the book is `portability: checkpoint` and pins "
                               "its model in `applies_to.models`.")
        uses_paths = sorted(n for n, expr in own.items() if _contains_path(expr))
        if uses_paths and inline and "schema_version" not in doc:
            version = max(version, PATHS_SCHEMA)
        if uses_paths and version < PATHS_SCHEMA:
            raise AddressError(f"{source}: the sets {uses_paths} are paths, which need address book schema "
                               f"{PATHS_SCHEMA}, not {version}: an older venngram would refuse them by name rather "
                               "than pass over them.")
        uses_heads = sorted(n for n, expr in own.items()
                            if any(t.units is not None and t.units.heads is not None for t in terms(expr)))
        if uses_heads and inline and "schema_version" not in doc:
            version = max(version, HEADS_SCHEMA)
        if uses_heads and version < HEADS_SCHEMA:
            raise AddressError(f"{source}: the sets {uses_heads} select whole heads, which needs address book "
                               f"schema {HEADS_SCHEMA}.")
        _check_graph_waypoints(own, graphs, portability, source)
        if uses_depth and version < DEPTH_SCHEMA:
            raise AddressError(f"{source}: the sets {uses_depth} give layers as a fraction of the depth, which needs "
                               f"address book schema {DEPTH_SCHEMA}, not {version}: an older venngram would refuse it "
                               "by name rather than pass over it.")
        if uses_axis and version < AXIS_SCHEMA:
            raise AddressError(f"{source}: the sets {uses_axis} use the `axis` field, which needs address book schema "
                               f"{AXIS_SCHEMA}, not {version}: an older venngram would refuse it by name rather than "
                               "pass over it.")
        raw_treatments = doc.get("treatments")
        with_weights = sorted(n for n, body in raw_treatments.items()
                              if isinstance(body, Mapping) and "weights" in body) \
            if isinstance(raw_treatments, Mapping) else []
        with_gradients = sorted(n for n, body in raw_treatments.items()
                                if isinstance(body, Mapping) and ("gradients" in body or _states_readout(body))) \
            if isinstance(raw_treatments, Mapping) else []
        if with_weights and inline and "schema_version" not in doc:
            version = max(version, WEIGHTS_SCHEMA)
        if with_gradients and inline and "schema_version" not in doc:
            version = max(version, GRADIENTS_SCHEMA)
        for names, key, needed in ((with_weights, "weights", WEIGHTS_SCHEMA),
                                   (with_gradients, "gradients` or a `readout", GRADIENTS_SCHEMA)):
            if names and version < needed:
                raise AddressError(f"{source}: the treatments {names} have `{key}`, which need address book schema "
                                   f"{needed}, not {version}: an older venngram would pass over them silently.")
        if uses_of and version < OF_SCHEMA:
            raise AddressError(f"{source}: the sets {uses_of} use the `of` field, which needs address book schema "
                               f"{OF_SCHEMA}, not {version}: an older venngram would refuse it by name rather than "
                               "pass over it.")
        imports = cls._load_imports(doc.get("imports") or {}, source, base_dir, _loading)
        known = {k for k in own} | {f"{ns}.{s}" for ns, book in imports.items() for s in book.sets()}
        extra = {k: v for k, v in doc.items() if k not in _KNOWN_KEYS}
        book = cls(name=str(doc.get("name", source)), schema_version=version, portability=portability,
                   applies_to=applies_to, requires=requires, own_sets=own, imports=imports, source=source,
                   description=doc.get("description"), extra=extra, graphs=graphs)
        for set_name, expr in own.items():
            missing = [r for r in references(expr) if r not in known]
            if missing:
                named = ", ".join("@" + m for m in missing)
                raise AddressError(f"{source}: set `{set_name}` references {named}, which "
                                   f"no set or import defines (known: {', '.join(sorted('@' + k for k in known))}).")
        book._check_cycles()
        book.pins()  # refuses imports that pin one model differently
        if doc.get("treatments") is not None:
            object.__setattr__(book, "treatments", _treatments(doc["treatments"], book, source, base_dir))
        return book

    @classmethod
    def from_file(cls, path: str | Path) -> "AddressBook":
        """Load a book from a YAML file; relative imports resolve against its directory."""
        path = Path(path)
        return cls.from_mapping(_load_yaml(path), source=str(path), base_dir=path.parent)

    @classmethod
    def _load_imports(cls, raw: Any, source: str, base_dir: str | Path | None,
                      loading: tuple[str, ...]) -> dict[str, "AddressBook"]:
        if not isinstance(raw, Mapping):
            raise AddressError(f"{source}: `imports` maps a namespace to a book path or URI.")
        out = {}
        for ns, target in raw.items():
            if not isinstance(ns, str) or not _SET_NAME.fullmatch(ns):
                raise AddressError(f"{source}: {ns!r} is not a valid import namespace.")
            if not isinstance(target, str):
                raise AddressError(f"{source}: import `{ns}` names a path or URI, got {target!r}.")
            from venngram.store import fetch, is_reference  # the store loads books, so it is imported here

            remote = _URI.match(target) or is_reference(target)
            key = target if remote else str((Path(base_dir or ".") / target).resolve())
            if key in loading or key == source:
                raise AddressError(f"{source}: imports form a cycle through {target}.")
            if is_reference(target):
                try:
                    path, _ = fetch(target)
                except VenngramError as e:
                    raise AddressError(f"{source}: import `{ns}` ({target}) did not fetch: {e}") from e
                out[ns] = cls.from_mapping(_load_yaml(path), source=target, base_dir=path.parent,
                                           _loading=loading + (key,))
            elif _URI.match(target):
                scheme = target.split("://", 1)[0]
                if scheme not in _URI_RESOLVERS:
                    raise AddressError(f"{source}: import `{ns}` uses `{scheme}://`, which has no registered resolver "
                                       "(register one with `venngram.book.register_uri_resolver`).")
                out[ns] = cls.from_mapping(_URI_RESOLVERS[scheme](target), source=target, _loading=loading + (key,))
            else:
                path = Path(key)
                if not path.is_file():
                    raise AddressError(f"{source}: import `{ns}` names {target}, which is not a file.")
                out[ns] = cls.from_mapping(_load_yaml(path), source=str(path), base_dir=path.parent,
                                           _loading=loading + (key,))
        return out

    def sets(self) -> dict[str, Expr]:
        """Every set this book can reference: its own by name, and each import's as ``<namespace>.<set>``."""
        out = dict(self.own_sets)
        for ns, book in self.imports.items():
            # an imported set's references are relative to its own book, so they take its namespace too
            out.update({f"{ns}.{name}": rename_refs(expr, lambda r, ns=ns: f"{ns}.{r}")
                        for name, expr in book.sets().items()})
        return out

    def _check_cycles(self) -> None:
        sets = self.sets()

        def visit(name: str, path: tuple[str, ...]) -> None:
            if name in path:
                cycle = path[path.index(name):] + (name,)
                chain = " -> ".join("@" + c for c in cycle)
                raise AddressError(f"{self.source}: set references form a cycle: {chain}.")
            for ref in references(sets[name]):
                if ref in sets:
                    visit(ref, path + (name,))

        for name in self.own_sets:
            visit(name, ())

    def check_requires(self, *, naming_scheme: str | None = None) -> None:
        """Refuse a consumer that cannot provide what this book, or any book it imports, requires.

        Args:
            naming_scheme: The naming scheme of the view the book's sets will resolve through (``None`` skips the
                check).

        Raises:
            AddressError: If a book requires a different naming scheme, naming the book and both schemes.
        """
        want = self.requires.get("naming_scheme")
        if naming_scheme is not None and want is not None and want != naming_scheme:
            raise AddressError(f"{self.source}: its sets are written in `{want}` names, so it requires a view whose"
                               f" naming scheme is `{want}`, not `{naming_scheme}`.")
        for book in self.imports.values():
            book.check_requires(naming_scheme=naming_scheme)

    def pins(self) -> dict[str, Any]:
        """The model pins this book and every book it imports declare, merged (``base_model``, ``latent_models``).

        Raises:
            AddressError: If two of them pin the same model differently, so that no model satisfies both.
        """
        out: dict[str, Any] = {}
        by: dict[str, str] = {}
        for who, pins in self._pin_sources("the book"):
            if "base_model" in pins:
                if "base_model" in out and out["base_model"] != pins["base_model"]:
                    raise AddressError(f"{self.source}: {by['base_model']} pins the base model "
                                       f"{_model(out['base_model'])} and {who} pins {_model(pins['base_model'])}, so "
                                       "no model satisfies both.")
                out["base_model"] = pins["base_model"]
                by.setdefault("base_model", who)
            for name, pin in pins.get("latent_models", {}).items():
                latents = out.setdefault("latent_models", {})
                if name in latents and latents[name] != pin:
                    raise AddressError(f"{self.source}: {by[name]} pins latent model `{name}` as {latents[name]} and "
                                       f"{who} as {pin}, so no model satisfies both.")
                latents[name] = pin
                by.setdefault(name, who)
        return out

    def _pin_sources(self, label: str) -> list[tuple[str, dict[str, Any]]]:
        own = {k: self.requires[k] for k in PIN_KEYS if k in self.requires}
        out = [(label, own)] if own else []
        for ns, book in self.imports.items():
            out.extend(book._pin_sources(f"import `{ns}`" if label == "the book" else f"{label}'s import `{ns}`"))
        return out

    def check_pins(self, *, base_model: Mapping[str, Any] | None = None,
                   latent_models: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        """Refuse a loaded model that differs from what this book, or any book it imports, was written against.

        Args:
            base_model: The loaded base model, ``{id, revision}`` (``None`` skips the check).
            latent_models: The loaded latent models by name, each ``{release: {...}, frames: {<frame>: <version>}}``
                in the form a book pins them (``None`` skips the check). A pinned latent model missing here, a pinned
                release key it does not report, and a pinned frame it does not declare all differ.

        Raises:
            PinMismatchError: If anything differs, naming every entry that does (its
                :attr:`~venngram.errors.PinMismatchError.entries`).
        """
        pins = self.pins()
        out: list[tuple[str, str]] = []
        want = pins.get("base_model")
        if want is not None and base_model is not None:
            got = {k: None if base_model.get(k) is None else str(base_model.get(k)) for k in ("id", "revision")}
            if got != want:
                out.append(("requires.base_model", f"written against {_model(want)}, the loaded model is "
                                                   f"{got['id']}@{got['revision']}"))
        if latent_models is not None:
            for name, pin in pins.get("latent_models", {}).items():
                entry = f"requires.latent_models.{name}"
                loaded = latent_models.get(name)
                if loaded is None:
                    out.append((entry, f"the book names latent model `{name}`, which is not loaded"))
                    continue
                release = loaded.get("release") or {}
                for key, value in pin["release"].items():
                    got = release.get(key)
                    if got is None or str(got) != value:
                        out.append((f"{entry}.release.{key}", f"pinned `{value}`, loaded "
                                                              f"{'nothing' if got is None else f'`{got}`'}"))
                frames = loaded.get("frames") or {}
                for frame, version in pin.get("frames", {}).items():
                    got = frames.get(frame)
                    if got is None:
                        out.append((f"{entry}.frames.{frame}", f"pinned version `{version}`, and the loaded latent "
                                                               "model declares no such frame"))
                    elif str(got) != version:
                        out.append((f"{entry}.frames.{frame}", f"pinned version `{version}`, loaded `{got}`"))
        if out:
            lines = "; ".join(f"`{entry}`: {why}" for entry, why in out)
            raise PinMismatchError(f"{self.source}: the loaded models differ from what the book was written against: "
                                   f"{lines}.", entries=tuple(entry for entry, _ in out))

    def check_applies(self, *, architecture: str | None = None, model_id: str | None = None,
                      revision: str | None = None, basis: str | None = None) -> None:
        """Refuse a model this book does not apply to.

        Raises:
            AddressError: If ``architecture`` is not among ``applies_to.architectures``, or, for a checkpoint book,
                ``model_id`` and ``revision`` match no pinned model, or ``basis`` differs from the pinned one.
        """
        archs = self.applies_to.get("architectures")
        if archs and architecture is not None and architecture not in archs:
            raise AddressError(f"{self.source}: the book applies to {', '.join(archs)}, not {architecture}.")
        if self.portability != "checkpoint":
            return
        for m in self.applies_to["models"]:
            if m["id"] == model_id and str(m["revision"]) == str(revision):
                pinned_basis = m.get("basis", "unprocessed")
                if basis is not None and basis != pinned_basis:
                    raise AddressError(f"{self.source}: its units were found in the `{pinned_basis}` basis, not "
                                       f"`{basis}`.")
                return
        pins = ", ".join(f"{m['id']}@{m['revision']}" for m in self.applies_to["models"])
        raise AddressError(f"{self.source}: a checkpoint book applies only to {pins}, not {model_id}@{revision}.")


#: The ``requires`` keys that pin the models a book was written against.
PIN_KEYS = ("base_model", "latent_models")
_PIN_VALUE = (str, int)


def _model(pin: Mapping[str, str]) -> str:
    return f"{pin['id']}@{pin['revision']}"


def _pin_value(value: Any, where: str, source: str) -> str:
    if isinstance(value, bool) or not isinstance(value, _PIN_VALUE) or not str(value).strip():
        raise AddressError(f"{source}: `{where}` is a non-empty string, got {value!r}.")
    return str(value)


def _pins(requires: Mapping[str, Any], source: str) -> dict[str, Any]:
    """``requires``' model pins, checked and with every value a string."""
    out: dict[str, Any] = {}
    base = requires.get("base_model")
    if base is not None:
        if not isinstance(base, Mapping) or set(base) != {"id", "revision"}:
            raise AddressError(f"{source}: `requires.base_model` is {{id, revision}}, got {base!r}.")
        out["base_model"] = {k: _pin_value(base[k], f"requires.base_model.{k}", source) for k in ("id", "revision")}
    latents = requires.get("latent_models")
    if latents is None:
        return out
    if not isinstance(latents, Mapping) or not latents:
        raise AddressError(f"{source}: `requires.latent_models` maps a latent model's name to its pin "
                           "{release, frames}.")
    if base is None:
        raise AddressError(f"{source}: `requires.latent_models` needs `requires.base_model`: a dictionary means "
                           "something only on the model it was trained against.")
    pins = {}
    for name, pin in latents.items():
        at = f"requires.latent_models.{name}"
        if not isinstance(name, str) or not _SET_NAME.fullmatch(name):
            raise AddressError(f"{source}: {name!r} is not a valid latent model name (letters, digits, `_` and `-`).")
        if not isinstance(pin, Mapping) or "release" not in pin or set(pin) - {"release", "frames"}:
            raise AddressError(f"{source}: `{at}` is {{release, frames}} (`frames` optional), got {pin!r}.")
        release = pin["release"]
        if not isinstance(release, Mapping) or not release:
            raise AddressError(f"{source}: `{at}.release` is a non-empty mapping, such as {{sae_lens: <release>, id: "
                               "<id>}} or {hub: <repository>, revision: <revision>}.")
        rel = {str(k): _pin_value(v, f"{at}.release.{k}", source) for k, v in release.items()}
        if "hub" in rel and "revision" not in rel:
            raise AddressError(f"{source}: `{at}.release` names the Hub repository {rel['hub']} with no `revision`: "
                               "a repository alone follows its latest snapshot, which pins nothing.")
        entry: dict[str, Any] = {"release": rel}
        frames = pin.get("frames")
        if frames is not None:
            if not isinstance(frames, Mapping) or not frames:
                raise AddressError(f"{source}: `{at}.frames` maps a frame's name to its exact version.")
            entry["frames"] = {str(k): _pin_value(v, f"{at}.frames.{k}", source) for k, v in frames.items()}
        pins[name] = entry
    out["latent_models"] = pins
    return out


def _check_graph_waypoints(own: Mapping[str, Expr], graphs: Mapping[str, Any], portability: str, source: str) -> None:
    """Refuse a path waypoint naming a graph the book does not carry or a node its graph lacks, and a book with graph
    waypoints that is not a checkpoint book (a graph subset's nodes belong to one checkpoint)."""
    used = []
    for set_name, expr in own.items():
        for path in _paths_in(expr):
            for w in path.waypoints:
                if w.graph is None:
                    continue
                used.append(set_name)
                if w.graph not in graphs:
                    raise AddressError(f"{source}: set `{set_name}` has a waypoint in graph `{w.graph}`, which the "
                                       f"book's `graphs` does not name (graphs: {sorted(graphs)}).")
                ids = {n.id for n in graphs[w.graph].nodes}
                missing = [n for n in w.nodes if n not in ids]
                if missing:
                    raise AddressError(f"{source}: set `{set_name}` names nodes {missing}, which graph `{w.graph}` "
                                       "does not have.")
    if used and portability != "checkpoint":
        raise AddressError(f"{source}: the sets {sorted(set(used))} route through a graph subset's nodes, which belong "
                           "to one checkpoint, so the book is `portability: checkpoint` and pins its model in "
                           "`applies_to.models`.")


def _paths_in(expr: Expr) -> list[PathExpr]:
    if isinstance(expr, PathExpr):
        return [expr]
    if isinstance(expr, Guarded):
        return _paths_in(expr.expr)
    if isinstance(expr, Minus):
        return _paths_in(expr.left) + _paths_in(expr.right)
    if isinstance(expr, (And, Or)):
        return [p for i in expr.items for p in _paths_in(i)]
    return []


def _states_readout(body: Mapping[str, Any]) -> bool:
    """Whether a treatment states a readout, for itself or for a stage."""
    stages = body.get("stages")
    return "readout" in body or (isinstance(stages, list) and any(isinstance(st, Mapping) and "readout" in st
                                                                  for st in stages))


_KNOWN_KEYS = ("schema_version", "name", "description", "portability", "applies_to", "requires", "imports", "sets",
               "treatments", "graphs")


def _graphs(raw: Any, source: str, base_dir: str | Path | None) -> dict[str, Any]:
    from venngram.graphs import GraphSubset

    if not isinstance(raw, Mapping) or not raw:
        raise AddressError(f"{source}: `graphs` maps a graph name to its document {{file, sha256}}.")
    out = {}
    for name, ref in raw.items():
        if not isinstance(name, str) or not _SET_NAME.fullmatch(name):
            raise AddressError(f"{source}: {name!r} is not a valid graph name (letters, digits, `_` and `-`).")
        if not isinstance(ref, Mapping) or set(ref) != {"file", "sha256"} or not all(isinstance(ref[k], str) and ref[k]
                                                                                    for k in ref):
            raise AddressError(f"{source}: graph `{name}` is {{file, sha256}}: a graph is cited by its hash, so a "
                               "changed document is refused rather than re-read.")
        try:
            out[name] = GraphSubset.from_file(Path(base_dir or ".") / ref["file"], sha256=ref["sha256"])
        except VenngramError as err:
            raise AddressError(f"{source}: graph `{name}`: {err}") from err
    return out


def _treatments(raw: Any, book: AddressBook, source: str, base_dir: str | Path | None) -> dict[str, Any]:
    from venngram.recipe import Recipe

    if not isinstance(raw, Mapping) or not raw:
        raise AddressError(f"{source}: `treatments` maps a treatment name to its recipe body (model, stages, ...).")
    out = {}
    for name, body in raw.items():
        if not isinstance(name, str) or not _SET_NAME.fullmatch(name):
            raise AddressError(f"{source}: {name!r} is not a valid treatment name (letters, digits, `_` and `-`).")
        out[name] = Recipe.treatment(body, book, name=name, source=f"{source}: treatment `{name}`", base_dir=base_dir)
    return out


def make_lock(selections: Mapping[str, Mapping[str, tuple[int, ...]]], *, model_id: str,
              revision: str) -> dict[str, Any]:
    """A lock entry recording what each set resolved to on one model revision.

    Args:
        selections: For each set, the selected tensors' names mapped to their shapes.
        model_id: The model the sets were resolved on.
        revision: Its revision.
    """
    return {"model": {"id": model_id, "revision": str(revision)},
            "sets": {s: {n: list(shape) for n, shape in members.items()} for s, members in selections.items()}}


def diff_lock(lock: Mapping[str, Any], selections: Mapping[str, Mapping[str, tuple[int, ...]]]) -> list[str]:
    """How ``selections`` differ from a lock made by :func:`make_lock` (empty when they agree): what a consumer checks
    to detect drift, a renamed module or a changed depth, instead of silently selecting something else."""
    out = []
    locked = lock.get("sets", {})
    for s in sorted(set(locked) | set(selections)):
        old = {n: tuple(v) for n, v in locked.get(s, {}).items()}
        new = {n: tuple(v) for n, v in selections.get(s, {}).items()}
        out += [f"@{s}: {n} is no longer selected" for n in sorted(set(old) - set(new))]
        out += [f"@{s}: {n} is newly selected" for n in sorted(set(new) - set(old))]
        out += [f"@{s}: {n} changed shape {old[n]} -> {new[n]}" for n in sorted(set(old) & set(new))
                if old[n] != new[n]]
    return out


__all__ = [
    "AddressBook",
    "AXIS_SCHEMA",
    "DEPTH_SCHEMA",
    "diff_lock",
    "GRAPHS_SCHEMA",
    "make_lock",
    "PORTABILITY",
    "register_uri_resolver",
    "SCHEMA_VERSION",
    "SCHEMA_WINDOW",
    "STRUCTURE_LEVELS",
]
