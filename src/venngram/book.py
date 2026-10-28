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
    requires: {structure: physical}
    imports: {base: ./common.yaml}
    sets:
      top:  {role: adapter, layers: "-4:"}
      rest: {role: adapter, exclude: "@top"}

**Versions.** ``schema_version`` is required and read within a window (:data:`SCHEMA_WINDOW`). A key an older reader
may ignore without misreading the book (``description``, ``charts``, any unknown top-level key) never changes the
version; a key that changes what a set selects does.

**Portability is declared.** A ``checkpoint`` book carries unit indices meaningful in one checkpoint only, so it must
pin ``applies_to.models`` with an ``id`` and a ``revision`` (and may give the ``basis`` its units were found in); it is
refused against any other model.

**References.** ``@<set>`` resolves within the book, ``@<import>.<set>`` within an import. Every reference is checked
when the book loads, and cycles are refused, naming the cycle.

**Imports** are paths relative to the importing book, or URIs whose scheme has a registered resolver
(:func:`register_uri_resolver`), which is how a consumer adds ``hub://``.

.. warning:: This is an experimental API and may change in a future release.
"""
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from venngram.address import Expr, parse, references, rename_refs
from venngram.errors import AddressError, VenngramError

SCHEMA_VERSION = 1
SCHEMA_WINDOW = (1, 1)
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
        raise AddressError(f"{source}: schema_version {version} was written by a newer venngram (this one reads "
                           f"{lo} to {hi}); upgrade venngram.")
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

    @classmethod
    def from_mapping(cls, doc: Mapping[str, Any], *, source: str = "<inline>", base_dir: str | Path | None = None,
                     inline: bool = False, _loading: tuple[str, ...] = ()) -> "AddressBook":
        """Load a book from its mapping.

        Args:
            doc: The book.
            source: Where it came from, for messages.
            base_dir: The directory relative imports resolve against (the current directory when ``None``).
            inline: A book written inside another document (a schedule's ``addresses`` key): ``schema_version`` and
                ``name`` default to the current version and ``source``.

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
        raw_sets = doc.get("sets")
        if not isinstance(raw_sets, Mapping) or not raw_sets:
            raise AddressError(f"{source}: `sets` is a non-empty mapping of set names to addresses.")
        own: dict[str, Expr] = {}
        for set_name, address in raw_sets.items():
            if not isinstance(set_name, str) or not _SET_NAME.fullmatch(set_name):
                raise AddressError(f"{source}: {set_name!r} is not a valid set name (letters, digits, `_` and `-`).")
            try:
                own[set_name] = parse(address)
            except VenngramError as err:
                raise AddressError(f"{source}: set `{set_name}`: {err}") from err
        imports = cls._load_imports(doc.get("imports") or {}, source, base_dir, _loading)
        known = {k for k in own} | {f"{ns}.{s}" for ns, book in imports.items() for s in book.sets()}
        extra = {k: v for k, v in doc.items() if k not in _KNOWN_KEYS}
        book = cls(name=str(doc.get("name", source)), schema_version=version, portability=portability,
                   applies_to=applies_to, requires=requires, own_sets=own, imports=imports, source=source,
                   description=doc.get("description"), extra=extra)
        for set_name, expr in own.items():
            missing = [r for r in references(expr) if r not in known]
            if missing:
                named = ", ".join("@" + m for m in missing)
                raise AddressError(f"{source}: set `{set_name}` references {named}, which "
                                   f"no set or import defines (known: {', '.join(sorted('@' + k for k in known))}).")
        book._check_cycles()
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
            key = target if _URI.match(target) else str((Path(base_dir or ".") / target).resolve())
            if key in loading or key == source:
                raise AddressError(f"{source}: imports form a cycle through {target}.")
            if _URI.match(target):
                scheme = target.split("://", 1)[0]
                if scheme not in _URI_RESOLVERS:
                    raise AddressError(f"{source}: import `{ns}` uses `{scheme}://`, which has no registered resolver "
                                       "(interpretune registers `hub://`).")
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


_KNOWN_KEYS = ("schema_version", "name", "description", "portability", "applies_to", "requires", "imports", "sets")


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
    "diff_lock",
    "make_lock",
    "PORTABILITY",
    "register_uri_resolver",
    "SCHEMA_VERSION",
    "SCHEMA_WINDOW",
    "STRUCTURE_LEVELS",
]
