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
Manifests
^^^^^^^^^

A directory of venngram documents (a Hub repository, a bucket prefix, a local folder) lists them in ``venngram.yaml``:

.. code-block:: yaml

    venngram_schema_version: 1
    requires: {venngram: ">=0.1"}
    documents:
      - {kind: address_book, path: books/chess.yaml, schema_version: 1, sha256: ..., needs: [books/units.safetensors]}
      - {kind: units, path: books/units.safetensors, sha256: ...}
      - {kind: recipe, path: recipes/chess.yaml, schema_version: 1, sha256: ...,
         needs: [books/chess.yaml, books/units.safetensors]}
    layout: {address_book: "books/{name}.yaml"}   # optional: how a writer names new documents of a kind

**Readers use the manifest only**, never a path convention. Every document is listed with its kind, its own schema
version, the sha256 of its content and the files it needs (an address book's sidecars and imported books, a recipe's
book), so a store fetches exactly a document and what it needs, and a layout can change without breaking a reader.

**Versions.** A manifest schema outside :data:`MANIFEST_WINDOW`, and a ``requires.venngram`` this version does not
satisfy, are refused by name. A document of an unknown ``kind`` is skipped with a warning (kept in
:attr:`Manifest.skipped`), so an older reader still uses the documents it understands.

:func:`build_manifest` writes one for a directory: it hashes each document and finds what it needs.

.. warning:: This is an experimental API and may change in a future release.
"""
import hashlib
import re
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from venngram.__about__ import __version__
from venngram.errors import StoreError

#: The file a directory of venngram documents lists them in.
MANIFEST_NAME = "venngram.yaml"

#: The manifest schema this code writes.
MANIFEST_SCHEMA_VERSION = 1

#: The manifest schemas this code reads.
MANIFEST_WINDOW = (1, 1)

#: The document kinds this version knows. ``units`` is a sidecar, read through the book that names it; ``index`` is a
#: registry index; ``format_adapter`` lists an installable format (:class:`venngram.sources.AdapterListing`) as data;
#: ``drafting_rules`` is a rule source for drafting maps (:mod:`venngram.maps.drafter`).
KINDS = ("address_book", "recipe", "component_map", "rules", "verification_record", "units", "index", "format_adapter",
         "drafting_rules")

_SHA256 = re.compile(r"[0-9a-f]{64}")
_CLAUSE = re.compile(r"^\s*(>=|<=|==|!=|>|<)\s*([0-9][0-9A-Za-z.]*)\s*$")


class UnknownKindWarning(UserWarning):
    """A manifest lists a document of a kind this version does not know; the document is skipped."""


@dataclass(frozen=True)
class DocumentEntry:
    """One listed document: its kind, path (relative, POSIX), content hash, schema version and the files it needs."""

    kind: str
    path: str
    sha256: str | None = None
    schema_version: int | None = None
    needs: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "path": self.path}
        if self.schema_version is not None:
            out["schema_version"] = self.schema_version
        if self.sha256 is not None:
            out["sha256"] = self.sha256
        if self.needs:
            out["needs"] = list(self.needs)
        return out


@dataclass(frozen=True)
class Manifest:
    """A directory's ``venngram.yaml`` (see the module docstring). Use :meth:`from_mapping` or :meth:`from_file`."""

    documents: tuple[DocumentEntry, ...]
    requires: Mapping[str, str] = field(default_factory=dict)
    layout: Mapping[str, str] = field(default_factory=dict)
    skipped: tuple[DocumentEntry, ...] = ()
    """Documents of kinds this version does not know, kept so a writer can carry them forward."""
    source: str = "<inline>"

    @classmethod
    def from_mapping(cls, doc: Any, *, source: str = "<inline>") -> "Manifest":
        """Read a manifest, refusing a malformed one by name.

        Raises:
            StoreError: If the schema is outside the window, ``requires.venngram`` is not satisfied, or an entry is
                malformed, repeated, or needs a file the manifest does not list.
        """
        if not isinstance(doc, Mapping):
            raise StoreError(f"{source}: a manifest is a mapping, got {type(doc).__name__}.")
        version = doc.get("venngram_schema_version")
        lo, hi = MANIFEST_WINDOW
        if not isinstance(version, int) or not lo <= version <= hi:
            raise StoreError(f"{source}: `venngram_schema_version` must be in {lo}..{hi} for this venngram "
                             f"({__version__}), got {version!r}.")
        requires = doc.get("requires") or {}
        if not isinstance(requires, Mapping):
            raise StoreError(f"{source}: `requires` maps a package to a version specifier.")
        spec = requires.get("venngram")
        if spec is not None and not satisfies(__version__, str(spec)):
            raise StoreError(f"{source}: requires venngram {spec}, this is {__version__}.")
        raw = doc.get("documents")
        if not isinstance(raw, list):
            raise StoreError(f"{source}: `documents` lists {{kind, path, ...}} entries.")
        known, skipped, seen = [], [], set()
        for item in raw:
            entry = _entry(item, source)
            if entry.path in seen:
                raise StoreError(f"{source}: `{entry.path}` is listed twice.")
            seen.add(entry.path)
            if entry.kind in KINDS:
                known.append(entry)
            else:
                warnings.warn(f"{source}: skipping `{entry.path}` of unknown kind `{entry.kind}` (this venngram "
                              f"knows {', '.join(KINDS)}).", UnknownKindWarning, stacklevel=2)
                skipped.append(entry)
        for entry in known:
            missing = [n for n in entry.needs if n not in seen]
            if missing:
                raise StoreError(f"{source}: `{entry.path}` needs {missing}, which the manifest does not list.")
        layout = doc.get("layout") or {}
        if not isinstance(layout, Mapping):
            raise StoreError(f"{source}: `layout` maps a kind to a path pattern.")
        return cls(tuple(known), {str(k): str(v) for k, v in requires.items()},
                   {str(k): str(v) for k, v in layout.items()}, tuple(skipped), source)

    @classmethod
    def from_file(cls, path: str | Path) -> "Manifest":
        """Read a manifest from YAML (loading runs no code)."""
        path = Path(path)
        with open(path, encoding="utf-8") as fh:
            return cls.from_mapping(yaml.safe_load(fh), source=str(path))

    def to_mapping(self) -> dict[str, Any]:
        doc: dict[str, Any] = {"venngram_schema_version": MANIFEST_SCHEMA_VERSION}
        if self.requires:
            doc["requires"] = dict(self.requires)
        doc["documents"] = [e.to_mapping() for e in (*self.documents, *self.skipped)]
        if self.layout:
            doc["layout"] = dict(self.layout)
        return doc

    def write(self, root: str | Path) -> Path:
        """Write the manifest into ``root`` as :data:`MANIFEST_NAME`."""
        path = Path(root) / MANIFEST_NAME
        path.write_text(yaml.safe_dump(self.to_mapping(), sort_keys=False), encoding="utf-8")
        return path

    def entry(self, path: str) -> DocumentEntry:
        """The entry for ``path``.

        Raises:
            StoreError: If the manifest does not list it (readers never fetch an unlisted path).
        """
        for e in self.documents:
            if e.path == path:
                return e
        if any(e.path == path for e in self.skipped):
            raise StoreError(f"{self.source}: `{path}` is of a kind this venngram does not know.")
        raise StoreError(f"{self.source}: `{path}` is not listed; listed: {[e.path for e in self.documents]}.")

    def closure(self, path: str) -> list[DocumentEntry]:
        """``path``'s entry and every entry it needs, transitively, each once."""
        out: dict[str, DocumentEntry] = {}
        todo = [path]
        while todo:
            entry = self.entry(todo.pop())
            if entry.path not in out:
                out[entry.path] = entry
                todo.extend(entry.needs)
        return list(out.values())

    def of_kind(self, kind: str) -> list[DocumentEntry]:
        return [e for e in self.documents if e.kind == kind]


def _relative(path: Any, source: str) -> str:
    if not isinstance(path, str) or not path:
        raise StoreError(f"{source}: a document path is a non-empty string, got {path!r}.")
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts or "\\" in path or path != p.as_posix():
        raise StoreError(f"{source}: `{path}` must be a relative POSIX path inside the directory (no `..`).")
    return path


def _entry(raw: Any, source: str) -> DocumentEntry:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("kind"), str):
        raise StoreError(f"{source}: a document entry is {{kind, path, ...}}, got {raw!r}.")
    path = _relative(raw.get("path"), source)
    sha = raw.get("sha256")
    if sha is not None and not (isinstance(sha, str) and _SHA256.fullmatch(sha)):
        raise StoreError(f"{source}: `{path}`: `sha256` is 64 lowercase hex digits, got {sha!r}.")
    version = raw.get("schema_version")
    if version is not None and not isinstance(version, int):
        raise StoreError(f"{source}: `{path}`: `schema_version` is an integer, got {version!r}.")
    needs = raw.get("needs") or []
    if not isinstance(needs, list):
        raise StoreError(f"{source}: `{path}`: `needs` lists paths.")
    unknown = set(raw) - {"kind", "path", "sha256", "schema_version", "needs"}
    if unknown:
        raise StoreError(f"{source}: `{path}`: unknown entry keys {sorted(unknown)}.")
    return DocumentEntry(raw["kind"], path, sha, version, tuple(_relative(n, source) for n in needs))


def _version_key(text: str) -> tuple[int, ...]:
    release = re.match(r"\d+(\.\d+)*", text)
    if not release:
        raise StoreError(f"`{text}` is not a version.")
    parts = [int(x) for x in release.group(0).split(".")]
    parts += [0] * (4 - len(parts))
    # a pre-release or dev version sorts before its release (0.1.0.dev0 < 0.1.0)
    return (*parts, 0 if release.group(0) == text else -1)


def satisfies(version: str, spec: str) -> bool:
    """Whether ``version`` satisfies ``spec``: comma-separated ``>=``, ``<=``, ``>``, ``<``, ``==``, ``!=`` clauses.

    >>> satisfies("0.2.1", ">=0.2, <0.3")
    True
    >>> satisfies("0.1.0.dev0", ">=0.1")
    False
    """
    ops = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b,
           "==": lambda a, b: a == b, "!=": lambda a, b: a != b}
    have = _version_key(version)
    for clause in spec.split(","):
        m = _CLAUSE.match(clause)
        if not m:
            raise StoreError(f"`{clause.strip()}` is not a version clause (>=, <=, >, <, ==, != and a version).")
        if not ops[m.group(1)](have, _version_key(m.group(2))):
            return False
    return True


def sha256_of(path: str | Path) -> str:
    """The sha256 of a file's content, as lowercase hex."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sidecars(units: Any) -> list[str]:
    """The sidecar files a ``units`` value names (flat ``{file, ...}`` or per layer)."""
    if not isinstance(units, Mapping):
        return []
    if "file" in units:
        return [str(units["file"])]
    return [str(v["file"]) for v in units.values() if isinstance(v, Mapping) and "file" in v]


def _is_local(target: str) -> bool:
    from venngram.store import is_reference  # a book's import may be a reference to another store's document

    return "://" not in target and not is_reference(target)


def _book_needs(doc: Any, base: Path) -> list[tuple[str, Path]]:
    """``(kind, path)`` for each file a book document needs: sidecars, and relative imports (with their needs)."""
    if not isinstance(doc, Mapping):
        return []
    out: list[tuple[str, Path]] = []
    for value in (doc.get("sets") or {}).values():
        if isinstance(value, Mapping):
            out += [("units", base / f) for f in _sidecars(value.get("units"))]
    for target in (doc.get("imports") or {}).values():
        if isinstance(target, str) and _is_local(target):
            path = base / target
            out.append(("address_book", path))
            out += _book_needs(_read_yaml(path), path.parent)
    return out


def _recipe_needs(doc: Any, base: Path) -> list[tuple[str, Path]]:
    if not isinstance(doc, Mapping):
        return []
    if isinstance(doc.get("compose"), list):
        out: list[tuple[str, Path]] = []
        for part in doc["compose"]:
            path = base / str(part)
            out += [("recipe", path), *_recipe_needs(_read_yaml(path), path.parent)]
        return out
    book = doc.get("book")
    if isinstance(book, str):
        path = base / book
        return [("address_book", path), *_book_needs(_read_yaml(path), path.parent)]
    return _book_needs(book, base)


def _read_yaml(path: Path) -> Any:
    if not path.is_file():
        raise StoreError(f"`{path}` is named by a document but is not a file.")
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


_NEEDS = {"address_book": _book_needs, "recipe": _recipe_needs}


def build_manifest(root: str | Path, documents: Iterable[tuple[str, str]], *, requires: Mapping[str, str] | None = None,
                   layout: Mapping[str, str] | None = None) -> Manifest:
    """A manifest for ``documents`` (``(kind, path)`` pairs, paths relative to ``root``).

    Each document is hashed, its schema version read, and the files it needs (an address book's sidecars and relative
    imports, a recipe's book) found and listed too, as ``units`` and ``address_book`` entries.

    Raises:
        StoreError: If a document or a file it needs is missing or outside ``root``.
    """
    root = Path(root).resolve()
    entries: dict[str, DocumentEntry] = {}
    todo = [(kind, root / path) for kind, path in documents]
    while todo:
        kind, path = todo.pop(0)
        rel = _inside(path, root)
        if rel in entries:
            continue
        if not path.is_file():
            raise StoreError(f"`{rel}` is not a file under {root}.")
        needs: list[tuple[str, Path]] = []
        version = None
        if kind != "units" and path.suffix in (".yaml", ".yml"):
            doc = _read_yaml(path)
            needs = _NEEDS.get(kind, lambda d, b: [])(doc, path.parent)
            if isinstance(doc, Mapping):
                v = doc.get("schema_version")
                version = v if isinstance(v, int) else None
        need_paths = tuple(dict.fromkeys(_inside(p, root) for _, p in needs))
        entries[rel] = DocumentEntry(kind, rel, sha256_of(path), version, need_paths)
        todo += needs
    return Manifest(tuple(entries.values()), dict(requires or {}), dict(layout or {}), source=str(root / MANIFEST_NAME))


def _inside(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        raise StoreError(f"`{path}` is outside the directory {root}; a manifest lists only what is inside it."
                         ) from None


__all__ = [
    "build_manifest",
    "DocumentEntry",
    "KINDS",
    "Manifest",
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA_VERSION",
    "MANIFEST_WINDOW",
    "satisfies",
    "sha256_of",
    "UnknownKindWarning",
]
