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
Sources
^^^^^^^

A **source** turns something outside venngram into venngram documents. There are two kinds, with different inputs and
one shared frame:

- a **map source** (:class:`MapSource`) derives a component map from something that knows an architecture's structure
  (a library's per-architecture adapters, a model's module tree);
- a **format** (:class:`Format`) translates a document another tool writes into venngram books and recipes, for a tool
  that does not integrate venngram itself. A package we own integrates venngram directly instead (it applies and
  exports documents and declares what it supports), so a format is for foreign documents only.

Both declare a ``name`` and ``version`` (recorded as :class:`Provenance` in what they produce) and what they cannot
express (``unexpressed``, data, so a comparison between sources counts what each covered). Both are found by entry point
(``venngram.map_sources`` and ``venngram.formats``), so a package ships a source without venngram depending on it, or
registered in process (:func:`register_map_source`, :func:`register_format`). Entry points are read on first use; one
that fails to load is skipped with a warning, so a broken package narrows what is available rather than breaking it.

A format not installed is still discoverable: a manifest may list a ``format_adapter`` document
(:class:`AdapterListing`) naming the formats it reads, its version and where to install it from, so a reader can say how
to read a foreign document without executing anything. Installing stays an explicit act.

A format's :meth:`~Format.read` returns a :class:`Translation`: documents, not objects, so :func:`write_translation`
puts it in a directory and :func:`venngram.manifest.build_manifest` and :func:`venngram.store.publish` take it from
there. A translation is lossless or refused: what venngram cannot carry is refused by name, unless the
:class:`ReadContext` is ``lenient``, when it is listed in :attr:`Translation.unexpressed` and kept under the recipe's
``metadata``, so nothing is dropped silently.

.. warning:: This is an experimental API and may change in a future release.
"""
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

from venngram.errors import Refusal, VenngramError
from venngram.units import write_units

#: Entry-point groups, by kind of source.
MAP_SOURCES_GROUP = "venngram.map_sources"
FORMATS_GROUP = "venngram.formats"


class SourceError(VenngramError):
    """Raised when a source is unknown, collides with another, or cannot produce what it was asked for."""


class SourceLoadWarning(UserWarning):
    """An entry point did not load; the source it names is skipped."""


@dataclass(frozen=True)
class Provenance:
    """What produced a document: the source's kind, name and version, and what it was produced from."""

    kind: str
    """``map_source`` or ``format``."""
    source: str
    version: str
    input_sha256: str | None = None
    """The sha256 of the foreign document a format read, so a translation can be re-run and checked."""
    revision_source: str | None = None
    """How a model revision was obtained when the input pinned none (``resolved-at-import``)."""

    def to_mapping(self) -> dict[str, Any]:
        return {k: v for k, v in vars(self).items() if v is not None}


@runtime_checkable
class MapSource(Protocol):
    """Derives component maps from something that knows architectures' structure."""

    name: str
    version: str
    unexpressed: tuple[str, ...]
    """What the source cannot express (opaque towers, say), as data."""

    def architectures(self) -> Iterable[str]:
        """The architectures this source can derive a map for."""
        ...

    def derive(self, architecture: str, *, config: Any = None, model: Any = None) -> Any:
        """A :class:`~venngram.maps.schema.ComponentMap` for ``architecture``, from a configuration or a model where
        the source needs one.

        Raises:
            SourceError: If the source cannot describe the architecture.
        """
        ...


@dataclass(frozen=True)
class ReadContext:
    """What a format's :meth:`~Format.read` may rely on from its caller."""

    resolve_revision: Callable[[str], str] | None = None
    """Model id -> its current commit, for a format whose documents pin no revision (venngram's ``hf`` extra provides
    one); a format never resolves revisions itself."""
    inline_threshold: int = 512
    """Units per layer above which a set goes to a sidecar."""
    lenient: bool = False
    """Refuse what venngram cannot carry (default), or record it in :attr:`Translation.unexpressed`."""


@dataclass(frozen=True)
class Translation:
    """What a format produced from one foreign document."""

    documents: Mapping[str, tuple[str, Mapping[str, Any]]]
    """Relative path -> (kind, document mapping): usually a book and a recipe."""
    provenance: Provenance
    sidecars: Mapping[str, Mapping[str, tuple[Sequence[int], int]]] = field(default_factory=dict)
    """Relative sidecar path -> {key: (units, axis length)}, written with :func:`venngram.units.write_units`; the
    documents reference them by path, key and the sha256 :func:`write_translation` fills in."""
    unexpressed: tuple[Refusal, ...] = ()
    """What the foreign document held that venngram did not carry (only under a lenient context)."""


@runtime_checkable
class Format(Protocol):
    """Translates documents another tool writes into venngram documents."""

    name: str
    version: str
    kinds: tuple[str, ...]
    """The document kinds a translation holds (``("address_book", "recipe")``)."""
    unexpressed: tuple[str, ...]

    def detect(self, doc: Any) -> bool:
        """Whether ``doc`` is in this format."""
        ...

    def read(self, doc: Any, ctx: ReadContext) -> Translation:
        """Translate ``doc``.

        Raises:
            SourceError: If ``doc`` holds something venngram cannot carry and ``ctx.lenient`` is not set.
        """
        ...


@dataclass(frozen=True)
class AdapterListing:
    """A ``format_adapter`` document: an installable format, described as data.

    .. code-block:: yaml

        schema_version: 1
        kind: format_adapter
        name: neuronary
        version: 0.1.0
        formats: [neuronary]
        install: {pip: venngram-neuronary}        # or {git: https://github.com/<owner>/<repo>, commit: <sha>}
    """

    name: str
    version: str
    formats: tuple[str, ...]
    install: Mapping[str, str]

    @classmethod
    def from_mapping(cls, doc: Any, *, source: str = "<inline>") -> "AdapterListing":
        """Read a listing, refusing a malformed one by name."""
        if not isinstance(doc, Mapping) or doc.get("kind") != "format_adapter" or doc.get("schema_version") != 1:
            raise SourceError(f"{source}: a format adapter listing is `kind: format_adapter`, `schema_version: 1`.")
        install = doc.get("install")
        if not isinstance(install, Mapping) or not ({"pip", "git"} & set(install)):
            raise SourceError(f"{source}: `install` names a `pip` requirement or a `git` URL (with a `commit`).")
        if "git" in install and not install.get("commit"):
            raise SourceError(f"{source}: a `git` install pins a `commit`, so the listing cites one adapter.")
        formats = doc.get("formats")
        if not isinstance(formats, list) or not formats or not all(isinstance(f, str) for f in formats):
            raise SourceError(f"{source}: `formats` lists the format names the adapter reads.")
        return cls(str(doc.get("name")), str(doc.get("version")), tuple(formats),
                   {str(k): str(v) for k, v in install.items()})

    @classmethod
    def from_file(cls, path: str | Path) -> "AdapterListing":
        with open(path, encoding="utf-8") as fh:
            return cls.from_mapping(yaml.safe_load(fh), source=str(path))


class _Registry:
    """Sources of one kind: registered in process, plus entry points read on first use."""

    def __init__(self, group: str, protocol: type) -> None:
        self.group, self.protocol = group, protocol
        self._sources: dict[str, Any] = {}
        self._loaded = False

    def register(self, source: Any, *, replace: bool = False) -> None:
        if not isinstance(source, self.protocol):
            raise SourceError(f"{source!r} does not implement {self.protocol.__name__}.")
        if source.name in self._sources and not replace:
            raise SourceError(f"A source named `{source.name}` is already registered in `{self.group}`; pass "
                              "replace=True to replace it.")
        self._sources[source.name] = source

    def _load_entry_points(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        for ep in entry_points(group=self.group):
            if ep.name in self._sources:
                continue  # an in-process registration wins over an installed one of the same name
            try:
                obj = ep.load()
                source = obj() if isinstance(obj, type) or not isinstance(obj, self.protocol) else obj
                self.register(source)
            except Exception as e:  # a broken package narrows what is available, it never breaks discovery
                warnings.warn(f"{self.group}: entry point `{ep.name}` ({ep.value}) did not load: {e}",
                              SourceLoadWarning, stacklevel=3)

    def all(self) -> dict[str, Any]:
        self._load_entry_points()
        return dict(self._sources)

    def get(self, name: str) -> Any:
        sources = self.all()
        if name not in sources:
            raise SourceError(f"No source `{name}` in `{self.group}`; known: {sorted(sources)}.")
        return sources[name]

    def unregister(self, name: str) -> None:
        self._sources.pop(name, None)


_MAP_SOURCES = _Registry(MAP_SOURCES_GROUP, MapSource)
_FORMATS = _Registry(FORMATS_GROUP, Format)


def register_map_source(source: MapSource, *, replace: bool = False) -> None:
    """Register a map source in process (installed packages register through the ``venngram.map_sources`` entry
    point)."""
    _MAP_SOURCES.register(source, replace=replace)


def map_sources() -> dict[str, MapSource]:
    """Every map source, by name: registered ones and installed entry points."""
    return _MAP_SOURCES.all()


def derive_map(source: str, architecture: str, *, config: Any = None, model: Any = None) -> tuple[Any, Provenance]:
    """A component map from the named source, and its provenance."""
    s = _MAP_SOURCES.get(source)
    return s.derive(architecture, config=config, model=model), Provenance("map_source", s.name, s.version)


def register_format(fmt: Format, *, replace: bool = False) -> None:
    """Register a format in process (installed packages register through the ``venngram.formats`` entry point)."""
    _FORMATS.register(fmt, replace=replace)


def formats() -> dict[str, Format]:
    """Every format, by name: registered ones and installed entry points."""
    return _FORMATS.all()


def format_for(doc: Any, *, name: str | None = None) -> Format:
    """The format to read ``doc`` with: the named one, or the one format whose ``detect`` accepts it.

    Raises:
        SourceError: If no format accepts ``doc``, or more than one does and none is named.
    """
    if name is not None:
        return _FORMATS.get(name)
    accepting = [f for f in formats().values() if f.detect(doc)]
    if len(accepting) != 1:
        found = sorted(f.name for f in accepting)
        raise SourceError(f"{'No format' if not accepting else 'Several formats'} accept this document "
                          f"({found or 'known: ' + str(sorted(formats()))}); name one with `name=`.")
    return accepting[0]


def unregister(name: str) -> None:
    """Remove an in-process registration of either kind (for tests and replacements)."""
    _MAP_SOURCES.unregister(name)
    _FORMATS.unregister(name)


def write_translation(translation: Translation, root: str | Path) -> list[tuple[str, str]]:
    """Write a translation's sidecars and documents under ``root``; return ``(kind, path)`` for each document, ready
    for :func:`venngram.manifest.build_manifest`.

    A document's sidecar reference whose ``sha256`` is ``None`` is filled in with the written file's hash, so a format
    names a sidecar before its content exists.
    """
    root = Path(root)
    digests: dict[str, str] = {}
    for rel, sets in translation.sidecars.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        digests[rel] = write_units(path, sets)
    written = []
    for rel, (kind, doc) in translation.documents.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        filled = _fill_sha(doc, digests, path.parent, root)
        path.write_text(yaml.safe_dump(filled, sort_keys=False), encoding="utf-8")
        written.append((kind, rel))
    return written


def _fill_sha(value: Any, digests: Mapping[str, str], doc_dir: Path, root: Path) -> Any:
    if isinstance(value, Mapping):
        out = {k: _fill_sha(v, digests, doc_dir, root) for k, v in value.items()}
        if "file" in out and "key" in out and out.get("sha256") is None:
            rel = (doc_dir / str(out["file"])).resolve().relative_to(root.resolve()).as_posix()
            if rel in digests:
                out["sha256"] = digests[rel]
        return out
    if isinstance(value, list):
        return [_fill_sha(v, digests, doc_dir, root) for v in value]
    return value


__all__ = [
    "AdapterListing",
    "derive_map",
    "Format",
    "format_for",
    "formats",
    "FORMATS_GROUP",
    "map_sources",
    "MAP_SOURCES_GROUP",
    "MapSource",
    "Provenance",
    "ReadContext",
    "register_format",
    "register_map_source",
    "SourceError",
    "SourceLoadWarning",
    "Translation",
    "unregister",
    "write_translation",
]
