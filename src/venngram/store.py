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
Stores
^^^^^^

A **document reference** names one document wherever it is stored::

    <store>:<ref>[@<revision>]:<path>[#<sha256>]

for example ``hub-model:speediedan/venngram-maps@3f1e...:maps/Qwen3ForCausalLM.yaml``. A :class:`Store` is the seam
between a reference and bytes: ``local`` (a directory), ``hub-model`` and ``hub-dataset`` (Hub repositories, versioned
by commit) and ``hub-bucket`` (a Hub bucket, unversioned). On the Hub stores ``<ref>`` may continue past the repository
or bucket id into a directory (``speediedan/venngram-examples/neuronary``) holding its own manifest, so one repository
can hold many small manifests rather than one large one. A new place to keep documents (an S3 prefix, a lab file share)
is a new store (:func:`register_store`), never a change to a document.

**Latest or exact.** A reference without a revision and a sha256 follows the latest published snapshot: a versioned
store resolves its current commit, an unversioned one reads its current manifest. ``@<revision>`` pins a commit (a
branch name resolves to one), and ``#<sha256>`` pins the content on any store. :func:`pin` returns the exact reference a
latest one resolves to, for a record that must be reproducible (a checkpoint book's provenance, a verification record).

**Fetching is manifest-first at one revision.** :func:`fetch` resolves the revision once (a versioned store resolves a
reference without one, or with a branch name, to a commit), reads the directory's manifest
(:mod:`venngram.manifest`) at that revision, refuses a path it does not list, then fetches the document and every file
it needs into a local cache and checks each against the manifest's sha256 and the reference's. On an unversioned store
every listed file must carry a sha256, so a fetch is checked even when the reference does not cite one.

:func:`load` reads a fetched document by its manifest kind (:func:`register_loader` adds kinds), from the cache only.
:func:`publish` writes a directory's documents and its manifest to a store. An address book may import another book by
reference, so books on different stores compose.

The Hub stores need ``huggingface_hub`` (the ``hf`` extra).

.. warning:: This is an experimental API and may change in a future release.
"""
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

from venngram.errors import StoreError
from venngram.manifest import build_manifest, DocumentEntry, Manifest, MANIFEST_NAME, sha256_of

_SHA256 = re.compile(r"[0-9a-f]{64}")
_STORE_NAME = re.compile(r"[a-z][a-z0-9-]+")


@runtime_checkable
class Store(Protocol):
    """Where documents live. ``fetch`` and ``push`` move whole sets of paths, since a document needs its files."""

    name: str
    versioned: bool
    """Whether a reference can pin a revision. An unversioned store's references carry a sha256."""

    def resolve_revision(self, ref: str, revision: str | None) -> str | None:
        """The immutable revision ``revision`` (or the current state) names; ``None`` for an unversioned store."""
        ...

    def fetch(self, ref: str, paths: Sequence[str], revision: str | None, cache: Path) -> Path:
        """Make ``paths`` available locally and return the root they sit under (``root / path``). ``cache`` is a
        directory reserved for this ref and revision; a store with a cache of its own may ignore it."""
        ...

    def push(self, ref: str, files: Mapping[str, Path], message: str, revision: str | None) -> str | None:
        """Write ``files`` (store path -> local file) and return the revision written, if the store has one."""
        ...


@dataclass(frozen=True)
class Reference:
    """A parsed document reference (see the module docstring)."""

    store: str
    ref: str
    path: str
    revision: str | None = None
    sha256: str | None = None

    @classmethod
    def parse(cls, text: str) -> "Reference":
        """Parse ``<store>:<ref>[@<revision>]:<path>[#<sha256>]``.

        Raises:
            StoreError: If the text is malformed or names no registered store.
        """
        name, sep, rest = text.partition(":")
        if not sep or name not in _STORES:
            raise StoreError(f"`{text}` is not a document reference: it starts with a store name ({sorted(_STORES)}) "
                             "and a colon.")
        rest, _, sha = rest.partition("#") if "#" in rest else (rest, "", "")
        if sha and not _SHA256.fullmatch(sha):
            raise StoreError(f"`{text}`: the part after `#` is a sha256 (64 lowercase hex digits).")
        ref, sep, path = rest.rpartition(":")
        if not sep or not ref or not path:
            raise StoreError(f"`{text}`: a reference is `<store>:<ref>[@<revision>]:<path>[#<sha256>]`.")
        revision = None
        if _STORES[name][0] and "@" in ref:
            ref, _, revision = ref.rpartition("@")
        return cls(name, ref, path, revision or None, sha or None)

    def __str__(self) -> str:
        rev = f"@{self.revision}" if self.revision else ""
        sha = f"#{self.sha256}" if self.sha256 else ""
        return f"{self.store}:{self.ref}{rev}:{self.path}{sha}"


#: Registered stores: name -> (versioned, the store or a factory that builds it on first use).
_STORES: dict[str, tuple[bool, "Store | Callable[[], Store]"]] = {}

#: Loaders by document kind: a fetched file's path -> the loaded document.
_LOADERS: dict[str, Callable[[Path], Any]] = {}


def register_store(name: str, store: "Store | Callable[[], Store]", *, versioned: bool, replace: bool = False) -> None:
    """Register a store (or a factory, called on first use, so a store's dependency is imported only when used).

    Raises:
        StoreError: If ``name`` is malformed, or registered and ``replace`` is not set.
    """
    if not _STORE_NAME.fullmatch(name):
        raise StoreError(f"`{name}` is not a store name (lowercase letters, digits and `-`, two or more).")
    if name in _STORES and not replace:
        raise StoreError(f"The store `{name}` is already registered; pass replace=True to replace it.")
    _STORES[name] = (versioned, store)


def get_store(name: str) -> Store:
    """The store registered as ``name``."""
    if name not in _STORES:
        raise StoreError(f"No store `{name}`; registered: {sorted(_STORES)}.")
    versioned, store = _STORES[name]
    if not isinstance(store, Store):
        store = store()
        _STORES[name] = (versioned, store)
    return store


def is_reference(text: str) -> bool:
    """Whether ``text`` starts like a document reference (a registered store's name and a colon)."""
    name, sep, _ = text.partition(":")
    return bool(sep) and name in _STORES


def register_loader(kind: str, loader: Callable[[Path], Any], *, replace: bool = False) -> None:
    """Register how :func:`load` reads a document of ``kind`` from its fetched path."""
    if kind in _LOADERS and not replace:
        raise StoreError(f"The kind `{kind}` already has a loader; pass replace=True to replace it.")
    _LOADERS[kind] = loader


def cache_dir() -> Path:
    """Where fetched documents are cached: ``$VENNGRAM_CACHE``, else ``~/.cache/venngram``."""
    return Path(os.environ.get("VENNGRAM_CACHE") or Path.home() / ".cache" / "venngram")


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "--", text).strip("-") or "_"


def _check(path: Path, expected: str | None, what: str) -> None:
    if expected is not None and sha256_of(path) != expected:
        raise StoreError(f"{what}: the content's sha256 is {sha256_of(path)}, the reference cites {expected}; the "
                         "document changed after it was cited.")


def fetch_manifest(store: str, ref: str, revision: str | None = None) -> tuple[Manifest, Path, str | None]:
    """The manifest at ``ref`` (one revision, resolved), the root it was fetched to, and that revision."""
    s = get_store(store)
    revision = s.resolve_revision(ref, revision)
    if revision is not None:
        cache = cache_dir() / s.name / _safe(ref) / _safe(revision)
        root = s.fetch(ref, [MANIFEST_NAME], revision, cache)
    else:
        # an unversioned store's cache is keyed by the manifest's own content, so a changed directory is a new root
        with tempfile.TemporaryDirectory() as tmp:
            got = s.fetch(ref, [MANIFEST_NAME], None, Path(tmp))
            if not (got / MANIFEST_NAME).is_file():
                raise StoreError(f"{store}:{ref}: no {MANIFEST_NAME}; a directory of documents lists them in one.")
            if got.resolve() != Path(tmp).resolve():
                root = got  # the store serves files in place (a local directory): nothing to cache
            else:
                root = cache_dir() / s.name / _safe(ref) / sha256_of(got / MANIFEST_NAME)
                root.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(got / MANIFEST_NAME, root / MANIFEST_NAME)
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise StoreError(f"{store}:{ref}: no {MANIFEST_NAME}; a directory of documents lists them in one.")
    return Manifest.from_file(path), root, revision


def fetch(reference: "str | Reference") -> tuple[Path, DocumentEntry]:
    """Fetch a document and every file it needs; return its local path and its manifest entry.

    Raises:
        StoreError: If the reference is malformed, the manifest does not list the path, an unversioned store's
            manifest lists a file without a sha256, or any fetched file's content differs from what the manifest or the
            reference cites.
    """
    ref = Reference.parse(reference) if isinstance(reference, str) else reference
    s = get_store(ref.store)
    manifest, root, revision = fetch_manifest(ref.store, ref.ref, ref.revision)
    entry = manifest.entry(ref.path)
    if ref.sha256 is not None and entry.sha256 is not None and ref.sha256 != entry.sha256:
        raise StoreError(f"{ref}: the manifest at this revision lists sha256 {entry.sha256} for `{ref.path}`; the "
                         "reference cites another content.")
    closure = manifest.closure(ref.path)
    unhashed = [e.path for e in closure if e.sha256 is None]
    if not s.versioned and unhashed:
        raise StoreError(f"{ref}: on `{ref.store}` every listed file carries a sha256; {unhashed} do not.")
    root = s.fetch(ref.ref, [e.path for e in closure], revision, root)
    for e in closure:
        local = root / e.path
        if not local.is_file():
            raise StoreError(f"{ref}: `{e.path}` is listed but the store did not provide it.")
        _check(local, e.sha256, f"{ref.store}:{ref.ref}:{e.path}")
    _check(root / ref.path, ref.sha256, str(ref))
    return root / ref.path, entry


def pin(reference: "str | Reference") -> Reference:
    """The exact reference ``reference`` resolves to now: its revision resolved to a commit (on a versioned store) and
    the document's sha256 from the manifest at that revision.

    Raises:
        StoreError: As :func:`fetch_manifest`, or if the manifest does not list the path or gives it no sha256.
    """
    ref = Reference.parse(reference) if isinstance(reference, str) else reference
    manifest, _, revision = fetch_manifest(ref.store, ref.ref, ref.revision)
    entry = manifest.entry(ref.path)
    sha = ref.sha256 or entry.sha256
    if sha is None:
        raise StoreError(f"{ref}: the manifest gives `{ref.path}` no sha256, so it cannot be pinned by content.")
    if entry.sha256 is not None and sha != entry.sha256:
        raise StoreError(f"{ref}: the manifest at this revision lists sha256 {entry.sha256} for `{ref.path}`; the "
                         "reference cites another content.")
    return Reference(ref.store, ref.ref, ref.path, revision, sha)


def load(reference: "str | Reference") -> Any:
    """Fetch a document (see :func:`fetch`) and read it by its manifest kind.

    Raises:
        StoreError: If no loader reads the document's kind.
    """
    path, entry = fetch(reference)
    if entry.kind not in _LOADERS:
        raise StoreError(f"{reference}: no loader reads kind `{entry.kind}` (loaders: {sorted(_LOADERS)}; a `units` "
                         "sidecar is read through the book that names it).")
    return _LOADERS[entry.kind](path)


def publish(store: str, ref: str, root: str | Path, *, documents: Sequence[tuple[str, str]] | None = None,
            message: str = "publish venngram documents", revision: str | None = None,
            requires: Mapping[str, str] | None = None) -> tuple[Manifest, str | None]:
    """Write a directory's documents and its manifest to a store.

    ``documents`` (``(kind, path)`` pairs) builds a new manifest (:func:`venngram.manifest.build_manifest`); without
    it, the directory's existing manifest is used and every listed file is checked against its hash first.

    Returns:
        The manifest written, and the revision the store wrote (a commit on the Hub), if it has one.
    """
    root = Path(root)
    if documents is not None:
        manifest = build_manifest(root, documents, requires=requires)
        manifest.write(root)
    else:
        manifest = Manifest.from_file(root / MANIFEST_NAME)
        for e in manifest.documents:
            _check(root / e.path, e.sha256, f"{root}: {e.path}")
    files = {e.path: root / e.path for e in (*manifest.documents, *manifest.skipped)}
    files[MANIFEST_NAME] = root / MANIFEST_NAME
    return manifest, get_store(store).push(ref, files, message, revision)


class LocalStore:
    """A directory on this machine: ``local:<directory>:<path>#<sha256>``. Unversioned; files are read in place."""

    name = "local"
    versioned = False

    def resolve_revision(self, ref: str, revision: str | None) -> str | None:
        if revision is not None:
            raise StoreError(f"local:{ref}: a local directory keeps no revisions; cite the sha256 instead.")
        return None

    def fetch(self, ref: str, paths: Sequence[str], revision: str | None, cache: Path) -> Path:
        return Path(ref)

    def push(self, ref: str, files: Mapping[str, Path], message: str, revision: str | None) -> str | None:
        dest = Path(ref)
        for rel, src in files.items():
            target = dest / rel
            if target.resolve() != Path(src).resolve():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, target)
        return None


def _split_hub(ref: str, store: str) -> tuple[str, str]:
    """``<owner>/<name>[/<directory>]`` -> the repository or bucket id and the directory (``""`` for the root)."""
    parts = ref.split("/")
    if len(parts) < 2 or not all(parts):
        raise StoreError(f"{store}:{ref}: a Hub reference is `<owner>/<name>[/<directory>]`.")
    return "/".join(parts[:2]), "/".join(parts[2:])


def _in(prefix: str, path: str) -> str:
    return f"{prefix}/{path}" if prefix else path


class HubRepoStore:
    """A Hub model or dataset repository, or a directory of one:
    ``hub-model:<owner>/<name>[/<dir>][@<revision>]:<path>``. Versioned by commit; files land in the Hub's own cache."""

    versioned = True

    def __init__(self, repo_type: str) -> None:
        self.repo_type = repo_type
        self.name = f"hub-{repo_type}"

    def resolve_revision(self, ref: str, revision: str | None) -> str | None:
        from huggingface_hub import HfApi  # an optional dependency: the `hf` extra

        repo, _ = _split_hub(ref, self.name)
        sha = HfApi().repo_info(repo, repo_type=self.repo_type, revision=revision).sha
        if not sha:
            raise StoreError(f"{self.name}:{ref}: could not resolve {revision or 'the current revision'} to a commit.")
        return sha

    def fetch(self, ref: str, paths: Sequence[str], revision: str | None, cache: Path) -> Path:
        from huggingface_hub import hf_hub_download

        repo, prefix = _split_hub(ref, self.name)
        root = None
        for rel in paths:
            full = _in(prefix, rel)
            local = Path(hf_hub_download(repo, full, repo_type=self.repo_type, revision=revision))
            root = local.parents[len(Path(full).parts) - 1]
        if root is None:
            return cache
        return root / prefix if prefix else root

    def push(self, ref: str, files: Mapping[str, Path], message: str, revision: str | None) -> str | None:
        from huggingface_hub import CommitOperationAdd, HfApi

        repo, prefix = _split_hub(ref, self.name)
        ops = [CommitOperationAdd(path_in_repo=_in(prefix, rel), path_or_fileobj=str(src))
               for rel, src in files.items()]
        info = HfApi().create_commit(repo, ops, commit_message=message, repo_type=self.repo_type, revision=revision)
        return getattr(info, "oid", None)


class HubBucketStore:
    """A Hub bucket, or a directory of one: ``hub-bucket:<owner>/<bucket>[/<dir>]:<path>[#<sha256>]``. Unversioned."""

    name = "hub-bucket"
    versioned = False

    def resolve_revision(self, ref: str, revision: str | None) -> str | None:
        if revision is not None:
            raise StoreError(f"hub-bucket:{ref}: a bucket keeps no revisions; cite the sha256 instead.")
        return None

    def fetch(self, ref: str, paths: Sequence[str], revision: str | None, cache: Path) -> Path:
        import huggingface_hub

        bucket, prefix = _split_hub(ref, self.name)
        want = [p for p in paths if not (cache / p).is_file()]
        for p in want:
            (cache / p).parent.mkdir(parents=True, exist_ok=True)
        if want:
            huggingface_hub.download_bucket_files(bucket, [(_in(prefix, p), cache / p) for p in want],
                                                  raise_on_missing_files=True)
        return cache

    def push(self, ref: str, files: Mapping[str, Path], message: str, revision: str | None) -> str | None:
        import huggingface_hub

        bucket, prefix = _split_hub(ref, self.name)
        huggingface_hub.batch_bucket_files(bucket, add=[(src, _in(prefix, rel)) for rel, src in files.items()])
        return None


def _read_yaml(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_book(path: Path) -> Any:
    from venngram.book import AddressBook

    return AddressBook.from_file(path)


def _load_recipe(path: Path) -> Any:
    from venngram.recipe import load_recipe

    return load_recipe(path)


def _load_map(path: Path) -> Any:
    from venngram.maps.schema import load_component_map

    return load_component_map(path)


def _load_adapter_listing(path: Path) -> Any:
    from venngram.sources import AdapterListing

    return AdapterListing.from_file(path)


def _load_rules(path: Path) -> Any:
    from venngram.maps.rules import load_rules

    return load_rules(path)


register_store("local", LocalStore(), versioned=False)
register_store("hub-model", lambda: HubRepoStore("model"), versioned=True)
register_store("hub-dataset", lambda: HubRepoStore("dataset"), versioned=True)
register_store("hub-bucket", HubBucketStore(), versioned=False)
for _kind, _loader in (("address_book", _load_book), ("recipe", _load_recipe), ("component_map", _load_map),
                       ("rules", _load_rules), ("verification_record", _read_yaml),
                       ("format_adapter", _load_adapter_listing)):
    register_loader(_kind, _loader)


__all__ = [
    "cache_dir",
    "fetch",
    "fetch_manifest",
    "get_store",
    "HubBucketStore",
    "HubRepoStore",
    "is_reference",
    "load",
    "LocalStore",
    "pin",
    "publish",
    "Reference",
    "register_loader",
    "register_store",
    "Store",
]
