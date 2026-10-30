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
Registry
^^^^^^^^

The **registry** is an index over venngram documents wherever they are published, not a container of them: one row per
document, each naming the document by an exact reference (a commit and a sha256), so documents stay with their
authors and the index can be rebuilt from them at any time.

**Federation.** An index is built by a crawler (:func:`crawl`) from a list of manifest locations, and optionally from
every Hub repository tagged ``venngram`` (:func:`discover_hub`). A component map drafted from a module tree, written by
hand, or derived from any map source is a row like any other, with its source recorded, so maps for arbitrary
architectures stay discoverable alongside the seeds. Several maps may serve one architecture; lookup takes the best
verified one (see below), and :func:`rows` lists every candidate.

**Layout** (on any store; one index per location ``<store>:<ref>``)::

    index.yaml                                         # header: schema, name, current generation, sources, counts
    generations/<id>/kind=<kind>/subjects.parquet      # one row per subject of that kind, with counts
    generations/<id>/kind=<kind>/subject=<s>/rows.parquet
    generations/<id>/mined_rules.yaml                  # drafting rules mined from every crawled map

The index is a **partitioned parquet dataset**, partitioned by kind and by subject (the architecture for family-level
documents such as maps and verification records; the model id for checkpoint documents such as books and recipes), so
a lookup reads one small file and a query over the Hub (DuckDB over ``hf://``) reads only its partitions. A rebuild
writes a new generation and then the header, so a reader never sees a half-written index and a removed document leaves
no stale row.

**Lookup** (:func:`lookup_map`, used by :func:`venngram.maps.component_map_for` after maps registered in the running
process): the configured registries in order (:func:`registries`: ``VENNGRAM_REGISTRIES``, else
:data:`DEFAULT_REGISTRIES`; set it empty for none). Within the first registry with a usable row, rows rank by status
(``verified``, then ``partial``, then ``unverified``; ``failed`` and ``invalidated`` are never served) and then by the
order of the sources the index was crawled from. ``strict`` serves only ``verified`` rows.

**Offline.** Each registry's resolved revision is remembered in venngram's cache. When the store cannot be reached
(or ``HF_HUB_OFFLINE`` is set), the last revision is served from the cache with a :class:`RegistryOfflineWarning`
naming it; a registry never fetched before fails by name.

Uses ``pyarrow`` and, for the Hub stores, ``huggingface_hub`` (both venngram dependencies, imported on first use).

.. code-block:: bash

    python -m venngram.registry build --source hub-dataset:<owner>/<repo>/maps --discover-hub --out ./index
    python -m venngram.registry publish ./index hub-dataset:<owner>/<repo>/index
    python -m venngram.registry lookup Qwen3ForCausalLM

.. warning:: This is an experimental API and may change in a future release.
"""
import argparse
import datetime
import json
import os
import sys
import urllib.parse
import uuid
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from venngram.errors import VenngramError
from venngram.store import _safe, cache_dir, fetch_manifest, get_store, load, Reference

INDEX_SCHEMA_VERSION = 1
HEADER_NAME = "index.yaml"

#: The registries consulted when ``VENNGRAM_REGISTRIES`` is unset: the seed index.
DEFAULT_REGISTRIES: tuple[str, ...] = ("hub-dataset:speediedan/venngram-registry/index",)

#: Statuses in serving order; the others are listed but never served.
SERVED = ("verified", "partial", "unverified")
STATUSES = (*SERVED, "failed", "invalidated")

#: Document kinds whose subject is an architecture; the others with a model are subjected by model id.
FAMILY_KINDS = ("component_map", "verification_record", "rules", "drafting_rules")


class RegistryError(VenngramError):
    """A registry cannot be read or built."""


class RegistryOfflineWarning(UserWarning):
    """A registry was served from the cache because its store could not be reached."""


@dataclass(frozen=True)
class Row:
    """One indexed document."""

    kind: str
    subject: str
    reference: str
    """The exact reference (``<store>:<ref>@<commit>:<path>#<sha256>``; no commit on an unversioned store)."""
    sha256: str
    architecture: str | None = None
    model_id: str | None = None
    revision: str | None = None
    status: str = "unverified"
    source: str | None = None
    """What produced the document (a map source's or format's name and version), when it says."""
    location: str = ""
    """The manifest location it was crawled from."""
    rank: int = 0
    """The crawl order of its location (lower is preferred)."""
    tags: tuple[str, ...] = ()


_COLUMNS = ("kind", "subject", "reference", "sha256", "architecture", "model_id", "revision", "status", "source",
            "location", "rank", "tags")


# ----- reading -----------------------------------------------------------------------------------------------------


_STATE: dict[str, tuple[str | None, dict[str, Any]]] = {}
_MAPS: dict[str, Any] = {}
_override: list[str] | None = None


def registries() -> list[str]:
    """The registry locations consulted, in order."""
    if _override is not None:
        return list(_override)
    raw = os.environ.get("VENNGRAM_REGISTRIES")
    if raw is None:
        return list(DEFAULT_REGISTRIES)
    return [r for r in raw.replace(",", " ").split() if r]


def set_registries(locations: Sequence[str] | None) -> None:
    """Use these registries in this process (``None`` returns to ``VENNGRAM_REGISTRIES`` and the default)."""
    global _override
    _override = list(locations) if locations is not None else None
    clear_cache()


def clear_cache() -> None:
    """Forget resolved revisions, headers and loaded maps (the next lookup re-reads each registry)."""
    _STATE.clear()
    _MAPS.clear()


def _split(location: str) -> tuple[str, str, str | None]:
    store, sep, rest = location.partition(":")
    if not sep or not rest:
        raise RegistryError(f"`{location}` is not a registry location; it is `<store>:<ref>[@<revision>]`.")
    ref, _, revision = rest.partition("@")
    return store, ref, revision or None


def _offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes", "on")


def _state(location: str) -> tuple[str | None, dict[str, Any]]:
    """The resolved revision and header of a registry, once per process; the last known one when offline."""
    if location in _STATE:
        return _STATE[location]
    store, ref, pinned = _split(location)
    s = get_store(store)
    memo = cache_dir() / "registry" / _safe(location) / "last_revision"
    try:
        if _offline() and s.versioned and pinned is None:
            raise RegistryError("HF_HUB_OFFLINE is set")
        revision = s.resolve_revision(ref, pinned)
        header = _header(location, revision)
    except Exception as err:  # the store's own errors vary (network, auth, missing repository)
        if not s.versioned or not memo.is_file():
            raise RegistryError(f"{location}: the registry cannot be read ({type(err).__name__}: {err}) and none was "
                                "fetched before; register maps in process or point VENNGRAM_REGISTRIES elsewhere.") \
                from err
        revision = memo.read_text(encoding="utf-8").strip()
        warnings.warn(f"{location}: serving revision {revision} from the cache ({type(err).__name__}: {err}).",
                      RegistryOfflineWarning, stacklevel=3)
        header = _header(location, revision)
    if revision is not None:
        memo.parent.mkdir(parents=True, exist_ok=True)
        memo.write_text(revision, encoding="utf-8")
    _STATE[location] = (revision, header)
    return _STATE[location]


def _fetch(location: str, revision: str | None, path: str) -> Path | None:
    """One file of a registry, or ``None`` when the registry has no such file."""
    store, ref, _ = _split(location)
    s = get_store(store)
    cache = cache_dir() / "registry" / _safe(location) / _safe(revision or "local")
    try:
        root = s.fetch(ref, [path], revision, cache)
    except Exception as err:
        if type(err).__name__ in ("EntryNotFoundError", "RemoteEntryNotFoundError", "LocalEntryNotFoundError",
                                  "FileNotFoundError"):
            return None
        raise
    local = root / path
    return local if local.is_file() else None


def _header(location: str, revision: str | None) -> dict[str, Any]:
    path = _fetch(location, revision, HEADER_NAME)
    if path is None:
        raise RegistryError(f"{location}: no {HEADER_NAME}; build one with `python -m venngram.registry build`.")
    with open(path, encoding="utf-8") as fh:
        header = yaml.safe_load(fh)
    if not isinstance(header, Mapping) or header.get("kind") != "index" \
            or header.get("schema_version") != INDEX_SCHEMA_VERSION or not header.get("current"):
        raise RegistryError(f"{location}: {HEADER_NAME} is not a schema {INDEX_SCHEMA_VERSION} index header.")
    return dict(header)


def _partition(kind: str, subject: str | None = None) -> str:
    if subject is None:
        return f"kind={kind}/subjects.parquet"
    return f"kind={kind}/subject={urllib.parse.quote(subject, safe='')}/rows.parquet"


def _parquet() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as err:
        raise RegistryError("reading a registry index needs pyarrow, a venngram dependency: reinstall venngram, or "
                            "register maps in process.") from err
    return pq


def _read_rows(path: Path) -> list[Row]:
    pq = _parquet()
    out = []
    for r in pq.read_table(path).to_pylist():
        r["tags"] = tuple(r.get("tags") or ())
        out.append(Row(**{k: r.get(k) for k in _COLUMNS if k in r}))
    return out


def rows(kind: str, subject: str, *, locations: Sequence[str] | None = None) -> list[Row]:
    """Every indexed document of ``kind`` about ``subject`` (an architecture or a model id), registry by registry in
    order, best first within each."""
    out: list[Row] = []
    for location in (registries() if locations is None else locations):
        revision, header = _state(location)
        path = _fetch(location, revision, f"{header['current']}/{_partition(kind, subject)}")
        if path is not None:
            out += sorted(_read_rows(path), key=_order)
    return out


def subjects(kind: str, *, locations: Sequence[str] | None = None) -> list[str]:
    """Every subject that has a document of ``kind`` in the registries (one small file per registry)."""
    pq = _parquet()
    seen: dict[str, None] = {}
    for location in (registries() if locations is None else locations):
        revision, header = _state(location)
        path = _fetch(location, revision, f"{header['current']}/{_partition(kind)}")
        if path is not None:
            seen.update(dict.fromkeys(pq.read_table(path).column("subject").to_pylist()))
    return sorted(seen)


def _order(row: Row) -> tuple[int, int]:
    return (SERVED.index(row.status) if row.status in SERVED else len(SERVED), row.rank)


def lookup_map(architecture: str, *, strict: bool = False) -> Any:
    """The best component map for ``architecture`` in the configured registries, or ``None`` when none has one.

    Raises:
        RegistryError: If a registry cannot be read (and is not in the cache).
    """
    for location in registries():
        candidates = [r for r in rows("component_map", architecture, locations=[location]) if r.status in SERVED]
        if strict:
            candidates = [r for r in candidates if r.status == "verified"]
        if candidates:
            best = candidates[0]
            if best.reference not in _MAPS:
                _MAPS[best.reference] = load(best.reference)
            return _MAPS[best.reference]
    return None


def mined_rules(*, locations: Sequence[str] | None = None) -> list[Any]:
    """The drafting rules each registry mined from its maps (for :mod:`venngram.maps.drafter`)."""
    from venngram.maps.drafter import load_drafting_rules

    out = []
    for location in (registries() if locations is None else locations):
        revision, header = _state(location)
        path = _fetch(location, revision, f"{header['current']}/mined_rules.yaml")
        if path is not None:
            out.append(load_drafting_rules(path))
    return out


# ----- building ----------------------------------------------------------------------------------------------------


def discover_hub(*, tag: str = "venngram", limit: int | None = None) -> list[str]:
    """Manifest locations in Hub repositories tagged ``tag`` (model repositories with ``library_name: venngram`` carry
    it too): one location per directory holding a ``venngram.yaml``."""
    from huggingface_hub import HfApi

    api, out = HfApi(), []
    for repo_type, listing in (("model", api.list_models(filter=tag, limit=limit)),
                               ("dataset", api.list_datasets(filter=tag, limit=limit))):
        for info in listing:
            for path in api.list_repo_files(info.id, repo_type=repo_type):
                if path == "venngram.yaml" or path.endswith("/venngram.yaml"):
                    directory = path[: -len("venngram.yaml")].rstrip("/")
                    out.append(f"hub-{repo_type}:{info.id}" + (f"/{directory}" if directory else ""))
    return out


def _subject_of(kind: str, doc: Any) -> tuple[str | None, str | None, str | None]:
    """``(architecture, model_id, revision)`` a document is about, read from its raw mapping."""
    if not isinstance(doc, Mapping):
        return None, None, None
    if kind in ("component_map", "verification_record"):
        model = None
        source = (doc.get("config") or {}).get("source") if kind == "verification_record" else None
        if isinstance(source, str) and source.startswith("hf:"):
            model = source[3:]
        return doc.get("architecture"), model, None
    if kind == "address_book":
        models = (doc.get("applies_to") or {}).get("models") or []
        first = models[0] if models and isinstance(models[0], Mapping) else {}
        return None, first.get("id"), first.get("revision")
    if kind == "recipe":
        model = doc.get("model") or {}
        return None, model.get("id"), model.get("revision")
    return None, None, None


def _source_of(doc: Any) -> str | None:
    prov = doc.get("provenance") if isinstance(doc, Mapping) else None
    if isinstance(prov, Mapping) and prov.get("source"):
        return f"{prov['source']} {prov.get('version', '')}".strip()
    return None


def crawl(locations: Iterable[str]) -> list[Row]:
    """One row per document in each manifest location, in order (the order ranks rows for lookup).

    A component map's status comes from verification records crawled beside it (same location and architecture):
    the best record's status, else ``unverified``. A location that cannot be read is skipped with a warning.
    """
    out: list[Row] = []
    for rank, location in enumerate(dict.fromkeys(locations)):
        try:
            store, ref, revision = _split(location)
            manifest, root, resolved = fetch_manifest(store, ref, revision)
        except (ValueError, OSError) as err:
            warnings.warn(f"{location}: not crawled ({err}).", stacklevel=2)
            continue
        here: list[Row] = []
        for entry in manifest.documents:
            if entry.kind == "units" or entry.sha256 is None:
                continue
            reference = str(Reference(store, ref, entry.path, resolved, entry.sha256))
            doc = None
            if entry.path.endswith((".yaml", ".yml")):
                local = root / entry.path
                if not local.is_file():
                    local = Path(get_store(store).fetch(ref, [entry.path], resolved, root)) / entry.path
                with open(local, encoding="utf-8") as fh:
                    doc = yaml.safe_load(fh)
            arch, model, model_rev = _subject_of(entry.kind, doc)
            subject = arch if entry.kind in FAMILY_KINDS else model
            status = doc.get("status") if entry.kind == "verification_record" and isinstance(doc, Mapping) else None
            here.append(Row(entry.kind, subject or "_", reference, entry.sha256, arch, model, model_rev,
                            status if status in STATUSES else "unverified", _source_of(doc), location, rank))
        best: dict[str, str] = {}
        for r in here:
            if r.kind == "verification_record" and r.architecture:
                prev = best.get(r.architecture)
                if prev is None or STATUSES.index(r.status) < STATUSES.index(prev):
                    best[r.architecture] = r.status
        out += [Row(**{**asdict(r), "status": best.get(r.architecture or "", "unverified")})
                if r.kind == "component_map" else r for r in here]
    return out


def write_index(index_rows: Sequence[Row], out: str | Path, *, name: str, sources: Sequence[str] = ()) -> Path:
    """Write ``index_rows`` as a new generation under ``out`` and point the header at it. Returns the header path."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from venngram.maps.drafter import mine_rules

    out = Path(out)
    generation = f"generations/{datetime.datetime.now(datetime.UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    by_kind: dict[str, dict[str, list[Row]]] = {}
    for r in index_rows:
        by_kind.setdefault(r.kind, {}).setdefault(r.subject, []).append(r)
    schema = pa.schema([(c, pa.list_(pa.string()) if c == "tags" else pa.int32() if c == "rank" else pa.string())
                        for c in _COLUMNS])
    counts = {}
    for kind, groups in sorted(by_kind.items()):
        for subject, group in groups.items():
            path = out / generation / _partition(kind, subject)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {c: [list(getattr(r, c)) if c == "tags" else getattr(r, c) for r in group] for c in _COLUMNS}
            pq.write_table(pa.table(data, schema=schema), path)
        summary = pa.table({"subject": sorted(groups), "documents": [len(groups[s]) for s in sorted(groups)]})
        pq.write_table(summary, out / generation / _partition(kind))
        counts[kind] = sum(len(g) for g in groups.values())
    maps = []  # mined in architecture order, so a rebuild over the same maps mines the same rules
    for r in sorted(index_rows, key=lambda r: (r.subject, r.rank)):
        if r.kind == "component_map" and r.status in SERVED:
            try:
                maps.append(load(r.reference))
            except (ValueError, OSError) as err:
                warnings.warn(f"{r.reference}: not mined ({err}).", stacklevel=2)
    if maps:
        doc = mine_rules(maps).to_mapping()
        (out / generation / "mined_rules.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    header = {"schema_version": INDEX_SCHEMA_VERSION, "kind": "index", "name": name, "current": generation,
              "built": f"{datetime.datetime.now(datetime.UTC):%Y-%m-%dT%H:%M:%SZ}", "partitioning": ["kind", "subject"],
              "sources": list(sources), "counts": counts}
    (out / HEADER_NAME).write_text(yaml.safe_dump(header, sort_keys=False), encoding="utf-8")
    return out / HEADER_NAME


def publish_index(root: str | Path, location: str, *,
                  message: str = "publish the venngram registry index") -> str | None:
    """Push the current generation, then the header, to ``location``. Returns the revision the store wrote."""
    root = Path(root)
    with open(root / HEADER_NAME, encoding="utf-8") as fh:
        current = yaml.safe_load(fh)["current"]
    store, ref, _ = _split(location)
    files = {p.relative_to(root).as_posix(): p for p in (root / current).rglob("*") if p.is_file()}
    files[HEADER_NAME] = root / HEADER_NAME
    return get_store(store).push(ref, files, message, None)


# ----- command line ------------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m venngram.registry",
                                     description="Build, publish and query venngram registry indexes.")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="crawl manifest locations into a new index generation")
    b.add_argument("--source", action="append", default=[],
                   help="<store>:<ref>[@<revision>] (repeatable, in rank order)")
    b.add_argument("--discover-hub", action="store_true", help="also crawl every Hub repository tagged venngram")
    b.add_argument("--out", required=True, type=Path)
    b.add_argument("--name", default="venngram")
    p = sub.add_parser("publish", help="push a built index to a location")
    p.add_argument("root", type=Path)
    p.add_argument("location")
    q = sub.add_parser("lookup", help="the map lookup would serve for an architecture, and every candidate")
    q.add_argument("architecture")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            locations = list(dict.fromkeys(list(args.source) + (discover_hub() if args.discover_hub else [])))
            found = crawl(locations)
            print(write_index(found, args.out, name=args.name, sources=locations))
            print(json.dumps({k: sum(r.kind == k for r in found) for k in sorted({r.kind for r in found})}))
        elif args.command == "publish":
            print(publish_index(args.root, args.location))
        else:
            for r in rows("component_map", args.architecture):
                print(f"{r.status:<11} {r.source or '-':<24} {r.reference}")
            cmap = lookup_map(args.architecture)
            print("serves:", cmap.source if cmap is not None else None)
    except (ValueError, OSError) as err:
        print(f"{args.command}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["clear_cache", "crawl", "DEFAULT_REGISTRIES", "discover_hub", "lookup_map", "main", "mined_rules",
           "publish_index", "RegistryError", "RegistryOfflineWarning", "registries", "Row", "rows", "set_registries",
           "subjects", "write_index"]
