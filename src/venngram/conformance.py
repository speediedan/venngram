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
Document Conformance
^^^^^^^^^^^^^^^^^^^^

Checks a published directory of venngram documents from the outside, the way a reader meets it: fetch its manifest by
reference at one revision, load every document through its kind's loader (which validates it against the core schema
and checks every file it needs by sha256), and check each recipe against consumers' support records
(:class:`venngram.ops.Support`), so a refusal surfaces before any consumer runs the recipe.

.. code-block:: bash

    python -m venngram.conformance hub-dataset:speediedan/venngram-examples/maps
    python -m venngram.conformance hub-model:<owner>/<repo>/<dir>@<revision> --support steerer.yaml

A location is ``<store>:<ref>[@<revision>]`` (a document reference without its path). ``--support`` takes a
``support_record`` file or a document reference to one, and may repeat. The command exits non-zero when a document
fails to load or a recipe asks for something a given consumer does not honour. It is meant to run in a format
adapter's CI and on a schedule over published example repositories.

Kinds that are read through another document (``units`` sidecars, through the book that names them) or that have no
loader yet are listed, not loaded. (Conformance of a tunable view to the tuning protocol is a different check:
:mod:`venngram.tuning.conformance`.)

.. warning:: This is an experimental API and may change in a future release.
"""
import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from venngram.errors import VenngramError
from venngram.ops import check_support, Support
from venngram.store import fetch_manifest, get_store, is_reference, load, loader_kinds, Reference


@dataclass(frozen=True)
class DocumentResult:
    """One document's outcome: whether it loaded and, for a recipe, each consumer's refusals by consumer name."""

    path: str
    kind: str
    error: str | None = None
    refusals: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and not any(self.refusals.values())


@dataclass(frozen=True)
class ConformanceReport:
    """The outcome for one location at one revision."""

    location: str
    revision: str | None
    documents: tuple[DocumentResult, ...]
    unloaded: tuple[tuple[str, str], ...] = ()
    """``(kind, path)`` of entries not loaded on their own (sidecars, kinds without a loader, unknown kinds)."""

    @property
    def ok(self) -> bool:
        return bool(self.documents) and all(d.ok for d in self.documents)

    def render(self) -> str:
        at = f" @ {self.revision}" if self.revision else ""
        n = len(self.documents)
        lines = [f"{self.location}{at}: {n} document{'' if n == 1 else 's'}, "
                 f"{sum(not d.ok for d in self.documents)} failing"]
        for d in self.documents:
            lines.append(f"  {'ok  ' if d.ok else 'FAIL'}  {d.kind:<20} {d.path}")
            if d.error:
                lines.append(f"        {d.error}")
            for consumer, refusals in d.refusals.items():
                lines += [f"        refused by {consumer}: {r}" for r in refusals]
        lines += [f"  --    {kind:<20} {path} (not loaded on its own)" for kind, path in self.unloaded]
        if not self.documents:
            lines.append("  FAIL  the manifest lists no document that can be loaded")
        return "\n".join(lines)


def parse_location(text: str) -> tuple[str, str, str | None]:
    """``<store>:<ref>[@<revision>]`` as ``(store, ref, revision)``.

    Raises:
        VenngramError: If the store is not registered.
    """
    store, sep, rest = text.partition(":")
    if not sep or not rest:
        raise VenngramError(f"`{text}` is not a location; it is `<store>:<ref>[@<revision>]`.")
    get_store(store)
    ref, _, revision = rest.partition("@")
    return store, ref, revision or None


def check(location: str, *, supports: Sequence[Support] = ()) -> ConformanceReport:
    """Fetch the manifest at ``location``, load every document at that one revision, and check every recipe against
    each of ``supports``.

    Raises:
        VenngramError: If the location is malformed or its manifest cannot be fetched or read (a document that fails
            to load is reported, not raised).
    """
    store, ref, revision = parse_location(location)
    manifest, _, resolved = fetch_manifest(store, ref, revision)
    loadable = loader_kinds()
    results, unloaded = [], []
    for entry in manifest.documents:
        if entry.kind not in loadable:
            unloaded.append((entry.kind, entry.path))
            continue
        try:
            doc = load(Reference(store, ref, entry.path, resolved, entry.sha256))
        except (ValueError, OSError) as err:  # VenngramError is a ValueError
            results.append(DocumentResult(entry.path, entry.kind, error=f"{type(err).__name__}: {err}"))
            continue
        refusals = {}
        if entry.kind == "recipe":
            refusals = {(s.name or f"support {i}"): tuple(check_support(doc, s)) for i, s in enumerate(supports)}
        results.append(DocumentResult(entry.path, entry.kind, refusals=refusals))
    unloaded += [(e.kind, e.path) for e in manifest.skipped]
    return ConformanceReport(location, resolved, tuple(results), tuple(unloaded))


def read_support(text: str) -> Support:
    """A support record from a file path or a document reference.

    Raises:
        VenngramError: If a reference does not name a ``support_record``.
    """
    if is_reference(text):
        doc = load(text)
        if not isinstance(doc, Support):
            raise VenngramError(f"{text}: not a support record (loaded {type(doc).__name__}).")
        return doc
    return Support.from_file(text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m venngram.conformance",
                                     description="Load every document a published venngram manifest lists, and check "
                                                 "its recipes against consumers' support records.")
    parser.add_argument("locations", nargs="+", help="<store>:<ref>[@<revision>], one or more")
    parser.add_argument("--support", action="append", default=[], metavar="FILE_OR_REFERENCE",
                        help="a support_record to check every recipe against (repeatable)")
    args = parser.parse_args(argv)
    try:
        supports = [read_support(s) for s in args.support]
    except (ValueError, OSError) as err:
        print(f"--support: {err}", file=sys.stderr)
        return 2
    failed = False
    for location in args.locations:
        try:
            report = check(location, supports=supports)
        except (ValueError, OSError) as err:
            print(f"{location}: FAIL  {err}")
            failed = True
            continue
        print(report.render())
        failed |= not report.ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["check", "ConformanceReport", "DocumentResult", "main", "parse_location", "read_support"]
