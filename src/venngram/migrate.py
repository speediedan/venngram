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
Document Migration
^^^^^^^^^^^^^^^^^^

Rewrites a document written against an older schema of its kind into the current one, one schema step at a time,
then reads the result with the kind's own reader, so a migration never writes a document this venngram would refuse.
Readers keep accepting their whole window (see the README's "Versions and evolution"), so migrating is never required
to read a document; it is how a publisher brings one up to date.

.. code-block:: bash

    python -m venngram.migrate old_map.yaml            # the migrated document on stdout, the steps on stderr
    python -m venngram.migrate old_map.yaml --write    # in place

The steps, by kind:

==================  =========  ==================================================================================
kind                step       change
==================  =========  ==================================================================================
component map       1 -> 2     ``facts`` is renamed ``properties``
component map       2 -> 3     none (schema 3 adds ``optional`` rows)
component map       3 -> 4     none (schema 4 adds the separate overlay shape)
component map       4 -> 5     none (schema 5 adds per-row ``layout``, which the map builder measures)
component map       5 -> 6     none (schema 6 adds per-row ``unit_block``, which the map builder measures)
component map       6 -> 7     none (schema 7 adds an optional execution map)
overlay map         4 -> 5     none (schema 5 adds templates)
recipe              1 -> 2     one ``method`` and its ``treatments`` become one stage of ops
                               (:func:`venngram.recipe.migrate`)
recipe              -> book    recipe -> address book: the recipe becomes its book's treatment (named by its
                               ``name``), since the ``kind: recipe`` document is retired
support record      1 -> 2     the top-level steering contract moves under ``sections.steering``
==================  =========  ==================================================================================

The ``kind: recipe`` document is retired (a treatment is written in its book), so a recipe of any schema is rewritten
as an address book: its inline ``book``, or the book its ``book`` path names (relative to ``base_dir``), with the recipe
added under ``treatments`` (:meth:`venngram.recipe.Recipe.to_treatment`) at the book schema the treatment needs (3 for
``weights``, 4 for ``gradients`` or a ``readout``, 5 for a graph target). A book that already has a treatment of the
recipe's name is refused by name. A book read from a path keeps its own paths, which resolve from its directory, so the
treatment's operand files are rewritten relative to it, and the command refuses ``--write`` for such a recipe when the
book sits in another directory: write the output beside the book instead.

An address book (schema 2 only adds ``of``, schema 3 treatments with ``weights``), a support record at schema 2 to 4, a
rule set, an ``op_specs`` document (2 adds parameter targets), a ``weight_patch`` document and a ``graph_subset``
document are current already. A book that names graphs reads them relative to ``base_dir`` (the document's directory,
from the command line). A document newer than this venngram reads is refused as
:class:`~venngram.errors.NewerSchemaError`.

.. warning:: This is an experimental API and may change in a future release.
"""
import argparse
import copy
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import yaml

#: The directory a document being migrated sits in, so a book's graphs resolve (:func:`migrate`'s ``base_dir``).
_BASE: ContextVar[Path | None] = ContextVar("venngram_migrate_base", default=None)


def kind_of(doc: Mapping[str, Any]) -> str:
    """The kind of a document: its ``kind`` key, or for the kinds that carry none, its shape (``overlay`` for an
    overlay map, ``architecture`` and ``components`` for a component map, ``sets`` for an address book)."""
    if isinstance(doc.get("kind"), str):
        return doc["kind"]
    if "overlay" in doc:
        return "overlay_map"
    if "architecture" in doc and "components" in doc:
        return "component_map"
    if "sets" in doc:
        return "address_book"
    raise ValueError("the document is not a kind venngram migrates: it has no `kind`, and is not a component map, an "
                     "overlay map or an address book by its keys.")


def _component_map(doc: dict[str, Any], source: str, steps: list[str]) -> dict[str, Any]:
    from venngram.maps.schema import COMPONENT_MAP_SCHEMA_VERSION, from_document

    from_document(doc, source)  # refuse a malformed or too-new map before rewriting it
    version = doc["schema_version"]
    if version < 2:
        if "facts" in doc:
            doc["properties"] = doc.pop("facts")
        steps.append("component map 1 -> 2: `facts` renamed `properties`")
    if version < COMPONENT_MAP_SCHEMA_VERSION:
        steps.append(f"component map {max(version, 2)} -> {COMPONENT_MAP_SCHEMA_VERSION}: version only")
        doc["schema_version"] = COMPONENT_MAP_SCHEMA_VERSION
    from_document(doc, source)
    return doc


def _overlay_map(doc: dict[str, Any], source: str, steps: list[str]) -> dict[str, Any]:
    from venngram.maps.overlay import OVERLAY_SCHEMA_VERSION, overlay_from_document

    overlay_from_document(doc, source)
    if doc["schema_version"] < OVERLAY_SCHEMA_VERSION:
        steps.append(f"overlay map {doc['schema_version']} -> {OVERLAY_SCHEMA_VERSION}: version only")
        doc["schema_version"] = OVERLAY_SCHEMA_VERSION
    overlay_from_document(doc, source)
    return doc


def _recipe(doc: dict[str, Any], source: str, steps: list[str]) -> dict[str, Any]:
    from venngram.book import AddressBook, GRADIENTS_SCHEMA, WEIGHTS_SCHEMA
    from venngram.recipe import migrate as migrate_recipe, Recipe, RECIPE_SCHEMA_VERSION

    if doc.get("schema_version") == 1:
        doc = migrate_recipe(doc, source=source)
        steps.append(f"recipe 1 -> {RECIPE_SCHEMA_VERSION}: `method` and `treatments` became one stage of ops")
    base = _BASE.get()
    recipe = Recipe.from_mapping(doc, source=source, base_dir=base)
    name = doc.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{source}: a recipe becomes a treatment named by its `name`, and this one has none.")
    if isinstance(doc["book"], str):
        book_path = Path(base or ".") / doc["book"]
        with open(book_path, encoding="utf-8") as fh:
            book, book_base, where = yaml.safe_load(fh), book_path.parent, f"its book `{doc['book']}`"
    else:
        book, book_base, where = dict(doc["book"]), base, "its inline book"
    treatments = dict(book.get("treatments") or {})
    if name in treatments:
        raise ValueError(f"{source}: {where} already has a treatment `{name}`, so the recipe cannot become one of that "
                         "name; rename the recipe or that treatment first.")
    needs = [recipe.book.schema_version]  # a graph target needs the book's `graphs`, which are schema 5 already
    if recipe.weights:
        needs.append(WEIGHTS_SCHEMA)
    if recipe.gradients or recipe.readout is not None or any(st.readout is not None for st in recipe.stages):
        needs.append(GRADIENTS_SCHEMA)
    out = {"schema_version": max(needs), **{k: v for k, v in book.items() if k != "schema_version"}}
    out.setdefault("name", name)
    out["treatments"] = {**treatments, name: _rebased(recipe.to_treatment(), base, book_base)}
    AddressBook.from_mapping(out, source=source, base_dir=book_base)
    steps.append(f"recipe -> address book: the recipe became treatment `{name}` of {where}")
    return out


def _rebased(body: dict[str, Any], base: Path | None, book_base: Path | None) -> dict[str, Any]:
    """``body`` with its operands' files relative to ``book_base`` (the book's directory) instead of ``base``."""
    here, there = Path(base or ".").resolve(), Path(book_base or ".").resolve()
    if here == there:
        return body
    for operand in (body.get("operands") or {}).values():
        for key in ("file", "patch"):
            if isinstance(operand.get(key), str):
                operand[key] = Path(os.path.relpath(here / operand[key], there)).as_posix()
    return body


def _book_elsewhere(doc: Any, base: Path) -> Path | None:
    """The directory of a recipe's book named by path, when it is not ``base`` (where ``--write`` would put it)."""
    if not isinstance(doc, Mapping) or doc.get("kind") != "recipe" or not isinstance(doc.get("book"), str):
        return None
    there = (base / doc["book"]).parent
    return None if there.resolve() == base.resolve() else there


def _support_record(doc: dict[str, Any], source: str, steps: list[str]) -> dict[str, Any]:
    from venngram.support import SCHEMA_VERSION, SupportRecord

    SupportRecord.from_mapping(doc, source=source)
    if doc.get("schema_version") == 1:
        head = {k: doc[k] for k in ("name", "description") if k in doc}
        steering = {k: v for k, v in doc.items() if k not in ("schema_version", "kind", "name", "description")}
        doc = {"schema_version": SCHEMA_VERSION, "kind": "support_record", **head, "sections": {"steering": steering}}
        steps.append(f"support record 1 -> {SCHEMA_VERSION}: the steering contract moved under `sections.steering`")
    SupportRecord.from_mapping(doc, source=source)
    return doc


def _current(read: Callable[[dict[str, Any], str], Any]) -> Callable[[dict[str, Any], str, list[str]], Any]:
    def step(doc: dict[str, Any], source: str, steps: list[str]) -> dict[str, Any]:
        read(doc, source)
        return doc

    return step


def _book(doc: dict[str, Any], source: str) -> Any:
    from venngram.book import AddressBook

    return AddressBook.from_mapping(doc, source=source, base_dir=_BASE.get())


def _graph(doc: dict[str, Any], source: str) -> Any:
    from venngram.graphs import GraphSubset

    return GraphSubset.from_mapping(doc, source=source, base_dir=_BASE.get())


def _op_specs(doc: dict[str, Any], source: str) -> Any:
    from venngram.ops import read_op_specs

    return read_op_specs(doc, source=source, register=False)


def _weight_patch(doc: dict[str, Any], source: str) -> Any:
    from venngram.patches import weight_patch_from_mapping

    return weight_patch_from_mapping(doc, source=source, check_file=False)  # migrating a document reads no sidecar


def _rules(doc: dict[str, Any], source: str) -> Any:
    from venngram.maps.rules import rules_from_document

    return rules_from_document(doc, source)


_STEPS: dict[str, Callable[[dict[str, Any], str, list[str]], dict[str, Any]]] = {
    "component_map": _component_map, "overlay_map": _overlay_map, "recipe": _recipe,
    "support_record": _support_record, "address_book": _current(_book), "rules": _current(_rules),
    "op_specs": _current(_op_specs), "weight_patch": _current(_weight_patch), "graph_subset": _current(_graph)}


def migrate(doc: Mapping[str, Any], *, source: str = "<document>",
            base_dir: str | Path | None = None) -> tuple[dict[str, Any], list[str]]:
    """``doc`` rewritten to its kind's current schema, and the steps taken (empty when it is current already).
    ``base_dir`` is the document's directory, which a book's graphs and a graph's sidecars resolve against.

    Raises:
        NewerSchemaError: If the document is newer than this venngram reads.
        ValueError: If its kind has no migration here, or it is malformed (a reader's own refusal, by name).
    """
    if not isinstance(doc, Mapping):
        raise ValueError(f"{source}: a document is a mapping, got {type(doc).__name__}.")
    kind = kind_of(doc)
    if kind not in _STEPS:
        raise ValueError(f"{source}: venngram migrates {sorted(_STEPS)}, not `{kind}`.")
    steps: list[str] = []
    token = _BASE.set(Path(base_dir) if base_dir is not None else None)
    try:
        return _STEPS[kind](copy.deepcopy(dict(doc)), source, steps), steps
    finally:
        _BASE.reset(token)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m venngram.migrate",
                                     description="Rewrite a venngram document to its kind's current schema.")
    parser.add_argument("path", help="a YAML document")
    parser.add_argument("--write", action="store_true", help="rewrite the file in place instead of printing it")
    args = parser.parse_args(argv)
    path = Path(args.path)
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        migrated, steps = migrate(doc, source=str(path), base_dir=path.parent)
    except (ValueError, OSError) as err:
        print(f"{path}: {err}", file=sys.stderr)
        return 1
    elsewhere = _book_elsewhere(doc, path.parent)
    if args.write and elsewhere is not None:
        print(f"{path}: the recipe's book is in {elsewhere}, and its paths resolve from there, so the migrated book "
              "belongs beside it: print it and write it there instead of using --write.", file=sys.stderr)
        return 1
    for step in steps or ["current already"]:
        print(f"{path}: {step}", file=sys.stderr)
    text = yaml.safe_dump(migrated, sort_keys=False)
    if args.write:
        path.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["kind_of", "main", "migrate"]
