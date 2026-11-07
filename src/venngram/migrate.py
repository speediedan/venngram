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
overlay map         4 -> 5     none (schema 5 adds templates)
recipe              1 -> 2     one ``method`` and its ``treatments`` become one stage of ops
                               (:func:`venngram.recipe.migrate`)
support record      1 -> 2     the top-level steering contract moves under ``sections.steering``
==================  =========  ==================================================================================

An address book (schema 2 only adds ``of``), a support record at schema 2 or 3, and a rule set are current already.
A document newer than this venngram reads is refused as :class:`~venngram.errors.NewerSchemaError`.

.. warning:: This is an experimental API and may change in a future release.
"""
import argparse
import copy
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


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
    from venngram.recipe import migrate as migrate_recipe, Recipe, RECIPE_SCHEMA_VERSION

    if doc.get("schema_version") == 1:
        doc = migrate_recipe(doc, source=source)
        steps.append(f"recipe 1 -> {RECIPE_SCHEMA_VERSION}: `method` and `treatments` became one stage of ops")
    Recipe.from_mapping(doc, source=source)
    return doc


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

    return AddressBook.from_mapping(doc, source=source)


def _rules(doc: dict[str, Any], source: str) -> Any:
    from venngram.maps.rules import rules_from_document

    return rules_from_document(doc, source)


_STEPS: dict[str, Callable[[dict[str, Any], str, list[str]], dict[str, Any]]] = {
    "component_map": _component_map, "overlay_map": _overlay_map, "recipe": _recipe,
    "support_record": _support_record, "address_book": _current(_book), "rules": _current(_rules)}


def migrate(doc: Mapping[str, Any], *, source: str = "<document>") -> tuple[dict[str, Any], list[str]]:
    """``doc`` rewritten to its kind's current schema, and the steps taken (empty when it is current already).

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
    return _STEPS[kind](copy.deepcopy(dict(doc)), source, steps), steps


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m venngram.migrate",
                                     description="Rewrite a venngram document to its kind's current schema.")
    parser.add_argument("path", help="a YAML document")
    parser.add_argument("--write", action="store_true", help="rewrite the file in place instead of printing it")
    args = parser.parse_args(argv)
    path = Path(args.path)
    try:
        migrated, steps = migrate(yaml.safe_load(path.read_text(encoding="utf-8")), source=str(path))
    except (ValueError, OSError) as err:
        print(f"{path}: {err}", file=sys.stderr)
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
