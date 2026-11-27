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
"""Golden fixtures of every published schema, read by today's readers and migrated to the current schema."""
import warnings
from pathlib import Path

import pytest
import yaml

from venngram import AddressError, NewerSchemaError
from venngram.book import AddressBook
from venngram.migrate import kind_of, main, migrate
from venngram.recipe import Recipe

SCHEMAS = Path(__file__).parent / "schemas"
FIXTURES = sorted(SCHEMAS.glob("*.yaml"))
CURRENT = {"component_map": 7, "overlay_map": 5, "support_record": None, "address_book": None, "rules": 2,
           "op_specs": None, "weight_patch": 1, "graph_subset": 1, "evidence_table": 1}
#: A retired kind migrates to another: a recipe to the book that carries it as a treatment.
BECOMES = {"recipe": "address_book"}
MODEL = {"id": "Qwen/Qwen3.5-4B", "revision": "abc123"}


def load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_every_published_schema_has_a_fixture():
    names = {p.stem for p in FIXTURES}
    assert {f"component_map_v{v}" for v in range(1, 8)} | {"overlay_map_v4", "overlay_map_v5", "recipe_v1",
                                                            "recipe_v2", "recipe_v3", "recipe_v4", "recipe_v5",
                                                            "support_record_v1",
                                                            "support_record_v2", "support_record_v3",
                                                            "support_record_v4", "support_record_v5",
                                                            "support_record_v6", "support_record_v7",
                                                            "support_record_v8",
                                                            "graph_subset_v1", "evidence_table_v1",
                                                            "address_book_v1",
                                                            "address_book_v2", "address_book_v3",
                                                            "address_book_v4", "address_book_v5",
                                                            "address_book_v6", "address_book_v7",
                                                            "address_book_v8", "address_book_v9",
                                                            "address_book_v10", "address_book_v11", "address_book_v12",
                                                            "rules_v2",
                                                            "op_specs_v1", "op_specs_v2", "op_specs_v3",
                                                            "weight_patch_v1"} == names


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_a_fixture_migrates_to_a_current_document_once(path):
    doc = load(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # a schema 1 recipe reads with a deprecation warning
        migrated, steps = migrate(doc, source=path.name, base_dir=path.parent)
    kind = kind_of(doc)
    assert kind == path.stem.rsplit("_v", 1)[0]
    if kind in BECOMES:
        assert kind_of(migrated) == BECOMES[kind] and steps[-1].startswith("recipe -> address book")
        kind = BECOMES[kind]
    else:
        assert bool(steps) == (migrated["schema_version"] != doc["schema_version"] or "facts" in doc)
    if CURRENT[kind] is not None:
        assert migrated["schema_version"] == CURRENT[kind]
    again, more = migrate(migrated, source=path.name, base_dir=path.parent)
    assert again == migrated and more == []  # a current document is left as it is
    assert load(path) == doc  # never in place


def test_the_steps_are_the_documented_changes():
    migrated, steps = migrate(load(SCHEMAS / "component_map_v1.yaml"))
    assert migrated["properties"] == {"golden_note": "schema 1"} and "facts" not in migrated
    assert steps == ["component map 1 -> 2: `facts` renamed `properties`", "component map 2 -> 7: version only"]
    record, steps = migrate(load(SCHEMAS / "support_record_v1.yaml"))
    assert record["sections"] == {"steering": {"ops": ["add", "scale"], "targets": ["units"], "tokens": ["all"]}}
    assert record["name"] == "golden-steerer" and len(steps) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        book, steps = migrate(load(SCHEMAS / "recipe_v1.yaml"))
    assert book == migrate(load(SCHEMAS / "recipe_v2.yaml"))[0]  # the schema 1 fixture migrates as the schema 2 one
    assert steps == ["recipe 1 -> 2: `method` and `treatments` became one stage of ops",
                     "recipe -> address book: the recipe became treatment `old` of its inline book"]


@pytest.mark.parametrize("doc, error, match", [
    ({"schema_version": 99, "architecture": "X", "components": {}}, NewerSchemaError, "newer than this venngram"),
    ({"schema_version": 1, "kind": "index"}, ValueError, "not `index`"),
    ({"name": "nothing"}, ValueError, "not a kind venngram migrates"),
    ({"schema_version": 2, "architecture": "X", "facts": {}, "components": {}}, ValueError, "renamed to `properties`"),
])
def test_what_cannot_be_migrated_is_refused_by_name(doc, error, match):
    with pytest.raises(error, match=match):
        migrate(doc)


def test_the_command_prints_or_writes_the_migrated_document(tmp_path, capsys):
    path = tmp_path / "map.yaml"
    path.write_text((SCHEMAS / "component_map_v1.yaml").read_text())
    assert main([str(path)]) == 0
    out = capsys.readouterr()
    assert yaml.safe_load(out.out)["schema_version"] == 7 and "`facts` renamed" in out.err
    assert load(path)["schema_version"] == 1  # printed, not written
    assert main([str(path), "--write"]) == 0 and load(path)["schema_version"] == 7
    path.write_text("schema_version: 1\nkind: index\n")
    assert main([str(path)]) == 1 and "not `index`" in capsys.readouterr().err


def recipe_doc(book, name="steer", **body):
    return {"schema_version": 2, "kind": "recipe", "name": name, "model": {**MODEL, "revision_source": "pinned"},
            "book": book, "stages": [{"site": "mlp.out.hook_in", "ops": [
                {"op": "add", "target": {"set": "chess"}, "value": {"dose": 0.5}}]}], **body}


BOOK = {"schema_version": 1, "name": "chess", "portability": "checkpoint", "applies_to": {"models": [MODEL]},
        "sets": {"chess": {"point": "mlp.out.hook_in", "layers": [13], "units": [7956]}}}


@pytest.mark.parametrize("fixture", ["recipe_v2", "recipe_v3", "recipe_v4", "recipe_v5"])
def test_a_recipe_becomes_its_inline_books_treatment_with_the_same_stages(fixture):
    doc = load(SCHEMAS / f"{fixture}.yaml")
    recipe = Recipe.from_mapping(doc, source=fixture, base_dir=SCHEMAS)
    migrated, _ = migrate(doc, source=fixture, base_dir=SCHEMAS)
    assert set(migrated["treatments"]) == {doc["name"]} and migrated["sets"] == doc["book"]["sets"]
    needs = {2: 1, 3: 3, 4: 4, 5: 5}[doc["schema_version"]]  # the book schema a treatment of that recipe schema needs
    assert migrated["schema_version"] == max(doc["book"]["schema_version"], needs)
    treatment = AddressBook.from_mapping(migrated, base_dir=SCHEMAS).treatments[doc["name"]]
    assert (treatment.stages, treatment.weights, treatment.gradients) == (recipe.stages, recipe.weights,
                                                                          recipe.gradients)
    assert treatment.to_treatment() == recipe.to_treatment()


def test_a_recipe_naming_its_book_by_path_becomes_that_book_with_the_treatment(tmp_path):
    (tmp_path / "books").mkdir()
    (tmp_path / "books" / "chess.yaml").write_text(yaml.safe_dump({**BOOK, "treatments": {"other": {
        "model": MODEL, "stages": [{"site": "mlp.out.hook_in", "ops": [{"op": "scale", "target": {"set": "chess"},
                                                                       "value": {"dose": 2.0}}]}]}}}))
    (tmp_path / "recipes").mkdir()
    doc = recipe_doc("../books/chess.yaml", operands={"dir": {"values": [1.0, 0.0], "shape": [2]}})
    recipe = Recipe.from_mapping(doc, base_dir=tmp_path / "recipes")
    migrated, steps = migrate(doc, base_dir=tmp_path / "recipes")
    assert steps == ["recipe -> address book: the recipe became treatment `steer` of its book `../books/chess.yaml`"]
    assert {k: v for k, v in migrated.items() if k != "treatments"} == BOOK and set(migrated["treatments"]) == {
        "other", "steer"}
    book = AddressBook.from_mapping(migrated, base_dir=tmp_path / "books")
    assert book.treatments["steer"].stages == recipe.stages and book.treatments["steer"].operands == recipe.operands
    assert migrate(migrated, base_dir=tmp_path / "books") == (migrated, [])


def test_a_path_books_operand_files_are_rewritten_relative_to_the_book(tmp_path):
    from venngram.operands import write_operands

    (tmp_path / "books").mkdir()
    (tmp_path / "books" / "chess.yaml").write_text(yaml.safe_dump(BOOK))
    (tmp_path / "recipes").mkdir()
    sha = write_operands(tmp_path / "recipes" / "dirs.safetensors", {"concept": ([1.0, 0.0], [2])})
    doc = recipe_doc("../books/chess.yaml", operands={"concept": {"file": "dirs.safetensors", "key": "concept",
                                                                  "sha256": sha, "shape": [2]}})
    migrated, _ = migrate(doc, base_dir=tmp_path / "recipes")
    assert migrated["treatments"]["steer"]["operands"]["concept"]["file"] == "../recipes/dirs.safetensors"
    steer = AddressBook.from_mapping(migrated, base_dir=tmp_path / "books").treatments["steer"]
    assert steer.operands["concept"].load(tmp_path / "books") == ((1.0, 0.0), (2,))


def test_a_recipe_whose_book_has_a_treatment_of_its_name_is_refused_by_name(tmp_path):
    body = {"model": MODEL, "stages": [{"site": "mlp.out.hook_in", "ops": [{"op": "add", "target": {"set": "chess"},
                                                                            "value": {"dose": 1.0}}]}]}
    with pytest.raises(ValueError, match="its inline book already has a treatment `steer`"):
        migrate(recipe_doc({**BOOK, "treatments": {"steer": body}}))
    (tmp_path / "chess.yaml").write_text(yaml.safe_dump({**BOOK, "treatments": {"steer": body}}))
    with pytest.raises(ValueError, match="its book `chess.yaml` already has a treatment `steer`"):
        migrate(recipe_doc("chess.yaml"), base_dir=tmp_path)
    nameless = recipe_doc(BOOK)
    del nameless["name"]
    with pytest.raises(ValueError, match="named by its `name`"):
        migrate(nameless)
    with pytest.raises(AddressError, match="is not a valid treatment name"):  # the book's own reader refuses it
        migrate(recipe_doc(BOOK, name="a recipe"))


def test_the_command_will_not_write_a_book_away_from_its_directory(tmp_path, capsys):
    (tmp_path / "books").mkdir()
    (tmp_path / "books" / "chess.yaml").write_text(yaml.safe_dump(BOOK))
    (tmp_path / "recipes").mkdir()
    path = tmp_path / "recipes" / "steer.yaml"
    path.write_text(yaml.safe_dump(recipe_doc("../books/chess.yaml")))
    assert main([str(path), "--write"]) == 1 and "belongs beside it" in capsys.readouterr().err
    assert load(path)["kind"] == "recipe"  # left as it was
    assert main([str(path)]) == 0 and "steer" in yaml.safe_load(capsys.readouterr().out)["treatments"]
    beside = tmp_path / "books" / "steer.yaml"
    beside.write_text(yaml.safe_dump(recipe_doc("chess.yaml")))
    assert main([str(beside), "--write"]) == 0 and set(load(beside)["treatments"]) == {"steer"}
