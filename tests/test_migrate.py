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

from venngram import NewerSchemaError
from venngram.migrate import kind_of, main, migrate

SCHEMAS = Path(__file__).parent / "schemas"
FIXTURES = sorted(SCHEMAS.glob("*.yaml"))
CURRENT = {"component_map": 4, "overlay_map": 5, "recipe": 2, "support_record": None, "address_book": None,
           "rules": 2, "op_specs": 1}


def load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_every_published_schema_has_a_fixture():
    names = {p.stem for p in FIXTURES}
    assert {f"component_map_v{v}" for v in (1, 2, 3, 4)} | {"overlay_map_v4", "overlay_map_v5", "recipe_v1",
                                                            "recipe_v2", "support_record_v1", "support_record_v2",
                                                            "support_record_v3", "address_book_v1",
                                                            "address_book_v2", "rules_v2", "op_specs_v1"} == names


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_a_fixture_migrates_to_a_current_document_once(path):
    doc = load(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # a schema 1 recipe reads with a deprecation warning
        migrated, steps = migrate(doc, source=path.name)
    kind = kind_of(doc)
    assert kind == path.stem.rsplit("_v", 1)[0]
    if CURRENT[kind] is not None:
        assert migrated["schema_version"] == CURRENT[kind]
    assert bool(steps) == (migrated["schema_version"] != doc["schema_version"] or "facts" in doc)
    again, more = migrate(migrated, source=path.name)
    assert again == migrated and more == []  # a current document is left as it is
    assert load(path) == doc  # never in place


def test_the_steps_are_the_documented_changes():
    migrated, steps = migrate(load(SCHEMAS / "component_map_v1.yaml"))
    assert migrated["properties"] == {"golden_note": "schema 1"} and "facts" not in migrated
    assert steps == ["component map 1 -> 2: `facts` renamed `properties`", "component map 2 -> 4: version only"]
    record, steps = migrate(load(SCHEMAS / "support_record_v1.yaml"))
    assert record["sections"] == {"steering": {"ops": ["add", "scale"], "targets": ["units"], "tokens": ["all"]}}
    assert record["name"] == "golden-steerer" and len(steps) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        recipe, _ = migrate(load(SCHEMAS / "recipe_v1.yaml"))
    assert recipe == load(SCHEMAS / "recipe_v2.yaml")  # the schema 1 fixture migrates to the schema 2 one


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
    assert yaml.safe_load(out.out)["schema_version"] == 4 and "`facts` renamed" in out.err
    assert load(path)["schema_version"] == 1  # printed, not written
    assert main([str(path), "--write"]) == 0 and load(path)["schema_version"] == 4
    path.write_text("schema_version: 1\nkind: index\n")
    assert main([str(path)]) == 1 and "not `index`" in capsys.readouterr().err
