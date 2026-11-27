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
import copy

import pytest
import yaml

from venngram import AddressBook
from venngram.conformance import check
from venngram.errors import AddressError, NewerAddressSchemaError
from venngram.tables import check_members, UnitTable, read_table_rows, write_table_rows
from venngram.manifest import build_manifest
from venngram.store import fetch, publish
from venngram.units import _sha256

MODEL = {"id": "Qwen/Qwen3.5-4B", "revision": "pinned-model"}
SITE = "mlp.out.hook_in"
SOURCE = {"name": "example-finder", "repo": "example/finder", "commit": "0f3a9c1", "license": "MIT"}
COLUMNS = {"dose": {"description": "signed steering dose", "measure": "dose_unit"},
           "confirm_hit_rate": {"description": "fraction of confirm prompts that hit", "measure": "fraction"},
           "confirm_n": {"description": "confirm prompts", "dtype": "int"}}
ROWS = [{"layer": 13, "unit": 4810, "dose": 1.5, "confirm_hit_rate": 0.85, "confirm_n": 20},
        {"layer": 15, "unit": 7, "dose": -0.8, "confirm_hit_rate": 0.6, "confirm_n": 12}]


def table_doc(**extra):
    doc = {"schema_version": 1, "kind": "unit_table", "name": "nf_anime", "model": dict(MODEL),
           "source": dict(SOURCE), "columns": copy.deepcopy(COLUMNS), "rows": copy.deepcopy(ROWS)}
    doc.update(extra)
    return doc


def body(**extra):
    doc = table_doc(**extra)
    for key in ("schema_version", "kind", "name"):
        doc.pop(key, None)
    return doc


def steer(table="nf_anime", column="dose", target="found"):
    return {"model": dict(MODEL), "stages": [{"site": SITE, "ops": [
        {"op": "add", "target": {"set": target}, "value": {"from": {"table": table, "column": column}}}]}]}


def book_doc(tables=None, treatment=None, units=None):
    doc = {"schema_version": 13 if tables is not None else 2, "name": "zoo", "portability": "checkpoint",
           "applies_to": {"models": [dict(MODEL)]},
           "sets": {"found": {"point": SITE, "units": units or {13: [4810], 15: [7]}}},
           "treatments": {"steer": treatment or steer()}}
    if tables is not None:
        doc["tables"] = tables
    return doc


def test_a_table_reads_round_trips_and_gives_its_columns_by_member():
    table = UnitTable.from_mapping(table_doc())
    assert table.to_mapping() == table_doc()
    assert table.table() == {"dose": {(13, 4810): 1.5, (15, 7): -0.8},
                             "confirm_hit_rate": {(13, 4810): 0.85, (15, 7): 0.6},
                             "confirm_n": {(13, 4810): 20, (15, 7): 12}}
    assert isinstance(table.column("confirm_n")[(13, 4810)], int) and table.columns["dose"].measure == "dose_unit"
    with pytest.raises(AddressError, match="no column `hits`"):
        table.column("hits")


@pytest.mark.parametrize("change, match", [
    ({"model": {"id": "Qwen/Qwen3.5-4B"}}, "checkpoint the values were measured on"),
    ({"source": {"name": "x", "commit": "abc"}}, "`license`"),
    ({"source": {"name": "x", "license": "MIT"}}, "one of \\['version', 'commit', 'revision'\\]"),
    ({"columns": {}}, "`columns` maps"),
    ({"columns": {"dose": {"measure": "x"}}}, "with a `description`"),
    ({"columns": {"layer": {"description": "x"}}}, "not a valid column name"),
    ({"columns": {**COLUMNS, "p": {"description": "x", "dtype": "bool"}}}, "`dtype` is one of"),
    ({"rows": []}, "at least one"),
    ({"rows": [{**ROWS[0], "extra": 1.0}]}, "not declared columns"),
    ({"rows": [{k: v for k, v in ROWS[0].items() if k != "confirm_n"}]}, "every row gives every column"),
    ({"rows": [ROWS[0], {**ROWS[0], "dose": 2.0}]}, "repeats \\(layer 13, unit 4810\\).*table of its own"),
    ({"rows": [{**ROWS[0], "confirm_n": 2.5}]}, "integer column"),
    ({"rows": [{**ROWS[0], "dose": float("nan")}]}, "finite number"),
    ({"rows": [{**ROWS[0], "dose": True}]}, "finite number"),
    ({"rows": [{**ROWS[0], "unit": -1}]}, "`unit` is an integer >= 0"),
    ({"edges": []}, "unknown keys \\['edges'\\]"),
    ({"key": ["layer", "unit", "pass"]}, "`key` names \\['pass'\\], which this venngram does not know"),
    ({"key": ["unit"]}, "includes `layer` and `unit`"),
    ({"key": ["layer", "unit", "unit"]}, "lists each coordinate once"),
    ({"key": "layer,unit"}, "`key` lists the coordinates"),
    ({"columns": {"stack": {"description": "x"}}}, "not a valid column name"),
    ({"rows": [{**ROWS[0], "stack": "vision"}]}, "gives a `stack`, and the table's key is \\['layer', 'unit'\\]"),
    ({"key": ["stack", "layer", "unit"], "rows": [{**ROWS[0], "stack": 3}]}, "`stack` is a block stack's name"),
    ({"key": ["stack", "layer", "unit"], "rows": [{**ROWS[0], "stack": "vision"}, {**ROWS[0], "stack": "vision"}]},
     "repeats \\(stack vision, layer 13, unit 4810\\)"),
    ({"kind": "graph_subset"}, "kind: unit_table"),
])
def test_a_malformed_table_is_refused_by_name(change, match):
    with pytest.raises(AddressError, match=match):
        UnitTable.from_mapping(table_doc(**change))


def test_a_too_new_table_is_refused_as_newer():
    with pytest.raises(NewerAddressSchemaError, match="newer venngram") as err:
        UnitTable.from_mapping(table_doc(schema_version=2))
    assert err.value.kind == "unit_table"


def test_a_large_table_writes_its_rows_to_a_sidecar_that_reads_back_typed(tmp_path):
    table = UnitTable.from_mapping(table_doc())
    path = table.write(tmp_path / "nf.yaml", threshold=1)
    doc = yaml.safe_load(path.read_text())
    assert doc["rows"] == {"file": "nf.rows.safetensors", "sha256": _sha256(tmp_path / "nf.rows.safetensors")}
    again = UnitTable.from_file(path)
    assert again.table() == table.table() and again.rows_ref == doc["rows"]
    assert isinstance(again.column("confirm_n")[(15, 7)], int) and isinstance(again.column("dose")[(15, 7)], float)
    assert again.to_mapping() == doc  # a sidecar reference round-trips as the reference
    assert read_table_rows(tmp_path / "nf.rows.safetensors") == ROWS
    with pytest.raises(AddressError, match="sha256 differs"):
        read_table_rows(tmp_path / "nf.rows.safetensors", sha256="0" * 64)
    (tmp_path / "bad.safetensors").write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ")
    with pytest.raises(AddressError, match="not a venngram unit table row sidecar"):
        read_table_rows(tmp_path / "bad.safetensors")


def test_a_sidecar_value_of_the_wrong_type_for_its_column_is_refused(tmp_path):
    floats = {**COLUMNS, "confirm_n": {"description": "confirm prompts"}}  # written as a float column
    table = UnitTable.from_mapping(table_doc(columns=floats))
    sha = write_table_rows(tmp_path / "rows.safetensors", table)
    with pytest.raises(AddressError, match="`value.confirm_n` is I64 with 2 rows, got F64"):  # the document's int
        UnitTable.from_mapping(table_doc(rows={"file": "rows.safetensors", "sha256": sha}), base_dir=tmp_path)


def write_table(tmp_path, name="nf_anime.yaml", **change):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(table_doc(**change)))
    return {"file": name, "sha256": _sha256(path)}


def test_a_book_carries_its_tables_by_reference_or_inline(tmp_path):
    ref = write_table(tmp_path)
    book = AddressBook.from_mapping(book_doc({"nf_anime": ref, "audit": body()}), base_dir=tmp_path)
    assert set(book.tables) == {"nf_anime", "audit"} and book.tables["audit"].name == "audit"
    tables = book.unit_tables()
    assert tables["nf_anime"] == tables["audit"] == UnitTable.from_mapping(table_doc()).table()
    assert book.schema_version == 13 and "tables" not in book.extra
    with pytest.raises(AddressError, match="`tables` needs address book schema 13, not 12"):
        AddressBook.from_mapping({**book_doc({"nf_anime": body()}), "schema_version": 12})
    inline = {k: v for k, v in book_doc({"nf_anime": body()}).items() if k != "schema_version"}
    assert AddressBook.from_mapping(inline, inline=True).schema_version == 13  # an inline book takes the schema


def test_a_book_citing_a_changed_or_malformed_table_is_refused(tmp_path):
    ref = write_table(tmp_path)
    with pytest.raises(AddressError, match="table `nf_anime`.*sha256 differs"):
        AddressBook.from_mapping(book_doc({"nf_anime": {**ref, "sha256": "0" * 64}}), base_dir=tmp_path)
    with pytest.raises(AddressError, match="cited as \\{file, sha256\\}"):
        AddressBook.from_mapping(book_doc({"nf_anime": {"file": ref["file"]}}), base_dir=tmp_path)
    with pytest.raises(AddressError, match="table `audit`: unknown keys \\['kind'\\]"):
        AddressBook.from_mapping(book_doc({"audit": {**body(), "kind": "unit_table"}}))
    with pytest.raises(AddressError, match="`tables` maps a table name"):
        AddressBook.from_mapping(book_doc({}))


@pytest.mark.parametrize("treatment, units, match", [
    (steer(table="nf_cartoon"), None, "reads table `nf_cartoon`, which the book's `tables` does not name"),
    (steer(column="hits"), None, "table `nf_anime` has no column \\['hits'\\]"),
    (None, {13: [4810, 9], 15: [7]}, "stage 0 \\(mlp.out.hook_in\\) op 0 .*no row for \\(layer 13, unit 9\\)"),
])
def test_a_treatment_reading_values_the_book_cannot_give_is_refused_when_the_book_loads(treatment, units, match):
    with pytest.raises(AddressError, match=match):
        AddressBook.from_mapping(book_doc({"nf_anime": body()}, treatment=treatment, units=units))


def test_a_book_without_a_tables_section_leaves_them_to_its_caller():
    book = AddressBook.from_mapping(book_doc(treatment=steer(table="anything", column="whatever")))
    assert book.tables == {} and book.unit_tables() == {}
    value = book.treatments["steer"].stages[0].ops[0].value
    assert (value.table, value.column) == ("anything", "whatever")


def test_a_target_whose_members_need_a_model_is_left_to_the_consumer():
    doc = book_doc({"nf_anime": body()})
    doc["sets"]["found"] = {"point": SITE, "layers": [13], "units": "0:8"}  # a slice: members known on a model only
    book = AddressBook.from_mapping(doc)
    assert set(book.treatments) == {"steer"}


def test_check_members_names_every_member_without_a_row():
    table = UnitTable.from_mapping(table_doc())
    check_members(table, "dose", [(13, 4810), (15, 7)], "here")
    with pytest.raises(AddressError, match="here: .*\\(layer 1, unit 1\\), .*and 1 more"):
        check_members(table, "dose", [(1, u) for u in range(1, 10)], "here")


def test_a_published_book_lists_its_tables_and_sidecars_and_fetches_with_them(tmp_path, monkeypatch):
    monkeypatch.setenv("VENNGRAM_CACHE", str(tmp_path / "cache"))
    src = tmp_path / "src"
    (src / "tables").mkdir(parents=True)
    UnitTable.from_mapping(table_doc()).write(src / "tables" / "nf_anime.yaml", threshold=0)
    ref = {"file": "tables/nf_anime.yaml", "sha256": _sha256(src / "tables" / "nf_anime.yaml")}
    inline = UnitTable.from_mapping(body(), name="audit")
    sha = write_table_rows(src / "audit.rows.safetensors", inline)
    (src / "book.yaml").write_text(yaml.safe_dump(book_doc(
        {"nf_anime": ref, "audit": body(rows={"file": "audit.rows.safetensors", "sha256": sha})})))
    manifest = build_manifest(src, [("address_book", "book.yaml")])
    assert {(e.kind, e.path) for e in manifest.documents} == {
        ("address_book", "book.yaml"), ("unit_table", "tables/nf_anime.yaml"),
        ("unit_table_rows", "tables/nf_anime.rows.safetensors"), ("unit_table_rows", "audit.rows.safetensors")}
    assert set(manifest.documents[0].needs) == {"tables/nf_anime.yaml", "tables/nf_anime.rows.safetensors",
                                                "audit.rows.safetensors"}
    publish("local", str(tmp_path / "pub"), src, documents=[("address_book", "book.yaml")])
    path, _ = fetch(f"local:{tmp_path / 'pub'}:book.yaml")
    book = AddressBook.from_file(path)
    assert book.unit_tables()["nf_anime"] == book.unit_tables()["audit"]
    report = check(f"local:{tmp_path / 'pub'}")
    assert report.ok, report.render()
    assert {d.kind for d in report.documents} >= {"address_book", "unit_table"}


def test_a_transform_reading_another_column_names_one_the_table_has():
    treatment = steer()
    treatment["stages"][0]["ops"][0]["value"]["transform"] = [{"sign_of": {"column": "nope"}}]
    with pytest.raises(AddressError, match="has no column \\['nope'\\]"):
        AddressBook.from_mapping(book_doc({"nf_anime": body()}, treatment=treatment))
    treatment["stages"][0]["ops"][0]["value"]["transform"] = [{"sign_of": {"column": "confirm_hit_rate"}}]
    assert AddressBook.from_mapping(book_doc({"nf_anime": body()}, treatment=treatment)).treatments


def test_a_table_read_from_a_sidecar_writes_its_rows_afresh_wherever_it_goes(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    big = UnitTable.from_file(UnitTable.from_mapping(table_doc()).write(tmp_path / "a" / "t.yaml", threshold=0))
    assert big.rows_ref is not None
    again = UnitTable.from_file(big.write(tmp_path / "b" / "t.yaml"))  # inline now: no dangling reference
    assert again.rows_ref is None and again.table() == big.table()
    assert isinstance(big.to_mapping(inline=True)["rows"], list)  # a book's inline table carries its rows


@pytest.mark.parametrize("damage, match", [
    (lambda h, body: (h["layer"].update(data_offsets=[0, 8]), body)[1], "do not tile"),
    (lambda h, body: (h["unit"].update(data_offsets=[16, 10 ** 6]), body)[1], "outside its"),
    (lambda h, body: (h["unit"].update(data_offsets="x"), body)[1], "data offsets 'x'"),
    (lambda h, body: body + b"\0" * 8, "do not tile"),
    (lambda h, body: body[:-8], "outside its|do not tile"),
])
def test_a_damaged_row_sidecar_is_refused_by_name(tmp_path, damage, match):
    import json
    import struct

    write_table_rows(tmp_path / "r.safetensors", UnitTable.from_mapping(table_doc()))
    raw = (tmp_path / "r.safetensors").read_bytes()
    (size,) = struct.unpack("<Q", raw[:8])
    header, body = json.loads(raw[8:8 + size]), raw[8 + size:]
    body = damage(header, body)
    head = json.dumps(header).encode()
    (tmp_path / "bad.safetensors").write_bytes(struct.pack("<Q", len(head)) + head + body)
    with pytest.raises(AddressError, match=match):
        read_table_rows(tmp_path / "bad.safetensors")


def test_an_integer_column_holds_64_bit_values():
    with pytest.raises(AddressError, match="64-bit"):
        UnitTable.from_mapping(table_doc(rows=[{**ROWS[0], "confirm_n": 1 << 63}]))


def test_a_treatments_free_form_notes_are_metadata_and_an_earlier_evidence_mapping_is_kept_there():
    treatment = {**steer(), "metadata": {"author": "x"}, "evidence": {"examples": ["a"]}}
    book = AddressBook.from_mapping(book_doc({"nf_anime": body()}, treatment=treatment))
    steered = book.treatments["steer"]
    assert steered.metadata == {"author": "x", "evidence": {"examples": ["a"]}}
    assert "evidence" not in steered.to_mapping() and steered.to_mapping()["metadata"] == steered.metadata


STACKED = [{**ROWS[0]}, {**ROWS[1], "stack": "vision"}, {**ROWS[0], "stack": "vision", "dose": 0.25}]


def test_a_table_keyed_by_stack_gives_each_stacks_rows_and_round_trips(tmp_path):
    table = UnitTable.from_mapping(table_doc(key=["unit", "stack", "layer"], rows=copy.deepcopy(STACKED)))
    assert table.key == ("stack", "layer", "unit") and table.stacks == [None, "vision"]
    assert table.column("dose") == {(13, 4810): 1.5}
    assert table.column("dose", stack="vision") == {(15, 7): -0.8, (13, 4810): 0.25}
    assert table.to_mapping() == table_doc(key=["stack", "layer", "unit"], rows=copy.deepcopy(STACKED))
    plain = UnitTable.from_mapping(table_doc())
    assert plain.key == ("layer", "unit") and "key" not in plain.to_mapping()  # the default key is not written
    with pytest.raises(AddressError, match="keyed by \\['layer', 'unit'\\], so its rows are the primary stack's"):
        plain.column("dose", stack="vision")
    path = table.write(tmp_path / "stacked.yaml", threshold=0)  # rows to a sidecar, the stacks by index
    again = UnitTable.from_file(path)
    assert again.rows == table.rows and again.key == table.key and again.stacks == [None, "vision"]
    assert read_table_rows(tmp_path / "stacked.rows.safetensors") == STACKED


def test_a_sidecar_with_stacks_needs_a_table_keyed_by_stack(tmp_path):
    table = UnitTable.from_mapping(table_doc(key=["stack", "layer", "unit"], rows=copy.deepcopy(STACKED)))
    sha = write_table_rows(tmp_path / "rows.safetensors", table)
    with pytest.raises(AddressError, match="row 1: gives a `stack`, and the table's key is \\['layer', 'unit'\\]"):
        UnitTable.from_mapping(table_doc(rows={"file": "rows.safetensors", "sha256": sha}), base_dir=tmp_path)


def vision_book(table, units=None):
    doc = book_doc({"nf_anime": table}, units=units)
    doc["sets"]["found"]["stack"] = "vision"
    return doc


def test_a_target_in_another_stack_reads_only_a_table_keyed_by_stack():
    with pytest.raises(AddressError, match="op 0 \\(`add`\\): unit table `nf_anime` is keyed by \\['layer', 'unit'\\], "
                                          "so its rows are the primary stack's, and the target is in stack `vision`"):
        AddressBook.from_mapping(vision_book(body()))
    stacked = body(key=["stack", "layer", "unit"], rows=copy.deepcopy(STACKED))
    book = AddressBook.from_mapping(vision_book(stacked, units={13: [4810], 15: [7]}))
    assert book.unit_tables(stack="vision") == {"nf_anime": book.tables["nf_anime"].table(stack="vision")}
    assert book.unit_tables()["nf_anime"]["dose"] == {(13, 4810): 1.5}
    with pytest.raises(AddressError, match="no row for \\(stack vision, layer 13, unit 9\\)"):
        AddressBook.from_mapping(vision_book(stacked, units={13: [4810, 9]}))
    primary = book_doc({"nf_anime": stacked}, units={13: [4810], 15: [7]})  # the primary stack lacks (15, 7)
    with pytest.raises(AddressError, match="no row for \\(layer 15, unit 7\\)"):
        AddressBook.from_mapping(primary)


def test_a_composed_unit_target_reads_its_sites_stack():
    doc = vision_book(body(key=["stack", "layer", "unit"], rows=copy.deepcopy(STACKED)), units={15: [7]})
    doc["sets"]["more"] = {"point": SITE, "stack": "vision", "units": {13: [4810]}}
    doc["sets"]["both"] = "@found | @more"
    doc["treatments"]["steer"] = steer(target="both")
    book = AddressBook.from_mapping(doc)
    op = book.treatments["steer"].stages[0].ops[0]
    assert op.unit_stack(book) == "vision" and op.unit_members(book) == [(15, 7), (13, 4810)]
