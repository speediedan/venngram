"""Schema evolution: readers refuse a too-new document distinctly from a malformed one, and a registry lookup passes
over documents newer than it reads while a malformed document of a readable schema still fails."""
import warnings

import pytest
import yaml

pytest.importorskip("pyarrow")

from test_registry import build, directory, TINY  # noqa: E402

from venngram import NewerSchemaError, registry  # noqa: E402
from venngram.book import AddressBook  # noqa: E402
from venngram.errors import AddressError  # noqa: E402
from venngram.maps import component_map_for, from_document  # noqa: E402
from venngram.maps.drafter import DraftingRules  # noqa: E402
from venngram.maps.overlay import overlay_from_document, OVERLAY_SCHEMA_VERSION  # noqa: E402
from venngram.maps.rules import rules_from_document  # noqa: E402
from venngram.recipe import Recipe  # noqa: E402
from venngram.support import SupportRecord  # noqa: E402

LORA = {"schema_version": 5, "overlay": "peft.lora", "applies_when": {"module_names": ["lora_A"]},
        "roles": {"lora_A": "adapter:{key}"}}


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VENNGRAM_CACHE", str(tmp_path / "cache"))
    registry.clear_cache()


@pytest.mark.parametrize("read, doc, kind", [
    (from_document, {**TINY, "schema_version": 99}, "component_map"),
    (overlay_from_document, {**LORA, "schema_version": OVERLAY_SCHEMA_VERSION + 1}, "component_map"),
    (rules_from_document, {"schema_version": 99, "kind": "rules", "name": "x", "verify": {}}, "rules"),
    (DraftingRules.from_mapping, {"schema_version": 99, "kind": "rules", "name": "x", "draft": {}}, "rules"),
    (SupportRecord.from_mapping, {"schema_version": 99, "kind": "support_record", "sections": {}}, "support_record"),
])
def test_a_too_new_document_is_refused_as_newer_not_as_malformed(read, doc, kind):
    with pytest.raises(NewerSchemaError, match="newer than this venngram reads") as err:
        read(doc)
    assert err.value.kind == kind and err.value.version > err.value.newest
    assert isinstance(err.value, ValueError)  # existing `except ValueError` handlers still see it


@pytest.mark.parametrize("read, doc", [
    (AddressBook.from_mapping, {"schema_version": 99, "sets": {}}),
    (Recipe.from_mapping, {"schema_version": 99, "kind": "recipe"}),
])
def test_a_too_new_book_or_recipe_is_both_newer_and_an_address_error(read, doc):
    with pytest.raises(NewerSchemaError) as err:
        read(doc)
    assert isinstance(err.value, AddressError)


def test_a_newer_map_is_passed_over_once_and_the_best_readable_one_served(tmp_path):
    newer = {**TINY, "schema_version": 99, "components": {"blocks.{i}": {"module": "h.{i}", "kind": "block"}}}
    first = directory(tmp_path / "newer", {"tiny.yaml": newer})  # ranked first: crawled first
    second = directory(tmp_path / "readable", {"tiny.yaml": TINY})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the index build cannot mine the newer map, and says so
        registry.set_registries([build(tmp_path, [first, second])])
    rows = registry.rows("component_map", "TinyDecoder")
    assert [r.schema_version for r in rows] == [99, 1]  # recorded on the row, so it is skipped unfetched
    with pytest.warns(registry.NewerSchemaWarning, match="schema 99 is newer than this venngram reads"):
        assert component_map_for("TinyDecoder").components["blocks.{i}"].module == "layers.{i}"
    with warnings.catch_warnings():
        warnings.simplefilter("error", registry.NewerSchemaWarning)  # once per document, not per lookup
        registry.lookup_map("TinyDecoder")


def test_a_row_without_a_recorded_schema_is_checked_when_loaded(tmp_path):
    import pyarrow.parquet as pq

    newer = {**TINY, "schema_version": 99}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        location = build(tmp_path, [directory(tmp_path / "n", {"tiny.yaml": newer}),
                                    directory(tmp_path / "r", {"tiny.yaml": TINY})])
    for part in (tmp_path / "idx").glob("generations/*/kind=component_map/subject=*/rows.parquet"):
        pq.write_table(pq.read_table(part).drop(["schema_version"]), part)  # an index built before the column
    registry.set_registries([location])
    assert [r.schema_version for r in registry.rows("component_map", "TinyDecoder")] == [None, None]
    with pytest.warns(registry.NewerSchemaWarning):
        assert component_map_for("TinyDecoder").schema_version == 1


def test_a_malformed_map_of_a_readable_schema_still_fails_by_name(tmp_path):
    broken = {**TINY, "components": {"blocks.{i}": {"module": "layers.{i}", "kind": "no-such-kind"}}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        registry.set_registries([build(tmp_path, [directory(tmp_path / "b", {"tiny.yaml": broken}),
                                                  directory(tmp_path / "r", {"tiny.yaml": TINY})])])
    with pytest.raises(ValueError, match="no-such-kind") as err:
        registry.lookup_map("TinyDecoder")
    assert not isinstance(err.value, NewerSchemaError)


def test_a_newer_overlay_is_passed_over_for_the_readable_one(tmp_path):
    newer = {**LORA, "schema_version": OVERLAY_SCHEMA_VERSION + 1, "roles": {"lora_A": "adapter:{key}:v6"}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        registry.set_registries([build(tmp_path, [directory(tmp_path / "n", {"lora.yaml": newer}),
                                                  directory(tmp_path / "r", {"lora.yaml": LORA})])])
    with pytest.warns(registry.NewerSchemaWarning, match="overlay"):
        overlay = registry.lookup_overlay("peft.lora")
    assert overlay is not None and overlay.to_mapping()["roles"] == LORA["roles"]


def test_mined_rules_newer_than_this_venngram_are_passed_over(tmp_path):
    location = build(tmp_path, [directory(tmp_path / "r", {"tiny.yaml": TINY})])
    (mined,) = (tmp_path / "idx").glob("generations/*/mined_rules.yaml")
    doc = yaml.safe_load(mined.read_text())
    mined.write_text(yaml.safe_dump({**doc, "schema_version": 99}))
    with pytest.warns(registry.NewerSchemaWarning, match="rules schema 99"):
        assert registry.mined_rules(locations=[location]) == []


def test_newest_readable_follows_each_kind_and_overlay_subjects():
    from venngram.maps.schema import COMPONENT_MAP_SCHEMA_VERSION

    assert registry.newest_readable("component_map", "LlamaForCausalLM") == COMPONENT_MAP_SCHEMA_VERSION
    assert registry.newest_readable("component_map", "overlay:peft.lora") == OVERLAY_SCHEMA_VERSION
    assert registry.newest_readable("verification_record") is None
