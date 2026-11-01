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
"""Support records by section: opt-in contracts, unknown sections skipped by name, declared extensions."""
import warnings

import pytest
import yaml

from venngram.book import AddressBook
from venngram.conformance import check
from venngram.ops import Support
from venngram.store import load, publish
from venngram.support import (as_record, CircuitSupport, register_support_section, support_sections, SupportRecord,
                              TuningSupport, UnknownSupportSectionWarning)

TUNER = {"schema_version": 2, "kind": "support_record", "name": "tuner", "sections": {
    "tuning": {"fields": ["role", "name", "module", "layers", "stack", "component", "leaf"], "targets": ["param"],
               "units": "none", "naming_schemes": ["physical"]}}}


def book(sets, **top):
    return AddressBook.from_mapping({"schema_version": 1, "name": "b", "sets": sets, **top}, source="b")


def test_a_tuning_section_refuses_what_its_consumer_cannot_tune():
    record = SupportRecord.from_mapping(TUNER)
    assert record.tuning is not None and record.steering is None and record.circuits is None
    assert record.check_book(book({"top": "role:adapter & layers:-4:", "attn": {"component": "attn.*"}})) == []
    refusals = record.check_book(book({"site": {"point": "resid_post", "layers": 3},
                                       "neurons": {"module": "mlp.down_proj", "units": [1, 2]}},
                                      requires={"naming_scheme": "tl"}))
    assert refusals == [
        "b: it is written in `tl` names; this consumer's views provide ['physical']",
        "b: set `site`: the fields ['point'] are not honoured (honoured: ['role', 'name', 'module', 'layers', "
        "'stack', 'component', 'leaf'])",
        "b: set `site`: it selects a `site`, which this consumer does not tune (it tunes ['param'])",
        "b: set `neurons`: the fields ['units'] are not honoured (honoured: ['role', 'name', 'module', 'layers', "
        "'stack', 'component', 'leaf'])",
        "b: set `neurons`: it selects units, and this consumer tunes no unit sets (`units: none`)"]


def test_a_shared_book_is_judged_by_the_sets_a_consumer_uses(tmp_path):
    (tmp_path / "probes.yaml").write_text(yaml.safe_dump({"schema_version": 1, "name": "probes", "sets": {
        "probe": {"point": "resid_post"}, "top": {"role": "adapter", "layers": "-2:"}}}))
    shared = book({"tune": "@lift - @steer", "lift": "role:adapter", "steer": {"point": "resid_post", "layers": 3},
                   "upper": "@p.top"}, imports={"p": str(tmp_path / "probes.yaml")})
    tuning = TuningSupport(fields=frozenset({"role", "layers", "point"}))
    assert len(tuning.check_book(shared)) == 2  # `steer`, and the import's `probe`, select sites
    assert tuning.check_book(shared, sets=["lift"]) == []
    (only,) = tuning.check_book(shared, sets=["tune"])  # `tune` references `steer`
    assert only.startswith("b: set `steer`")
    assert tuning.check_book(shared, sets=["upper"]) == []  # the import's `top`, not its `probe`
    assert tuning.check_book(shared, sets=["p.probe"])[0].startswith("import `p`: ")
    with pytest.raises(ValueError, match=r"no set is named \['nope'\]"):
        tuning.check_book(shared, sets=["nope"])


def test_imported_books_are_checked_under_their_namespace(tmp_path):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump({"schema_version": 1, "name": "base",
                                                        "sets": {"probe": {"point": "resid_post"}}}))
    (tmp_path / "top.yaml").write_text(yaml.safe_dump({"schema_version": 1, "name": "top",
                                                       "imports": {"base": "./base.yaml"},
                                                       "sets": {"top": "role:adapter"}}))
    tuning = TuningSupport(fields=frozenset({"role", "point"}))
    (line,) = tuning.check_book(AddressBook.from_file(tmp_path / "top.yaml"))
    assert line.startswith("import `base`: ") and "selects a `site`" in line


def test_a_section_the_record_omits_is_a_contract_it_does_not_offer():
    record = SupportRecord.from_mapping(TUNER)
    assert record.check_recipe(object()) == ["recipe `?`: `tuner` declares no `steering` section, so it offers no "
                                             "steering contract"]
    steerer = as_record(Support(ops=frozenset({"add"}), name="steerer"))
    assert steerer.check_book(book({"x": "role:adapter"})) == [
        "b: `steerer` declares no `tuning` section, so it offers no tuning contract"]


def test_schema_1_records_read_as_a_steering_section():
    record = SupportRecord.from_mapping({"schema_version": 1, "kind": "support_record", "name": "s", "ops": ["add"],
                                         "tokens": ["all", "prompt"]})
    assert set(record.sections) == {"steering"} and record.steering.tokens == frozenset({"all", "prompt"})
    assert record.to_mapping()["sections"]["steering"]["tokens"] == ["all", "prompt"]


def test_unknown_sections_are_skipped_by_name_and_kept_as_written():
    doc = {**TUNER, "sections": {**TUNER["sections"], "attestation": {"signed": True},
                                 "acme.probes": {"heads": [3]}},
           "extensions": {"acme.probes": {"schema": "hub-model:acme/probe-support:schema.yaml"}}}
    with pytest.warns(UnknownSupportSectionWarning) as caught:
        record = SupportRecord.from_mapping(doc, source="r.yaml")
    messages = [str(w.message) for w in caught]
    assert any("`attestation` is skipped: this venngram does not know the section type" in m for m in messages)
    assert any("`acme.probes` is skipped: it is an extension (schema hub-model:acme/probe-support:schema.yaml)" in m
               for m in messages)
    assert set(record.sections) == {"tuning"} and record.skipped == {"attestation": {"signed": True},
                                                                     "acme.probes": {"heads": [3]}}
    assert record.to_mapping()["sections"]["acme.probes"] == {"heads": [3]}  # written back as it was read


def test_a_registered_extension_section_is_parsed(monkeypatch):
    from venngram import support

    monkeypatch.setattr(support, "_SECTIONS", dict(support._SECTIONS))
    register_support_section("acme.probes", lambda doc, source: frozenset(doc["heads"]))
    assert "acme.probes" in support_sections()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnknownSupportSectionWarning)
        record = SupportRecord.from_mapping({**TUNER, "sections": {"acme.probes": {"heads": [3, 5]}},
                                             "extensions": {"acme.probes": {"schema": "https://acme.example/s"}}})
    assert record.sections["acme.probes"] == frozenset({3, 5})
    for name in ("tuning", "probes", "Acme.probes"):
        with pytest.raises(ValueError, match="namespaced"):
            register_support_section(name, lambda doc, source: doc)
    with pytest.raises(ValueError, match="already registered"):
        register_support_section("acme.probes", lambda doc, source: doc)


def test_a_circuits_section_names_graphs_nodes_features_and_evidence():
    record = SupportRecord.from_mapping({"schema_version": 2, "kind": "support_record", "name": "tracer", "sections": {
        "circuits": {"graphs": ["attribution"], "nodes": ["feature", "error", "embedding", "logit"],
                     "features": ["clt", "transcoder"], "evidence": ["traced", "pruned"]}}})
    assert record.circuits == CircuitSupport(graphs=frozenset({"attribution"}),
                                             nodes=frozenset({"feature", "error", "embedding", "logit"}),
                                             features=frozenset({"clt", "transcoder"}),
                                             evidence=frozenset({"traced", "pruned"}))
    assert SupportRecord.from_mapping(record.to_mapping()) == record


@pytest.mark.parametrize("doc, match", [
    ({**TUNER, "schema_version": 3}, "schemas 1 to 2"),
    ({**TUNER, "ops": ["add"]}, "contracts go under `sections`"),
    ({**TUNER, "sections": {}}, "maps each contract"),
    ({**TUNER, "sections": {"tuning": {"units": "lora"}}}, "`units` is one of"),
    ({**TUNER, "sections": {"tuning": {"fields": ["neuron"]}}}, "does not define"),
    ({**TUNER, "sections": {"circuits": {"evidence": ["guessed"]}}}, "does not define"),
    ({**TUNER, "sections": {"steering": {"ops": ["add"], "speed": 1}}}, "unknown"),
    ({**TUNER, "extensions": {"probes": {"schema": "x"}}}, "not namespaced"),
    ({**TUNER, "extensions": {"acme.probes": {}}}, "gives the `schema`"),
])
def test_malformed_records_are_refused_by_name(doc, match):
    with pytest.raises(ValueError, match=match):
        SupportRecord.from_mapping(doc)


def test_conformance_checks_books_against_tuning_sections(tmp_path, monkeypatch):
    monkeypatch.setenv("VENNGRAM_CACHE", str(tmp_path / "cache"))
    src = tmp_path / "src"
    src.mkdir()
    (src / "lora.yaml").write_text(yaml.safe_dump({"schema_version": 1, "name": "lora",
                                                   "sets": {"top": "role:adapter & layers:-4:"}}))
    (src / "probe.yaml").write_text(yaml.safe_dump({"schema_version": 1, "name": "probe",
                                                    "sets": {"probe": {"point": "resid_post"}}}))
    (src / "tuner.yaml").write_text(yaml.safe_dump(TUNER))
    publish("local", str(tmp_path / "pub"), src, documents=[("address_book", "lora.yaml"),
                                                           ("address_book", "probe.yaml"),
                                                           ("support_record", "tuner.yaml")])
    location = f"local:{tmp_path / 'pub'}"
    tuner = load(f"{location}:tuner.yaml")
    assert isinstance(tuner, SupportRecord)
    steerer = Support(ops=frozenset({"add"}), name="steerer")  # offers no tuning contract: does not judge books
    report = check(location, supports=[tuner, steerer])
    results = {d.path: d.refusals for d in report.documents}
    assert results["lora.yaml"] == {"tuner": ()}
    assert set(results["probe.yaml"]) == {"tuner"} and len(results["probe.yaml"]["tuner"]) == 2
    assert not report.ok
