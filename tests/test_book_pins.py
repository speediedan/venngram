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
"""A book's ``requires`` pins the base model, dictionary releases and frame versions it was written against."""
import pytest
import yaml

from venngram import AddressBook, AddressError, PinMismatchError
from venngram.support import SupportRecord

BASE = {"id": "google/gemma-2-2b", "revision": "c5ebcd4"}
SAE = {"release": {"sae_lens": "gemma-scope-2b-pt-res", "id": "layer_12/width_16k/average_l0_82"},
       "frames": {"weekday": "3"}}
BSF = {"release": {"hub": "org/bsf-l17", "revision": "9ab1f0e"}}
PINS = {"base_model": BASE, "latent_models": {"sae": SAE, "bsf_l17": BSF}}
LOADED = {"sae": {"release": {"sae_lens": "gemma-scope-2b-pt-res", "id": "layer_12/width_16k/average_l0_82"},
                  "frames": {"weekday": "3", "month": "1"}},
          "bsf_l17": {"release": {"hub": "org/bsf-l17", "revision": "9ab1f0e"}}}


def pinned(requires=None, **top):
    doc = {"schema_version": 8, "name": "b", "sets": {"sae": "module:latent_models.sae"},
           "requires": PINS if requires is None else requires, **top}
    return AddressBook.from_mapping(doc, source="b")


def write(path, doc):
    path.write_text(yaml.safe_dump(doc))
    return path


def test_a_book_loaded_on_what_it_pins_passes():
    book = pinned()
    assert book.pins() == PINS
    # a release key or frame the book does not pin is the consumer's own business
    book.check_pins(base_model=BASE, latent_models=LOADED)
    # each check is skipped when the consumer does not report that part
    book.check_pins()
    book.check_pins(base_model=BASE)


def test_every_differing_pin_is_named():
    loaded = {"sae": {"release": {"sae_lens": "gemma-scope-2b-pt-res", "id": "layer_20/width_16k/average_l0_82"},
                      "frames": {"weekday": "4"}}}
    with pytest.raises(PinMismatchError) as err:
        pinned().check_pins(base_model={**BASE, "revision": "0000000"}, latent_models=loaded)
    assert err.value.entries == ("requires.base_model", "requires.latent_models.sae.release.id",
                                 "requires.latent_models.sae.frames.weekday", "requires.latent_models.bsf_l17")
    message = str(err.value)
    assert "written against google/gemma-2-2b@c5ebcd4, the loaded model is google/gemma-2-2b@0000000" in message
    assert "pinned `layer_12/width_16k/average_l0_82`, loaded `layer_20/width_16k/average_l0_82`" in message
    assert "pinned version `3`, loaded `4`" in message
    assert "names latent model `bsf_l17`, which is not loaded" in message
    assert isinstance(err.value, AddressError)


@pytest.mark.parametrize("loaded, entry", [
    ({**LOADED, "sae": {"release": LOADED["sae"]["release"]}}, "requires.latent_models.sae.frames.weekday"),
    ({**LOADED, "bsf_l17": {"release": {"hub": "org/bsf-l17"}}}, "requires.latent_models.bsf_l17.release.revision"),
    ({**LOADED, "bsf_l17": {"release": {"hub": "org/other", "revision": "9ab1f0e"}}},
     "requires.latent_models.bsf_l17.release.hub"),
])
def test_an_unreported_pin_differs(loaded, entry):
    with pytest.raises(PinMismatchError) as err:
        pinned().check_pins(base_model=BASE, latent_models=loaded)
    assert err.value.entries == (entry,)


def test_pin_values_compare_as_strings():
    book = pinned({"base_model": {"id": "org/m", "revision": 7},
                   "latent_models": {"sae": {"release": {"hub": "org/sae", "revision": 3}, "frames": {"f": 2}}}})
    book.check_pins(base_model={"id": "org/m", "revision": "7"},
                    latent_models={"sae": {"release": {"hub": "org/sae", "revision": "3"}, "frames": {"f": 2}}})


@pytest.mark.parametrize("requires, match", [
    ({"base_model": {"id": "org/m"}}, r"`requires.base_model` is \{id, revision\}"),
    ({"base_model": {"id": "org/m", "revision": "a", "basis": "x"}}, r"`requires.base_model` is \{id, revision\}"),
    ({"base_model": {"id": "", "revision": "a"}}, "`requires.base_model.id` is a non-empty string"),
    ({"base_model": {"id": "org/m", "revision": True}}, "`requires.base_model.revision` is a non-empty string"),
    ({"latent_models": {"sae": SAE}}, "needs `requires.base_model`"),
    ({"base_model": BASE, "latent_models": {}}, "maps a latent model's name to its pin"),
    ({"base_model": BASE, "latent_models": {"a.b": SAE}}, "not a valid latent model name"),
    ({"base_model": BASE, "latent_models": {"sae": {"frames": {"f": "1"}}}}, r"is \{release, frames\}"),
    ({"base_model": BASE, "latent_models": {"sae": {**SAE, "basis": "x"}}}, r"is \{release, frames\}"),
    ({"base_model": BASE, "latent_models": {"sae": {"release": {}}}}, "`requires.latent_models.sae.release` is a non"),
    ({"base_model": BASE, "latent_models": {"sae": {"release": {"hub": "org/sae"}}}}, "with no `revision`"),
    ({"base_model": BASE, "latent_models": {"sae": {"release": {"sae_lens": None}}}},
     "`requires.latent_models.sae.release.sae_lens` is a non-empty string"),
    ({"base_model": BASE, "latent_models": {"sae": {**SAE, "frames": {}}}}, "maps a frame's name to its exact version"),
])
def test_malformed_pins_are_refused(requires, match):
    with pytest.raises(AddressError, match=match):
        pinned(requires)


def test_pins_need_schema_8_and_an_inline_book_takes_it():
    with pytest.raises(AddressError, match="needs address book schema 8, not 7: an older venngram would pass over"):
        pinned(schema_version=7)
    inline = AddressBook.from_mapping({"sets": {"a": "role:latent"}, "requires": {"base_model": BASE}}, inline=True)
    assert inline.schema_version == 8
    assert AddressBook.from_mapping({"sets": {"a": "role:latent"}}, inline=True).schema_version == 1


def test_a_checkpoint_book_pins_a_base_model_among_its_models():
    top = {"portability": "checkpoint", "applies_to": {"models": [{"id": "google/gemma-2-2b", "revision": "c5ebcd4"}]}}
    pinned(**top)
    with pytest.raises(AddressError, match="not among the checkpoint's pinned `applies_to.models`"):
        pinned({"base_model": {**BASE, "revision": "0000000"}}, **top)


def test_pins_bind_through_imports(tmp_path):
    write(tmp_path / "sae.yaml", {"schema_version": 8, "requires": {"base_model": BASE, "latent_models": {"sae": SAE}},
                                  "sets": {"sae": "module:latent_models.sae"}})
    outer = AddressBook.from_file(write(tmp_path / "outer.yaml", {"schema_version": 1, "imports": {"s": "sae.yaml"},
                                                                  "sets": {"top": "@s.sae"}}))
    assert outer.pins() == {"base_model": BASE, "latent_models": {"sae": SAE}}
    with pytest.raises(PinMismatchError) as err:
        outer.check_pins(base_model={**BASE, "id": "google/gemma-2-9b"})
    assert err.value.entries == ("requires.base_model",)


def test_imports_that_pin_one_model_differently_are_refused(tmp_path):
    write(tmp_path / "a.yaml", {"schema_version": 8, "requires": {"base_model": BASE}, "sets": {"a": "role:latent"}})
    write(tmp_path / "b.yaml", {"schema_version": 8, "requires": {"base_model": {**BASE, "revision": "0000000"}},
                                "sets": {"b": "role:latent"}})
    write(tmp_path / "c.yaml", {"schema_version": 8,
                                "requires": {"base_model": BASE, "latent_models": {"sae": {"release": BSF["release"]}}},
                                "sets": {"c": "role:latent"}})
    with pytest.raises(AddressError, match=r"import `a` pins the base model google/gemma-2-2b@c5ebcd4 and import `b` "
                                           r"pins google/gemma-2-2b@0000000, so no model satisfies both"):
        AddressBook.from_file(write(tmp_path / "ab.yaml", {"schema_version": 1, "sets": {"x": "@a.a"},
                                                           "imports": {"a": "a.yaml", "b": "b.yaml"}}))
    with pytest.raises(AddressError, match=r"the book pins latent model `sae` as .* and import `c` as"):
        AddressBook.from_file(write(tmp_path / "own.yaml", {"schema_version": 8, "imports": {"c": "c.yaml"},
                                                            "requires": PINS, "sets": {"x": "@c.c"}}))


TUNING = {"fields": ["role", "name", "module", "layers", "component", "leaf"], "targets": ["param"], "units": "none"}


def record(tuning, version=7):
    return SupportRecord.from_mapping({"schema_version": version, "kind": "support_record", "name": "tuner",
                                       "sections": {"tuning": tuning}})


def test_a_consumer_that_does_not_check_pins_refuses_a_pinned_book():
    assert record(TUNING, version=2).check_book(pinned()) == [
        "b: it pins the models it was written against (`requires.base_model`, `requires.latent_models`), which this "
        "consumer does not check (`pins: false`)"]
    checks = record({**TUNING, "pins": True})
    assert checks.check_book(pinned()) == []
    assert checks.to_mapping()["schema_version"] == 7
    assert checks.to_mapping()["sections"]["tuning"]["pins"] is True
    assert record(TUNING, version=2).to_mapping()["schema_version"] == 2
    with pytest.raises(ValueError, match="`pins` needs support record schema 7, not 6"):
        record({**TUNING, "pins": True}, version=6)
    with pytest.raises(ValueError, match="`pins` is true or false"):
        record({**TUNING, "pins": "yes"})
