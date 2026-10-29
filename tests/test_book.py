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
import pytest
import yaml
from conftest import adapter_roles

from venngram import AddressBook, AddressError, diff_lock, make_lock, ParamSpace, register_uri_resolver, select
from venngram.book import _URI_RESOLVERS
from venngram.physical import PhysicalStructure

COMMON = {"schema_version": 1, "name": "common", "sets": {"top": {"role": "adapter", "layers": "-2:"},
                                                          "rest": {"role": "adapter", "exclude": "@top"}}}


def write(path, doc):
    path.write_text(yaml.safe_dump(doc))
    return path


def test_a_book_loads_and_its_sets_select(model, tmp_path):
    book = AddressBook.from_file(write(tmp_path / "book.yaml", COMMON))
    space = ParamSpace([n for n, _ in model.named_parameters()], roles=adapter_roles(model),
                       structure=PhysicalStructure(model))
    top, rest = select("@top", space, book.sets()), select("@rest", space, book.sets())
    assert top and rest and not set(top) & set(rest)


def test_imports_are_namespaced_and_keep_their_own_references(model, tmp_path):
    write(tmp_path / "common.yaml", COMMON)
    outer = {"schema_version": 1, "name": "outer", "imports": {"base": "common.yaml"},
             "sets": {"top": {"role": "adapter", "layers": 0}, "theirs": "@base.rest"}}
    book = AddressBook.from_file(write(tmp_path / "outer.yaml", outer))
    assert set(book.sets()) == {"top", "theirs", "base.top", "base.rest"}
    space = ParamSpace([n for n, _ in model.named_parameters()], roles=adapter_roles(model),
                       structure=PhysicalStructure(model))
    # base.rest excludes base.top (the top two blocks), not the outer book's own `top` (block 0)
    theirs = select("@theirs", space, book.sets())
    assert any(n.startswith("layers.0.") for n in theirs) and not any(n.startswith("layers.5.") for n in theirs)


@pytest.mark.parametrize("doc, match", [
    ({"name": "x", "sets": {"a": "role:adapter"}}, "`schema_version` is required"),
    ({"schema_version": 2, "sets": {"a": "role:adapter"}}, "newer venngram"),
    ({"schema_version": 0, "sets": {"a": "role:adapter"}}, "older than"),
    ({"schema_version": 1, "sets": {}}, "non-empty mapping"),
    ({"schema_version": 1, "sets": {"a b": "role:adapter"}}, "not a valid set name"),
    ({"schema_version": 1, "sets": {"a": "@b"}}, "no set or import defines"),
    ({"schema_version": 1, "sets": {"a": "@b", "b": "@a"}}, "cycle"),
    ({"schema_version": 1, "sets": {"a": "role:nope"}}, "set `a`"),
    ({"schema_version": 1, "portability": "checkpoint", "sets": {"a": "role:adapter"}}, "pins"),
    ({"schema_version": 1, "portability": "anywhere", "sets": {"a": "role:adapter"}}, "portability"),
    ({"schema_version": 1, "requires": {"structure": "magic"}, "sets": {"a": "role:adapter"}}, "requires.structure"),
    ({"schema_version": 1, "requires": {"naming_scheme": ""}, "sets": {"a": "role:adapter"}}, "requires.naming_scheme"),
    ({"schema_version": 1, "requires": {"naming_scheme": 3}, "sets": {"a": "role:adapter"}}, "requires.naming_scheme"),
    ({"schema_version": 1, "imports": {"x": "hub://org/repo"}, "sets": {"a": "role:adapter"}}, "no registered"),
])
def test_malformed_books_are_refused(doc, match):
    with pytest.raises(AddressError, match=match):
        AddressBook.from_mapping(doc)


def test_import_cycles_are_refused(tmp_path):
    write(tmp_path / "a.yaml", {"schema_version": 1, "imports": {"b": "b.yaml"}, "sets": {"x": "role:adapter"}})
    write(tmp_path / "b.yaml", {"schema_version": 1, "imports": {"a": "a.yaml"}, "sets": {"y": "role:adapter"}})
    with pytest.raises(AddressError, match="cycle"):
        AddressBook.from_file(tmp_path / "a.yaml")


def test_inline_books_default_the_version_and_keep_unknown_keys_as_descriptive():
    book = AddressBook.from_mapping({"sets": {"a": "role:adapter"}, "charts": {"a": {"kind": "circle"}}}, inline=True)
    assert book.schema_version == 1 and book.extra == {"charts": {"a": {"kind": "circle"}}}


def test_checkpoint_books_apply_only_to_their_pinned_model():
    book = AddressBook.from_mapping({
        "schema_version": 1, "portability": "checkpoint", "sets": {"n": "point:mlp.out.hook_in & units:[3]"},
        "applies_to": {"architectures": ["Qwen3ForCausalLM"], "models": [{"id": "org/m", "revision": "abc"}]}})
    book.check_applies(architecture="Qwen3ForCausalLM", model_id="org/m", revision="abc", basis="unprocessed")
    with pytest.raises(AddressError, match="applies only to org/m@abc"):
        book.check_applies(model_id="org/m", revision="def")
    with pytest.raises(AddressError, match="`unprocessed` basis"):
        book.check_applies(model_id="org/m", revision="abc", basis="tl_processed")
    with pytest.raises(AddressError, match="applies to Qwen3ForCausalLM"):
        book.check_applies(architecture="LlamaForCausalLM", model_id="org/m", revision="abc")


def test_uri_resolvers(monkeypatch):
    monkeypatch.setattr("venngram.book._URI_RESOLVERS", dict(_URI_RESOLVERS))
    register_uri_resolver("mem", lambda uri: COMMON)
    with pytest.raises(AddressError, match="already has a resolver"):
        register_uri_resolver("mem", lambda uri: COMMON)
    book = AddressBook.from_mapping({"schema_version": 1, "imports": {"c": "mem://common"}, "sets": {"a": "@c.top"}})
    assert "c.rest" in book.sets()


def test_locks_detect_drift():
    lock = make_lock({"top": {"layers.5.w": (4, 4)}}, model_id="org/m", revision="abc")
    assert diff_lock(lock, {"top": {"layers.5.w": (4, 4)}}) == []
    drift = diff_lock(lock, {"top": {"layers.5.w": (8, 4), "layers.6.w": (4, 4)}})
    assert drift == ["@top: layers.6.w is newly selected", "@top: layers.5.w changed shape (4, 4) -> (8, 4)"]


def test_a_book_written_in_one_vocabulary_is_refused_through_a_view_with_another(tmp_path):
    write(tmp_path / "tl.yaml", {"schema_version": 1, "requires": {"naming_scheme": "tl"},
                                 "sets": {"q": {"name": r"blocks\.\d+\.attn\.W_Q"}}})
    outer = write(tmp_path / "outer.yaml", {"schema_version": 1, "imports": {"tl": "tl.yaml"},
                                            "sets": {"top": {"role": "adapter", "layers": "-2:"}}})
    book = AddressBook.from_file(outer)
    book.check_requires(naming_scheme="tl")
    book.check_requires(naming_scheme=None)
    # the requirement is the imported book's, and it binds through the import
    with pytest.raises(AddressError, match="written in `tl` names, so it requires a view whose naming scheme is `tl`, "
                                           "not `physical`"):
        book.check_requires(naming_scheme="physical")
    # a book of vocabulary-neutral sets declares none and applies through any view
    AddressBook.from_mapping({"schema_version": 1, "sets": {"a": "role:adapter"}}).check_requires(naming_scheme="tl")
