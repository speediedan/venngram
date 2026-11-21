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
"""Gradient paths: routes between ordered waypoints, declared by venngram and placed by a consumer's route resolver."""
import hashlib
from pathlib import Path as FsPath

import pytest
from test_resolve import space_of

from venngram import AddressBook, select
from venngram.address import parse, Path, references, to_mapping, to_string, Waypoint
from venngram.errors import AddressError, AddressUnresolvable
from venngram.ops import check_support, op_spec_from_mapping, OpSpec, ParamSpec, read_op_specs, Support, write_op_specs
from venngram.paths import is_path, paths, RouteResolver, waypoint_kinds
from venngram.support import SupportRecord

SCHEMAS = FsPath(__file__).parent / "schemas"
EMBED = {"component": "embed", "anchor": {"kind": "token_match", "token_ids": [5, 9]}}
HEADS = {"component": "attn.*", "layers": "0.5d:0.75d"}
PATH = {"path": [EMBED, HEADS, "component:unembed"]}
MODEL = {"id": "Qwen/Qwen3.5-4B", "revision": "abc123"}


def book(sets, version=10, **top):
    return AddressBook.from_mapping({"schema_version": version, "name": "b", "sets": sets, **top}, source="b",
                                    base_dir=SCHEMAS)


def test_a_path_reads_its_waypoints_in_order_and_round_trips():
    expr = parse(PATH)
    assert isinstance(expr, Path) and len(expr.waypoints) == 3
    first = expr.waypoints[0]
    assert first.anchor is not None and first.anchor.kind == "token_match" and first.anchor.token_ids == (5, 9)
    assert parse(to_mapping(expr)) == expr
    assert to_string(expr) == ("path(component:embed @ token_match -> (layers:0.5d:0.75d & component:attn.*) -> "
                               "component:unembed)")
    assert waypoint_kinds(expr) == {"address", "anchor"}
    graph = parse({"path": [{"graph": "g", "nodes": ["f0"]}, "@heads"]})
    assert graph.waypoints[0] == Waypoint(graph="g", nodes=("f0",)) and references(graph) == ["heads"]


@pytest.mark.parametrize("raw, match", [
    ({"path": ["component:embed"]}, "at least two waypoints"),
    ({"path": ["component:embed", "component:unembed"], "role": "base"}, "alone"),
    ({"path": ["component:embed", {"path": ["component:embed", "component:unembed"]}]}, "itself a path"),
    ({"path": ["component:embed", "component:unembed & of:grad"]}, "sets \\['of'\\]"),
    ({"path": [{"graph": "g", "nodes": []}, "component:unembed"]}, "non-empty list of distinct node ids"),
    ({"path": [{"graph": "g", "nodes": ["a", "a"]}, "component:unembed"]}, "non-empty list of distinct node ids"),
    ({"path": [{"component": "embed", "anchor": {"kind": "nowhere"}}, "component:unembed"]}, "an anchor is"),
])
def test_malformed_paths_are_refused_by_name(raw, match):
    with pytest.raises(AddressError, match=match):
        parse(raw)


def test_paths_unite_and_refuse_other_algebra():
    sets = {"a": PATH, "b": {"path": ["component:embed", "component:unembed"]}, "both": {"any": ["@a", "@b"]},
            "heads": HEADS}
    assert [len(p.waypoints) for p in paths("@both", sets)] == [3, 2]
    assert is_path("@both", sets) and not is_path("@heads", sets)
    with pytest.raises(AddressUnresolvable, match="intersects or subtracts paths"):
        paths({"all": ["@a", "@b"]}, sets)
    with pytest.raises(AddressUnresolvable, match="is not a path"):
        paths({"any": ["@a", "@heads"]}, sets)
    with pytest.raises(AddressUnresolvable, match="no set is named `@missing`"):
        paths("@missing", sets)
    with pytest.raises(AddressUnresolvable, match="form a cycle"):
        paths("@x", {"x": {"any": ["@y"]}, "y": {"any": ["@x"]}})


def test_a_path_is_not_a_set_of_tensors(model):
    with pytest.raises(AddressUnresolvable, match="is a path, which a consumer's route resolver places"):
        select(PATH, space_of(model))


def test_a_book_with_paths_is_schema_10_and_references_resolve_through_waypoints(tmp_path):
    loaded = book({"heads": HEADS, "p": {"path": [EMBED, "@heads", "component:unembed"]}})
    assert paths("@p", loaded.sets())[0].waypoints[1].address == parse("@heads")
    with pytest.raises(AddressError, match="are paths, which need address book schema 10, not 9"):
        book({"p": PATH}, version=9)
    with pytest.raises(AddressError, match="references @nowhere"):
        book({"p": {"path": ["@nowhere", "component:unembed"]}})
    assert AddressBook.from_mapping({"sets": {"p": PATH}}, inline=True).schema_version == 10
    # an import's path keeps referencing its own sets, under the import's namespace
    (tmp_path / "inner.yaml").write_text("schema_version: 10\nsets:\n  heads: {component: 'attn.*'}\n"
                                         "  p: {path: [component:embed, '@heads']}\n")
    outer = AddressBook.from_mapping({"schema_version": 1, "imports": {"i": "inner.yaml"}, "sets": {"q": "@i.p"}},
                                     base_dir=tmp_path)
    assert paths("@q", outer.sets())[0].waypoints[1].address == parse("@i.heads")


def graph_ref():
    path = SCHEMAS / "graph_subset_v1.yaml"
    return {"g": {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}


def test_graph_waypoints_name_the_books_graph_nodes_and_make_a_checkpoint_book():
    checkpoint = {"portability": "checkpoint", "applies_to": {"models": [MODEL]}, "graphs": graph_ref()}
    via = {"path": [{"graph": "g", "nodes": ["f0", "b0"]}, "component:unembed"]}
    book({"p": via}, **checkpoint)
    with pytest.raises(AddressError, match="which the book's `graphs` does not name"):
        book({"p": {"path": [{"graph": "h", "nodes": ["f0"]}, "component:unembed"]}}, **checkpoint)
    with pytest.raises(AddressError, match="names nodes \\['zz'\\], which graph `g` does not have"):
        book({"p": {"path": [{"graph": "g", "nodes": ["zz"]}, "component:unembed"]}}, **checkpoint)
    with pytest.raises(AddressError, match="so the book is `portability: checkpoint`"):
        book({"p": via}, graphs=graph_ref())


def treatment(gradients, sets=None):
    return book({"heads": HEADS, "p": PATH, "w": "role:base & leaf:weight & of:grad", **(sets or {})},
                treatments={"t": {"model": MODEL, "gradients": gradients}}).treatments["t"]


def test_stop_and_route_act_on_a_path_in_the_gradients_section():
    t = treatment([{"op": "stop", "target": {"set": "p"}}, {"op": "route", "target": {"set": "p"},
                                                            "params": {"reduce": "sum"}}])
    assert [o.op for o in t.gradients] == ["stop", "route"] and t.gradients[1].params == {"reduce": "sum"}
    assert t.gradients[0].target_kind(t.book) == "path"
    assert t.to_treatment()["gradients"][1] == {"op": "route", "target": {"set": "p"}, "params": {"reduce": "sum"}}


@pytest.mark.parametrize("op, match", [
    ({"op": "stop", "target": {"set": "p"}, "params": {"reduce": "max"}}, "one of \\['each', 'sum', 'mean'\\]"),
    ({"op": "replace", "target": {"set": "p"}, "value": {"dose": 0.0}}, "not `path`"),
    ({"op": "stop", "target": {"set": "w"}}, "not `param`"),
    ({"op": "stop", "target": {"set": "p"}, "value": {"dose": 1.0}}, "takes no `value`"),
    ({"op": "stop", "target": {"set": "p"}, "tokens": "last"}, "waypoints' anchors"),
])
def test_path_ops_are_refused_outside_their_contract(op, match):
    with pytest.raises(AddressError, match=match):
        treatment([op])


def test_a_path_is_acted_on_only_in_the_gradients_section():
    with pytest.raises(AddressError, match="a path is acted on in a treatment's `gradients` section"):
        book({"p": PATH}, treatments={"t": {"model": MODEL, "weights": [{"op": "scale", "target": {"set": "p"},
                                                                          "value": {"dose": 0.5}}]}})


def test_a_consumer_declares_the_waypoint_kinds_its_route_resolver_places():
    t = treatment([{"op": "stop", "target": {"set": "p"}}])
    support = Support(ops=frozenset(), targets=frozenset({"path"}), gradient_ops=frozenset({"stop"}),
                      routes=frozenset({"address"}))
    assert check_support(t, support) == [
        "gradients op 0 (`stop`): its path uses the waypoint kinds ['anchor'], which this consumer's route resolver "
        "does not place (`routes`: ['address'])"]
    assert check_support(t, Support(ops=frozenset(), targets=frozenset({"path"}), gradient_ops=frozenset({"stop"}),
                                    routes=frozenset({"address", "anchor"}))) == []
    record = SupportRecord.from_mapping({"schema_version": 8, "kind": "support_record", "name": "r", "sections": {
        "steering": {"ops": [], "targets": ["path"], "gradient_ops": ["stop", "route"],
                     "routes": ["address", "anchor"]}}})
    assert record.to_mapping()["schema_version"] == 8
    with pytest.raises(ValueError, match="`routes` needs support record schema 8, not 7"):
        SupportRecord.from_mapping({"schema_version": 7, "kind": "support_record", "sections": {
            "steering": {"ops": [], "routes": ["address"]}}})


def test_a_declared_path_op_is_op_specs_schema_3():
    spec = OpSpec("detour", ("none",), {"how": ParamSpec(str, choices=("a", "b"))}, (), ("path",), "none",
                  of=("grad",))
    doc = write_op_specs([spec], name="paths", description=None)
    assert doc["schema_version"] == 3 and doc["ops"][0]["params"]["how"]["choices"] == ["a", "b"]
    assert op_spec_from_mapping(doc["ops"][0]) == spec
    with pytest.raises(ValueError, match="needs op_specs schema 3, not 2"):
        read_op_specs({**doc, "schema_version": 2}, register=False)


def test_a_route_resolver_is_a_protocol_a_consumer_implements():
    class Tiny:
        def routes(self, path, *, graphs=None):
            return [(i, i + 1) for i in range(len(path.waypoints) - 1)]

    assert isinstance(Tiny(), RouteResolver) and Tiny().routes(parse(PATH)) == [(0, 1), (1, 2)]
