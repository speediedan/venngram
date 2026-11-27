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
from venngram.graphs import Anchor, anchor_cases, GraphSubset, read_graph_table, write_graph_table
from venngram.manifest import build_manifest
from venngram.recipe import Recipe
from venngram.store import publish
from venngram.support import SupportRecord
from venngram.units import _sha256

MODEL = {"id": "example/model", "revision": "pinned-model"}
TOKENIZER = {"id": "example/tokenizer", "revision": "pinned-tokenizer"}
SITE = "mlp.out.hook_out"


def node(node_id, feature, position=2, anchor=None, activation=3.0, baseline=1.0, site=SITE, layer=1, **extra):
    out = {"id": node_id, "kind": "feature", "site": site, "space": "tc", "layer": layer, "position": position,
           "index": {"feature": feature}, "activation": activation, "baseline": baseline,
           "anchor": anchor or {"kind": "from_end", "offset": 0}}
    return {**out, **extra}


def graph_doc(nodes=None, **extra):
    doc = {"schema_version": 1, "kind": "graph_subset", "name": "reasoning", "model": dict(MODEL),
           "spaces": {"tc": {"id": "example/transcoders", "revision": "pinned", "kind": "transcoder"},
                      "bsf": {"id": "example/bsf", "revision": "pinned", "kind": "block_sparse", "index": ["block"],
                              "value_shape": [3]}},
           "source": {"tokenizer": dict(TOKENIZER), "token_ids": [11, 22, 33],
                      "attribution_targets": [{"kind": "logit", "token_id": 44, "position": 2}],
                      "pruning": {"method": "threshold", "threshold": 0.01}},
           "nodes": nodes if nodes is not None else [
               node("f0", 7), node("f1", 8, position=1, anchor={"kind": "token_match", "token_ids": [22]},
                                   activation=2.0, baseline=0.5),
               {"id": "e0", "kind": "error", "site": SITE, "layer": 1, "position": 2, "activation": 0.25},
               {"id": "out", "kind": "logit", "token_id": 44, "position": 2, "activation": 5.0}],
           "edges": [{"source": "f1", "target": "f0", "attribution": 0.5}] if nodes is None else []}
    doc.update(extra)
    return doc


def bsf_node(node_id, block, activation, baseline=None, position=2):
    out = {"id": node_id, "kind": "feature", "site": SITE, "space": "bsf", "layer": 1, "position": position,
           "index": {"block": block}, "activation": activation, "anchor": {"kind": "absolute", "position": 0}}
    if baseline is not None:
        out["baseline"] = baseline
    return out


@pytest.mark.parametrize("case", anchor_cases(), ids=lambda c: f"{c['anchor']['kind']}-{c['token_ids']}")
def test_anchors_resolve_as_their_conformance_vectors_say(case):
    anchor = Anchor.from_mapping(case["anchor"], "case")
    if "positions" in case:
        assert anchor.resolve(case["token_ids"]) == tuple(case["positions"])
    else:
        with pytest.raises(AddressError, match=case["refuses"]):
            anchor.resolve(case["token_ids"])


@pytest.mark.parametrize("raw, match", [
    (None, "an anchor is"),
    ({"kind": "nearest"}, "an anchor is"),
    ({"kind": "all", "offset": 1}, "has no keys"),
    ({"kind": "absolute"}, "gives its `position`"),
    ({"kind": "absolute", "position": -1}, "integer >= 0"),
    ({"kind": "absolute", "position": True}, "integer >= 0"),
    ({"kind": "token_match", "token_ids": []}, "nonempty"),
    ({"kind": "token_match", "token_ids": [22], "offset": 1}, "outside the matched span"),
])
def test_malformed_anchors_are_refused(raw, match):
    with pytest.raises(AddressError, match=match):
        Anchor.from_mapping(raw, "case")


def test_a_graph_loads_its_nodes_edges_spaces_and_provenance():
    graph = GraphSubset.from_mapping(graph_doc())
    assert [n.id for n in graph.nodes] == ["f0", "f1", "e0", "out"]
    assert graph.node("f0").index == (7,) and graph.node("f0").values == {"activation": (3.0,), "baseline": (1.0,)}
    assert graph.sites() == (SITE,) and graph.edges[0].attribution == 0.5
    assert graph.provenance["pruning"]["method"] == "threshold"
    again = GraphSubset.from_mapping(graph.to_mapping())
    assert again.nodes == graph.nodes and again.edges == graph.edges and again.spaces == graph.spaces


def _with(change):
    doc = graph_doc()
    change(doc)
    return doc


@pytest.mark.parametrize("change, match", [
    (lambda d: d.update(kind="graph"), "kind: graph_subset"),
    (lambda d: d.update(schema_version=0), "schema_version"),
    (lambda d: d["model"].pop("revision"), "one revision only"),
    (lambda d: d.update(nodes=[]), "at least one node"),
    (lambda d: d.update(extra=1), "unknown keys"),
    (lambda d: d["nodes"].append(copy.deepcopy(d["nodes"][0])), "repeated"),
    (lambda d: d["nodes"].append({**d["nodes"][0], "id": "twin"}), "one source coordinate"),
    (lambda d: d["nodes"][0].pop("layer"), "lacks \\['layer'\\]"),
    (lambda d: d["nodes"][0].update(space="sae"), "not declared"),
    (lambda d: d["nodes"][0].update(index={"unit": 7}), "gives the axes \\['feature'\\]"),
    (lambda d: d["nodes"][0].pop("activation"), "source `activation`"),
    (lambda d: d["nodes"][0].update(activation=float("nan")), "finite"),
    (lambda d: d["nodes"][0].update(activation=[1.0]), "is a number"),
    (lambda d: d["nodes"][0].update(position=3), "outside the source prompt"),
    (lambda d: d["nodes"][0].update(kind="token"), "`kind` is one of"),
    (lambda d: d["edges"].append({"source": "f0", "target": "nowhere", "attribution": 1.0}), "not node ids"),
    (lambda d: d["edges"].append({"source": "f0", "target": "f1", "attribution": float("inf")}), "finite"),
    (lambda d: d["spaces"]["tc"].update(index=["position"]), "cannot name an index axis"),
    (lambda d: d["spaces"]["tc"].update(value_shape=[2, 2]), "value_shape"),
    (lambda d: d["spaces"]["tc"].pop("revision"), "pinned by id and revision"),
    (lambda d: d["source"].update(tokenizer={"id": "t"}), "source.tokenizer"),
])
def test_malformed_graphs_are_refused(change, match):
    with pytest.raises(AddressError, match=match):
        GraphSubset.from_mapping(_with(change))


def test_a_newer_graph_schema_is_refused_as_newer():
    with pytest.raises(NewerAddressSchemaError, match="newer venngram"):
        GraphSubset.from_mapping(graph_doc(schema_version=2))


def test_a_plan_places_feature_nodes_by_anchor_with_values_or_deltas():
    graph = GraphSubset.from_mapping(graph_doc())
    plan = graph.plan([5, 22, 9, 9], tokenizer=TOKENIZER)
    assert [(p.node, p.position, p.index, p.value) for p in plan] == [("f0", 3, (7,), (3.0,)), ("f1", 1, (8,), (2.0,))]
    delta = graph.plan([5, 22, 9, 9], baseline="baseline", tokenizer=TOKENIZER)
    assert [p.value for p in delta] == [(2.0,), (1.5,)]


def test_an_all_anchor_expands_and_a_collision_is_refused_not_resolved():
    graph = GraphSubset.from_mapping(graph_doc(nodes=[node("f0", 7, anchor={"kind": "all"})]))
    assert [p.position for p in graph.plan([1, 2, 3])] == [0, 1, 2]
    clash = GraphSubset.from_mapping(graph_doc(nodes=[node("f0", 7, anchor={"kind": "all"}),
                                                      node("f9", 7, position=0, anchor={"kind": "from_end"})]))
    with pytest.raises(AddressError, match="node `f9` lands on the coordinate of node `f0`"):
        clash.plan([1, 2, 3])


@pytest.mark.parametrize("nodes, kwargs, match", [
    ([node("f0", 7), node("f1", 8, anchor={"kind": "absolute", "position": 9})], {}, "node `f1`.*outside the prompt"),
    ([node("f0", 7, anchor=None)], {}, None),
    ([node("f0", 7, activation=1e308, baseline=-1e308)], {"baseline": "baseline"}, "not finite"),
    ([node("f0", 7)], {"baseline": "missing"}, "no `missing` column"),
    ([node("f0", 7)], {"sites": ["elsewhere"]}, "no feature node"),
    ([node("f0", 7, anchor={"kind": "token_match", "token_ids": [22]})], {}, "pinned tokenizer"),
    ([node("f0", 7, anchor={"kind": "token_match", "token_ids": [22]})],
     {"tokenizer": {"id": "other", "revision": "x"}}, "pinned tokenizer"),
])
def test_a_plan_refuses_the_whole_subset_naming_the_node(nodes, kwargs, match):
    if match is None:  # a node without an anchor is context only, and cannot be placed
        nodes[0].pop("anchor")
        match = "has no anchor"
    graph = GraphSubset.from_mapping(graph_doc(nodes=nodes))
    with pytest.raises(AddressError, match=match):
        graph.plan([5, 22, 9], **kwargs)


def test_block_sparse_nodes_carry_a_vector_per_block():
    graph = GraphSubset.from_mapping(graph_doc(nodes=[bsf_node("b0", 3, [0.5, -1.0, 2.0], [0.5, 0.0, 1.0])]))
    (placed,) = graph.plan([1, 2], baseline="baseline")
    assert (placed.space, placed.index, placed.shape, placed.value) == ("bsf", (3,), (3,), (0.0, -1.0, 1.0))
    assert graph.to_mapping()["nodes"][0]["activation"] == [0.5, -1.0, 2.0]
    with pytest.raises(AddressError, match="a list of 3 numbers"):
        GraphSubset.from_mapping(graph_doc(nodes=[bsf_node("b0", 3, 0.5)]))


def test_a_two_tier_index_addresses_a_coordinate_within_a_block():
    doc = graph_doc(nodes=[{**bsf_node("b0", 3, 0.5), "space": "units", "index": {"block": 3, "coordinate": 1}}])
    doc["spaces"]["units"] = {"id": "example/bsf", "revision": "pinned", "kind": "block_sparse",
                              "index": ["block", "coordinate"]}
    (placed,) = GraphSubset.from_mapping(doc).plan([1])
    assert placed.index == (3, 1) and placed.value == (0.5,)


def test_a_restriction_is_the_finite_universe_outside_the_plan():
    graph = GraphSubset.from_mapping(graph_doc(nodes=[node("f0", 7)]))
    plan = graph.plan([1, 2])
    rest = graph.restriction(plan, {(SITE, "tc", 1): [(7,), (8,), (9,)]}, positions=2)
    assert len(rest) == 5 and (SITE, "tc", 1, 1, (7,)) not in rest and (SITE, "tc", 1, 0, (7,)) in rest
    assert (SITE, "tc", 1, 2, (8,)) not in rest and len(list(rest)) == 5
    with pytest.raises(AddressError, match="outside the universe"):
        graph.restriction(plan, {(SITE, "tc", 1): [(8,)]}, positions=2)
    with pytest.raises(AddressError, match="without the 1 axes"):
        graph.restriction(plan, {(SITE, "tc", 1): [(7, 0)]}, positions=2)


def test_a_table_sidecar_round_trips_and_is_refused_when_changed(tmp_path):
    graph = GraphSubset.from_mapping(graph_doc())
    digest = write_graph_table(tmp_path / "g.table.safetensors", graph)
    table = read_graph_table(tmp_path / "g.table.safetensors", sha256=digest)
    doc = graph_doc(nodes={"file": "g.table.safetensors", "sha256": digest},
                    edges={"file": "g.table.safetensors", "sha256": digest})
    again = GraphSubset.from_mapping(doc, base_dir=tmp_path)
    assert again.nodes == graph.nodes and again.edges == graph.edges and len(table["nodes"]) == 4
    assert again.to_mapping()["nodes"] == {"file": "g.table.safetensors", "sha256": digest}
    with open(tmp_path / "g.table.safetensors", "r+b") as fh:
        fh.seek(-1, 2)
        fh.write(b"\x01")
    with pytest.raises(AddressError, match="sha256 differs"):
        GraphSubset.from_mapping(doc, base_dir=tmp_path)


def test_integer_identities_survive_a_table_exactly(tmp_path):
    big = 16777217  # not representable in F32, the operand sidecar's dtype
    graph = GraphSubset.from_mapping(graph_doc(nodes=[node("f0", big)], source={"token_ids": [1, 2, 3]}))
    path = graph.write(tmp_path / "g.yaml", threshold=0)
    assert yaml.safe_load(path.read_text())["nodes"]["file"] == "g.table.safetensors"
    assert GraphSubset.from_file(path).node("f0").index == (big,)
    small = GraphSubset.from_mapping(graph_doc()).write(tmp_path / "small.yaml")
    assert isinstance(yaml.safe_load(small.read_text())["nodes"], list)


def test_a_table_of_vector_values_round_trips(tmp_path):
    graph = GraphSubset.from_mapping(graph_doc(nodes=[bsf_node("b0", 3, [0.5, -1.0, 2.0]),
                                                      bsf_node("b1", 4, [1.0, 0.0, 0.25], position=1)]))
    graph.write(tmp_path / "v.yaml", threshold=0)
    assert GraphSubset.from_file(tmp_path / "v.yaml").nodes == graph.nodes


def write_graph(tmp_path, doc=None, name="reasoning.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc or graph_doc()))
    return {"file": name, "sha256": _sha256(path)}


def transfer_op(**value):
    return {"op": "replace", "target": {"graph": "reasoning"},
            "value": value or {"from": {"graph": "reasoning", "column": "activation"}}}


def book_with(tmp_path, stages, *, sets=None, graph=None, version=5, model=MODEL):
    return {"schema_version": version, "name": "transfer", "sets": sets or {},
            "graphs": {"reasoning": write_graph(tmp_path, graph)},
            "treatments": {"transfer": {"model": dict(model), "stages": stages}}}


def test_a_book_names_graphs_and_its_treatments_apply_them(tmp_path):
    doc = book_with(tmp_path, [{"site": SITE, "ops": [transfer_op(), {
        "op": "add", "target": {"graph": "reasoning"},
        "value": {"from": {"graph": "reasoning", "column": "activation", "baseline": "baseline"},
                  "transform": [{"scale": {"factor": 0.5}}]}}]}])
    book = AddressBook.from_mapping(doc, base_dir=tmp_path)
    assert book.schema_version == 5 and book.graphs["reasoning"].name == "reasoning" and not book.own_sets
    recipe = book.treatments["transfer"]
    first, second = recipe.stages[0].ops
    assert first.target_kind(book) == "graph_nodes" and first.target_of(book) == "value"
    assert second.value.baseline == "baseline" and second.value.source == "graph"
    written = recipe.to_mapping()
    assert written["schema_version"] == 5
    assert written["stages"][0]["ops"][1]["target"] == {"graph": "reasoning"}
    assert written["stages"][0]["ops"][1]["value"]["from"] == {"graph": "reasoning", "column": "activation",
                                                                "baseline": "baseline"}
    assert Recipe.treatment(recipe.to_treatment(), book, name="again").stages == recipe.stages


def test_a_restriction_names_a_finite_feature_universe_of_the_book(tmp_path):
    sets = {"universe": {"features": "tc", "layers": [1], "units": [7, 8, 9]}}
    restrict = {"op": "replace", "target": {"graph": "reasoning", "outside": {"set": "universe", "tokens": "prompt"}},
                "value": {"dose": 0.0}}
    book = AddressBook.from_mapping(book_with(tmp_path, [{"site": SITE, "ops": [transfer_op(), restrict]}],
                                              sets=sets), base_dir=tmp_path)
    op = book.treatments["transfer"].stages[0].ops[1]
    assert op.target_kind(book) == "graph_complement" and op.outside == "universe"
    assert op.to_mapping()["target"] == restrict["target"]


def test_a_restriction_universe_refuses_execution_passes(tmp_path):
    sets = {"universe": "features:tc & point:resid_post & layers:[1] & units:[7] & passes:2"}
    restrict = {"op": "replace", "target": {"graph": "reasoning", "outside": {"set": "universe", "tokens": "prompt"}},
                "value": {"dose": 0.0}}
    doc = book_with(tmp_path, [{"site": SITE, "ops": [restrict]}], sets=sets, version=12)
    with pytest.raises(AddressError, match=r"set `universe` uses `passes`"):
        AddressBook.from_mapping(doc, base_dir=tmp_path)


def _refused(tmp_path, stages, match, **kwargs):
    with pytest.raises(AddressError, match=match):
        AddressBook.from_mapping(book_with(tmp_path, stages, **kwargs), base_dir=tmp_path)


@pytest.mark.parametrize("stages, match", [
    ([{"site": SITE, "tokens": "last", "ops": [transfer_op()]}], "gives no `tokens`"),
    ([{"site": SITE, "ops": [{**transfer_op(), "tokens": "last"}]}], "takes no \\['tokens'\\]"),
    ([{"site": SITE, "ops": [{**transfer_op(), "target": {"graph": "missing"}}]}], "names a graph of the book"),
    ([{"site": SITE, "ops": [{**transfer_op(), "target": {"graph": "reasoning", "tokens": "all"}}]}],
     "contains only `graph` and `outside`"),
    ([{"site": SITE, "ops": [{**transfer_op(), "op": "scale"}]}], "admits \\['replace', 'add'\\]"),
    ([{"site": SITE, "ops": [transfer_op(**{"from": {"graph": "reasoning", "column": "score"}})]}], "no column"),
    ([{"site": SITE, "ops": [transfer_op(**{"from": {"graph": "other", "column": "activation"}})]}],
     "reads graph `other`"),
    ([{"site": SITE, "ops": [transfer_op(**{"from": {"table": "e", "column": "c"}})]}], "not a `from` value"),
    ([{"site": SITE, "ops": [transfer_op(**{"from": {"graph": "reasoning", "column": "activation"},
                                            "transform": [{"normalize": {"by": "max_abs"}}]})]}],
     "act on each value alone"),
    ([{"site": "elsewhere", "ops": [transfer_op()]}], "no feature node there"),
    ([{"site": SITE, "ops": [{"op": "replace", "target": {"graph": "reasoning", "outside": {
        "set": "universe", "tokens": "prompt"}}, "value": {"from": {"graph": "reasoning", "column": "activation"}}}]}],
     "replaces the universe's other members with a `dose`"),
    ([{"site": SITE, "ops": [{"op": "add", "target": {"graph": "reasoning", "outside": {
        "set": "universe", "tokens": "prompt"}}, "value": {"dose": 0.0}}]}], "admits \\['replace'\\]"),
    ([{"site": SITE, "ops": [{"op": "replace", "target": {"graph": "reasoning", "outside": {
        "set": "universe", "tokens": "all"}}, "value": {"dose": 0.0}}]}], "tokens: prompt"),
    ([{"site": SITE, "ops": [{"op": "replace", "target": {"graph": "reasoning", "outside": {
        "set": "rows", "tokens": "prompt"}}, "value": {"dose": 0.0}}]}], "features of the graph's spaces"),
])
def test_malformed_graph_treatments_are_refused(tmp_path, stages, match):
    sets = {"universe": {"features": "tc", "layers": [1], "units": [7, 8, 9]},
            "rows": {"point": SITE, "layers": [1], "units": [7]}}
    _refused(tmp_path, stages, match, sets=sets)


def test_a_graph_must_pin_the_treatments_model_and_be_reached_at_every_site(tmp_path):
    _refused(tmp_path, [{"site": SITE, "ops": [transfer_op()]}], "pins example/model@pinned-model",
             model={"id": "example/model", "revision": "other"})
    two_sites = graph_doc(nodes=[node("f0", 7), node("f1", 8, site="attn.out")])
    _refused(tmp_path, [{"site": SITE, "ops": [transfer_op()]}], "feature nodes at \\['attn.out'\\]", graph=two_sites)


def test_graphs_need_book_schema_5_and_cite_a_hash(tmp_path):
    _refused(tmp_path, [{"site": SITE, "ops": [transfer_op()]}], "needs address book schema 5", version=4)
    doc = book_with(tmp_path, [{"site": SITE, "ops": [transfer_op()]}])
    doc["graphs"]["reasoning"]["sha256"] = "0" * 64
    with pytest.raises(AddressError, match="sha256 differs"):
        AddressBook.from_mapping(doc, base_dir=tmp_path)
    doc["graphs"]["reasoning"] = {"file": "reasoning.yaml"}
    with pytest.raises(AddressError, match="cited by its hash"):
        AddressBook.from_mapping(doc, base_dir=tmp_path)
    inline = book_with(tmp_path, [{"site": SITE, "ops": [transfer_op()]}])
    inline.pop("schema_version")
    assert AddressBook.from_mapping(inline, base_dir=tmp_path, inline=True).schema_version == 5


def test_a_legacy_recipe_with_a_graph_target_needs_schema_5(tmp_path):
    book = book_with(tmp_path, [{"site": SITE, "ops": [transfer_op()]}])
    book.pop("treatments")
    doc = {"schema_version": 4, "kind": "recipe", "name": "r", "model": dict(MODEL), "book": book,
           "stages": [{"site": SITE, "ops": [transfer_op()]}]}
    with pytest.raises(AddressError, match="needs recipe schema 5"):
        Recipe.from_mapping(doc, base_dir=tmp_path)
    assert Recipe.from_mapping({**doc, "schema_version": 5}, base_dir=tmp_path).to_mapping()["schema_version"] == 5


def circuits_record(**subsets):
    sections = {"circuits": {"graphs": ["attribution"], **({"subsets": subsets} if subsets is not None else {})}}
    return SupportRecord.from_mapping({"schema_version": 6, "kind": "support_record", "name": "tracer",
                                       "sections": sections})


def test_graph_treatments_are_checked_against_circuits_subsets(tmp_path):
    graph = graph_doc(nodes=[node("f0", 7), node("f1", 8, anchor={"kind": "token_match", "token_ids": [22]}),
                             bsf_node("b0", 3, [0.5, -1.0, 2.0], [0.0, 0.0, 0.0])])
    sets = {"universe": {"features": "tc", "layers": [1], "units": [7, 8, 9]}}
    restrict = {"op": "replace", "target": {"graph": "reasoning", "outside": {"set": "universe", "tokens": "prompt"}},
                "value": {"dose": 0.0}}
    book = AddressBook.from_mapping(book_with(tmp_path, [{"site": SITE, "ops": [transfer_op(), restrict]}], sets=sets,
                                              graph=graph), base_dir=tmp_path)
    recipe = book.treatments["transfer"]
    full = circuits_record(ops=["replace"], anchors=["from_end", "token_match", "absolute"],
                           featurizers=["transcoder", "block_sparse"], values=["scalar", "vector"], complement=True)
    assert full.check_recipe(recipe) == []
    refusals = circuits_record(ops=["add"], anchors=["from_end"], featurizers=["transcoder"]).check_recipe(recipe)
    assert any("not applied to graph targets" in r for r in refusals)
    assert any("anchor `token_match` is not resolved" in r for r in refusals)
    assert any("`block_sparse` featurizer, which is not placed" in r for r in refusals)
    assert any("vector values, which are not applied" in r for r in refusals)
    assert any("restriction (`outside`) is not applied" in r for r in refusals)
    reads_only = SupportRecord.from_mapping({"schema_version": 2, "kind": "support_record", "name": "reader",
                                             "sections": {"circuits": {"graphs": ["attribution"]}}})
    assert reads_only.check_recipe(recipe) == ["recipe `transfer`: its graph targets need `circuits.subsets`, which "
                                               "this consumer does not declare (reading graphs is not applying them)"]
    steering_only = SupportRecord.from_mapping({"schema_version": 2, "kind": "support_record", "name": "steerer",
                                                "sections": {"steering": {"ops": ["replace"]}}})
    assert steering_only.check_recipe(recipe) == ["recipe `transfer`: `steerer` declares no `circuits` section, so it "
                                                  "offers no circuits contract"]


def test_subsets_need_support_record_schema_6_and_round_trip():
    with pytest.raises(ValueError, match="needs support record schema 6"):
        SupportRecord.from_mapping({"schema_version": 5, "kind": "support_record", "sections": {
            "circuits": {"subsets": {"ops": ["replace"]}}}})
    with pytest.raises(ValueError, match="names \\['nearest'\\]"):
        circuits_record(anchors=["nearest"])
    with pytest.raises(ValueError, match="true or false"):
        circuits_record(complement="yes")
    record = circuits_record(ops=["replace"], anchors=["all"], complement=True)
    written = record.to_mapping()
    assert written["schema_version"] == 6
    assert SupportRecord.from_mapping(written).to_mapping() == written


def test_a_published_book_lists_its_graph_and_table_and_passes_conformance(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    graph = GraphSubset.from_mapping(graph_doc())
    graph.write(src / "reasoning.yaml", threshold=0)
    ref = {"file": "reasoning.yaml", "sha256": _sha256(src / "reasoning.yaml")}
    (src / "book.yaml").write_text(yaml.safe_dump({
        "schema_version": 5, "name": "transfer", "sets": {}, "graphs": {"reasoning": ref},
        "treatments": {"transfer": {"model": dict(MODEL), "stages": [{"site": SITE, "ops": [transfer_op()]}]}}}))
    manifest = build_manifest(src, [("address_book", "book.yaml")])
    assert [(e.kind, e.path) for e in manifest.documents] == [
        ("address_book", "book.yaml"), ("graph_subset", "reasoning.yaml"),
        ("graph_table", "reasoning.table.safetensors")]
    publish("local", str(tmp_path / "pub"), src, documents=[("address_book", "book.yaml")])
    tracer = circuits_record(ops=["replace"], anchors=["from_end", "token_match"])
    report = check(f"local:{tmp_path / 'pub'}", supports=[tracer])
    assert report.ok, report.render()
    assert {d.kind for d in report.documents} == {"address_book", "graph_subset"}
    assert report.documents[0].refusals == {"tracer": ()}
