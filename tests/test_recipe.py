import copy

import pytest
import yaml

from venngram import ParamSpace, select
from venngram.errors import AddressError
from venngram.operands import read_operand, write_operands
from venngram.ops import check_support, OPS, OpSpec, register_op, Support
from venngram.recipe import compose, load_recipe, migrate, Recipe, RecipeSchemaWarning

MODEL = {"id": "Qwen/Qwen3.5-4B", "revision": "abc123"}


def book_doc(sets, model=MODEL):
    """A checkpoint book; ``sets`` maps a name to ``{layer: [unit, ...]}``, or to a ready set mapping."""
    out = {}
    for name, units in sets.items():
        if isinstance(units, dict) and all(isinstance(k, int) for k in units):
            out[name] = {"point": "mlp.out.hook_in", "layers": sorted(units),
                         "units": {k: sorted(v) for k, v in sorted(units.items())}}
        else:
            out[name] = units
    return {"schema_version": 1, "name": "book", "portability": "checkpoint",
            "applies_to": {"models": [{"id": model["id"], "revision": model["revision"]}]}, "sets": out}


def recipe_doc(stages, sets=None, operands=None, model=MODEL, name="r"):
    doc = {"schema_version": 2, "kind": "recipe", "name": name, "model": {**model, "revision_source": "pinned"},
           "book": book_doc(sets or {"chess": {13: [7956], 14: [8949]}}, model), "stages": stages}
    if operands:
        doc["operands"] = operands
    return doc


def add(target, value, site="mlp.out.hook_in"):
    return {"site": site, "ops": [{"op": "add", "target": {"set": target}, "value": value}]}


CONCEPT = {"values": [1.0, 0.0, 0.0, 0.0], "shape": [4], "basis": "unfolded"}
POLES = {"values": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], "shape": [2, 4], "basis": "paper"}


def test_a_schema_2_recipe_reads_and_round_trips():
    doc = recipe_doc([
        add("chess", {"doses": {13: {7956: -1.0}, 14: {8949: 1.0}}}),
        {"site": "blocks.resid_post", "tokens": "last", "ops": [
            {"op": "reject", "operand": "concept", "params": {"fraction": 0.5}},
            {"op": "patch", "operand": "poles"},
            {"op": "clamp", "operand": "concept", "params": {"max": 2.0}}]}],
        operands={"concept": CONCEPT, "poles": POLES})
    recipe = Recipe.from_mapping(doc)
    first, second = recipe.stages
    assert first.ops[0].unit_doses(recipe.book) == {(13, 7956): -1.0, (14, 8949): 1.0}
    assert second.tokens.kind == "last" and [o.op for o in second.ops] == ["reject", "patch", "clamp"]
    assert recipe.operands["poles"].kind == "pair" and second.ops[0].params == {"fraction": 0.5}
    again = Recipe.from_mapping(recipe.to_mapping())
    assert again.to_mapping() == recipe.to_mapping() and again.stages == recipe.stages


def test_values_from_evidence_and_features_targets():
    sets = {"top": {"space": "transcoder", "features": "3,7", "layers": [10]}}
    doc = recipe_doc([{"site": "transcoder.features", "tokens": {"positions": [5, 6]}, "ops": [
        {"op": "replace", "target": {"set": "top"},
         "value": {"from": {"evidence": "scores", "column": "score"},
                   "transform": [{"sign_aware_scale": {"factor": 4.0}}, "abs"]}}]}], sets=sets)
    recipe = Recipe.from_mapping(doc)
    op = recipe.stages[0].ops[0]
    assert op.target_kind(recipe.book) == "features" and [t.name for t in op.value.transform] == [
        "sign_aware_scale", "abs"]
    assert recipe.stages[0].tokens.positions == (5, 6)


@pytest.mark.parametrize("op, match", [
    ({"op": "lerp"}, "unknown op 'lerp'"),
    ({"op": "mul", "value": {"dose": 2.0}}, "`mul` is now `scale`"),
    ({"op": "reject", "operand": "nobasis"}, "must declare its `basis`"),
    ({"op": "patch", "operand": "concept"}, "is not a pair operand"),
    ({"op": "reject", "operand": "concept", "params": {"fraction": 2.0}}, "lies in \\[0.0, 1.0\\]"),
    ({"op": "reject", "operand": "concept", "params": {"width": 1.0}}, "unknown params"),
    ({"op": "clamp", "operand": "concept"}, "at least one of \\['min', 'max'\\]"),
    ({"op": "reject"}, "needs an operand"),
    ({"op": "add"}, "needs a `value` or an operand"),
    ({"op": "scale", "operand": "concept", "value": {"dose": 1.0}}, "is not a  operand|is not a valid operand"),
    ({"op": "project", "operand": "concept", "value": {"dose": 1.0}}, "takes no `value`"),
    ({"op": "reject", "operand": "concept", "target": {"set": "chess"}}, "acts on \\['activation'\\]"),
    ({"op": "add", "target": {"set": "missing"}, "value": {"dose": 1.0}}, "names a set of the book"),
    ({"op": "add", "value": {"dose": 1.0, "doses": {}}}, "exactly one of"),
    ({"op": "add", "value": {"dose": 1.0, "transform": ["cube"]}}, "unknown transform 'cube'"),
    ({"op": "add", "value": {"doses": {1: {2: 1.0}}}}, "per-unit `doses` need a target set with `units`"),
    ({"op": "add", "value": {"dose": 1.0}, "tokens": "last"}, "cannot override its stage's `tokens`"),
])
def test_misused_ops_are_refused_by_name(op, match):
    doc = recipe_doc([{"site": "s", "ops": [op]}],
                     operands={"concept": CONCEPT, "nobasis": {"values": [1.0, 0.0], "shape": [2]}})
    with pytest.raises(AddressError, match=match):
        Recipe.from_mapping(doc)


@pytest.mark.parametrize("change, match", [
    (lambda r: r["model"].pop("revision"), "one revision only"),
    (lambda r: r["model"].update(revision="other"), "the book pins"),
    (lambda r: r.update(schema_version=3), "must be in 1..2"),
    (lambda r: r.update(stages=[]), "at least one"),
    (lambda r: r["stages"][0].update(tokens="first"), "`tokens` is all, last"),
    (lambda r: r.update(operands={"x": {"values": [1.0], "shape": [2]}}), "1 values for shape"),
])
def test_malformed_recipes_are_refused(change, match):
    doc = copy.deepcopy(recipe_doc([add("chess", {"dose": 1.0})]))
    change(doc)
    with pytest.raises(AddressError, match=match):
        Recipe.from_mapping(doc)


SCHEMA_1 = {"schema_version": 1, "kind": "recipe", "name": "old", "model": {**MODEL, "revision_source": "pinned"},
            "book": book_doc({"chess": {13: [7956], 14: [8949]}}),
            "method": {"op": "mul", "site": "mlp.out.hook_in", "normalization": "layer_median",
                       "apply_to": "all_tokens"},
            "treatments": [{"set": "chess", "dose": 0.8}]}
SCHEMA_1_AS_2 = {"schema_version": 2, "kind": "recipe", "name": "old", "model": SCHEMA_1["model"],
                 "book": SCHEMA_1["book"],
                 "stages": [{"site": "mlp.out.hook_in", "tokens": "all", "ops": [
                     {"op": "scale", "target": {"set": "chess"},
                      "value": {"dose": 0.8, "transform": [{"statistic": {"name": "layer_median"}}]}}]}]}


def test_schema_1_migrates_to_schema_2_and_reads_with_a_warning():
    assert migrate(SCHEMA_1) == SCHEMA_1_AS_2  # the golden rewrite
    with pytest.warns(RecipeSchemaWarning, match="re-write it in schema 2"):
        recipe = Recipe.from_mapping(SCHEMA_1)
    assert recipe.to_mapping()["stages"] == SCHEMA_1_AS_2["stages"]
    with pytest.raises(AddressError, match="no schema 2 token scope"):
        migrate({**SCHEMA_1, "method": {**SCHEMA_1["method"], "apply_to": "first_token"}})


@pytest.mark.parametrize("apply_to, scope", [("prompt_tokens", "prompt"), ("generated_tokens", "generated")])
def test_prompt_and_generated_scopes_read_migrate_and_need_support(apply_to, scope):
    migrated = migrate({**SCHEMA_1, "method": {**SCHEMA_1["method"], "apply_to": apply_to}})
    assert migrated["stages"][0]["tokens"] == scope
    recipe = Recipe.from_mapping(migrated)
    assert recipe.stages[0].tokens.kind == scope and recipe.to_mapping()["stages"][0]["tokens"] == scope
    refusals = check_support(recipe, Support(ops=frozenset(OPS), transforms=frozenset({"statistic"})))
    assert refusals == [f"stage 0 (mlp.out.hook_in) op 0 (`scale`): token scope `{scope}` is not supported"]
    assert not check_support(recipe, Support(ops=frozenset(OPS), tokens=frozenset({"all", scope}),
                                             transforms=frozenset({"statistic"})))


def test_operands_round_trip_through_a_sidecar(tmp_path):
    sha = write_operands(tmp_path / "dirs.safetensors", {"concept": ([0.5, -1.0, 2.0, 0.0], [4])})
    assert read_operand(tmp_path / "dirs.safetensors", "concept", sha256=sha) == ((0.5, -1.0, 2.0, 0.0), (4,))
    doc = recipe_doc([{"site": "s", "ops": [{"op": "reject", "operand": "concept"}]}], operands={
        "concept": {"file": "dirs.safetensors", "key": "concept", "sha256": sha, "shape": [4], "basis": "unfolded"}})
    (tmp_path / "r.yaml").write_text(yaml.safe_dump(doc))
    recipe = load_recipe(tmp_path / "r.yaml")
    assert recipe.operands["concept"].load(recipe.base_dir) == ((0.5, -1.0, 2.0, 0.0), (4,))
    with pytest.raises(AddressError, match="sha256 differs"):
        write_operands(tmp_path / "dirs.safetensors", {"concept": ([1.0, 1.0, 1.0, 1.0], [4])})
        recipe.operands["concept"].load(recipe.base_dir)


def test_support_records_refuse_by_name():
    doc = recipe_doc([add("chess", {"dose": 1.0}),
                      {"site": "resid", "tokens": "last", "ops": [{"op": "reject", "operand": "concept"}]}],
                     operands={"concept": CONCEPT})
    recipe = Recipe.from_mapping(doc)
    tuning = Support(ops=frozenset({"add", "scale"}), targets=frozenset({"units"}),
                     sites=frozenset({"mlp.out.hook_in"}))
    refusals = check_support(recipe, tuning)
    assert any("op is not supported" in r for r in refusals) and any("token scope `last`" in r for r in refusals)
    assert any("site `resid`" in r for r in refusals) and any("target `activation`" in r for r in refusals)
    steering = Support(ops=frozenset(OPS), tokens=frozenset({"all", "last"}), bases=frozenset({"unfolded"}))
    assert check_support(recipe, steering) == []


def r(op, doses, name="r", revision="abc123"):
    sets = {f"s{i}": {layer: [u]} for i, ((layer, u), _) in enumerate(doses.items())}
    stages = [{"site": "mlp.out.hook_in", "ops": [{"op": op, "target": {"set": f"s{i}"}, "value": {"dose": d}}
                                                   for i, (_, d) in enumerate(doses.items())]}]
    return Recipe.from_mapping(recipe_doc(stages, sets, model={**MODEL, "revision": revision}, name=name),
                               source=name)


def test_same_op_recipes_merge_and_non_commuting_ops_need_an_order():
    a, b = r("add", {(15, 1): 0.5, (15, 2): 1.0}, "a"), r("add", {(15, 1): 0.25}, "b")
    (stage,) = compose([a, b])
    assert stage.unit_doses("add") == {(15, 1): 0.75, (15, 2): 1.0}
    m = r("scale", {(15, 1): 2.0}, "m")
    with pytest.raises(AddressError, match="does not commute"):
        compose([a, m])
    (ordered,) = compose([a, m], ordered=True)
    assert [b.op.op for b in ordered.ops] == ["add", "add", "scale"]
    disjoint = r("scale", {(16, 9): 2.0}, "d")
    (both,) = compose([a, disjoint])
    assert both.unit_doses("scale") == {(16, 9): 2.0}
    with pytest.raises(AddressError, match="over one revision"):
        compose([a, r("add", {(1, 1): 1.0}, "x", revision="other")])


def test_a_composition_document_loads_its_parts_in_order(tmp_path):
    for name, op in (("a", "add"), ("b", "scale")):
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(r(op, {(15, 1): 2.0}, name).to_mapping()))
    (tmp_path / "both.yaml").write_text(yaml.safe_dump({"schema_version": 2, "compose": ["a.yaml", "b.yaml"],
                                                         "order": "listed"}))
    (stage,) = load_recipe(tmp_path / "both.yaml").stages()
    assert [b.op.op for b in stage.ops] == ["add", "scale"]


def test_a_new_op_is_registered_as_a_spec():
    register_op(OpSpec("softcap", ("none",), value="required", composition="min"))
    try:
        (stage,) = compose([r("softcap", {(3, 3): 0.2}, "x"), r("softcap", {(3, 3): 0.9}, "y")])
        assert stage.unit_doses("softcap") == {(3, 3): 0.2}
        with pytest.raises(ValueError, match="already registered"):
            register_op(OpSpec("softcap"))
        with pytest.raises(ValueError, match="unknown"):
            register_op(OpSpec("weird", ("tensor",)), replace=True)
    finally:
        OPS.pop("softcap")


def test_a_recipe_selection_lands_on_the_qwen35_mlp_out_it_names():
    pytest.importorskip("transformers")
    from test_maps import tiny

    from venngram.maps import MapStructure, map_for

    model = tiny("Qwen3_5ForConditionalGeneration")
    recipe = Recipe.from_mapping(recipe_doc([add("chess", {"dose": 1.0})], {"chess": {1: [31], 2: [5]}}))
    space = ParamSpace([n for n, _ in model.named_parameters()], structure=MapStructure(model, map_for(model)))
    chess = recipe.book.sets()["chess"]
    for layer in (1, 2):
        (weight,) = select(f"component:mlp.out & layers:{layer}", space)
        width = dict(model.named_parameters())[weight].shape[1]  # down_proj's input axis is the units' axis
        assert chess.units.for_layer(layer, width) == {1: (31,), 2: (5,)}[layer]
