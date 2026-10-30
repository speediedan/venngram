import copy

import pytest
import yaml

from venngram import ParamSpace, select
from venngram.errors import AddressError
from venngram.recipe import compose, from_neuronary, load_recipe, OPS, Recipe, register_op

# the shape of a neuronary recipe (schema/recipe.schema.json): "L13N7956-1.0 L14N8949+1.0" on Qwen3.5-4B
NEURONARY = {
    "id": "two-neurons", "name": "Two neurons", "description": "a made-up recipe in neuronary's shape",
    "model": "Qwen/Qwen3.5-4B", "tags": ["test"], "author": {"handle": "someone"}, "date": "2026-10-30",
    "neurons": [{"layer": 13, "neuron": 7956, "dose": -1.0}, {"layer": 14, "neuron": 8949, "dose": 1.0}],
    "method": {"op": "add", "site": "mlp.down_proj.input"},
    "sampling": {"temperature": 0.7},
    "examples": [{"prompt": "hi", "output": "hello"}],
}


def imported(doc=NEURONARY, revision="abc123"):
    book, recipe = from_neuronary(doc, revision=revision)
    return book, Recipe.from_mapping(recipe)


def test_a_neuronary_recipe_imports_losslessly():
    book, recipe = imported()
    assert book["portability"] == "checkpoint" and book["applies_to"]["models"] == [
        {"id": "Qwen/Qwen3.5-4B", "revision": "abc123"}]
    assert recipe.method.site == "mlp.out.hook_in" and recipe.method.op == "add"
    assert recipe.stage().doses == {(13, 7956): -1.0, (14, 8949): 1.0}
    assert recipe.evidence["examples"] == NEURONARY["examples"] and recipe.sampling == {"temperature": 0.7}
    assert recipe.metadata["imported_from"] == "neuronary" and recipe.metadata["author"] == {"handle": "someone"}
    assert recipe.revision_source == "given"
    assert Recipe.from_mapping(recipe.to_mapping()).stage() == recipe.stage()


def test_one_dose_for_every_neuron_is_kept_as_one_dose_and_an_unpinned_revision_is_resolved():
    doc = {**NEURONARY, "neurons": [{"layer": 15, "neuron": n, "dose": 0.8} for n in (4041, 17)]}
    _, recipe = from_neuronary(doc, resolve_revision=lambda model_id: f"resolved-{model_id}")
    assert recipe["treatments"] == [{"set": "two-neurons", "dose": 0.8}]
    assert recipe["model"] == {"id": "Qwen/Qwen3.5-4B", "revision": "resolved-Qwen/Qwen3.5-4B",
                               "revision_source": "resolved-at-import"}


def test_experts_and_unknown_sites_are_refused_by_name():
    with pytest.raises(AddressError, match="MoE expert"):
        from_neuronary({**NEURONARY, "neurons": [{"layer": 1, "neuron": 2, "dose": 1, "expert": 3}]}, revision="r")
    with pytest.raises(AddressError, match="unknown neuronary site"):
        from_neuronary({**NEURONARY, "method": {"site": "attn.o.input"}}, revision="r")


def test_a_recipe_and_its_book_round_trip_through_files(tmp_path):
    book, recipe = from_neuronary(NEURONARY, revision="abc123")
    (tmp_path / "book.yaml").write_text(yaml.safe_dump(book))
    (tmp_path / "recipe.yaml").write_text(yaml.safe_dump({**recipe, "book": "book.yaml"}))
    loaded = load_recipe(tmp_path / "recipe.yaml")
    assert isinstance(loaded, Recipe) and loaded.stage().doses == {(13, 7956): -1.0, (14, 8949): 1.0}
    assert loaded.to_mapping()["book"] == "book.yaml"


@pytest.mark.parametrize("change, match", [
    (lambda r: r["method"].update(op="lerp"), "unknown op 'lerp'"),
    (lambda r: r["model"].pop("revision"), "one revision only"),
    (lambda r: r["model"].update(revision="other"), "the book pins"),
    (lambda r: r["treatments"][0].update(dose=1.0), "not both or neither"),
    (lambda r: r["treatments"][0].update(set="missing"), "names a set of the book"),
    (lambda r: r.update(schema_version=2), "`schema_version` must be 1"),
])
def test_malformed_recipes_are_refused(change, match):
    _, recipe = from_neuronary(NEURONARY, revision="abc123")
    recipe = copy.deepcopy(recipe)
    change(recipe)
    with pytest.raises(AddressError, match=match):
        Recipe.from_mapping(recipe)


def recipe_with(op, doses, revision="abc123", site="mlp.down_proj.input"):
    neurons = [{"layer": layer, "neuron": n, "dose": d} for (layer, n), d in doses.items()]
    _, doc = from_neuronary({**NEURONARY, "neurons": neurons, "method": {"op": op, "site": site}}, revision=revision)
    doc["method"]["op"] = op
    return Recipe.from_mapping(doc)


def test_same_op_recipes_merge_and_mixed_ops_need_an_order():
    a = recipe_with("add", {(15, 1): 0.5, (15, 2): 1.0})
    b = recipe_with("add", {(15, 1): 0.25})
    (stage,) = compose([a, b])
    assert stage.doses == {(15, 1): 0.75, (15, 2): 1.0}
    m = recipe_with("mul", {(15, 1): 2.0})
    with pytest.raises(AddressError, match="do not commute"):
        compose([a, m])
    assert [(s.method.op, dict(s.doses)) for s in compose([a, m], ordered=True)] == [
        ("add", {(15, 1): 0.5, (15, 2): 1.0}), ("mul", {(15, 1): 2.0})]
    disjoint = recipe_with("mul", {(16, 9): 2.0})
    assert {s.method.op for s in compose([a, disjoint])} == {"add", "mul"}


def test_a_composition_is_over_one_revision_and_one_site():
    with pytest.raises(AddressError, match="over one revision"):
        compose([recipe_with("add", {(1, 1): 1.0}), recipe_with("add", {(1, 1): 1.0}, revision="other")])
    elsewhere = recipe_with("add", {(1, 1): 1.0}).to_mapping()
    elsewhere["method"]["site"] = "attn.o.hook_in"
    with pytest.raises(AddressError, match="only the op may differ"):
        compose([recipe_with("add", {(1, 1): 1.0}), Recipe.from_mapping(elsewhere)])


def test_a_composition_document_loads_its_parts_in_order(tmp_path):
    for name, op in (("a", "add"), ("b", "mul")):
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(recipe_with(op, {(15, 1): 2.0}).to_mapping()))
    (tmp_path / "both.yaml").write_text(yaml.safe_dump({"schema_version": 1, "compose": ["a.yaml", "b.yaml"],
                                                         "order": "listed"}))
    assert [s.method.op for s in load_recipe(tmp_path / "both.yaml").stages()] == ["add", "mul"]


def test_a_new_op_is_registered_with_how_its_doses_combine():
    register_op("clamp", max)
    try:
        stage, = compose([recipe_with("clamp", {(3, 3): 0.2}), recipe_with("clamp", {(3, 3): 0.9})])
        assert stage.doses == {(3, 3): 0.9}
        with pytest.raises(ValueError, match="already registered"):
            register_op("clamp", min)
    finally:
        OPS.pop("clamp")


def test_an_imported_selection_lands_on_the_qwen35_mlp_out_it_names():
    pytest.importorskip("transformers")
    from test_maps import tiny

    from venngram.maps import MapStructure, map_for

    model = tiny("Qwen3_5ForConditionalGeneration")
    _, recipe = imported({**NEURONARY, "neurons": [{"layer": 1, "neuron": 31, "dose": 1.0},
                                                   {"layer": 2, "neuron": 5, "dose": 1.0}]})
    space = ParamSpace([n for n, _ in model.named_parameters()], structure=MapStructure(model, map_for(model)))
    chess = recipe.book.sets()["two-neurons"]
    for layer in (1, 2):
        (weight,) = select(f"component:mlp.out & layers:{layer}", space)
        width = dict(model.named_parameters())[weight].shape[1]  # down_proj's input axis is the units' axis
        assert chess.units.for_layer(layer, width) == {1: (31,), 2: (5,)}[layer]
