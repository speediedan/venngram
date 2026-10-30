import copy

import pytest
import yaml

from venngram import ParamSpace, select
from venngram.errors import AddressError
from venngram.recipe import compose, load_recipe, OPS, Recipe, register_op

MODEL = {"id": "Qwen/Qwen3.5-4B", "revision": "abc123"}


def book_doc(name, units, model=MODEL):
    """A checkpoint book holding one set, ``units`` as ``{layer: [unit, ...]}``."""
    return {"schema_version": 1, "name": name, "portability": "checkpoint",
            "applies_to": {"models": [{"id": model["id"], "revision": model["revision"]}]},
            "sets": {name: {"point": "mlp.out.hook_in", "layers": sorted(units),
                            "units": {k: sorted(v) for k, v in sorted(units.items())}}}}


def recipe_doc(doses, op="add", model=MODEL, site="mlp.out.hook_in", name="two-units"):
    """A recipe with an inline book, ``doses`` as ``{(layer, unit): dose}``: one dose when they agree, else a table."""
    units: dict[int, list[int]] = {}
    for layer, unit in doses:
        units.setdefault(layer, []).append(unit)
    values = set(doses.values())
    if len(values) == 1:
        treatment = {"set": name, "dose": values.pop()}
    else:
        table: dict[int, dict[int, float]] = {}
        for (layer, unit), dose in doses.items():
            table.setdefault(layer, {})[unit] = dose
        treatment = {"set": name, "doses": table}
    return {"schema_version": 1, "kind": "recipe", "name": name,
            "model": {**model, "revision_source": "pinned"}, "book": book_doc(name, units, model),
            "method": {"op": op, "site": site, "normalization": "raw", "apply_to": "all_tokens"},
            "treatments": [treatment], "sampling": {"temperature": 0.7},
            "evidence": {"examples": [{"prompt": "hi", "output": "hello"}]}, "metadata": {"author": "someone"}}


TWO = {(13, 7956): -1.0, (14, 8949): 1.0}


def test_a_recipe_reads_and_round_trips():
    recipe = Recipe.from_mapping(recipe_doc(TWO))
    assert recipe.method.site == "mlp.out.hook_in" and recipe.method.op == "add"
    assert recipe.stage().doses == TWO
    assert recipe.evidence["examples"] == [{"prompt": "hi", "output": "hello"}]
    assert recipe.sampling == {"temperature": 0.7}
    assert recipe.metadata == {"author": "someone"} and recipe.revision_source == "pinned"
    assert Recipe.from_mapping(recipe.to_mapping()).stage() == recipe.stage()


def test_one_dose_covers_every_unit_of_a_set():
    recipe = Recipe.from_mapping(recipe_doc({(15, 4041): 0.8, (15, 17): 0.8}))
    assert recipe.to_mapping()["treatments"] == [{"set": "two-units", "dose": 0.8}]
    assert recipe.stage().doses == {(15, 17): 0.8, (15, 4041): 0.8}


def test_a_recipe_and_its_book_round_trip_through_files(tmp_path):
    doc = recipe_doc(TWO)
    (tmp_path / "book.yaml").write_text(yaml.safe_dump(doc["book"]))
    (tmp_path / "recipe.yaml").write_text(yaml.safe_dump({**doc, "book": "book.yaml"}))
    loaded = load_recipe(tmp_path / "recipe.yaml")
    assert isinstance(loaded, Recipe) and loaded.stage().doses == TWO
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
    recipe = copy.deepcopy(recipe_doc(TWO))
    change(recipe)
    with pytest.raises(AddressError, match=match):
        Recipe.from_mapping(recipe)


def recipe_with(op, doses, revision="abc123"):
    return Recipe.from_mapping(recipe_doc(doses, op=op, model={**MODEL, "revision": revision}))


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


def test_a_recipe_selection_lands_on_the_qwen35_mlp_out_it_names():
    pytest.importorskip("transformers")
    from test_maps import tiny

    from venngram.maps import MapStructure, map_for

    model = tiny("Qwen3_5ForConditionalGeneration")
    recipe = Recipe.from_mapping(recipe_doc({(1, 31): 1.0, (2, 5): 1.0}))
    space = ParamSpace([n for n, _ in model.named_parameters()], structure=MapStructure(model, map_for(model)))
    chess = recipe.book.sets()["two-units"]
    for layer in (1, 2):
        (weight,) = select(f"component:mlp.out & layers:{layer}", space)
        width = dict(model.named_parameters())[weight].shape[1]  # down_proj's input axis is the units' axis
        assert chess.units.for_layer(layer, width) == {1: (31,), 2: (5,)}[layer]
