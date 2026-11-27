import copy

import pytest
import yaml

from venngram import ParamSpace, select
from venngram.errors import AddressError
from venngram.operands import read_operand, write_operands
from venngram.ops import check_support, OPS, OpSpec, register_op, Support
from venngram.book import AddressBook
from venngram.treatment import compose, Composition, Treatment

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


def treatment_doc(stages, sets=None, operands=None, model=MODEL, name="r"):
    """A treatment's body with the ``book`` it treats (a mapping) and its ``name``, as :func:`read_treatment` takes
    them."""
    doc = {"name": name, "model": {**model, "revision_source": "pinned"},
           "book": book_doc(sets or {"chess": {13: [7956], 14: [8949]}}, model), "stages": stages}
    if operands:
        doc["operands"] = operands
    return doc


def read_treatment(doc, *, source="<inline>", base_dir=None):
    """The treatment ``doc`` (see :func:`treatment_doc`) describes, read over its book."""
    body = {k: v for k, v in doc.items() if k not in ("book", "name")}
    book = AddressBook.from_mapping(doc["book"], source=f"{source}: book", base_dir=base_dir, inline=True)
    return Treatment.from_mapping(body, book, name=doc.get("name"), source=source, base_dir=base_dir)


def add(target, value, site="mlp.out.hook_in"):
    return {"site": site, "ops": [{"op": "add", "target": {"set": target}, "value": value}]}


CONCEPT = {"values": [1.0, 0.0, 0.0, 0.0], "shape": [4], "basis": "unfolded"}
POLES = {"values": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], "shape": [2, 4], "basis": "paper"}


def test_a_treatment_reads_and_round_trips():
    doc = treatment_doc([
        add("chess", {"doses": {13: {7956: -1.0}, 14: {8949: 1.0}}}),
        {"site": "blocks.resid_post", "tokens": "last", "ops": [
            {"op": "reject", "operand": "concept", "params": {"fraction": 0.5}},
            {"op": "patch", "operand": "poles"},
            {"op": "clamp", "operand": "concept", "params": {"max": 2.0}}]}],
        operands={"concept": CONCEPT, "poles": POLES})
    treatment = read_treatment(doc)
    first, second = treatment.stages
    assert first.ops[0].unit_doses(treatment.book) == {(13, 7956): -1.0, (14, 8949): 1.0}
    assert second.tokens.kind == "last" and [o.op for o in second.ops] == ["reject", "patch", "clamp"]
    assert treatment.operands["poles"].kind == "pair" and second.ops[0].params == {"fraction": 0.5}
    again = Treatment.from_mapping(treatment.to_mapping(), treatment.book, name=treatment.name)
    assert again.to_mapping() == treatment.to_mapping() and again.stages == treatment.stages


def test_values_from_tables_and_features_targets():
    sets = {"top": {"space": "transcoder", "features": "3,7", "layers": [10]}}
    doc = treatment_doc([{"site": "transcoder.features", "tokens": {"positions": [5, 6]}, "ops": [
        {"op": "replace", "target": {"set": "top"},
         "value": {"from": {"table": "scores", "column": "score"},
                   "transform": [{"sign_aware_scale": {"factor": 4.0}}, "abs"]}}]}], sets=sets)
    treatment = read_treatment(doc)
    op = treatment.stages[0].ops[0]
    assert op.target_kind(treatment.book) == "features" and [t.name for t in op.value.transform] == [
        "sign_aware_scale", "abs"]
    assert treatment.stages[0].tokens.positions == (5, 6)


UNIT_SETS = {"a": {12: [1, 2, 3]}, "b": {12: [3, 4]}, "c": {14: [5, 6]}, "ctl": {12: [2], 14: [6]}}


def composed(stages, sets, **kwargs):
    """A treatment over a book of schema 2 (the composition tests also use ``of``)."""
    doc = treatment_doc(stages, sets, **kwargs)
    doc["book"]["schema_version"] = 2
    return read_treatment(doc, source=kwargs.get("name", "<inline>"))


@pytest.mark.parametrize("expr, pairs", [
    ("@a & @b", [(12, 3)]),
    ("@a | @b", [(12, 1), (12, 2), (12, 3), (12, 4)]),
    ("@a - @b", [(12, 1), (12, 2)]),
    ("(@a | @c) - @ctl", [(12, 1), (12, 3), (14, 5)]),
    ("(@a | @c) & layers:14", [(14, 5), (14, 6)]),
    ("(@a | @c) - layers:14", [(12, 1), (12, 2), (12, 3)]),
    ("@either - @ctl", [(12, 1), (12, 3), (12, 4)]),
    ({"all": ["@a", "@b"], "expect": 1}, [(12, 3)]),
])
def test_a_unit_target_composes_sets_with_the_set_algebra(expr, pairs):
    sets = {**UNIT_SETS, "either": "@a | @b", "t": expr}
    treatment = composed([add("t", {"dose": 0.5})], sets)
    op = treatment.stages[0].ops[0]
    assert op.target_kind(treatment.book) == "units"
    assert list(op.unit_doses(treatment.book)) == pairs and set(op.unit_doses(treatment.book).values()) == {0.5}


def test_a_composed_unit_target_takes_per_unit_doses_commutes_when_disjoint_and_reads_of():
    sets = {**UNIT_SETS, "both": "@a & @b", "rest": "@a - @b"}
    doses = {"doses": {12: {3: 2.0, 1: 9.0}}}
    treatment = composed([add("both", doses)], sets)
    assert treatment.stages[0].ops[0].unit_doses(treatment.book) == {(12, 3): 2.0}
    scaled = composed([{"site": "mlp.out.hook_in", "ops": [
        {"op": "scale", "target": {"set": "rest"}, "value": {"dose": 2.0}}]}], sets, name="s")
    (stage,) = compose([treatment, scaled])
    assert stage.unit_doses("add") == {(12, 3): 2.0} and stage.unit_doses("scale") == {(12, 1): 2.0, (12, 2): 2.0}
    grads = composed([add("g", {"dose": 1.0})], {**UNIT_SETS, "g": "(@a & @b) & of:grad"})
    assert grads.stages[0].ops[0].target_of(grads.book) == "grad"
    assert grads.stages[0].ops[0].unit_doses(grads.book) == {(12, 3): 1.0}
    values_only = Support(ops=frozenset({"add"}), targets=frozenset({"units"}))
    assert check_support(treatment, values_only) == []
    assert any("addresses `of: grad`" in r for r in check_support(grads, values_only))


def test_a_unit_targets_members_do_not_depend_on_its_value():
    sets = {**UNIT_SETS, "both": "(@a | @c) - @ctl", "top": {"space": "transcoder", "features": "3,7", "layers": [10]}}
    from_table = {"from": {"table": "scores", "column": "score"}, "transform": ["abs"]}
    members = [(12, 1), (12, 3), (14, 5)]
    for value in (from_table, {"dose": 0.5}, {"doses": {12: {3: 2.0}}}):
        treatment = composed([add("both", value)], sets)
        assert treatment.stages[0].ops[0].unit_members(treatment.book) == members
    (op,) = composed([add("both", from_table)], sets).stages[0].ops
    with pytest.raises(AddressError, match="has no unit doses"):
        op.unit_doses(treatment.book)
    single = composed([add("a", from_table)], sets)
    assert single.stages[0].ops[0].unit_members(single.book) == [(12, 1), (12, 2), (12, 3)]
    whole = composed([{"site": "resid.post", "ops": [{"op": "scale", "value": {"dose": 2.0}}]}], sets)
    with pytest.raises(AddressError, match="has no unit members"):
        whole.stages[0].ops[0].unit_members(whole.book)
    top = composed([{"site": "transcoder.features", "ops": [
        {"op": "replace", "target": {"set": "top"}, "value": {"dose": 1.0}}]}], sets)
    with pytest.raises(AddressError, match="must be one address with `units`"):
        top.stages[0].ops[0].unit_members(top.book)


@pytest.mark.parametrize("expr, match", [
    ("@a - @a", "set `t` names no units: its composition is empty"),
    ("@a | @other_site", r"composes units at different sites \(point:mlp.in.hook_out, point:mlp.out.hook_in\)"),
    ("@a & @top", "composes the feature set .* feature sets do not compose in a treatment"),
    ("@a | layers:12", "unites a `layers`-only term"),
    ("@a & layers:-4:", "depends on the model's depth"),
    ("@a & point:mlp.out.hook_in", "names no explicit units"),
    ("@a | @deep", "names its units for every layer of a slice or parity"),
    ("(@a & of:value) | @g", "its terms disagree on `of`"),
    ({"all": ["@a", "@b"], "expect": 2}, r"name 1 units where `expect` requires 2"),
])
def test_a_composed_unit_target_is_refused_by_name(expr, match):
    sets = {**UNIT_SETS, "t": expr, "top": {"space": "transcoder", "features": "3,7", "layers": [12]},
            "other_site": {"point": "mlp.in.hook_out", "layers": [12], "units": [9]},
            "deep": {"point": "mlp.out.hook_in", "layers": "-4:", "units": [1]},
            "g": {"point": "mlp.out.hook_in", "layers": [12], "units": [1], "of": "grad"}}
    with pytest.raises(AddressError, match=match):
        composed([add("t", {"dose": 1.0})], sets)


@pytest.mark.parametrize("op, match", [
    ({"op": "lerp"}, "unknown op 'lerp'"),
    ({"op": "mul", "value": {"dose": 2.0}}, "`mul` is now `scale`"),
    ({"op": "reject", "operand": "nobasis"}, "must declare its `basis`"),
    ({"op": "patch", "operand": "concept"}, "is not a pair operand"),
    ({"op": "reject", "operand": "concept", "params": {"fraction": -1.0}}, "lies in \\[0.0, inf\\]"),
    ({"op": "reject", "operand": "concept", "params": {"width": 1.0}}, "unknown params"),
    ({"op": "clamp", "operand": "concept"}, "at least one of \\['min', 'max'\\]"),
    ({"op": "reject"}, "needs an operand"),
    ({"op": "add"}, "needs a `value` or an operand"),
    ({"op": "scale", "operand": "concept", "value": {"dose": 1.0}}, "is not a  operand|is not a valid operand"),
    ({"op": "project", "operand": "concept", "value": {"dose": 1.0}}, "takes no `value`"),
    ({"op": "reject", "operand": "concept", "target": {"set": "chess"}}, "acts on \\['activation', 'param'\\]"),
    ({"op": "add", "target": {"set": "missing"}, "value": {"dose": 1.0}}, "names a set of the book"),
    ({"op": "add", "value": {"dose": 1.0, "doses": {}}}, "exactly one of"),
    ({"op": "add", "value": {"dose": 1.0, "transform": ["cube"]}}, "unknown transform 'cube'"),
    ({"op": "add", "value": {"doses": {1: {2: 1.0}}}}, "per-unit `doses` need a target set with `units`"),
    ({"op": "add", "value": {"dose": 1.0}, "tokens": "last"}, "cannot override its stage's `tokens`"),
])
def test_misused_ops_are_refused_by_name(op, match):
    doc = treatment_doc([{"site": "s", "ops": [op]}],
                     operands={"concept": CONCEPT, "nobasis": {"values": [1.0, 0.0], "shape": [2]}})
    with pytest.raises(AddressError, match=match):
        read_treatment(doc)


@pytest.mark.parametrize("target", [
    {"set": "selected", "graph": "missing"},
    {"set": "selected", "units": [2]},
    {"set": "selected", 1: "ignored"},
    {"set": ["selected"]},
    {"set": {"selected": True}},
    {"set": None},
    {"sets": "selected"},
    "selected",
])
@pytest.mark.parametrize("section", ["stages", "weights", "gradients"])
@pytest.mark.parametrize("in_book", [False, True], ids=["treatment", "book-treatment"])
def test_malformed_target_is_refused_in_every_op_section(target, section, in_book):

    address = {"layers": [1], "units": [1]}
    if section == "stages":
        address["point"] = "mlp.out.hook_in"
    elif section == "gradients":
        address["of"] = "grad"
    doc = treatment_doc([], sets={"selected": address})
    doc["book"]["schema_version"] = 4
    op = {"op": "replace", "target": target, "value": {"dose": 0.0}}
    doc[section] = [{"site": "mlp.out.hook_in", "ops": [op]}] if section == "stages" else [op]
    if in_book:
        book = {**doc["book"], "treatments": {"invalid": {"model": doc["model"], section: doc[section]}}}
        load = lambda: AddressBook.from_mapping(book)
    else:
        load = lambda: read_treatment(doc)
    with pytest.raises(AddressError, match=r"`target` names a set of the book.*contains only `set`"):
        load()


@pytest.mark.parametrize("change, match", [
    (lambda r: r["model"].pop("revision"), "one revision only"),
    (lambda r: r["model"].update(revision="other"), "the book pins"),
    (lambda r: r.update(schema_version=2), r"is written in the book it treats, so it has no \['schema_version'\]"),
    (lambda r: r.update(stages=[]), "at least one"),
    (lambda r: r["stages"][0].update(tokens="first"), "`tokens` is all, last"),
    (lambda r: r.update(operands={"x": {"values": [1.0], "shape": [2]}}), "1 values for shape"),
    (lambda r: r["stages"][0]["ops"][0]["value"].update(transform=[{"sign_of": {"column": "score"}}]),
     r"\['sign_of'\] read the `from` table, so apply only to a `from` value, not `dose`"),
    (lambda r: r["stages"][0]["ops"][0].update(value={"doses": {13: {7956: 1.0}},
                                                      "transform": [{"sign_aware_scale": {"factor": 2.0}}]}),
     r"\['sign_aware_scale'\] read the `from` table.*not `doses`"),
])
def test_malformed_treatments_are_refused(change, match):
    doc = copy.deepcopy(treatment_doc([add("chess", {"dose": 1.0})]))
    change(doc)
    with pytest.raises(AddressError, match=match):
        read_treatment(doc)


@pytest.mark.parametrize("scope", ["prompt", "generated"])
def test_prompt_and_generated_scopes_read_and_need_support(scope):
    stage = {"site": "mlp.out.hook_in", "tokens": scope, "ops": [
        {"op": "scale", "target": {"set": "chess"},
         "value": {"dose": 0.8, "transform": [{"statistic": {"name": "layer_median"}}]}}]}
    treatment = read_treatment(treatment_doc([stage]))
    assert treatment.stages[0].tokens.kind == scope and treatment.to_mapping()["stages"][0]["tokens"] == scope
    refusals = check_support(treatment, Support(ops=frozenset(OPS), transforms=frozenset({"statistic"})))
    assert refusals == [f"stage 0 (mlp.out.hook_in) op 0 (`scale`): token scope `{scope}` is not supported"]
    assert not check_support(treatment, Support(ops=frozenset(OPS), tokens=frozenset({"all", scope}),
                                                transforms=frozenset({"statistic"})))


def test_operands_round_trip_through_a_sidecar(tmp_path):
    sha = write_operands(tmp_path / "dirs.safetensors", {"concept": ([0.5, -1.0, 2.0, 0.0], [4])})
    assert read_operand(tmp_path / "dirs.safetensors", "concept", sha256=sha) == ((0.5, -1.0, 2.0, 0.0), (4,))

    body = {"model": MODEL, "stages": [{"site": "s", "ops": [{"op": "reject", "operand": "concept"}]}], "operands": {
        "concept": {"file": "dirs.safetensors", "key": "concept", "sha256": sha, "shape": [4], "basis": "unfolded"}}}
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({**book_doc({"chess": {13: [7956]}}), "treatments": {"r": body}}))
    treatment = AddressBook.from_file(tmp_path / "b.yaml").treatments["r"]
    assert treatment.operands["concept"].load(treatment.base_dir) == ((0.5, -1.0, 2.0, 0.0), (4,))
    with pytest.raises(AddressError, match="sha256 differs"):
        write_operands(tmp_path / "dirs.safetensors", {"concept": ([1.0, 1.0, 1.0, 1.0], [4])})
        treatment.operands["concept"].load(treatment.base_dir)


def test_support_records_refuse_by_name():
    doc = treatment_doc([add("chess", {"dose": 1.0}),
                      {"site": "resid", "tokens": "last", "ops": [{"op": "reject", "operand": "concept"}]}],
                     operands={"concept": CONCEPT})
    treatment = read_treatment(doc)
    tuning = Support(ops=frozenset({"add", "scale"}), targets=frozenset({"units"}),
                     sites=frozenset({"mlp.out.hook_in"}))
    refusals = check_support(treatment, tuning)
    assert any("op is not supported" in r for r in refusals) and any("token scope `last`" in r for r in refusals)
    assert any("site `resid`" in r for r in refusals) and any("target `activation`" in r for r in refusals)
    steering = Support(ops=frozenset(OPS), tokens=frozenset({"all", "last"}), bases=frozenset({"unfolded"}))
    assert check_support(treatment, steering) == []


def r(op, doses, name="r", revision="abc123"):
    sets = {f"s{i}": {layer: [u]} for i, ((layer, u), _) in enumerate(doses.items())}
    stages = [{"site": "mlp.out.hook_in", "ops": [{"op": op, "target": {"set": f"s{i}"}, "value": {"dose": d}}
                                                   for i, (_, d) in enumerate(doses.items())]}]
    return read_treatment(treatment_doc(stages, sets, model={**MODEL, "revision": revision}, name=name),
                               source=name)


def test_same_op_treatments_merge_and_non_commuting_ops_need_an_order():
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


def test_a_books_treatments_compose_in_the_order_given():

    treatments = {name: r(op, {(15, 1): 2.0}, name).to_mapping() for name, op in (("a", "add"), ("b", "scale"))}
    book = AddressBook.from_mapping({**book_doc({"s0": {15: [1]}}), "treatments": treatments})
    (stage,) = Composition((book.treatments["a"], book.treatments["b"]), ordered=True).stages()
    assert [b.op.op for b in stage.ops] == ["add", "scale"]
    with pytest.raises(AddressError, match="does not commute"):
        Composition((book.treatments["a"], book.treatments["b"])).stages()


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


def test_a_treatment_selection_lands_on_the_qwen35_mlp_out_it_names():
    pytest.importorskip("transformers")
    from test_maps import tiny

    from venngram.maps import MapStructure, map_for

    model = tiny("Qwen3_5ForConditionalGeneration")
    treatment = read_treatment(treatment_doc([add("chess", {"dose": 1.0})], {"chess": {1: [31], 2: [5]}}))
    space = ParamSpace([n for n, _ in model.named_parameters()], structure=MapStructure(model, map_for(model)))
    chess = treatment.book.sets()["chess"]
    for layer in (1, 2):
        (weight,) = select(f"component:mlp.out & layers:{layer}", space)
        width = dict(model.named_parameters())[weight].shape[1]  # down_proj's input axis is the units' axis
        assert chess.units.for_layer(layer, width) == {1: (31,), 2: (5,)}[layer]


def test_a_book_carries_its_treatments_by_name():
    from venngram.support import SupportRecord

    body = {"model": {**MODEL, "revision_source": "pinned"}, "stages": [add("chess", {"dose": 0.5})]}
    book = AddressBook.from_mapping({**book_doc({"chess": {13: [7956], 14: [8949]}}),
                                     "treatments": {"steer": body}})
    steer = book.treatments["steer"]
    assert isinstance(steer, Treatment) and steer.name == "steer" and steer.book is book
    assert steer.stages[0].ops[0].unit_doses(book) == {(13, 7956): 0.5, (14, 8949): 0.5}
    again = Treatment.from_mapping(steer.to_mapping(), book, name="steer")
    assert again.to_mapping() == steer.to_mapping() and again.stages == steer.stages
    assert "treatments" not in book.extra and "book" not in steer.to_mapping()
    adds = SupportRecord.from_mapping({"schema_version": 2, "kind": "support_record", "name": "adds",
                                       "sections": {"steering": {"ops": ["add"]}}})
    scales = SupportRecord.from_mapping({"schema_version": 2, "kind": "support_record", "name": "scales",
                                         "sections": {"steering": {"ops": ["scale"]}}})
    assert adds.check_treatments(book) == []
    assert scales.check_treatments(book) and scales.check_treatments(book)[0].startswith("treatment `steer`: ")


def test_a_treatment_is_refused_by_name_when_it_carries_a_header_or_names_no_set():

    base = book_doc({"chess": {13: [7956]}})
    body = {"model": MODEL, "stages": [add("chess", {"dose": 0.5})]}
    with pytest.raises(AddressError, match="treatment `t`: a treatment is written in the book it treats"):
        AddressBook.from_mapping({**base, "treatments": {"t": {**body, "book": "./other.yaml"}}})
    with pytest.raises(AddressError, match="names a set of the book"):
        AddressBook.from_mapping({**base, "treatments": {"t": {**body, "stages": [add("nope", {"dose": 1.0})]}}})
    with pytest.raises(AddressError, match="`treatments` maps a treatment name"):
        AddressBook.from_mapping({**base, "treatments": []})
