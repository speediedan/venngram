"""Parameter targets: a treatment's `weights` section, weight patches, their schema steps and conformance vectors."""
import copy
import hashlib

import pytest
import torch
import yaml
from test_treatment import add, CONCEPT, MODEL, read_treatment, treatment_doc

from venngram.book import AddressBook
from venngram.errors import AddressError
from venngram.manifest import build_manifest
from venngram.ops import (check_ops, check_support, op_cases, op_spec_from_mapping, read_op_specs, Support,
                          write_op_specs)
from venngram.patches import read_weight_patch, tensor_sha256, weight_patch_from_mapping, write_weight_patch
from venngram.support import SupportRecord
from venngram.treatment import compose_weights, Composition, Treatment

# tensors, addressed without a site: a whole matrix, and neurons along mlp.out's unit axis
PARAM_SETS = {"down": {"component": "mlp.out", "layers": [13], "leaf": "weight"},
              "neurons": {"component": "mlp.out", "layers": [13], "leaf": "weight", "units": [7, 9]},
              "others": {"component": "mlp.out", "layers": [13], "leaf": "weight", "units": [9, 11]},
              "both": "@neurons | @others",
              "chess": {13: [7956]}}


def weights_doc(weights, *, stages=None, operands=None, sets=PARAM_SETS, name="w"):
    doc = treatment_doc(stages or [], sets, operands=operands, name=name)
    doc["weights"] = weights
    if stages is None:
        doc.pop("stages")
    return doc


def zero(target):
    return {"op": "scale", "target": {"set": target}, "value": {"dose": 0.0}}


def test_a_weights_op_targets_a_tensor_or_its_units_and_round_trips():
    treatment = read_treatment(weights_doc([zero("neurons"), {"op": "scale", "target": {"set": "down"},
                                                                "value": {"dose": 0.5}}]))
    units, whole = treatment.weights
    assert (units.target_kind(treatment.book), whole.target_kind(treatment.book)) == ("param_units", "param")
    assert units.unit_doses(treatment.book) == {(13, 7): 0.0, (13, 9): 0.0} and treatment.stages == ()
    doc = treatment.to_mapping()
    assert treatment.schema_version == 3 and "stages" not in doc and len(doc["weights"]) == 2
    assert Treatment.from_mapping(doc, treatment.book).weights == treatment.weights
    composed = read_treatment(weights_doc([zero("both")]))
    assert list(composed.weights[0].unit_doses(composed.book)) == [(13, 7), (13, 9), (13, 11)]
    plain = read_treatment(treatment_doc([add("chess", {"dose": 1.0})]))
    assert plain.schema_version == 2 and plain.weights == ()  # schema 3 only when it is needed


@pytest.mark.parametrize("op, match", [
    ({"op": "scale", "value": {"dose": 0.0}}, "a `weights` op names the tensors it acts on"),
    ({**zero("chess")}, "addresses a site .* names no `point`, `features` or `subspace`"),
    ({**zero("neurons"), "tokens": "last"}, "its token scope is `all`, not `last`"),
    ({"op": "patch", "target": {"set": "down"}, "operand": "poles"}, "acts on \\['activation'\\], not `param`"),
    ({"op": "reject", "target": {"set": "neurons"}, "operand": "concept"}, "not `param_units`"),
    ({"op": "replace_with_error", "target": {"set": "down"}, "value": {"dose": 1.0}}, "not `param`"),
    ({"op": "scale", "target": {"set": "grads"}, "value": {"dose": 1.0}}, "a `weights` op changes a tensor's value"),
])
def test_a_misused_weights_op_is_refused_by_name(op, match):
    poles = {"values": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], "shape": [2, 4], "basis": "paper"}
    sets = {**PARAM_SETS, "grads": {"component": "mlp.out", "layers": [13], "of": "grad"}}
    doc = weights_doc([op], sets=sets, operands={"concept": CONCEPT, "poles": poles})
    doc["book"]["schema_version"] = 2
    with pytest.raises(AddressError, match=match):
        read_treatment(doc)


def test_a_book_treatment_with_weights_needs_book_schema_3():
    book = {**treatment_doc([], PARAM_SETS)["book"],
            "treatments": {"off": {"model": MODEL, "weights": [zero("neurons")]}}}
    with pytest.raises(AddressError, match="treatments \\['off'\\] have `weights`, which need address book schema 3"):
        AddressBook.from_mapping(book)
    loaded = AddressBook.from_mapping({**book, "schema_version": 3})
    assert loaded.treatments["off"].weights[0].target_kind(loaded) == "param_units"
    assert loaded.treatments["off"].to_mapping()["weights"] == [zero("neurons")]


def test_a_delta_operand_is_a_weight_patch_applied_only_by_a_weights_op(tmp_path):
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    patch = write_weight_patch(tmp_path / "edit.yaml", name="rank-one", target="component:mlp.out & layers:13",
                               shape=(2, 3), form="low_rank", tensors={"u": [1.0, 2.0], "v": [0.0, 1.0, 0.0]},
                               base_sha256=tensor_sha256(base))
    assert patch.rank == 1 and patch.delta() == (0.0, 1.0, 0.0, 0.0, 2.0, 0.0)
    assert read_weight_patch(tmp_path / "edit.yaml") == patch
    doc = weights_doc([{"op": "add", "target": {"set": "down"}, "operand": "edit"}], operands={"edit": {
        "patch": "edit.yaml"}})
    treatment = read_treatment(doc, base_dir=tmp_path)
    edit = treatment.operands["edit"]
    assert edit.kind == "delta" and edit.shape == (2, 3) and edit.load(tmp_path)[0] == patch.delta()
    assert treatment.to_mapping()["operands"] == {"edit": {"patch": "edit.yaml"}}
    in_a_stage = treatment_doc([{"site": "mlp.out.hook_in", "ops": [{"op": "add", "operand": "edit"}]}],
                            operands={"edit": {"patch": "edit.yaml"}})
    with pytest.raises(AddressError, match="is a weight patch, which only a `weights` op applies"):
        read_treatment(in_a_stage, base_dir=tmp_path)
    (tmp_path / "book.yaml").write_text(yaml.safe_dump({**doc["book"], "schema_version": 3,
                                                         "treatments": {"edit": treatment.to_mapping()}}))
    entries = {e.path: e.kind for e in build_manifest(tmp_path, [("address_book", "book.yaml")]).documents}
    assert entries == {"book.yaml": "address_book", "edit.yaml": "weight_patch", "edit.safetensors": "weight_delta"}


@pytest.mark.parametrize("form, tensors, delta", [
    ("sparse", {"indices": [0, 5], "values": [9.0, -1.0]}, (9.0, 0.0, 0.0, 0.0, 0.0, -1.0)),
    ("dense", {"delta": [1.0, 0.0, 0.0, 0.0, 0.0, 2.0]}, (1.0, 0.0, 0.0, 0.0, 0.0, 2.0)),
])
def test_every_form_reads_back_its_delta(tmp_path, form, tensors, delta):
    write_weight_patch(tmp_path / "p.yaml", name="p", target="t", shape=(2, 3), form=form, tensors=tensors,
                       base_sha256="00" * 32)
    assert read_weight_patch(tmp_path / "p.yaml").delta() == delta


@pytest.mark.parametrize("change, match", [
    (lambda d: d.update(form="svd"), "`form` is one of"),
    (lambda d: d.update(shape=[2, 3, 4]), "one or two positive dimensions"),
    (lambda d: d.pop("base_sha256"), "gives \\['base_sha256'\\]"),
    (lambda d: d.update(dtype="bfloat16"), "`dtype` is `float32`"),
    (lambda d: d.update(sha256="11" * 32), "its sha256 differs"),
    (lambda d: d.update(bogus=1), "unknown keys \\['bogus'\\]"),
    (lambda d: d.update(schema_version=2), "weight_patch schema 2 is newer"),
    (lambda d: d.update(shape=[3, 2]), "are \\[rows, r\\] and \\[cols, r\\]"),
])
def test_a_malformed_weight_patch_is_refused_by_name(tmp_path, change, match):
    write_weight_patch(tmp_path / "p.yaml", name="p", target="t", shape=(2, 3), form="low_rank",
                       tensors={"u": [1.0, 2.0], "v": [0.0, 1.0, 0.0]}, base_sha256="00" * 32)
    doc = yaml.safe_load((tmp_path / "p.yaml").read_text())
    change(doc)
    with pytest.raises(Exception, match=match):
        weight_patch_from_mapping(doc, base_dir=tmp_path)


def test_the_tensor_hash_is_of_its_float32_bytes():
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert tensor_sha256(t) == hashlib.sha256(t.numpy().astype("<f4").tobytes()).hexdigest()
    assert tensor_sha256(t.t()) != tensor_sha256(t) and tensor_sha256(t.double()) == tensor_sha256(t)


def test_weight_ops_from_two_treatments_compose_when_they_commute():
    a = read_treatment(weights_doc([zero("neurons")], name="a"), source="a")
    b = read_treatment(weights_doc([{"op": "scale", "target": {"set": "neurons"}, "value": {"dose": 2.0}}],
                                        name="b"), source="b")
    assert [x.treatment.name for x in compose_weights([a, b])] == ["a", "b"]  # two scales multiply, in either order
    c = read_treatment(weights_doc([{"op": "replace", "target": {"set": "others"}, "value": {"dose": 1.0}}],
                                        name="c"), source="c")
    with pytest.raises(AddressError, match="weights op `replace` does not commute with `scale`"):
        compose_weights([a, c])  # unit 9 is in both
    assert len(Composition((a, c), ordered=True).weights()) == 2


def test_a_consumer_declares_parameter_targets_and_the_record_takes_schema_4():
    treatment = read_treatment(weights_doc([zero("neurons")]))
    steering = Support(ops=frozenset({"scale"}))
    assert check_support(treatment, steering) == ["weights op 0 (`scale`): target `param_units` is not supported"]
    weights = Support(ops=frozenset({"scale"}), targets=frozenset({"units", "param_units"}))
    assert check_support(treatment, weights) == []
    record = SupportRecord(sections={"treatments": weights})
    doc = record.to_mapping()
    assert doc["schema_version"] == 9 and SupportRecord.from_mapping(doc).treatments == weights
    old = {"schema_version": 4, "kind": "support_record", "sections": {"steering": doc["sections"]["treatments"]}}
    assert SupportRecord.from_mapping(old).treatments == weights
    with pytest.raises(ValueError, match="parameter targets \\['param_units'\\] need support record schema 4"):
        SupportRecord.from_mapping({**old, "schema_version": 3})


def test_an_op_declaring_a_parameter_target_needs_op_specs_schema_2():
    spec = op_spec_from_mapping({"name": "soft_zero", "targets": ["param_units"], "value": "none"})
    doc = write_op_specs([spec], name="weight-ops")
    assert doc["schema_version"] == 2 and read_op_specs(doc, register=False) == (spec,)
    with pytest.raises(ValueError, match="needs op_specs schema 2, not 1"):
        read_op_specs({**doc, "schema_version": 1}, register=False)
    assert op_spec_from_mapping({"name": "plain"}).targets == ("activation", "units", "features")


def _evaluate(case):
    """A reference evaluator for the parameter vectors, in torch."""
    i = case.inputs
    w = torch.tensor(i["w"]).reshape([int(d) for d in i["shape"]])
    if case.op == "scale":
        dim, units = int(i["dim"][0]), [int(u) for u in i["units"]]
        edited = w.clone()
        edited.index_copy_(dim, torch.tensor(units), w.index_select(dim, torch.tensor(units)) * i["dose"][0])
        out = {"w": edited.flatten().tolist()}
        if "x" in i:
            x = torch.tensor(i["x"])
            scaled = x.clone()
            scaled[units] *= i["dose"][0]
            out.update(y_weight=(edited @ x).tolist(), y_activation=(w @ scaled).tolist())
        return out
    if case.op == "add" and "b" in i:
        b, x, j, d = torch.tensor(i["b"]), torch.tensor(i["x"]), int(i["units"][0]), i["dose"][0]
        patched = b + d * w[:, j]
        shifted = x.clone()
        shifted[j] += d
        return {"b": patched.tolist(), "y_weight": (w @ x + patched).tolist(),
                "y_activation": (w @ shifted + b).tolist()}
    if case.op == "add":
        return {"w": (w + torch.outer(torch.tensor(i["u"]), torch.tensor(i["v"]))).flatten().tolist()}
    if case.op == "reject":
        u = torch.tensor(i["u"])
        return {"w": ((torch.eye(len(u)) - i["fraction"][0] * torch.outer(u, u)) @ w).flatten().tolist()}
    flat = w.flatten().clone()
    flat[[int(k) for k in i["indices"]]] = torch.tensor(i["values"])
    return {"w": flat.tolist()}


def test_the_parameter_vectors_hold_and_state_each_activation_counterpart():
    cases = op_cases(target="param") + op_cases(target="param_units")
    assert len(cases) == 7 and all(c.op in ("scale", "add", "reject", "replace") for c in cases)
    assert check_ops(_evaluate, target="param") == [] and check_ops(_evaluate, target="param_units") == []
    for case in cases:
        if "y_weight" in case.expect:
            assert case.expect["y_weight"] == case.expect["y_activation"]

    def unscaled(case):
        out = _evaluate(case)
        return {**out, "w": case.inputs["w"]} if case.op == "scale" else out
    assert {f.split(":")[0] for f in check_ops(unscaled, target="param_units")} == {
        "param-units-scale-scales-the-units-slices",
        "param-units-scale-along-the-first-dimension-of-a-transposed-weight"}


def test_fraction_may_exceed_one_on_activations_too():
    doc = treatment_doc([{"site": "s", "ops": [{"op": "reject", "operand": "concept", "params": {"fraction": 6.0}}]}],
                     operands={"concept": CONCEPT})
    assert read_treatment(copy.deepcopy(doc)).stages[0].ops[0].params == {"fraction": 6.0}
