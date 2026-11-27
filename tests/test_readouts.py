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
"""Readouts: what a treatment's gradients are taken of, as weighted terms of registered reductions over addresses."""
import math

import pytest
from test_treatment import CONCEPT, MODEL, read_treatment, treatment_doc

from venngram.book import AddressBook
from venngram.errors import AddressError
from venngram.ops import check_support, Support
from venngram.readouts import check_readouts, READOUTS, readout_cases, register_readout
from venngram.support import SupportRecord
from venngram.treatment import Treatment

# the logits of an answer and a foil token (unembed's output units are the vocabulary), a residual site, and a
# gradient target at layer 12's MLP output
SETS = {"answer": {"point": "unembed.hook_out", "units": [42]},
        "foil": {"point": "unembed.hook_out", "units": [7]},
        "resid": {"point": "blocks.resid_post", "layers": [20]},
        "mlp_grad": {"point": "mlp.out.hook_in", "layers": [12], "units": [3, 4], "of": "grad"},
        "mlp_val": {"point": "mlp.out.hook_in", "layers": [12], "units": [3, 4]},
        "weights": {"component": "mlp.out", "layers": [12], "leaf": "weight"},
        "answer_grad": {"point": "unembed.hook_out", "units": [42], "of": "grad"}}
LOGIT_DIFF = {"terms": [{"reduce": "sum", "set": "answer", "tokens": "last", "weight": 1.0},
                        {"reduce": "sum", "set": "foil", "tokens": "last", "weight": -1.0}]}
ZERO_GRAD = {"site": "mlp.out.hook_in", "ops": [{"op": "scale", "target": {"set": "mlp_grad"}, "value": {"dose": 0.0}}]}
STEER = {"site": "mlp.out.hook_in", "ops": [{"op": "add", "target": {"set": "mlp_val"}, "value": {"dose": 1.0}}]}


def readout_doc(stages, *, readout=LOGIT_DIFF, operands=None):
    doc = treatment_doc(stages, SETS, operands=operands)
    doc["book"]["schema_version"] = 2
    if readout is not None:
        doc["readout"] = readout
    return doc


def _evaluate(case):
    """A reference evaluator: the weighted sum of each term's reduction."""
    total = 0.0
    for t in case.terms:
        if t.reduce == "sum":
            v = sum(sum(row) for row in t.values)
        elif t.reduce == "project":
            v = sum(sum(x * o for x, o in zip(row, t.operand)) for row in t.values)
        elif t.reduce == "norm":
            v = sum(math.sqrt(sum(x * x for x in row)) for row in t.values)
        else:
            v = sum(sum(row) for row in t.values) / sum(len(row) for row in t.values)  # `mean`, registered below
        total += t.weight * v
    return total


def test_the_core_reductions_have_meanings_and_vectors_a_consumer_is_checked_against():
    assert {"sum", "project", "norm"} <= set(READOUTS) and all(READOUTS[n].meaning for n in ("sum", "project", "norm"))
    assert {t.reduce for c in readout_cases() for t in c.terms} >= {"sum", "project", "norm"}
    assert check_readouts(_evaluate) == []
    wrong = check_readouts(lambda case: _evaluate(case) + (1.0 if case.name == "logit_difference" else 0.0))
    assert wrong == ["logit_difference: expected 1.75, got 2.75"]


def test_a_registered_reduction_states_its_meaning_and_brings_its_vectors():
    case = {"name": "mean_over_units_and_tokens", "terms": [{"reduce": "mean", "values": [[1.0, 3.0], [5.0, 7.0]]}],
            "expect": 4.0}
    with pytest.raises(ValueError, match="state what it computes"):
        register_readout("mean", meaning=" ", cases=[case])
    with pytest.raises(ValueError, match="no conformance case uses it"):
        register_readout("mean", meaning="the mean", cases=[{**case, "terms": [{"reduce": "sum", "values": [[1.0]]}]}])
    with pytest.raises(ValueError, match="gives `mean` an operand it does not take"):
        register_readout("mean", meaning="the mean",
                         cases=[{**case, "terms": [{**case["terms"][0], "operand": [1, 1]}]}])
    with pytest.raises(ValueError, match="already registered"):
        register_readout("sum", meaning="again", cases=[case])
    register_readout("mean", meaning="the addressed values' mean over their units and tokens", cases=[case])
    try:
        assert check_readouts(_evaluate) == [] and "mean" in READOUTS
        treatment = read_treatment(readout_doc([ZERO_GRAD], readout={"terms": [{"reduce": "mean", "set": "resid"}]}))
        assert treatment.readout.terms[0].reduce == "mean"
    finally:
        from venngram import readouts

        readouts.READOUTS.pop("mean")
        readouts._REGISTERED_CASES.pop("mean")


def test_a_treatment_states_its_readout_and_a_stage_may_override_it():
    project = {"terms": [{"reduce": "project", "set": "resid", "operand": "probe"}]}
    treatment = read_treatment(readout_doc([ZERO_GRAD, {**ZERO_GRAD, "readout": project}, STEER],
                                           operands={"probe": CONCEPT}))
    assert [t.weight for t in treatment.readout.terms] == [1.0, -1.0]
    assert treatment.stages[0].readout is None and treatment.stages[1].readout.terms[0].operand == "probe"
    doc = treatment.to_mapping()
    assert treatment.schema_version == 4
    assert doc["readout"]["terms"][1] == {"reduce": "sum", "set": "foil", "tokens": "last", "weight": -1.0}
    assert Treatment.from_mapping(doc, treatment.book).readout == treatment.readout
    only_stage = read_treatment(readout_doc([{**ZERO_GRAD, "readout": LOGIT_DIFF}], readout=None))
    assert only_stage.readout is None and only_stage.schema_version == 4


@pytest.mark.parametrize("stages, readout, match", [
    ([ZERO_GRAD, {**ZERO_GRAD, "readout": LOGIT_DIFF}], None,
     "stage 0 acts on gradients .* states the `readout` they are gradients of"),
    ([STEER], LOGIT_DIFF, "the treatment states a readout, but no stage takes its gradients from it"),
    ([{**STEER, "readout": LOGIT_DIFF}], None, "stage 0 states a readout, but none of its ops acts on a gradient"),
    ([ZERO_GRAD], {"terms": [{"reduce": "softmax", "set": "answer"}]}, "unknown reduction 'softmax'"),
    ([ZERO_GRAD], {"terms": [{"reduce": "sum", "set": "weights"}]}, "addresses tensors .* a readout reads activations"),
    ([ZERO_GRAD], {"terms": [{"reduce": "sum", "set": "answer_grad"}]}, "a readout reads values"),
    ([ZERO_GRAD], {"terms": [{"reduce": "project", "set": "resid"}]}, "`project` takes an operand"),
    ([ZERO_GRAD], {"terms": [{"reduce": "sum", "set": "answer", "operand": "probe"}]}, "`sum` takes no operand"),
    ([ZERO_GRAD], {"terms": [{"reduce": "sum", "set": "answer", "weight": "1"}]}, "`weight` is a finite number"),
    ([ZERO_GRAD], {"expr": "logit[42] - logit[7]"}, "it carries no expression"),
])
def test_a_misplaced_or_malformed_readout_is_refused_by_name(stages, readout, match):
    with pytest.raises(AddressError, match=match):
        read_treatment(readout_doc(stages, readout=readout, operands={"probe": CONCEPT}))


def test_a_gradient_stage_without_a_readout_keeps_reading_below_schema_4():
    treatment = read_treatment(readout_doc([ZERO_GRAD], readout=None))  # the consumer's own scalar
    assert treatment.readout is None and treatment.schema_version == 2


def test_a_book_treatment_with_a_readout_is_book_schema_4():
    body = {"model": MODEL, "readout": LOGIT_DIFF, "stages": [ZERO_GRAD]}
    book = {**readout_doc([])["book"], "schema_version": 3, "treatments": {"diff": body}}
    with pytest.raises(AddressError, match="have `gradients` or a `readout`, which need address book schema 4"):
        AddressBook.from_mapping(book)
    loaded = AddressBook.from_mapping({**book, "schema_version": 4})
    assert loaded.treatments["diff"].readout.terms[0].set == "answer"
    assert loaded.treatments["diff"].to_mapping()["readout"]["terms"][0]["reduce"] == "sum"


def test_a_treatment_readout_refuses_execution_passes():
    body = {"model": MODEL, "readout": LOGIT_DIFF, "stages": [ZERO_GRAD]}
    book = {"schema_version": 12, "sets": {**SETS, "answer": {**SETS["answer"], "passes": 2}},
            "treatments": {"diff": body}}
    with pytest.raises(AddressError, match=r"readout term 0: set `answer` uses `passes`"):
        AddressBook.from_mapping(book)


def test_a_consumer_declares_the_reductions_it_computes():
    treatment = read_treatment(readout_doc([ZERO_GRAD]))
    grads = Support(ops=frozenset({"scale"}), of=frozenset({"value", "grad"}))
    assert check_support(treatment, grads) == ["readout term 0: reduction `sum` is not computed (`readouts`: [])",
                                            "readout term 1: reduction `sum` is not computed (`readouts`: [])"]
    computes = Support(ops=frozenset({"scale"}), of=frozenset({"value", "grad"}), readouts=frozenset({"sum"}))
    assert check_support(treatment, computes) == []
    doc = SupportRecord(sections={"treatments": computes}).to_mapping()
    assert doc["schema_version"] == 9 and doc["sections"]["treatments"]["readouts"] == ["sum"]
    assert SupportRecord.from_mapping(doc).treatments == computes
    old = {"schema_version": 5, "kind": "support_record", "sections": {"steering": doc["sections"]["treatments"]}}
    with pytest.raises(ValueError, match="`readouts` needs support record schema 5, not 4"):
        SupportRecord.from_mapping({**old, "schema_version": 4})
