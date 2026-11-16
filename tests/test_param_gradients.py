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
"""Parameter gradients: a recipe's `gradients` section, its schema steps and its support declaration."""
import pytest
import torch
from test_param_targets import PARAM_SETS
from test_recipe import CONCEPT, MODEL, recipe_doc

from venngram.book import AddressBook
from venngram.errors import AddressError
from venngram.ops import check_support, Support
from venngram.patches import tensor_sha256, write_weight_patch
from venngram.recipe import compose_weights, Composition, Recipe
from venngram.support import SupportRecord

GRAD_SETS = {**PARAM_SETS,
             "down_grad": "@down & of:grad",
             "neuron_grads": "@neurons & of:grad",
             "other_grads": "@others & of:grad"}


def gradients_doc(gradients, *, sets=GRAD_SETS, operands=None, name="g"):
    doc = recipe_doc([], sets, operands=operands, name=name)
    doc["schema_version"] = 4
    doc["book"]["schema_version"] = 2
    doc["gradients"] = gradients
    doc.pop("stages")
    return doc


def scale(target, dose):
    return {"op": "scale", "target": {"set": target}, "value": {"dose": dose}}


def test_a_gradients_op_targets_a_tensors_gradient_or_its_units_and_round_trips():
    masked = {"op": "replace", "target": {"set": "neuron_grads"}, "value": {"dose": 0.0}}
    recipe = Recipe.from_mapping(gradients_doc([masked, scale("down_grad", 0.5)]))
    units, whole = recipe.gradients
    assert (units.target_kind(recipe.book), whole.target_kind(recipe.book)) == ("param_units", "param")
    assert units.target_of(recipe.book) == whole.target_of(recipe.book) == "grad"
    assert units.unit_doses(recipe.book) == {(13, 7): 0.0, (13, 9): 0.0} and recipe.weights == ()
    doc = recipe.to_mapping()
    assert doc["schema_version"] == 4 and "stages" not in doc and doc["gradients"] == [masked, scale("down_grad", 0.5)]
    assert Recipe.from_mapping(doc).gradients == recipe.gradients
    projected = Recipe.from_mapping(gradients_doc([{"op": "project", "target": {"set": "down_grad"},
                                                    "operand": "concept"}], operands={"concept": CONCEPT}))
    assert projected.gradients[0].operand == "concept"


@pytest.mark.parametrize("op, match", [
    ({"op": "scale", "value": {"dose": 0.0}}, "a `gradients` op names the tensors it acts on"),
    (scale("down", 0.5), "addresses `of: value`; a `gradients` op acts on a tensor's gradient"),
    (scale("neuron_grads", 0.5) | {"tokens": "last"}, "acts on a step's accumulated gradient.*not `last`"),
    ({"op": "patch", "target": {"set": "down_grad"}, "operand": "poles"}, "acts on \\['activation'\\], not `param`"),
    ({"op": "replace_with_error", "target": {"set": "down_grad"}, "value": {"dose": 1.0}},
     "addresses `of: grad`, and this op acts on \\['value'\\]"),
])
def test_a_misused_gradients_op_is_refused_by_name(op, match):
    poles = {"values": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], "shape": [2, 4], "basis": "paper"}
    with pytest.raises(AddressError, match=match):
        Recipe.from_mapping(gradients_doc([op], operands={"concept": CONCEPT, "poles": poles}))


def test_a_weight_patch_applies_to_a_value_never_to_a_gradient(tmp_path):
    base = torch.zeros(2, 3)
    write_weight_patch(tmp_path / "edit.yaml", name="rank-one", target="component:mlp.out & layers:13", shape=(2, 3),
                       form="low_rank", tensors={"u": [1.0, 2.0], "v": [0.0, 1.0, 0.0]},
                       base_sha256=tensor_sha256(base))
    doc = gradients_doc([{"op": "add", "target": {"set": "down_grad"}, "operand": "edit"}],
                        operands={"edit": {"patch": "edit.yaml"}})
    with pytest.raises(AddressError, match="is a weight patch, which only a `weights` op applies"):
        Recipe.from_mapping(doc, base_dir=tmp_path)


def test_gradients_need_recipe_schema_4_and_a_book_treatment_with_them_book_schema_4():
    doc = gradients_doc([scale("neuron_grads", 0.5)])
    doc["schema_version"] = 3
    with pytest.raises(AddressError, match="`gradients` needs recipe schema 4, not 3"):
        Recipe.from_mapping(doc)
    book = {**gradients_doc([])["book"], "schema_version": 3,
            "treatments": {"damped": {"model": MODEL, "gradients": [scale("neuron_grads", 0.5)]}}}
    with pytest.raises(AddressError, match="treatments \\[.damped.\\] have `gradients` or a `readout`, which "
                                           "need address book schema 4"):
        AddressBook.from_mapping(book)
    read = AddressBook.from_mapping({**book, "schema_version": 4})
    assert read.treatments["damped"].gradients[0].target_kind(read) == "param_units"
    assert read.treatments["damped"].to_treatment()["gradients"] == [scale("neuron_grads", 0.5)]


def test_gradient_ops_from_two_recipes_compose_when_they_commute():
    a = Recipe.from_mapping(gradients_doc([scale("neuron_grads", 0.5)], name="a"), source="a")
    b = Recipe.from_mapping(gradients_doc([scale("neuron_grads", 2.0)], name="b"), source="b")
    assert [x.recipe.name for x in Composition((a, b)).gradients()] == ["a", "b"]
    c = Recipe.from_mapping(gradients_doc([{"op": "replace", "target": {"set": "other_grads"}, "value": {"dose": 0.0}}],
                                          name="c"), source="c")
    with pytest.raises(AddressError, match="gradients op `replace` does not commute with `scale`"):
        compose_weights([a, c], section="gradients")  # unit 9 is in both
    assert len(Composition((a, c), ordered=True).gradients()) == 2 and Composition((a, c)).weights() == ()


def test_a_consumer_declares_the_ops_it_applies_to_parameter_gradients_at_support_record_schema_5():
    recipe = Recipe.from_mapping(gradients_doc([scale("neuron_grads", 0.5)]))
    weights_only = Support(ops=frozenset({"scale"}), targets=frozenset({"param", "param_units"}))
    assert check_support(recipe, weights_only) == [
        "gradients op 0 (`scale`): the op is not applied to parameter gradients (`gradient_ops`: [])"]
    grads = Support(ops=frozenset({"scale"}), targets=frozenset({"param_units"}), gradient_ops=frozenset({"scale"}))
    assert check_support(recipe, grads) == []
    doc = SupportRecord(sections={"steering": grads}).to_mapping()
    assert doc["schema_version"] == 5 and doc["sections"]["steering"]["gradient_ops"] == ["scale"]
    assert SupportRecord.from_mapping(doc).steering == grads
    with pytest.raises(ValueError, match="`gradient_ops` needs support record schema 5, not 4"):
        SupportRecord.from_mapping({**doc, "schema_version": 4})
    assert SupportRecord(sections={"steering": weights_only}).to_mapping()["schema_version"] == 4
