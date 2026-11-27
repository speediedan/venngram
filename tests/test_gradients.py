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
"""Gradient addresses: the ``of`` field qualifies what an address selects without filtering it."""
import copy

import pytest
from test_treatment import read_treatment, treatment_doc
from test_resolve import space_of

from venngram import AddressBook, AddressUnresolvable, parse, select
from venngram.address import to_mapping, to_string
from venngram.errors import AddressError
from venngram.ops import check_support, Support
from venngram.resolve import gradient_masks, gradient_of, split_of
from venngram.support import SupportRecord, TuningSupport


def test_of_qualifies_without_filtering(model):
    space = space_of(model)
    assert select("role:adapter & layers:-2: & of:grad", space) == select("role:adapter & layers:-2:", space)
    assert gradient_of("role:adapter & layers:-2: & of:grad") == "grad"
    assert gradient_of("role:adapter") == "value" and gradient_of({"role": "adapter", "of": "value"}) == "value"


def test_of_round_trips_in_both_forms():
    expr = parse({"role": "adapter", "layers": "-4:", "of": "grad"})
    assert to_string(expr) == "role:adapter & layers:-4: & of:grad"
    assert to_mapping(parse(to_string(expr))) == {"role": "adapter", "layers": "-4:", "of": "grad"}
    with pytest.raises(AddressError, match="`of` field is value or grad"):
        parse("of:hessian")


def test_of_follows_references_and_refuses_a_mix():
    sets = {"top": "role:adapter & layers:-4:", "top_q": "@top & module:q_proj", "grads": "@top & of:grad",
            "values": "@top & of:value"}
    assert gradient_of("(@top - @top_q) & of:grad", sets) == "grad"
    assert gradient_of("@grads - @top_q", sets) == "grad"
    with pytest.raises(AddressUnresolvable, match="disagree on `of`.*`grad` in @grads.*`value` in @values"):
        gradient_of("@grads | @values", sets)
    with pytest.raises(AddressUnresolvable, match="no set is named `@missing`"):
        gradient_of("@missing & of:grad", sets)


def test_site_gradients_qualify_a_point():
    assert gradient_of("point:resid_post & layers:10 & of:grad") == "grad"
    assert parse("point:resid_post & layers:10 & of:grad").target == "site"  # still a site, resolved by a consumer


Q = "role:base & module:q_proj & leaf:weight & layers:-2:"


def widths(model):
    return {n: p.shape[0] for n, p in model.named_parameters()}


def test_units_under_of_grad_are_a_gradient_mask_over_the_first_axis(model):
    space = space_of(model)
    masked = f"{Q} & units:[0, 2] & of:grad"
    assert select(masked, space) == select(Q, space) == ["layers.4.attn.q_proj.weight", "layers.5.attn.q_proj.weight"]
    assert gradient_masks(masked, space, widths=widths(model)) == {"layers.4.attn.q_proj.weight": (0, 2),
                                                                   "layers.5.attn.q_proj.weight": (0, 2)}
    per_layer = gradient_masks(f"{Q} & units:{{4: [1], 5: '2:'}} & of:grad", space, widths=widths(model))
    assert per_layer == {"layers.4.attn.q_proj.weight": (1,), "layers.5.attn.q_proj.weight": (2, 3)}
    only_4 = gradient_masks(f"{Q} & units:{{4: [1]}} & of:grad", space, widths=widths(model))
    assert only_4["layers.5.attn.q_proj.weight"] == ()  # a layer the set does not name keeps no gradient
    whole = gradient_masks(f"{Q} & of:grad", space, widths=widths(model))
    assert whole["layers.4.attn.q_proj.weight"] == (0, 1, 2, 3)


def test_a_mask_follows_references_into_a_book(model):
    sets = {"q": Q, "q_rows": "@q & units:[3]", "q_grad": "@q_rows & of:grad"}
    assert set(gradient_masks("@q_grad", space_of(model), sets, widths=widths(model)).values()) == {(3,)}


def test_unit_sets_combine_exactly_under_intersections_and_differences(model):
    space, w = space_of(model), widths(model)
    nested = gradient_masks(f"(({Q} & units:[0]) | role:adapter) & layers:-1: & of:grad", space, widths=w)
    assert nested["layers.5.attn.q_proj.weight"] == (0,)
    assert all(v == tuple(range(w[n])) for n, v in nested.items() if n != "layers.5.attn.q_proj.weight")
    rest = gradient_masks(f"role:base & of:grad - ({Q} & units:[0])", space, widths=w)
    assert rest["layers.4.attn.q_proj.weight"] == rest["layers.5.attn.q_proj.weight"] == (1, 2, 3)
    assert gradient_masks(f"{Q} & units:[0] & units:[1] & of:grad", space, widths=w) == {
        "layers.4.attn.q_proj.weight": (), "layers.5.attn.q_proj.weight": ()}
    with pytest.raises(AddressUnresolvable, match="needs an `of: grad` address"):
        gradient_masks(f"{Q} & units:[0]", space, widths=w)


K = "role:base & module:k_proj & leaf:weight & layers:-1:"


def test_a_union_takes_each_disjuncts_mask_and_unites_them_per_tensor(model):
    space, w = space_of(model), widths(model)
    rows = f"({Q} & units:[0, 2] & of:grad) | ({K} & units:[1] & of:grad)"
    assert gradient_masks(rows, space, widths=w) == {"layers.4.attn.q_proj.weight": (0, 2),
                                                    "layers.5.attn.q_proj.weight": (0, 2),
                                                    "layers.5.attn.k_proj.weight": (1,)}
    overlap = f"({Q} & units:[0] & of:grad) | ({Q} & layers:-1: & units:[3] & of:grad)"
    assert gradient_masks(overlap, space, widths=w) == {"layers.4.attn.q_proj.weight": (0,),
                                                       "layers.5.attn.q_proj.weight": (0, 3)}
    whole = f"({Q} & units:[1] & of:grad) | {Q} & layers:-1: & of:grad"  # a disjunct without units keeps every index
    assert gradient_masks(whole, space, widths=w)["layers.5.attn.q_proj.weight"] == (0, 1, 2, 3)
    sets = {"q_rows": f"{Q} & units:[0] & of:grad", "k_rows": f"{K} & units:[1] & of:grad", "rows": "@q_rows | @k_rows"}
    assert gradient_masks("@rows", space, sets, widths=w) == (  # a union opened through a reference
        gradient_masks(f"({Q} & units:[0] & of:grad) | ({K} & units:[1] & of:grad)", space, widths=w))


def test_a_union_may_unite_a_value_part_and_a_gradient_part(model):
    space = space_of(model)
    mixed = f"({K} & of:value) | ({Q} & units:[1] & of:grad)"
    value, grad = split_of(mixed)
    assert (value, grad) == (parse(f"{K} & of:value"), parse(f"{Q} & units:[1] & of:grad"))
    assert select(mixed, space) == select(f"{K} | {Q}", space)  # units in the gradient part mask, they do not filter
    with pytest.raises(AddressUnresolvable, match="disagree on `of`.*unites a value part and a gradient part"):
        gradient_of(mixed)
    assert split_of(f"{K} | {Q} & of:grad") == (None, parse(f"{K} | {Q} & of:grad"))  # unstated `of` follows the set
    assert split_of(K) == (parse(K), None)
    sets = {"v": f"{K} & of:value", "g": f"{Q} & of:grad", "both": "@v | @g", "plain": f"{K} | {Q}"}
    value, grad = split_of("@both", sets)
    assert (value, grad) == (parse("@v"), parse("@g"))
    assert split_of("@plain & of:grad", sets) == (None, parse("@plain & of:grad"))  # an agreeing set stays whole


@pytest.mark.parametrize("address, match", [
    (f"({K} & of:value) | {Q} | ({Q} & of:grad)", "`role:base & module:q_proj & layers:-2: & leaf:weight` states no "
                                                  "`of` beside disjuncts that differ on it"),
    (f"({K} & of:value) & ({Q} & of:grad)", "mixes them in an intersection or a difference"),
    (f"({K} & of:value & units:[0]) | ({Q} & of:grad)", "tuned through a unit adapter"),
])
def test_a_mixed_set_that_cannot_be_split_is_refused_by_name(model, address, match):
    with pytest.raises(AddressUnresolvable, match=match):
        select(address, space_of(model))


def test_units_on_tensors_without_of_grad_keep_their_refusal(model):
    with pytest.raises(AddressUnresolvable, match="tuned through a unit adapter.*or masked as gradients"):
        select(f"{Q} & units:[0]", space_of(model))


BOOK = {"name": "grads", "sets": {"top": "role:adapter & layers:-4:", "top_grad": "@top & of:grad"}}


def test_a_book_using_of_is_schema_2_and_others_stay_readable_as_1():
    assert AddressBook.from_mapping({**BOOK, "schema_version": 2}).schema_version == 2
    with pytest.raises(AddressError, match=r"the sets \['top_grad'\] use the `of` field.*schema 2, not 1"):
        AddressBook.from_mapping({**BOOK, "schema_version": 1})
    assert AddressBook.from_mapping({"schema_version": 1, "sets": {"top": "role:adapter"}}).schema_version == 1
    assert AddressBook.from_mapping({"sets": BOOK["sets"]}, inline=True).schema_version == 2


def test_tuning_consumers_opt_in_to_gradients():
    book = AddressBook.from_mapping({**BOOK, "schema_version": 2})
    assert TuningSupport().check_book(book, sets=["top"]) == []
    (refusal,) = TuningSupport().check_book(book, sets=["top_grad"])
    assert "the fields ['of'] are not honoured" in refusal
    opted_in = TuningSupport(fields=TuningSupport().fields | {"of"})
    assert opted_in.check_book(book) == [] and "of" in opted_in.to_mapping()["fields"]


def gradient_treatment():
    doc = copy.deepcopy(treatment_doc([{"site": "mlp.out.hook_in", "ops": [
        {"op": "scale", "target": {"set": "chess"}, "value": {"dose": 0.0}}]}]))
    doc["book"]["schema_version"] = 2
    doc["book"]["sets"]["chess"]["of"] = "grad"
    return read_treatment(doc)


def test_a_treatment_may_act_on_gradients_and_a_consumer_must_declare_it():
    treatment = gradient_treatment()
    (refusal,) = check_support(treatment, Support(ops=frozenset({"scale"})))
    assert "addresses `of: grad`, which this consumer does not act on (it declares ['value'])" in refusal
    assert check_support(treatment, Support(ops=frozenset({"scale"}), of=frozenset({"value", "grad"}))) == []


def test_a_steering_of_declaration_is_support_record_schema_3():
    steering = {"ops": ["scale"], "of": ["value", "grad"]}
    record = SupportRecord.from_mapping({"schema_version": 3, "kind": "support_record", "sections": {
        "steering": steering}})
    assert record.treatments.of == {"value", "grad"} and record.to_mapping()["schema_version"] == 9
    assert record.to_mapping()["sections"]["treatments"]["of"] == ["grad", "value"]
    assert SupportRecord.from_mapping(record.to_mapping()).treatments.of == record.treatments.of
    with pytest.raises(ValueError, match="`of` needs support record schema 3, not 2"):
        SupportRecord.from_mapping({"schema_version": 2, "kind": "support_record", "sections": {"steering": steering}})
    with pytest.raises(ValueError, match="not at the top level of a schema 1 one"):
        SupportRecord.from_mapping({"schema_version": 1, "kind": "support_record", **steering})
    plain = SupportRecord.from_mapping({"schema_version": 2, "kind": "support_record", "sections": {
        "steering": {"ops": ["scale"]}}})
    assert plain.to_mapping()["schema_version"] == 9 and "of" not in plain.to_mapping()["sections"]["treatments"]
