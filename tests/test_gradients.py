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
from test_recipe import recipe_doc
from test_resolve import space_of

from venngram import AddressBook, AddressUnresolvable, parse, select
from venngram.address import to_mapping, to_string
from venngram.errors import AddressError
from venngram.recipe import Recipe
from venngram.resolve import gradient_of
from venngram.support import TuningSupport


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


def test_a_gradient_mask_over_tensor_units_is_named_not_resolved(model):
    with pytest.raises(AddressUnresolvable, match="is a gradient mask, and this version does not resolve one yet"):
        select("module:q_proj & units:[0, 1] & of:grad", space_of(model))


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


def test_a_recipe_refuses_a_gradient_target():
    doc = recipe_doc([{"site": "mlp.out.hook_in", "ops": [{"op": "add", "target": {"set": "chess"},
                                                            "value": {"dose": 1.0}}]}])
    doc = copy.deepcopy(doc)
    doc["book"]["schema_version"] = 2
    doc["book"]["sets"]["chess"]["of"] = "grad"
    with pytest.raises(AddressError, match="addresses gradients"):
        Recipe.from_mapping(doc)
