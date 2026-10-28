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
import pytest

from venngram.address import (And, is_address, LayerSet, Minus, Or, parse, Ref, references, rename_refs, Term,
                              to_mapping, to_string, UnitSet)
from venngram.errors import AddressError, VenngramError

FORMS = [
    "role:adapter",
    "role:adapter:top",
    "role:adapter & layers:-4:",
    "role:adapter - @top",
    "role:adapter & module:q_proj | @lora.top",
    'role:adapter & name:"layers\\.2[4-7]\\."',
    "point:mlp.out.hook_in & layers:12 & units:[431, 1007]",
    "role:adapter - layers:0:3 - layers:-2:",
    "(role:adapter | role:intervention) & layers:even",
    "features:bsf_l17 & space:jlens & point:resid_post & layers:17 & units:[412]",
    "point:resid_post & units:{12: [431], 14: [88]}",
    "layers:0:8 & layers:4: & role:adapter",
    "subspace:entailment_dir & point:resid_post & layers:17",
]


@pytest.mark.parametrize("text", FORMS)
def test_the_string_and_mapping_forms_round_trip(text):
    expr = parse(text)
    assert to_string(parse(to_string(expr))) == to_string(expr)
    mapping = to_mapping(expr)
    assert to_mapping(parse(mapping)) == mapping


def test_disjoint_conjunctions_are_one_record_and_precedence_is_and_minus_or():
    assert parse("role:adapter & layers:-4:") == Term(role="adapter", layers="-4:")
    assert to_mapping(parse("role:adapter & layers:-4:")) == {"role": "adapter", "layers": "-4:"}
    expr = parse("@a | @b - @c & role:adapter")
    assert isinstance(expr, Or) and isinstance(expr.items[1], Minus) and isinstance(expr.items[1].right, And)
    assert isinstance(parse("layers:0:8 & layers:4:"), And)  # a repeated field cannot merge


@pytest.mark.parametrize("entry", ["model.layers.0.weight", r"model\.layers\.2[4-7]\..*lora_.*", "(q|k)_proj",
                                   "layer.0.weight", ".*", "embed|head", "blocks.(9|1[0-1]).*"])
def test_bare_names_and_regexes_are_not_addresses(entry):
    assert not is_address(entry)
    with pytest.raises(AddressError, match="is not an address"):
        parse(entry)


@pytest.mark.parametrize("entry", ["role:adapter", "@top", "(role:adapter | @x)", " ( @x)", {"role": "adapter"}])
def test_addresses_are_recognized(entry):
    assert is_address(entry)


@pytest.mark.parametrize("bad, match", [
    ("role:bogus", "unknown role kind"),
    ("layers:4-11", "slices are half-open"),
    ("role:adapter &", "ends where a term was expected"),
    ("name:abc", "quoted"),
    ("(role:adapter", "Unbalanced"),
    ("role:adapter && layers:1", "Unexpected"),
    ("units:odd", "not a parity"),
    ('name:"("', "not a valid regex"),
    ({"role": "adapter", "colour": "red"}, "Unknown address keys"),
    ({"optional": True}, "selects nothing"),
    ({"role": "adapter", "expect": -1}, "member count"),
    ("@bad..name", "Cannot parse|not a valid set reference"),
])
def test_malformed_addresses_are_refused(bad, match):
    with pytest.raises(VenngramError, match=match):
        parse(bad)


def test_layer_sets():
    assert LayerSet.parse("-4:").select(28) == (24, 25, 26, 27)
    assert LayerSet.parse(":9").select(12) == tuple(range(9))
    assert LayerSet.parse("even").select(5) == (0, 2, 4)
    assert LayerSet.parse([3, -1]).select(12) == (3, 11)
    assert LayerSet.parse("13").select(28) == (13,)
    with pytest.raises(AddressError, match="outside a stack of 12"):
        LayerSet.parse(12).select(12)
    with pytest.raises(AddressError, match="boolean"):
        LayerSet.parse(True)


def test_unit_sets():
    assert UnitSet.parse("[431, 1007]").for_layer(12, 4096) == (431, 1007)
    assert UnitSet.parse("0:4").for_layer(None, 4096) == (0, 1, 2, 3)
    per = UnitSet.parse({12: [431], 14: "8:10"})
    assert per.layers == (12, 14) and per.for_layer(14, 64) == (8, 9) and per.for_layer(13, 64) == ()


def test_mapping_combinators():
    expr = parse({"role": "adapter", "exclude": {"any": ["@top", "@mid"]}, "optional": True})
    assert to_string(expr) == "role:adapter - (@top | @mid)"
    assert references(expr) == ["top", "mid"]
    assert references(rename_refs(expr, lambda r: f"lora.{r}")) == ["lora.top", "lora.mid"]
    assert parse({"all": ["@a", "@b"]}) == And((Ref("a"), Ref("b")))


def test_target_kinds():
    assert parse("role:adapter").target == "param"
    assert parse("point:resid_post & layers:3").target == "site"
    assert parse("features:sae & point:resid_post").target == "feature"
    assert parse("subspace:x & point:resid_post").target == "subspace"
    with pytest.raises(AddressError, match="not both"):
        parse("features:sae & subspace:x & point:resid_post").check_target()
    with pytest.raises(AddressError, match="needs a `point`"):
        parse("space:jlens & layers:3").check_target()
