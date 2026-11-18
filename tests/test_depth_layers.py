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
from torch import nn

from venngram import AddressBook, ParamSpace, select
from venngram.address import LayerSet, parse, to_mapping, to_string
from venngram.errors import AddressError
from venngram.physical import PhysicalStructure


def stack(depth):
    root = nn.Module()
    root.layers = nn.ModuleList([nn.Linear(1, 1, bias=False, device="meta") for _ in range(depth)])
    return ParamSpace([n for n, _ in root.named_parameters()], structure=PhysicalStructure(root))


@pytest.mark.parametrize("depth, band", [(28, range(7, 21)), (16, range(4, 12))])
def test_a_depth_relative_band_selects_the_same_band_on_two_depths(depth, band):
    assert LayerSet.parse("0.25d:0.75d").select(depth) == tuple(band)
    assert select("layers:0.25d:0.75d", stack(depth)) == [f"layers.{i}.weight" for i in band]


@pytest.mark.parametrize("spec, depth, expected", [
    ("0.5d", 28, (14,)), ("0.75d", 28, (21,)), ("0.5d", 16, (8,)), ("0.75d", 16, (12,)),   # the paper's pairs
    ("[0.5d, 0.75d]", 28, (14, 21)), ("0.75d:", 28, tuple(range(21, 28))), (":0.5d", 10, (0, 1, 2, 3, 4)),
    ("0.75d", 10, (7,)), ("0.25d:0.75d", 10, (2, 3, 4, 5, 6)),                              # floor, not round
    ("0.29d", 100, (29,)), ("0.7d:", 10, (7, 8, 9)),                                         # exact, not float
    ("0d:-2", 6, (0, 1, 2, 3)), ("-2:1d", 6, (4, 5)), (["0.5d", 3], 16, (3, 8)), ("1d", 4, None),
])
def test_a_depth_fraction_is_the_floor_of_its_share_of_the_depth(spec, depth, expected):
    layers = LayerSet.parse(spec)
    assert layers.depth_relative and layers.indices is None and layers.bounds is None
    if expected is None:
        with pytest.raises(AddressError, match="outside a stack of 4 blocks"):
            layers.select(depth)
    else:
        assert layers.select(depth) == expected


@pytest.mark.parametrize("spec, match", [
    ("0.5", "fraction of the depth"), ("1.5d", "more than the whole depth"), ("d", "not a fraction"),
    ("-0.5d", "not a fraction"), ("0.5d:x", "Cannot read the layer set"),
])
def test_malformed_depth_fractions_are_refused(spec, match):
    with pytest.raises(AddressError, match=match):
        LayerSet.parse(spec)


def test_a_depth_fraction_round_trips_and_is_refused_for_units():
    for text in ("role:adapter & layers:0.25d:0.75d", "role:adapter & layers:[0.5d,0.75d]"):
        expr = parse(text)
        assert to_string(expr) == text and parse(to_mapping(expr)) == expr
    with pytest.raises(AddressError, match="not a fraction of a depth"):
        parse("module:down & units:0.5d")


def test_a_book_with_depth_fractions_needs_schema_7():
    sets = {"band": {"role": "adapter", "layers": "0.25d:0.75d"}}
    with pytest.raises(AddressError, match="needs address book schema 7"):
        AddressBook.from_mapping({"schema_version": 6, "name": "b", "sets": sets})
    assert AddressBook.from_mapping({"schema_version": 7, "name": "b", "sets": sets}).own_sets["band"] == \
        parse(sets["band"])
    assert AddressBook.from_mapping({"name": "b", "sets": sets}, inline=True).schema_version == 7
