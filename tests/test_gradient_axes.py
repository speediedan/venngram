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

from venngram import AddressBook, ParamSpace
from venngram.address import parse, to_mapping, to_string
from venngram.errors import AddressError, AddressUnresolvable
from venngram.maps import from_document, MapStructure
from venngram.recipe import Recipe
from venngram.resolve import gradient_forms, gradient_masks, gradient_selections, GradientSelection
from venngram.support import SupportRecord

MLP = "component:mlp.* & leaf:weight & units:[1] & of:grad"


def mlp_space(layout):
    """One block whose MLP projections are (out, in) weights for ``out_in`` (``nn.Linear``) and their transposes for
    ``in_out`` (GPT-2's ``Conv1D``), so the down projection's unit axis is the column of one and the row of the
    other."""
    root, block = nn.Module(), nn.Module()
    up_shape, down_shape = ((5, 3), (3, 5)) if layout == "out_in" else ((3, 5), (5, 3))
    block.up = nn.Module()
    block.up.weight = nn.Parameter(nn.init.zeros_(nn.Linear(1, 1, device="meta").weight.new_empty(up_shape)))
    block.up.bias = nn.Parameter(nn.Linear(1, 5, device="meta").bias.new_empty((5,)))
    block.down = nn.Module()
    block.down.weight = nn.Parameter(nn.Linear(1, 1, device="meta").weight.new_empty(down_shape))
    block.down.bias = nn.Parameter(nn.Linear(1, 3, device="meta").bias.new_empty((3,)))
    root.layers = nn.ModuleList([block, block.__class__()])
    second = root.layers[1]
    second.up, second.down = block.up.__class__(), block.down.__class__()
    for name in ("up", "down"):
        src, dst = getattr(block, name), getattr(second, name)
        dst.weight = nn.Parameter(src.weight.new_empty(src.weight.shape))
        dst.bias = nn.Parameter(src.bias.new_empty(src.bias.shape))
    doc = {"schema_version": 5, "architecture": f"AxisProbe_{layout}", "components": {
        "blocks.{i}": {"module": "layers.{i}", "kind": "block"},
        "blocks.{i}.mlp.in": {"module": "layers.{i}.up", "kind": "linear", "layout": layout},
        "blocks.{i}.mlp.out": {"module": "layers.{i}.down", "kind": "linear", "layout": layout}}}
    structure = MapStructure(root, from_document(doc))
    assert structure.refusal is None, structure.refusal
    params = dict(root.named_parameters())
    return ParamSpace(list(params), structure=structure), {n: tuple(p.shape) for n, p in params.items()}


@pytest.mark.parametrize("layout, down_axis", [("out_in", 1), ("in_out", 0)])
def test_a_unit_axis_selects_neurons_on_linear_and_transposed_weights_alike(layout, down_axis):
    space, shapes = mlp_space(layout)
    got = gradient_selections(MLP.replace("of:grad", "of:grad & axis:unit") + " & layers:0", space, shapes=shapes)
    up_axis = 1 - down_axis
    assert got == {"layers.0.up.weight": GradientSelection(up_axis, (1,)),
                   "layers.0.down.weight": GradientSelection(down_axis, (1,))}


def test_an_unqualified_mask_keeps_the_first_axis_in_both_helpers():
    space, shapes = mlp_space("out_in")
    address = MLP + " & layers:0"
    legacy = gradient_masks(address, space, widths={n: s[0] for n, s in shapes.items()})
    assert legacy == {"layers.0.up.weight": (1,), "layers.0.down.weight": (1,)}
    assert gradient_selections(address, space, shapes=shapes) == {n: GradientSelection(0, v) for n, v in legacy.items()}


def test_a_physical_axis_selects_that_dimension_on_any_model():
    space, shapes = mlp_space("in_out")
    got = gradient_selections("component:mlp.out & leaf:weight & layers:0 & units:[0, 2] & of:grad & axis:1", space,
                              shapes=shapes)
    assert got == {"layers.0.down.weight": GradientSelection(1, (0, 2))}


def test_the_first_axis_helper_refuses_an_explicit_axis_rather_than_drop_it():
    space, shapes = mlp_space("out_in")
    for axis in ("unit", "0"):
        with pytest.raises(AddressUnresolvable, match="gradient_selections"):
            gradient_masks(f"{MLP} & axis:{axis}", space, widths={n: s[0] for n, s in shapes.items()})


def test_unions_unite_along_one_axis_and_refuse_two():
    space, shapes = mlp_space("out_in")
    rows = "component:mlp.out & leaf:weight & layers:0 & of:grad"
    same = gradient_selections(f"{rows} & units:[0] & axis:1 | {rows} & units:[3] & axis:1", space, shapes=shapes)
    assert same == {"layers.0.down.weight": GradientSelection(1, (0, 3))}
    with pytest.raises(AddressUnresolvable, match="dimensions \\[0, 1\\] by different disjuncts"):
        gradient_selections(f"{rows} & units:[0] & axis:0 | {rows} & units:[3] & axis:1", space, shapes=shapes)
    whole = f"{rows} | {rows} & units:[3] & axis:1"
    assert gradient_selections(whole, space, shapes=shapes) == {"layers.0.down.weight": GradientSelection()}
    with pytest.raises(AddressUnresolvable, match="no dimension 2"):  # a whole-tensor branch hides no invalid one
        gradient_selections(f"{rows} | {rows} & units:[3] & axis:2", space, shapes=shapes)


def test_per_layer_units_and_slices_count_along_the_chosen_dimension():
    space, shapes = mlp_space("out_in")
    got = gradient_selections("component:mlp.out & leaf:weight & units:{0: '-2:'} & of:grad & axis:unit", space,
                              shapes=shapes)
    assert got == {"layers.0.down.weight": GradientSelection(1, (3, 4)),
                   "layers.1.down.weight": GradientSelection(1, ())}
    with pytest.raises(AddressUnresolvable, match="layers.0.down.weight"):
        gradient_selections("component:mlp.out & leaf:weight & layers:0 & units:[5] & of:grad & axis:unit", space,
                            shapes=shapes)


@pytest.mark.parametrize("address, match", [
    ("component:mlp.out & leaf:bias & layers:0 & units:[0] & of:grad & axis:unit", "not its units|has no unit axis"),
    ("component:mlp.out & leaf:weight & layers:0 & units:[0] & of:grad & axis:3", "no dimension 3"),
])
def test_a_unit_axis_a_tensor_lacks_is_refused_by_name(address, match):
    space, shapes = mlp_space("out_in")
    with pytest.raises(AddressUnresolvable, match=match):
        gradient_selections(address, space, shapes=shapes)


def test_a_structure_without_unit_axes_refuses_axis_unit():
    _, shapes = mlp_space("out_in")
    bare = ParamSpace(list(shapes))
    with pytest.raises(AddressUnresolvable, match="provides no unit axes"):
        gradient_selections("name:\"layers\\.0\\.down\\.weight\" & units:[0] & of:grad & axis:unit", bare,
                            shapes=shapes)


@pytest.mark.parametrize("raw, match", [
    ({"module": "down", "axis": "unit", "of": "grad"}, "needs `units`"),
    ({"point": "resid_post", "units": [1], "axis": "unit"}, "tensor target"),
    ({"module": "down", "units": [1], "axis": -1}, "nonnegative dimension"),
    ({"module": "down", "units": [1], "axis": "rows"}, "nonnegative dimension"),
])
def test_malformed_axes_are_refused(raw, match):
    with pytest.raises(AddressError, match=match):
        parse(raw).check_target()


def test_axis_round_trips_in_both_forms_and_names_its_gradient_form():
    expr = parse("module:down & units:[1] & of:grad & axis:1")
    assert to_mapping(expr)["axis"] == 1 and parse(to_mapping(expr)).axis == 1
    assert to_string(parse(to_mapping(expr))).endswith("& of:grad & axis:1")
    assert gradient_forms("module:down & units:[1] & of:grad & axis:unit") == {"unit-axis"}
    assert gradient_forms("module:down & units:[1] & of:grad & axis:0 | module:up & units:[0] & of:grad") == {
        "physical-axis", "union-masks"}


def test_a_book_using_axis_needs_schema_6_and_a_consumer_that_honours_it():
    sets = {"neurons": {"component": "mlp.out", "leaf": "weight", "units": [1], "of": "grad", "axis": "unit"}}
    with pytest.raises(AddressError, match="needs address book schema 6"):
        AddressBook.from_mapping({"schema_version": 5, "name": "b", "sets": sets})
    book = AddressBook.from_mapping({"schema_version": 6, "name": "b", "sets": sets})
    assert AddressBook.from_mapping({"name": "b", "sets": sets}, inline=True).schema_version == 6
    old = SupportRecord.from_mapping({"schema_version": 5, "kind": "support_record", "name": "masker", "sections": {
        "tuning": {"fields": ["component", "leaf", "units", "of"], "units": "unit-adapter"}}})
    assert any("`axis`" in r or "'axis'" in r for r in old.check_book(book))
    new = SupportRecord.from_mapping({"schema_version": 5, "kind": "support_record", "name": "masker", "sections": {
        "tuning": {"fields": ["component", "leaf", "units", "of", "axis"], "units": "unit-adapter",
                   "gradients": ["unit-axis"]}}})
    assert new.check_book(book) == []
    physical = AddressBook.from_mapping({"schema_version": 6, "name": "p", "sets": {
        "cols": {"component": "mlp.out", "leaf": "weight", "units": [1], "of": "grad", "axis": 1}}})
    assert any("physical-axis" in r for r in new.check_book(physical))


def test_a_recipe_refuses_an_axis_it_does_not_read():
    model = {"id": "m", "revision": "r"}
    book = {"schema_version": 6, "name": "b", "portability": "checkpoint", "applies_to": {"models": [model]},
            "sets": {"cols": {"component": "mlp.out", "leaf": "weight", "layers": [0], "units": [1], "of": "grad",
                              "axis": "unit"}}}
    doc = {"schema_version": 4, "kind": "recipe", "name": "r", "model": model, "book": book,
           "gradients": [{"op": "replace", "target": {"set": "cols"}, "value": {"dose": 0.0}}]}
    with pytest.raises(AddressError, match="no recipe reads `axis` yet"):
        Recipe.from_mapping(doc)
