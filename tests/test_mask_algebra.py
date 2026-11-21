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
"""Exact gradient mask algebra (intersections and differences along one axis) and element masks."""
import json
import struct

import pytest
from test_gradient_axes import mlp_space

from venngram import AddressBook, select
from venngram.address import parse, to_string
from venngram.errors import AddressError, AddressUnresolvable
from venngram.resolve import gradient_forms, gradient_masks, gradient_selections, GradientSelection
from venngram.support import SupportRecord
from venngram.units import ElementMask, read_elements, read_units, sidecar_root, write_units

DOWN = "component:mlp.out & leaf:weight & layers:0 & of:grad"  # (3, 5) on the out_in layout
D0 = "layers.0.down.weight"


def sel(address, layout="out_in", sets=None):
    space, shapes = mlp_space(layout)
    return gradient_selections(address, space, sets, shapes=shapes)


# Mask algebra (#111)

def test_an_intersection_of_masks_along_one_axis_keeps_their_common_indices():
    assert sel(f"{DOWN} & units:[0, 1, 2] & units:[1, 2]") == {D0: GradientSelection(0, (1, 2))}
    sets = {"rows": f"{DOWN} & units:[0, 2]", "important": f"{DOWN} & units:[2]"}
    assert sel("@rows & @important", sets=sets) == {D0: GradientSelection(0, (2,))}
    # a union inside an intersection is resolved per tensor before intersecting
    assert sel(f"({DOWN} & units:[0] | {DOWN} & units:[2]) & units:[2, 1]") == {D0: GradientSelection(0, (2,))}


def test_a_difference_of_masks_subtracts_the_right_sides_indices():
    sets = {"rows": f"{DOWN} & units:[0, 1, 2]", "control": f"{DOWN} & units:[1]"}
    assert sel("@rows - @control", sets=sets) == {D0: GradientSelection(0, (0, 2))}
    # a whole tensor minus a mask keeps the complement along the mask's axis
    assert sel(f"{DOWN} - ({DOWN} & units:[1] & axis:1)") == {D0: GradientSelection(1, (0, 2, 3, 4))}
    # a right side keeping the whole tensor drops it, as a difference of tensors does
    space, shapes = mlp_space("out_in")
    up_and_down = "component:mlp.* & leaf:weight & layers:0 & of:grad"
    assert select(f"{up_and_down} - ({DOWN} & units:[1])", space) == ["layers.0.up.weight", D0]
    assert select(f"{up_and_down} - {DOWN}", space) == ["layers.0.up.weight"]
    assert gradient_selections(f"{up_and_down} & units:[0] - {DOWN}", space, shapes=shapes) == {
        "layers.0.up.weight": GradientSelection(0, (0,))}


def test_the_first_axis_helper_reads_the_same_algebra():
    space, shapes = mlp_space("out_in")
    widths = {n: s[0] for n, s in shapes.items()}
    assert gradient_masks(f"{DOWN} - ({DOWN} & units:[1])", space, widths=widths) == {D0: (0, 2)}


@pytest.mark.parametrize("address, match", [
    (f"({DOWN} & units:[0]) & ({DOWN} & units:[1] & axis:1)", "dimensions \\[0, 1\\] intersect to a block"),
    (f"({DOWN} & units:[0, 1]) - ({DOWN} & units:[1] & axis:1)", "dimensions \\[0, 1\\] subtract to a block"),
])
def test_masks_along_two_dimensions_are_refused_rather_than_approximated(address, match):
    with pytest.raises(AddressUnresolvable, match=match):
        sel(address)


def test_algebra_is_a_gradient_form_a_consumer_declares():
    assert "mask-algebra" in gradient_forms(f"{DOWN} & units:[0] & units:[1]")
    assert "mask-algebra" in gradient_forms(f"{DOWN} - ({DOWN} & units:[1])")
    assert "mask-algebra" not in gradient_forms(f"{DOWN} & units:[0] | {DOWN} & units:[1]")
    book = AddressBook.from_mapping({"schema_version": 2, "name": "b",
                                     "sets": {"s": f"{DOWN} & units:[0] & units:[1]"}}, source="b")
    tuning = {"fields": ["role", "name", "module", "layers", "component", "leaf", "units", "of"], "targets": ["param"],
              "units": "unit-adapter"}
    record = SupportRecord.from_mapping({"schema_version": 5, "kind": "support_record", "name": "t",
                                         "sections": {"tuning": tuning}})
    assert any("mask-algebra" in line for line in record.check_book(book))
    record = SupportRecord.from_mapping({"schema_version": 5, "kind": "support_record", "name": "t",
                                         "sections": {"tuning": {**tuning, "gradients": ["mask-algebra"]}}})
    assert record.check_book(book) == []


# Element masks (#110)

@pytest.fixture
def masks(tmp_path):
    """A sidecar with element masks for the (3, 5) down weight and the (5, 3) up weight, and a unit set."""
    down = ElementMask.from_indices((3, 5), [0, 6, 14])
    other = ElementMask.from_indices((3, 5), [6, 7])
    up = ElementMask.from_indices((5, 3), [1])
    digest = write_units(tmp_path / "m.safetensors", {"down": down, "other": other, "up": up, "rows": ([1], 3)})
    ref = {"file": "m.safetensors", "sha256": digest}
    return tmp_path, ref, down, other


def elements(ref, key):
    return {"elements": {**ref, "key": key}}


def test_an_element_mask_round_trips_through_a_sidecar(tmp_path):
    mask = ElementMask.from_indices((2, 3, 4), [0, 5, 23])
    digest = write_units(tmp_path / "e.safetensors", {"m": mask})
    assert read_elements(tmp_path / "e.safetensors", "m", sha256=digest) == mask
    assert mask.to_bytes() == bytes([0b00100001, 0, 0b10000000])
    assert ElementMask.full((2, 2)) - mask.__class__.from_indices((2, 2), [1]) == ElementMask((2, 2), 0b1101)
    raw = (tmp_path / "e.safetensors").read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    meta = json.loads(raw[8:8 + n])["__metadata__"]
    assert meta["venngram.schema_version"] == "2"  # a file holding an element mask is schema 2
    write_units(tmp_path / "u.safetensors", {"u": ([1], 4)})
    raw = (tmp_path / "u.safetensors").read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    assert json.loads(raw[8:8 + n])["__metadata__"]["venngram.schema_version"] == "1"  # and any other schema 1
    with pytest.raises(AddressError, match="is an element mask, not a unit set"):
        read_units(tmp_path / "e.safetensors", "m")
    with pytest.raises(AddressError, match="is a unit set, not an element mask"):
        read_elements(tmp_path / "u.safetensors", "u")
    with pytest.raises(ValueError, match="combine only at one shape"):
        _ = mask | ElementMask((2, 3))


def test_an_element_mask_keeps_individual_elements_of_a_tensors_gradient(masks):
    root, ref, down, other = masks
    with sidecar_root(root):
        got = sel(f"{DOWN} & units:{elements(ref, 'down')}")
        assert got == {D0: GradientSelection(elements=down)}
        assert got[D0].elements.indices() == (0, 6, 14) and not got[D0].whole
        either = sel(f"{DOWN} & units:{elements(ref, 'down')} | {DOWN} & units:{elements(ref, 'other')}")
        assert either[D0].elements == down | other
        both = sel(f"{DOWN} & units:{elements(ref, 'down')} & units:{elements(ref, 'other')}")
        assert both[D0].elements.indices() == (6,)
        rest = sel(f"{DOWN} - ({DOWN} & units:{elements(ref, 'down')})")
        assert rest[D0].elements == ElementMask.full((3, 5)) - down
        per_layer = sel(f"component:mlp.out & leaf:weight & of:grad & units:{{0: {elements(ref, 'down')}}}")
        assert per_layer["layers.0.down.weight"].elements == down
        assert per_layer["layers.1.down.weight"].elements == ElementMask((3, 5))  # a layer it does not name: nothing


@pytest.mark.parametrize("address, match", [
    ("component:mlp.* & leaf:weight & layers:0 & of:grad & units:{down}", "shape \\(3, 5\\), but the tensor"),
    (f"{DOWN} & units:{{down}} | {DOWN} & units:[1]", "element mask and a mask along one dimension do not combine"),
])
def test_element_masks_refuse_another_shape_and_a_mix_with_axis_masks(masks, address, match):
    root, ref, _, _ = masks
    with sidecar_root(root), pytest.raises(AddressUnresolvable, match=match):
        sel(address.replace("{down}", str(elements(ref, "down"))))


def test_element_masks_need_a_tensor_gradient_and_no_axis(masks):
    root, ref, _, _ = masks
    with sidecar_root(root):
        space, shapes = mlp_space("out_in")
        with pytest.raises(AddressUnresolvable, match="gradient_selections"):
            widths = {n: s[0] for n, s in shapes.items()}
            gradient_masks(f"{DOWN} & units:{elements(ref, 'down')}", space, widths=widths)
        with pytest.raises(AddressError, match="takes no `axis:1`"):
            parse({"module": "down", "units": elements(ref, "down"), "axis": 1}).check_target()
        with pytest.raises(AddressError, match="needs a tensor target"):
            parse({"point": "resid_post", "units": elements(ref, "down")}).check_target()
        with pytest.raises(AddressError, match="either unit indices or element masks"):
            parse({"module": "down", "units": {0: elements(ref, "down"), 1: [2]}})
        with pytest.raises(AddressError, match="is an element mask"):
            parse({"module": "down", "units": elements(ref, "down")}).units.for_layer(None, 3)
        expr = parse(f"{DOWN} & units:{elements(ref, 'down')}")
        assert parse(to_string(expr)) == expr
        assert "element-masks" in gradient_forms(expr)


def test_a_book_with_an_element_mask_is_a_schema_9_checkpoint_book(masks):
    root, ref, down, _ = masks
    sets = {"d": {"component": "mlp.out", "leaf": "weight", "of": "grad", "units": elements(ref, "down")}}
    pinned = {"portability": "checkpoint", "applies_to": {"models": [{"id": "org/m", "revision": "abc"}]}}
    book = AddressBook.from_mapping({"schema_version": 9, "name": "b", "sets": sets, **pinned}, base_dir=root)
    space, shapes = mlp_space("out_in")
    with sidecar_root(root):
        assert gradient_selections("@d", space, book.sets(), shapes=shapes)["layers.0.down.weight"].elements == down
    with pytest.raises(AddressError, match="needs address book schema 9, not 8"):
        AddressBook.from_mapping({"schema_version": 8, "name": "b", "sets": sets, **pinned}, base_dir=root)
    with pytest.raises(AddressError, match="so the book is `portability: checkpoint`"):
        AddressBook.from_mapping({"schema_version": 9, "name": "b", "sets": sets}, base_dir=root)
    inline = AddressBook.from_mapping({"sets": sets, **pinned}, base_dir=root, inline=True)
    assert inline.schema_version == 9
