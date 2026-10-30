import json
import struct

import pytest
import yaml

from venngram import AddressBook, parse, to_mapping, to_string, UnitSet
from venngram.errors import AddressError
from venngram.units import INLINE_THRESHOLD, read_units, SidecarRef, sidecar_root, units_value, write_units

SPARSE = [3, 17, 4041, 9000]
DENSE = list(range(0, 9216, 2))


@pytest.mark.parametrize("form", ["indices", "bitmask", "auto"])
def test_a_sidecar_round_trips_in_either_form(tmp_path, form):
    path = tmp_path / "units.safetensors"
    digest = write_units(path, {"L15": (SPARSE, 9216), "L16": (DENSE, 9216)}, form=form)
    assert read_units(path, "L15", sha256=digest) == (tuple(SPARSE), 9216)
    assert read_units(path, "L16", sha256=digest) == (tuple(DENSE), 9216)
    if form == "auto":  # each set takes its smaller form: indices for the sparse one, a mask for the dense one
        meta = json.loads(_header(path)["__metadata__"]["venngram.sets"])
        assert (meta["L15"]["form"], meta["L16"]["form"]) == ("indices", "bitmask")


def _header(path):
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    return json.loads(raw[8:8 + n])


def test_the_file_is_standard_safetensors(tmp_path):
    numpy = pytest.importorskip("safetensors.numpy")
    path = tmp_path / "units.safetensors"
    write_units(path, {"a": (SPARSE, 9216), "b": (DENSE, 9216)}, form="auto")
    tensors = numpy.load_file(str(path))
    assert tensors["a"].tolist() == SPARSE and tensors["a"].dtype.name == "int32"
    assert tensors["b"].dtype.name == "uint8" and len(tensors["b"]) == 9216 // 8


def test_a_changed_file_a_wrong_axis_and_a_missing_hash_are_refused(tmp_path):
    path = tmp_path / "units.safetensors"
    digest = write_units(path, {"L15": (SPARSE, 9216)})
    units = UnitSet.parse({"file": str(path), "key": "L15", "sha256": digest})
    assert units.for_layer(15, 9216) == tuple(SPARSE)
    with pytest.raises(AddressError, match="indexes an axis of 9216 units, but the target at layer 15 has 4096"):
        units.for_layer(15, 4096)
    with pytest.raises(AddressError, match="sha256"):
        UnitSet.parse({"file": str(path), "key": "L15"})
    write_units(path, {"L15": ([1], 9216)})
    with pytest.raises(AddressError, match="sha256 differs"):
        UnitSet.parse({"file": str(path), "key": "L15", "sha256": digest})
    with pytest.raises(AddressError, match="no unit set `L99`"):
        read_units(path, "L99")


def _rewrite(path, key, payload: bytes, dtype: str, form: str, axis: int):
    header = {key: {"dtype": dtype, "shape": [len(payload) // (4 if dtype == "I32" else 1)],
                    "data_offsets": [0, len(payload)]},
              "__metadata__": {"venngram.schema_version": "1",
                               "venngram.sets": json.dumps({key: {"form": form, "axis": axis}})}}
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def test_malformed_sets_are_refused(tmp_path):
    path = tmp_path / "bad.safetensors"
    _rewrite(path, "k", struct.pack("<3i", 5, 2, 9), "I32", "indices", 16)
    with pytest.raises(AddressError, match="sorted, unique indices"):
        read_units(path, "k")
    _rewrite(path, "k", bytes([0, 0b0001_0000]), "U8", "bitmask", 12)  # bit 12 lies past an axis of 12
    with pytest.raises(AddressError, match="past its axis"):
        read_units(path, "k")
    _rewrite(path, "k", bytes(3), "U8", "bitmask", 12)
    with pytest.raises(AddressError, match="2 U8 bytes"):
        read_units(path, "k")
    path.write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")
    with pytest.raises(AddressError, match="not a venngram sidecar"):
        read_units(path, "k")


def test_a_book_resolves_sidecars_beside_it_and_keeps_the_reference(tmp_path):
    (tmp_path / "sets").mkdir()
    value = units_value({15: DENSE, 16: [7, 8]}, 9216, tmp_path / "sets" / "chess.safetensors", relative_to=tmp_path)
    assert value[15] == {"file": "sets/chess.safetensors", "key": "L15", "sha256": value[15]["sha256"]}
    book_path = tmp_path / "chess.yaml"
    book_path.write_text(yaml.safe_dump({
        "schema_version": 1, "name": "chess", "portability": "checkpoint",
        "applies_to": {"models": [{"id": "Qwen/Qwen3.5-4B", "revision": "abc123"}]},
        "sets": {"chess": {"point": "mlp.out.hook_in", "units": value}}}))
    chess = AddressBook.from_file(book_path).sets()["chess"]
    assert chess.units.for_layer(15, 9216) == tuple(DENSE) and chess.units.for_layer(16, 9216) == (7, 8)
    assert to_mapping(chess)["units"] == value  # the book keeps the reference, never the inlined units
    with sidecar_root(tmp_path):
        assert parse(to_string(chess)) == chess  # and the string form round-trips


def test_small_sets_stay_inline():
    assert units_value([5, 3, 3], 9216, "/nonexistent.safetensors") == [3, 5]
    assert units_value({1: list(range(INLINE_THRESHOLD))}, 9216, "/nonexistent.safetensors") == {
        1: list(range(INLINE_THRESHOLD))}


def test_a_sidecar_reference_is_strict_about_its_keys():
    with pytest.raises(AddressError, match="not \\['bytes'\\]"):
        SidecarRef.parse({"file": "a", "key": "b", "sha256": "c", "bytes": 1})
