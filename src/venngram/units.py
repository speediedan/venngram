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
r"""
Sidecar Unit Sets
^^^^^^^^^^^^^^^^^

Large unit sets stored beside an address book, so a set of one neuron and a mask of a hundred thousand share one
grammar: inline ``units`` lists stay as they are, and a ``units`` value may instead name a sidecar file:

.. code-block:: yaml

    units: {file: units.safetensors, key: L15, sha256: 9f2c...}         # one set, every selected layer
    units: {15: {file: units.safetensors, key: L15, sha256: 9f2c...},   # per layer, sidecar or inline
            16: [431, 1007]}

A sidecar is a `safetensors <https://huggingface.co/docs/safetensors>`_ file, read and written here with the standard
library alone (no ``torch``, no pickle, nothing executes). Each key holds one set in one of two forms, named in the
file's metadata with the axis length it indexes:

- ``indices``: a sorted, unique ``I32`` index array (sparse sets);
- ``bitmask``: a packed ``U8`` mask, one bit per unit of the axis: unit ``j`` is bit ``j % 8`` of byte ``j // 8``
  (little bit order; dense sets);
- ``elements`` (sidecar schema 2): an element mask, a packed ``U8`` mask of one bit per element of a tensor of the
  ``shape`` recorded beside it, elements in row-major order (:class:`ElementMask`). It keeps individual weights'
  gradients, so it is named as ``units: {elements: {file, key, sha256}}`` and refused on a tensor of another shape.

A file holding no element mask is written at schema 1, so an older venngram still reads it.

A reference must carry the file's ``sha256``, so it cites exactly one content wherever the file is stored, and a
changed file is refused rather than silently re-read. The axis length is read from the header before any data, so a
set written for one width is refused on another (:meth:`~venngram.address.UnitSet.for_layer`). Relative paths resolve
against the address book's directory (:func:`sidecar_root`).

.. warning:: This is an experimental API and may change in a future release.
"""
import hashlib
import json
import os
import struct
import sys
from array import array
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from venngram.errors import AddressError

#: The sidecar format's schema, recorded in each file's metadata and read within a window.
SIDECAR_SCHEMA_VERSION = 2
#: The first sidecar schema that may hold an element mask (form ``elements``).
ELEMENTS_SIDECAR_SCHEMA = 2

#: Below this many units per layer, :func:`units_value` writes a set inline rather than to a sidecar.
INLINE_THRESHOLD = 512

FORMS = ("indices", "bitmask")
#: The forms a sidecar set may take: unit sets (:data:`FORMS`) and element masks.
ALL_FORMS = (*FORMS, "elements")
_ROOT: ContextVar[Path | None] = ContextVar("venngram_sidecar_root", default=None)


@contextmanager
def sidecar_root(path: str | Path | None) -> Iterator[None]:
    """Resolve relative sidecar paths against ``path`` (an address book's directory) inside the block."""
    token = _ROOT.set(Path(path) if path is not None else None)
    try:
        yield
    finally:
        _ROOT.reset(token)


@dataclass(frozen=True)
class SidecarRef:
    """A reference to one set in a sidecar file: its path (relative to the book), key and content hash."""

    file: str
    key: str
    sha256: str

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "SidecarRef":
        """Read a ``{file, key, sha256}`` mapping, refusing anything else by name."""
        unknown = set(raw) - {"file", "key", "sha256"}
        if unknown:
            raise AddressError(f"A sidecar unit set takes `file`, `key` and `sha256`, not {sorted(unknown)}.")
        missing = [k for k in ("file", "key", "sha256") if not isinstance(raw.get(k), str) or not raw.get(k)]
        if missing:
            hint = " (the content hash makes the reference cite one file; write sidecars with `write_units`)" \
                if "sha256" in missing else ""
            raise AddressError(f"A sidecar unit set needs a non-empty string {', '.join(f'`{m}`' for m in missing)}"
                               f"{hint}, got {dict(raw)!r}.")
        return cls(raw["file"], raw["key"], raw["sha256"].lower())

    def mapping(self) -> dict[str, str]:
        """The reference as a book stores it."""
        return {"file": self.file, "key": self.key, "sha256": self.sha256}

    def path(self) -> Path:
        """The file, resolved against the current :func:`sidecar_root` when relative."""
        path = Path(self.file)
        root = _ROOT.get()
        return path if path.is_absolute() or root is None else root / path

    def load(self) -> tuple[tuple[int, ...], int]:
        """The set's sorted unit indices and the axis length they index."""
        return read_units(self.path(), self.key, sha256=self.sha256)

    def load_elements(self) -> "ElementMask":
        """The element mask stored under this reference."""
        return read_elements(self.path(), self.key, sha256=self.sha256)


@dataclass(frozen=True)
class ElementMask:
    """Which elements of one tensor's gradient are kept: a bit per element of ``shape``, in row-major order (element
    ``i`` of the flattened tensor is bit ``i`` of ``bits``). Masks of one shape unite, intersect and subtract exactly.

    >>> m = ElementMask.from_indices((2, 3), [0, 4])
    >>> m.indices(), m.count, (m | ElementMask.from_indices((2, 3), [5])).indices()
    ((0, 4), 2, (0, 4, 5))
    """

    shape: tuple[int, ...]
    bits: int = 0

    def __post_init__(self) -> None:
        shape = tuple(int(d) for d in self.shape)
        if not shape or any(d < 0 for d in shape):
            raise ValueError(f"an element mask's shape is a non-empty tuple of non-negative sizes, got {self.shape!r}")
        if self.bits < 0 or self.bits >> self.size:
            raise ValueError(f"an element mask of shape {shape} keeps elements outside it")
        object.__setattr__(self, "shape", shape)

    @property
    def size(self) -> int:
        """The number of elements of ``shape``."""
        out = 1
        for d in self.shape:
            out *= d
        return out

    @property
    def count(self) -> int:
        """The number of kept elements."""
        return bin(self.bits).count("1")

    @classmethod
    def from_indices(cls, shape: Sequence[int], indices: Sequence[int]) -> "ElementMask":
        """The mask keeping the elements at ``indices`` of the flattened tensor."""
        bits = 0
        for i in indices:
            bits |= 1 << int(i)
        return cls(tuple(shape), bits)

    @classmethod
    def full(cls, shape: Sequence[int]) -> "ElementMask":
        """The mask keeping every element."""
        mask = cls(tuple(shape))
        return cls(mask.shape, (1 << mask.size) - 1)

    def indices(self) -> tuple[int, ...]:
        """The kept elements' indices in the flattened tensor, ascending."""
        bits, out, i = self.bits, [], 0
        while bits:
            if bits & 1:
                out.append(i)
            bits >>= 1
            i += 1
        return tuple(out)

    def to_bytes(self) -> bytes:
        """The packed mask, one bit per element in row-major order, little bit order (element ``j`` is bit ``j % 8`` of
        byte ``j // 8``), as a sidecar stores it."""
        return self.bits.to_bytes((self.size + 7) // 8, "little")

    def _same(self, other: "ElementMask") -> None:
        if not isinstance(other, ElementMask) or other.shape != self.shape:
            raise ValueError(f"element masks combine only at one shape, not {self.shape} and "
                             f"{getattr(other, 'shape', other)}")

    def __or__(self, other: "ElementMask") -> "ElementMask":
        self._same(other)
        return ElementMask(self.shape, self.bits | other.bits)

    def __and__(self, other: "ElementMask") -> "ElementMask":
        self._same(other)
        return ElementMask(self.shape, self.bits & other.bits)

    def __sub__(self, other: "ElementMask") -> "ElementMask":
        self._same(other)
        return ElementMask(self.shape, self.bits & ~other.bits)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _header(path: Path) -> tuple[dict[str, Any], dict[str, Any], int]:
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise AddressError(f"{path}: not a safetensors file (no header length).")
        (n,) = struct.unpack("<Q", raw)
        if n > 100 * (1 << 20):
            raise AddressError(f"{path}: implausible safetensors header length {n}.")
        try:
            header = json.loads(fh.read(n))
        except ValueError as err:
            raise AddressError(f"{path}: unreadable safetensors header: {err}.") from None
    meta = header.pop("__metadata__", None) or {}
    try:
        version = int(meta.get("venngram.schema_version", ""))
        sets = json.loads(meta.get("venngram.sets", ""))
    except ValueError:
        raise AddressError(f"{path}: not a venngram sidecar (no `venngram.schema_version` and `venngram.sets` "
                           "metadata).") from None
    if not 1 <= version <= SIDECAR_SCHEMA_VERSION:
        raise AddressError(f"{path}: sidecar schema {version} is not one this venngram reads (1-"
                           f"{SIDECAR_SCHEMA_VERSION}); upgrade venngram, or re-write the sidecar.")
    return header, sets, 8 + n


def _entry(path: Path, key: str, sha256: str | None) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    """The header entry, the venngram metadata and the data of the set stored under ``key``."""
    if not path.is_file():
        raise AddressError(f"Sidecar unit set {path} does not exist.")
    if sha256 is not None and _sha256(path) != sha256.lower():
        raise AddressError(f"{path}: its sha256 differs from the reference's; the sidecar changed after the book "
                           "named it, so it is refused rather than re-read.")
    header, sets, start = _header(path)
    if key not in header or key not in sets:
        raise AddressError(f"{path}: no unit set `{key}`; it holds {sorted(k for k in header if k in sets)}.")
    entry, info = header[key], sets[key]
    if info.get("form") not in ALL_FORMS:
        raise AddressError(f"{path}: set `{key}` has form {info.get('form')!r}; expected one of {ALL_FORMS}.")
    lo, hi = entry["data_offsets"]
    with open(path, "rb") as fh:
        fh.seek(start + lo)
        data = fh.read(hi - lo)
    return entry, info, data


def read_units(path: str | Path, key: str, *, sha256: str | None = None) -> tuple[tuple[int, ...], int]:
    """The unit indices stored under ``key`` in a sidecar, and the axis length they index.

    Raises:
        AddressError: If the file's hash differs from ``sha256``, the key is absent or holds an element mask, or the
            stored set is malformed (unsorted, repeated or out-of-range indices; a mask of the wrong length or with bits
            past the axis).
    """
    path = Path(path)
    entry, info, data = _entry(path, key, sha256)
    form, axis = info.get("form"), info.get("axis")
    if form == "elements":
        raise AddressError(f"{path}: set `{key}` is an element mask, not a unit set: name it as "
                           "`units: {elements: {file, key, sha256}}`.")
    if not isinstance(axis, int) or axis < 0:
        raise AddressError(f"{path}: set `{key}` has axis {axis!r}; expected a non-negative axis length.")
    if form == "indices":
        if entry["dtype"] != "I32" or len(data) % 4:
            raise AddressError(f"{path}: set `{key}` is `indices`, which is stored as I32, not {entry['dtype']}.")
        values = array("i")
        values.frombytes(data)
        if sys.byteorder != "little":
            values.byteswap()
        units = tuple(values)
        if any(b <= a for a, b in zip(units, units[1:])) or (units and (units[0] < 0 or units[-1] >= axis)):
            raise AddressError(f"{path}: set `{key}` must hold sorted, unique indices in [0, {axis}).")
        return units, axis
    if entry["dtype"] != "U8" or len(data) != (axis + 7) // 8:
        raise AddressError(f"{path}: set `{key}` is a `bitmask` of {axis} units, which is {(axis + 7) // 8} U8 bytes.")
    if axis % 8 and data[-1] >> (axis % 8):
        raise AddressError(f"{path}: set `{key}` sets bits past its axis of {axis} units.")
    return tuple(8 * i + j for i, byte in enumerate(data) if byte for j in range(8) if byte >> j & 1), axis


def read_elements(path: str | Path, key: str, *, sha256: str | None = None) -> ElementMask:
    """The element mask stored under ``key`` in a sidecar.

    Raises:
        AddressError: If the file's hash differs from ``sha256``, the key is absent or holds a unit set, or the stored
            mask is malformed (a shape that is not a list of non-negative sizes, data of the wrong length, or bits past
            the tensor's last element).
    """
    path = Path(path)
    entry, info, data = _entry(path, key, sha256)
    if info.get("form") != "elements":
        raise AddressError(f"{path}: set `{key}` is a unit set, not an element mask: name it as "
                           "`units: {file, key, sha256}`.")
    shape = info.get("shape")
    if not isinstance(shape, list) or not shape or not all(isinstance(d, int) and d >= 0 for d in shape):
        raise AddressError(f"{path}: element mask `{key}` has shape {shape!r}; expected a list of non-negative sizes.")
    size = ElementMask(tuple(shape)).size
    if entry["dtype"] != "U8" or len(data) != (size + 7) // 8:
        raise AddressError(f"{path}: element mask `{key}` of shape {tuple(shape)} is {(size + 7) // 8} U8 bytes.")
    bits = int.from_bytes(data, "little")
    if bits >> size:
        raise AddressError(f"{path}: element mask `{key}` sets bits past its {size} elements.")
    return ElementMask(tuple(shape), bits)


def write_units(path: str | Path, sets: Mapping[str, "tuple[Sequence[int], int] | ElementMask"], *,
                form: str = "auto") -> str:
    """Write sets (key -> (unit indices, axis length), or key -> an :class:`ElementMask`) to a sidecar and return its
    sha256. A file holding an element mask is sidecar schema 2; any other, schema 1.

    ``form`` is ``indices``, ``bitmask``, or ``auto``: per unit set, whichever is smaller (a mask takes one byte per
    eight units of the axis, an index array four bytes per unit). An element mask is always stored packed.
    """
    if form not in (*FORMS, "auto"):
        raise ValueError(f"form must be one of {FORMS} or 'auto', got {form!r}")
    header: dict[str, Any] = {}
    info: dict[str, Any] = {}
    blobs: list[bytes] = []
    offset = 0
    for key, value in sets.items():
        if isinstance(value, ElementMask):
            blob = value.to_bytes()
            header[key] = {"dtype": "U8", "shape": [len(blob)], "data_offsets": [offset, offset + len(blob)]}
            info[key] = {"form": "elements", "shape": list(value.shape), "bitorder": "little", "order": "row-major"}
            blobs.append(blob)
            offset += len(blob)
            continue
        units, axis = value
        units = sorted(set(int(u) for u in units))
        if units and (units[0] < 0 or units[-1] >= axis):
            raise ValueError(f"set {key!r}: units must lie in [0, {axis})")
        chosen = form if form != "auto" else ("bitmask" if 4 * len(units) > (axis + 7) // 8 else "indices")
        if chosen == "indices":
            values = array("i", units)
            if sys.byteorder != "little":
                values.byteswap()
            blob, dtype, shape = values.tobytes(), "I32", [len(units)]
        else:
            mask = bytearray((axis + 7) // 8)
            for u in units:
                mask[u // 8] |= 1 << (u % 8)
            blob, dtype, shape = bytes(mask), "U8", [len(mask)]
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(blob)]}
        info[key] = {"form": chosen, "axis": axis, **({"bitorder": "little"} if chosen == "bitmask" else {})}
        blobs.append(blob)
        offset += len(blob)
    version = ELEMENTS_SIDECAR_SCHEMA if any(i["form"] == "elements" for i in info.values()) else 1
    header["__metadata__"] = {"venngram.schema_version": str(version),
                              "venngram.sets": json.dumps(info, sort_keys=True)}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)  # safetensors aligns the data to eight bytes
    path = Path(path)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))
    return _sha256(path)


def units_value(units: Mapping[int, Sequence[int]] | Sequence[int], axis: int, path: str | Path, *,
                relative_to: str | Path | None = None, threshold: int = INLINE_THRESHOLD, form: str = "auto") -> Any:
    """The ``units`` value a book should store for ``units`` (flat, or per layer).

    Inline when every layer holds at most ``threshold`` units; otherwise a sidecar written to ``path`` with one key per
    layer (``L<layer>``, or ``units`` for a flat set), referenced relative to ``relative_to`` (the book's directory).
    """
    per_layer = dict(units) if isinstance(units, Mapping) else None
    sizes = [len(v) for v in per_layer.values()] if per_layer is not None else [len(units)]  # type: ignore[arg-type]
    if max(sizes, default=0) <= threshold:
        return ({layer: sorted(set(v)) for layer, v in per_layer.items()} if per_layer is not None
                else sorted(set(units)))  # type: ignore[arg-type]
    keyed = {f"L{layer}": (v, axis) for layer, v in per_layer.items()} if per_layer is not None \
        else {"units": (units, axis)}
    digest = write_units(path, keyed, form=form)  # type: ignore[arg-type]
    file = os.path.relpath(path, relative_to) if relative_to is not None else str(path)
    file = Path(file).as_posix()
    if per_layer is None:
        return SidecarRef(file, "units", digest).mapping()
    return {layer: SidecarRef(file, f"L{layer}", digest).mapping() for layer in per_layer}


__all__ = [
    "ALL_FORMS",
    "ElementMask",
    "ELEMENTS_SIDECAR_SCHEMA",
    "FORMS",
    "INLINE_THRESHOLD",
    "read_elements",
    "read_units",
    "SIDECAR_SCHEMA_VERSION",
    "SidecarRef",
    "sidecar_root",
    "units_value",
    "write_units",
]
