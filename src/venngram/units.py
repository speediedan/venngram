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
  (little bit order; dense sets).

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
SIDECAR_SCHEMA_VERSION = 1

#: Below this many units per layer, :func:`units_value` writes a set inline rather than to a sidecar.
INLINE_THRESHOLD = 512

FORMS = ("indices", "bitmask")
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


def read_units(path: str | Path, key: str, *, sha256: str | None = None) -> tuple[tuple[int, ...], int]:
    """The unit indices stored under ``key`` in a sidecar, and the axis length they index.

    Raises:
        AddressError: If the file's hash differs from ``sha256``, the key is absent, or the stored set is malformed
            (unsorted, repeated or out-of-range indices; a mask of the wrong length or with bits past the axis).
    """
    path = Path(path)
    if not path.is_file():
        raise AddressError(f"Sidecar unit set {path} does not exist.")
    if sha256 is not None and _sha256(path) != sha256.lower():
        raise AddressError(f"{path}: its sha256 differs from the reference's; the sidecar changed after the book "
                           "named it, so it is refused rather than re-read.")
    header, sets, start = _header(path)
    if key not in header or key not in sets:
        raise AddressError(f"{path}: no unit set `{key}`; it holds {sorted(k for k in header if k in sets)}.")
    entry, info = header[key], sets[key]
    form, axis = info.get("form"), info.get("axis")
    if form not in FORMS or not isinstance(axis, int) or axis < 0:
        raise AddressError(f"{path}: set `{key}` has form {form!r} and axis {axis!r}; expected one of {FORMS} and a "
                           "non-negative axis.")
    lo, hi = entry["data_offsets"]
    with open(path, "rb") as fh:
        fh.seek(start + lo)
        data = fh.read(hi - lo)
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


def write_units(path: str | Path, sets: Mapping[str, tuple[Sequence[int], int]], *, form: str = "auto") -> str:
    """Write sets (key -> (unit indices, axis length)) to a sidecar and return its sha256.

    ``form`` is ``indices``, ``bitmask``, or ``auto``: per set, whichever is smaller (a mask takes one byte per eight
    units of the axis, an index array four bytes per unit).
    """
    if form not in (*FORMS, "auto"):
        raise ValueError(f"form must be one of {FORMS} or 'auto', got {form!r}")
    header: dict[str, Any] = {}
    info: dict[str, Any] = {}
    blobs: list[bytes] = []
    offset = 0
    for key, (units, axis) in sets.items():
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
    header["__metadata__"] = {"venngram.schema_version": str(SIDECAR_SCHEMA_VERSION),
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
    "FORMS",
    "INLINE_THRESHOLD",
    "read_units",
    "SIDECAR_SCHEMA_VERSION",
    "SidecarRef",
    "sidecar_root",
    "units_value",
    "write_units",
]
