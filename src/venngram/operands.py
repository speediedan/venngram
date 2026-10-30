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
Operands
^^^^^^^^

A recipe op's **operand** is a named tensor (a direction, a pair of directions, the basis of a subspace) declared in the
recipe with its ``shape`` and the ``basis`` it is written in. A small operand may be inline (``values``); a large one is
a safetensors **operand sidecar**, referenced like a unit sidecar by ``{file, key, sha256}``, holding ``F32`` tensors.
It is read and written with the standard library (no ``torch``, no pickle).

.. warning:: This is an experimental API and may change in a future release.
"""
import json
import math
import struct
import sys
from array import array
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from venngram.errors import AddressError
from venngram.units import _sha256

#: The operand sidecar schema this code writes and reads.
OPERAND_SCHEMA_VERSION = 1

#: Inline ``values`` above this many numbers are refused: write a sidecar instead.
INLINE_LIMIT = 4096


def _numel(shape: Sequence[int]) -> int:
    return math.prod(shape) if shape else 1


def write_operands(path: str | Path, tensors: Mapping[str, tuple[Sequence[float], Sequence[int]]]) -> str:
    """Write tensors (key -> (values in row-major order, shape)) as ``F32`` to a sidecar and return its sha256."""
    header: dict[str, Any] = {}
    blobs: list[bytes] = []
    offset = 0
    for key, (values, shape) in tensors.items():
        shape = [int(d) for d in shape]
        if len(values) != _numel(shape):
            raise ValueError(f"operand {key!r}: {len(values)} values for shape {shape}")
        data = array("f", (float(v) for v in values))
        if sys.byteorder != "little":
            data.byteswap()
        blob = data.tobytes()
        header[key] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    header["__metadata__"] = {"venngram.schema_version": str(OPERAND_SCHEMA_VERSION), "venngram.kind": "operands"}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    path = Path(path)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))
    return _sha256(path)


def read_operand(path: str | Path, key: str, *, sha256: str | None = None) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """The values (row-major) and shape stored under ``key`` in an operand sidecar.

    Raises:
        AddressError: If the file's hash differs from ``sha256``, it is not a venngram operand sidecar, the key is
            absent, or the tensor is not ``F32`` of its declared shape.
    """
    path = Path(path)
    if not path.is_file():
        raise AddressError(f"Operand sidecar {path} does not exist.")
    if sha256 is not None and _sha256(path) != sha256.lower():
        raise AddressError(f"{path}: its sha256 differs from the reference's; the operand changed after the recipe "
                           "named it.")
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(n))
        meta = header.pop("__metadata__", None) or {}
        if meta.get("venngram.kind") != "operands" or int(meta.get("venngram.schema_version", 0)) != 1:
            raise AddressError(f"{path}: not a venngram operand sidecar (schema {OPERAND_SCHEMA_VERSION}).")
        if key not in header:
            raise AddressError(f"{path}: no operand `{key}`; it holds {sorted(header)}.")
        entry = header[key]
        lo, hi = entry["data_offsets"]
        fh.seek(8 + n + lo)
        data = fh.read(hi - lo)
    shape = tuple(int(d) for d in entry["shape"])
    if entry["dtype"] != "F32" or len(data) != 4 * _numel(shape):
        raise AddressError(f"{path}: operand `{key}` must be F32 of shape {list(shape)}.")
    values = array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    return tuple(values), shape


__all__ = ["INLINE_LIMIT", "OPERAND_SCHEMA_VERSION", "read_operand", "write_operands"]
