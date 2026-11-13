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
Weight Patches
^^^^^^^^^^^^^^

A **weight patch** is the operand of a recipe's ``weights`` op (``add`` or ``replace`` on a ``param`` target): a change
to one tensor, as a ``kind: weight_patch`` document and a safetensors sidecar holding its delta.

.. code-block:: yaml

    schema_version: 1
    kind: weight_patch
    name: neuron-1024-write
    target: "component:mlp.out & layers:13 & leaf:weight"   # the address of one tensor, or its name
    shape: [2560, 9728]                                      # the tensor's shape, as stored
    dtype: float32                                           # the dtype the hashes are taken in
    form: low_rank                                           # low_rank | sparse | dense
    file: neuron-1024.safetensors
    sha256: ...                                              # the sidecar's
    base_sha256: ...                                         # the tensor it applies to
    edited_sha256: ...                                       # the tensor after ``add`` (optional)

The delta is in the tensor's own coordinates, as stored (not its ``in``/``out`` sides): ``low_rank`` holds ``u``
(``[rows, r]``) and ``v`` (``[cols, r]``) for :math:`\Delta = u v^{\top}`; ``sparse`` holds ``indices`` (``I64``,
row-major flat positions) and ``values``; ``dense`` holds ``delta`` of the tensor's shape. ``add`` gives
:math:`W + \Delta`; ``replace`` sets the tensor to :math:`\Delta` (``low_rank``, ``dense``) or the listed entries to
their values (``sparse``).

``base_sha256`` names the tensor the patch was made for, so a consumer checks it before applying and refuses a patch
that would land on another base; :func:`tensor_sha256` is that hash (SHA-256 of the tensor's bytes, contiguous and
little-endian, in ``dtype``). venngram reads and writes patches and never applies one.

.. warning:: This is an experimental API and may change in a future release.
"""
import hashlib
import json
import math
import struct
import sys
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from venngram.errors import AddressError
from venngram.units import _sha256

#: The ``weight_patch`` document schemas this code reads.
WEIGHT_PATCH_WINDOW = (1, 1)

#: The forms a delta is stored in, with the sidecar keys each holds.
FORMS: dict[str, tuple[str, ...]] = {"low_rank": ("u", "v"), "sparse": ("indices", "values"), "dense": ("delta",)}

_DTYPES = {"F32": ("f", 4), "I64": ("q", 8)}
_KEYS = ("schema_version", "kind", "name", "description", "target", "shape", "dtype", "form", "file", "sha256",
         "base_sha256", "edited_sha256")


@dataclass(frozen=True)
class WeightPatch:
    """A weight patch document (see the module docstring). Use :func:`read_weight_patch`."""

    name: str
    target: str
    shape: tuple[int, ...]
    form: str
    file: str
    sha256: str
    base_sha256: str
    dtype: str = "float32"
    edited_sha256: str | None = None
    description: str | None = None
    source: str = "<inline>"
    base_dir: str | None = None

    def tensors(self) -> dict[str, tuple[tuple[float, ...] | tuple[int, ...], tuple[int, ...]]]:
        """The sidecar's tensors (key -> (values in row-major order, shape)), its hash checked first."""
        return _read(Path(self.base_dir or ".") / self.file, sha256=self.sha256)

    @property
    def rank(self) -> int | None:
        """The rank of a ``low_rank`` patch (``None`` for the other forms)."""
        return self.tensors()["u"][1][1] if self.form == "low_rank" else None

    def delta(self) -> tuple[float, ...]:
        """The delta as a dense row-major tensor of :attr:`shape` (for checking a small patch; a consumer applying a
        large one uses :meth:`tensors` in its own form)."""
        t = self.tensors()
        n = math.prod(self.shape)
        if self.form == "dense":
            return tuple(float(v) for v in t["delta"][0])
        if self.form == "sparse":
            out = [0.0] * n
            for i, v in zip(t["indices"][0], t["values"][0]):
                out[int(i)] += float(v)
            return tuple(out)
        (u, (rows, r)), (v, (cols, _)) = t["u"], t["v"]
        return tuple(sum(u[i * r + k] * v[j * r + k] for k in range(r)) for i in range(rows) for j in range(cols))

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {"schema_version": WEIGHT_PATCH_WINDOW[1], "kind": "weight_patch", "name": self.name}
        if self.description is not None:
            out["description"] = self.description
        out.update(target=self.target, shape=list(self.shape), dtype=self.dtype, form=self.form, file=self.file,
                   sha256=self.sha256, base_sha256=self.base_sha256)
        if self.edited_sha256 is not None:
            out["edited_sha256"] = self.edited_sha256
        return out


def weight_patch_from_mapping(doc: Any, *, source: str = "<inline>", base_dir: str | Path | None = None,
                              check_file: bool = True) -> WeightPatch:
    """Read a ``weight_patch`` document, refusing a malformed one by name.

    Raises:
        NewerSchemaError: If its schema is newer than this venngram reads.
        AddressError: If a key is unknown, missing or malformed, the form is unknown, or (``check_file``) the sidecar
            is missing, differs from ``sha256``, or holds tensors that do not fit the shape and form.
    """
    from venngram.errors import NewerSchemaError

    if not isinstance(doc, Mapping) or doc.get("kind") != "weight_patch":
        raise AddressError(f"{source}: a weight patch is `kind: weight_patch`.")
    version, (lo, hi) = doc.get("schema_version"), WEIGHT_PATCH_WINDOW
    if isinstance(version, int) and not isinstance(version, bool) and version > hi:
        raise NewerSchemaError(f"{source}: weight_patch schema {version} is newer than this venngram reads ({lo} to "
                               f"{hi}); upgrade venngram.", kind="weight_patch", version=version, newest=hi)
    if not isinstance(version, int) or isinstance(version, bool) or version < lo:
        raise AddressError(f"{source}: this venngram reads weight_patch schemas {lo} to {hi}, got {version!r}.")
    unknown = sorted(set(doc) - set(_KEYS))
    if unknown:
        raise AddressError(f"{source}: unknown keys {unknown} (known: {list(_KEYS)}).")
    missing = [k for k in ("name", "target", "shape", "form", "file", "sha256", "base_sha256") if not doc.get(k)]
    if missing:
        raise AddressError(f"{source}: a weight patch gives {missing}.")
    shape = doc["shape"]
    if not isinstance(shape, list) or not 1 <= len(shape) <= 2 or not all(
            isinstance(d, int) and not isinstance(d, bool) and d > 0 for d in shape):
        raise AddressError(f"{source}: `shape` is the tensor's shape, one or two positive dimensions, got {shape!r}.")
    if doc["form"] not in FORMS:
        raise AddressError(f"{source}: `form` is one of {list(FORMS)}, got {doc['form']!r}.")
    if doc["form"] == "low_rank" and len(shape) != 2:
        raise AddressError(f"{source}: a `low_rank` patch is of a matrix, not a tensor of shape {shape}.")
    if doc.get("dtype", "float32") != "float32":
        raise AddressError(f"{source}: `dtype` is `float32` (the hashes are taken in it), got {doc['dtype']!r}.")
    patch = WeightPatch(name=str(doc["name"]), target=str(doc["target"]), shape=tuple(shape), form=doc["form"],
                        file=str(doc["file"]), sha256=str(doc["sha256"]).lower(),
                        base_sha256=str(doc["base_sha256"]).lower(), edited_sha256=doc.get("edited_sha256"),
                        description=doc.get("description"), source=source,
                        base_dir=str(base_dir) if base_dir is not None else None)
    if check_file:
        _check_tensors(patch, patch.tensors())
    return patch


def read_weight_patch(path: str | Path) -> WeightPatch:
    """Read a ``weight_patch`` document from YAML; its sidecar resolves against the document's directory."""
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        return weight_patch_from_mapping(yaml.safe_load(fh), source=str(path), base_dir=path.parent)


def write_weight_patch(path: str | Path, *, name: str, target: str, shape: Sequence[int], form: str,
                       tensors: Mapping[str, Sequence[float] | Sequence[int]], base_sha256: str,
                       edited_sha256: str | None = None, description: str | None = None) -> WeightPatch:
    """Write a weight patch: its sidecar beside ``path`` (``<stem>.safetensors``) and the document at ``path``.

    ``tensors`` holds the form's keys as flat row-major values: ``u`` and ``v`` (``low_rank``; their rank is inferred
    from ``shape``), ``indices`` and ``values`` (``sparse``), or ``delta`` (``dense``).

    Raises:
        AddressError: If the form is unknown or the tensors do not fit the shape and form.
    """
    path = Path(path)
    shape = tuple(int(d) for d in shape)
    if form not in FORMS:
        raise AddressError(f"{path}: `form` is one of {list(FORMS)}, got {form!r}.")
    if set(tensors) != set(FORMS[form]):
        raise AddressError(f"{path}: a `{form}` patch holds {list(FORMS[form])}, got {sorted(tensors)}.")
    if form == "low_rank":
        if len(shape) != 2 or len(tensors["u"]) % shape[0] or len(tensors["u"]) // shape[0] * shape[1] != len(
                tensors["v"]):
            raise AddressError(f"{path}: `u` and `v` are [rows, r] and [cols, r] for a matrix of shape {list(shape)}.")
        r = len(tensors["u"]) // shape[0]
        laid = {"u": ("F32", tensors["u"], (shape[0], r)), "v": ("F32", tensors["v"], (shape[1], r))}
    elif form == "sparse":
        laid = {"indices": ("I64", tensors["indices"], (len(tensors["indices"]),)),
                "values": ("F32", tensors["values"], (len(tensors["values"]),))}
    else:
        laid = {"delta": ("F32", tensors["delta"], shape)}
    sidecar = path.with_suffix(".safetensors")
    digest = _write(sidecar, laid)
    patch = WeightPatch(name=name, target=target, shape=shape, form=form, file=sidecar.name, sha256=digest,
                        base_sha256=base_sha256.lower(), edited_sha256=edited_sha256, description=description,
                        source=str(path), base_dir=str(path.parent))
    _check_tensors(patch, patch.tensors())
    path.write_text(yaml.safe_dump(patch.to_mapping(), sort_keys=False), encoding="utf-8")
    return patch


def tensor_sha256(tensor: Any) -> str:
    """The hash a weight patch names a tensor by: SHA-256 of its ``float32`` bytes, contiguous and little-endian.

    >>> import torch
    >>> tensor_sha256(torch.zeros(2)) == hashlib.sha256(bytes(8)).hexdigest()
    True
    """
    import torch

    t = tensor.detach().to("cpu", torch.float32).contiguous().clone()
    data = bytes(t.untyped_storage())
    if sys.byteorder != "little":
        swapped = array("f")
        swapped.frombytes(data)
        swapped.byteswap()
        data = swapped.tobytes()
    return hashlib.sha256(data).hexdigest()


def _check_tensors(patch: WeightPatch, tensors: Mapping[str, Any]) -> None:
    where = f"{patch.source}: weight patch `{patch.name}`"
    if set(tensors) != set(FORMS[patch.form]):
        raise AddressError(f"{where}: a `{patch.form}` sidecar holds {list(FORMS[patch.form])}, got {sorted(tensors)}.")
    n = math.prod(patch.shape)
    if patch.form == "dense" and tensors["delta"][1] != patch.shape:
        raise AddressError(f"{where}: `delta` has shape {list(tensors['delta'][1])}, the tensor {list(patch.shape)}.")
    if patch.form == "sparse":
        (idx, (k,)), (_, (m,)) = tensors["indices"], tensors["values"]
        if k != m:
            raise AddressError(f"{where}: {k} indices for {m} values.")
        if any(not 0 <= i < n for i in idx) or len(set(idx)) != len(idx):
            raise AddressError(f"{where}: `indices` are distinct flat positions in [0, {n}).")
    if patch.form == "low_rank":
        (_, us), (_, vs) = tensors["u"], tensors["v"]
        if len(us) != 2 or len(vs) != 2 or us[0] != patch.shape[0] or vs[0] != patch.shape[1] or us[1] != vs[1]:
            raise AddressError(f"{where}: `u` {list(us)} and `v` {list(vs)} are [rows, r] and [cols, r] for a matrix "
                               f"of shape {list(patch.shape)}.")


def _write(path: Path, tensors: Mapping[str, tuple[str, Sequence[float] | Sequence[int], tuple[int, ...]]]) -> str:
    header: dict[str, Any] = {}
    blobs: list[bytes] = []
    offset = 0
    for key, (dtype, values, shape) in tensors.items():
        if len(values) != math.prod(shape):
            raise AddressError(f"{path}: `{key}` holds {len(values)} values for shape {list(shape)}.")
        code, _ = _DTYPES[dtype]
        data = array(code, (int(v) if dtype == "I64" else float(v) for v in values))
        if sys.byteorder != "little":
            data.byteswap()
        blob = data.tobytes()
        header[key] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    header["__metadata__"] = {"venngram.schema_version": "1", "venngram.kind": "weight_patch"}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))
    return _sha256(path)


def _read(path: Path, *, sha256: str) -> dict[str, tuple[Any, tuple[int, ...]]]:
    if not path.is_file():
        raise AddressError(f"Weight patch sidecar {path} does not exist.")
    if _sha256(path) != sha256:
        raise AddressError(f"{path}: its sha256 differs from the weight patch's; the delta changed after the patch "
                           "named it.")
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(n))
        body = fh.read()
    meta = header.pop("__metadata__", None) or {}
    if meta.get("venngram.kind") != "weight_patch":
        raise AddressError(f"{path}: not a venngram weight patch sidecar.")
    out: dict[str, tuple[Any, tuple[int, ...]]] = {}
    for key, entry in header.items():
        if entry.get("dtype") not in _DTYPES:
            raise AddressError(f"{path}: `{key}` is {entry.get('dtype')}; a weight patch holds F32 and I64 tensors.")
        code, size = _DTYPES[entry["dtype"]]
        shape = tuple(int(d) for d in entry["shape"])
        lo, hi = entry["data_offsets"]
        if hi - lo != size * math.prod(shape):
            raise AddressError(f"{path}: `{key}` holds {hi - lo} bytes for {entry['dtype']} of shape {list(shape)}.")
        values = array(code)
        values.frombytes(body[lo:hi])
        if sys.byteorder != "little":
            values.byteswap()
        out[key] = (tuple(values), shape)
    return out


__all__ = ["FORMS", "read_weight_patch", "tensor_sha256", "weight_patch_from_mapping", "WEIGHT_PATCH_WINDOW",
           "WeightPatch", "write_weight_patch"]
