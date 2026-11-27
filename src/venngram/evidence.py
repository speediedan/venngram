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
Evidence Tables
^^^^^^^^^^^^^^^

An **evidence table** is recorded evidence about units, kept as a document of its own (``kind: evidence_table``): one
row per ``(layer, unit)``, typed columns with a stated meaning, and the provenance of the measurement. A treatment's
value may come ``from`` one of its columns (:mod:`venngram.recipe`: ``value: {from: {evidence: <name>, column:
<column>}}``), and an address book names the tables its treatments read (``evidence:``, :mod:`venngram.book`), so a
book is published with the evidence it reads and a table is checked against the book when it loads.

.. code-block:: yaml

    schema_version: 1
    kind: evidence_table
    name: nf_anime
    model: {id: Qwen/Qwen3.5-4B, revision: pinned-model-commit}    # the checkpoint the evidence was measured on
    source: {name: neuron-finder, repo: example/neuron-finder, commit: 0f3a9c1, license: MIT}
    columns:
      dose: {description: signed steering dose at the best confirm hit rate, unit: dose_unit}
      confirm_hit_rate: {description: fraction of confirm prompts that hit the target, unit: fraction}
      confirm_n: {description: confirm prompts, dtype: int}
    rows:
      - {layer: 13, unit: 4810, dose: 1.5, confirm_hit_rate: 0.85, confirm_n: 20}

**Rows are keyed by** ``(layer, unit)``, with each key at most once, and every row gives every column: a unit the
evidence says nothing about has no row. A table holds one finding per unit, so a unit that is a finding for two
targets (with different statistics) is a row in two tables, one per finding, and a book carries one table per finding.

**Columns state their meaning**: each has a ``description``, optionally a ``unit`` (what its numbers measure, such as
``fraction`` or a named statistic), and a ``dtype`` (``float``, the default, or ``int``). Values are finite numbers.

**Provenance is required**: ``model`` is the checkpoint the evidence was measured on, pinned by ``{id, revision}`` (it
may differ from the model a treatment steers, for example a quantized variant; a consumer decides whether that
matters), and ``source`` names where the evidence came from, with its ``license`` and a ``version``, ``commit`` or
``revision``. Other ``source`` keys are free-form.

**Large tables** go to a sidecar (``rows: {file, sha256}``): a safetensors file of typed columns with integer
``layer`` and ``unit`` columns (:func:`write_evidence_rows`), read and written with the standard library.

**Read-time checks.** In a book with an ``evidence`` section, a treatment's ``from`` value names one of the book's
tables and one of its columns, and every member of its target has a row (where the target's members can be listed
without a model), or the book is refused by name (:func:`check_members`). A book without the section leaves evidence
to its caller, as before: a consumer that applies a ``from`` value then needs the caller's table, and refuses without
it. :meth:`EvidenceTable.table` gives the table in the shape such a caller passes
(``{column: {(layer, unit): value}}``).

.. warning:: This is an experimental API and may change in a future release.
"""
import json
import math
import re
import struct
import sys
from array import array
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TypeGuard

from venngram.errors import AddressError, NewerAddressSchemaError
from venngram.units import _sha256

#: The evidence table document schema this code writes, and the window it reads.
SCHEMA_VERSION = 1
SCHEMA_WINDOW = (1, 1)
#: The row sidecar schema.
ROWS_SCHEMA_VERSION = 1
DTYPES = ("float", "int")
#: Row keys that are not value columns.
KEYS = ("layer", "unit")
#: Below this many rows, :meth:`EvidenceTable.write` keeps the rows inline.
INLINE_THRESHOLD = 512
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")
_TOP_KEYS = ("schema_version", "kind", "name", "description", "model", "source", "columns", "rows")
#: The keys of a table written inline in a book (the document without its header).
BODY_KEYS = ("description", "model", "source", "columns", "rows")
_VERSION_KEYS = ("version", "commit", "revision")

Key = tuple[int, int]


@dataclass(frozen=True)
class Column:
    """One column of an evidence table: what its values mean."""

    name: str
    description: str
    unit: str | None = None
    """What its numbers measure (``fraction``, a named statistic, ...), where it has one."""
    dtype: str = "float"

    @classmethod
    def from_mapping(cls, name: Any, raw: Any, where: str) -> "Column":
        if not isinstance(name, str) or not _NAME.fullmatch(name) or name in KEYS:
            raise AddressError(f"{where}: {name!r} is not a valid column name (letters, digits, `_`, `.` and `-`; "
                               f"not {list(KEYS)}, which key the rows).")
        if not isinstance(raw, Mapping) or not isinstance(raw.get("description"), str) or not raw["description"]:
            raise AddressError(f"{where}: column `{name}` is {{description, unit, dtype}}, with a `description` "
                               "stating what its values mean.")
        unknown = sorted(set(raw) - {"description", "unit", "dtype"})
        if unknown:
            raise AddressError(f"{where}: column `{name}` has unknown keys {unknown}.")
        unit = raw.get("unit")
        if unit is not None and (not isinstance(unit, str) or not unit):
            raise AddressError(f"{where}: column `{name}`: `unit` is a name, got {unit!r}.")
        dtype = raw.get("dtype", "float")
        if dtype not in DTYPES:
            raise AddressError(f"{where}: column `{name}`: `dtype` is one of {list(DTYPES)}, got {dtype!r}.")
        return cls(name, raw["description"], unit, dtype)

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {"description": self.description}
        if self.unit is not None:
            out["unit"] = self.unit
        if self.dtype != "float":
            out["dtype"] = self.dtype
        return out

    def read(self, value: Any, where: str) -> float | int:
        if self.dtype == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise AddressError(f"{where}: `{self.name}` is an integer column, got {value!r}.")
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise AddressError(f"{where}: `{self.name}` is a finite number, got {value!r}.")
        return float(value)


@dataclass(frozen=True)
class EvidenceTable:
    """A loaded evidence table (see the module docs). Use :meth:`from_mapping` or :meth:`from_file`."""

    name: str
    model: Mapping[str, str]
    """The checkpoint the evidence was measured on, ``{id, revision}``."""
    columns: Mapping[str, Column]
    rows: Mapping[Key, Mapping[str, float | int]]
    """Each row's values by column, keyed by ``(layer, unit)``, in the document's order."""
    provenance: Mapping[str, Any] = field(default_factory=dict)
    """The ``source`` section: where the evidence came from, its version and its license."""
    description: str | None = None
    source: str = "<inline>"
    rows_ref: Mapping[str, str] | None = None
    """The ``{file, sha256}`` sidecar the rows were read from, kept so the document round-trips."""

    def column(self, name: str) -> dict[Key, float | int]:
        """``(layer, unit) -> value`` of one column.

        Raises:
            AddressError: If the table has no such column.
        """
        if name not in self.columns:
            raise AddressError(f"{self.source}: evidence table `{self.name}` has no column `{name}` (columns: "
                               f"{sorted(self.columns)}).")
        return {key: row[name] for key, row in self.rows.items()}

    def table(self) -> dict[str, dict[Key, float | int]]:
        """Every column as ``{column: {(layer, unit): value}}``: the shape a consumer that reads ``from`` values takes
        its evidence in."""
        return {name: self.column(name) for name in self.columns}

    @classmethod
    def from_mapping(cls, doc: Any, *, source: str = "<inline>", base_dir: str | Path | None = None,
                     name: str | None = None) -> "EvidenceTable":
        """Read an evidence table document, refusing a malformed one by name. With ``name``, ``doc`` is a table
        written inline in a book: the document's body (:data:`BODY_KEYS`) without its header.

        Raises:
            AddressError: If a key is missing or malformed, provenance is incomplete, a row repeats a key, lacks a
                column or carries one the table does not declare, a value is not of its column's type or not finite,
                or a sidecar differs from its hash.
        """
        if name is None:
            if not isinstance(doc, Mapping) or doc.get("kind") != "evidence_table":
                raise AddressError(f"{source}: an evidence table is a mapping with `kind: evidence_table`.")
            version = doc.get("schema_version")
            lo, hi = SCHEMA_WINDOW
            if isinstance(version, int) and not isinstance(version, bool) and version > hi:
                raise NewerAddressSchemaError(f"{source}: evidence table schema {version} was written by a newer "
                                              f"venngram (this one reads {lo} to {hi}); upgrade venngram.",
                                              kind="evidence_table", version=version, newest=hi)
            if not isinstance(version, int) or isinstance(version, bool) or not lo <= version <= hi:
                raise AddressError(f"{source}: `schema_version` is an integer in {lo}..{hi}, got {version!r}.")
            known: Sequence[str] = _TOP_KEYS
        else:
            if not isinstance(doc, Mapping):
                raise AddressError(f"{source}: an inline evidence table is a mapping of {list(BODY_KEYS)}.")
            known = BODY_KEYS
        unknown = sorted(set(doc) - set(known))
        if unknown:
            raise AddressError(f"{source}: unknown keys {unknown} (known: {list(known)}).")
        model = doc.get("model")
        if not isinstance(model, Mapping) or not all(isinstance(model.get(k), str) and model[k]
                                                     for k in ("id", "revision")):
            raise AddressError(f"{source}: `model` is {{id, revision}}: the checkpoint the evidence was measured on.")
        provenance = doc.get("source")
        if not isinstance(provenance, Mapping) or not isinstance(provenance.get("license"), str) \
                or not provenance["license"] or not any(provenance.get(k) for k in _VERSION_KEYS):
            raise AddressError(f"{source}: `source` names where the evidence came from, with its `license` and one of "
                               f"{list(_VERSION_KEYS)}, so the table can be published and reproduced.")
        raw_columns = doc.get("columns")
        if not isinstance(raw_columns, Mapping) or not raw_columns:
            raise AddressError(f"{source}: `columns` maps each column name to {{description, unit, dtype}}.")
        columns = {str(c): Column.from_mapping(c, raw, source) for c, raw in raw_columns.items()}
        raw_rows = doc.get("rows")
        rows_ref = None
        if _is_sidecar(raw_rows):
            rows_ref = {"file": raw_rows["file"], "sha256": raw_rows["sha256"]}
            raw_rows = read_evidence_rows(Path(base_dir or ".") / raw_rows["file"], sha256=raw_rows["sha256"])
        if not isinstance(raw_rows, list) or not raw_rows:
            raise AddressError(f"{source}: `rows` lists at least one {{layer, unit, <column>: ...}}, or names a "
                               "sidecar {file, sha256}.")
        rows: dict[Key, dict[str, float | int]] = {}
        for i, raw in enumerate(raw_rows):
            key, values = _row(raw, i, columns, source)
            if key in rows:
                raise AddressError(f"{source}: row {i} repeats (layer {key[0]}, unit {key[1]}): a table holds one "
                                   "finding per unit, so a second finding for a unit belongs in a table of its own.")
            rows[key] = values
        return cls(name=str(name or doc.get("name") or source), model=dict(model), columns=columns, rows=rows,
                   provenance=dict(provenance), description=doc.get("description"), source=source,
                   rows_ref=rows_ref)

    @classmethod
    def from_file(cls, path: str | Path, *, sha256: str | None = None) -> "EvidenceTable":
        """Read an evidence table document from YAML; its sidecar resolves against its directory.

        Raises:
            AddressError: As :meth:`from_mapping`, or if the file differs from ``sha256``.
        """
        import yaml

        path = Path(path)
        if not path.is_file():
            raise AddressError(f"Evidence table {path} does not exist.")
        if sha256 is not None and _sha256(path) != sha256.lower():
            raise AddressError(f"{path}: its sha256 differs from the reference's; the table changed after the book "
                               "named it, so it is refused rather than re-read.")
        with open(path, encoding="utf-8") as fh:
            return cls.from_mapping(yaml.safe_load(fh), source=str(path), base_dir=path.parent)

    def to_mapping(self, *, inline: bool = False) -> dict[str, Any]:
        """The document (with ``inline``, the body a book writes under its ``evidence`` section); rows read from a
        sidecar are written as the same reference (see :func:`write_evidence_rows` for writing one)."""
        doc: dict[str, Any] = {} if inline else {"schema_version": SCHEMA_VERSION, "kind": "evidence_table",
                                                 "name": self.name}
        if self.description is not None:
            doc["description"] = self.description
        doc["model"] = dict(self.model)
        doc["source"] = dict(self.provenance)
        doc["columns"] = {name: c.to_mapping() for name, c in self.columns.items()}
        doc["rows"] = dict(self.rows_ref) if self.rows_ref is not None else \
            [{"layer": layer, "unit": unit, **values} for (layer, unit), values in self.rows.items()]
        return doc

    def write(self, path: str | Path, *, threshold: int = INLINE_THRESHOLD) -> Path:
        """Write the document to ``path`` (YAML), its rows inline up to ``threshold`` rows and otherwise to a sidecar
        beside it (``<stem>.rows.safetensors``)."""
        import yaml

        path = Path(path)
        table = self
        if len(self.rows) > threshold:
            sidecar = path.with_name(f"{path.stem}.rows.safetensors")
            table = replace(self, rows_ref={"file": sidecar.name, "sha256": write_evidence_rows(sidecar, self)})
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(table.to_mapping(), fh, sort_keys=False)
        return path


def check_members(table: EvidenceTable, column: str, members: Iterable[Key], where: str) -> None:
    """Refuse, naming them, the ``(layer, unit)`` members that have no row in ``table`` (whose ``column`` a value
    reads).

    Raises:
        AddressError: If the table has no such column, or a member has no row.
    """
    table.column(column)
    missing = [m for m in members if m not in table.rows]
    if missing:
        shown = ", ".join(f"(layer {layer}, unit {unit})" for layer, unit in missing[:8])
        more = f" and {len(missing) - 8} more" if len(missing) > 8 else ""
        raise AddressError(f"{where}: evidence table `{table.name}` has no row for {shown}{more}, so column "
                           f"`{column}` gives them no value.")


def _is_sidecar(raw: Any) -> TypeGuard[Mapping[str, str]]:
    return isinstance(raw, Mapping) and set(raw) == {"file", "sha256"} and all(isinstance(raw[k], str) and raw[k]
                                                                               for k in raw)


def _row(raw: Any, i: int, columns: Mapping[str, Column], source: str) -> tuple[Key, dict[str, float | int]]:
    where = f"{source}: row {i}"
    if not isinstance(raw, Mapping):
        raise AddressError(f"{where}: a row is {{layer, unit, <column>: <value>, ...}}, got {raw!r}.")
    key = []
    for k in KEYS:
        v = raw.get(k)
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise AddressError(f"{where}: `{k}` is an integer >= 0, got {v!r}.")
        key.append(v)
    given = set(raw) - set(KEYS)
    unknown, missing = sorted(given - set(columns)), sorted(set(columns) - given)
    if unknown:
        raise AddressError(f"{where}: {unknown} are not declared columns (columns: {sorted(columns)}).")
    if missing:
        raise AddressError(f"{where}: lacks {missing}: every row gives every column (a unit the evidence says nothing "
                           "about has no row).")
    return (key[0], key[1]), {c: col.read(raw[c], where) for c, col in columns.items()}


# ---- the row sidecar ------------------------------------------------------------------------------------------------

_CODES = {"I64": ("q", 8), "F64": ("d", 8)}
_COLUMN_DTYPE = {"int": "I64", "float": "F64"}


def write_evidence_rows(path: str | Path, table: EvidenceTable) -> str:
    """Write ``table``'s rows to a sidecar (safetensors: ``layer`` and ``unit`` as I64, each column as ``value.<name>``,
    I64 or F64 by its dtype) and return its sha256. Reference it from the document as ``rows: {file, sha256}``."""
    n = len(table.rows)
    blobs: dict[str, tuple[str, bytes]] = {}

    def put(name: str, dtype: str, values: Iterable[Any]) -> None:
        data = array(_CODES[dtype][0], values)
        if sys.byteorder != "little":
            data.byteswap()
        blobs[name] = (dtype, data.tobytes())

    put("layer", "I64", (k[0] for k in table.rows))
    put("unit", "I64", (k[1] for k in table.rows))
    for name, col in table.columns.items():
        put(f"value.{name}", _COLUMN_DTYPE[col.dtype], (row[name] for row in table.rows.values()))
    header: dict[str, Any] = {}
    offset = 0
    for name, (dtype, blob) in blobs.items():
        header[name] = {"dtype": dtype, "shape": [n], "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
    header["__metadata__"] = {"venngram.schema_version": str(ROWS_SCHEMA_VERSION), "venngram.kind": "evidence_rows",
                              "venngram.table": json.dumps({"columns": list(table.columns), "rows": n})}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    path = Path(path)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)) + raw + b"".join(blob for _, blob in blobs.values()))
    return _sha256(path)


def read_evidence_rows(path: str | Path, *, sha256: str | None = None) -> list[dict[str, Any]]:
    """A sidecar's rows as document mappings (what :meth:`EvidenceTable.from_mapping` reads inline); the document's
    ``columns`` then check their names and types.

    Raises:
        AddressError: If the file differs from ``sha256``, is not a venngram evidence row sidecar, or a column has the
            wrong dtype or length.
    """
    path = Path(path)
    if not path.is_file():
        raise AddressError(f"Evidence rows {path} do not exist.")
    if sha256 is not None and _sha256(path) != sha256.lower():
        raise AddressError(f"{path}: its sha256 differs from the reference's; the rows changed after the table named "
                           "them.")
    with open(path, "rb") as fh:
        head = fh.read(8)
        if len(head) != 8:
            raise AddressError(f"{path}: not a safetensors file (no header length).")
        (size,) = struct.unpack("<Q", head)
        if size > 100 * (1 << 20):
            raise AddressError(f"{path}: implausible safetensors header length {size}.")
        try:
            header = json.loads(fh.read(size))
            meta_raw = header.pop("__metadata__", None) or {}
            if meta_raw.get("venngram.kind") != "evidence_rows" or \
                    int(meta_raw.get("venngram.schema_version", 0)) != ROWS_SCHEMA_VERSION:
                raise ValueError("not evidence rows")
            meta = json.loads(meta_raw["venngram.table"])
            n, names = int(meta["rows"]), [str(c) for c in meta["columns"]]
        except (ValueError, KeyError, TypeError):
            raise AddressError(f"{path}: not a venngram evidence row sidecar (schema {ROWS_SCHEMA_VERSION}).") from None
        body = fh.read()

    def column(name: str, dtypes: Sequence[str]) -> list[Any]:
        entry = header.get(name)
        if not isinstance(entry, Mapping):
            raise AddressError(f"{path}: no column `{name}`.")
        dtype = str(entry.get("dtype"))
        lo, hi = entry.get("data_offsets", (0, -1))
        if dtype not in dtypes or entry.get("shape") != [n] or hi - lo != n * _CODES[dtype][1]:
            raise AddressError(f"{path}: column `{name}` is {' or '.join(dtypes)} with {n} rows, got {dtype} "
                               f"{entry.get('shape')}.")
        data = array(_CODES[dtype][0])
        data.frombytes(body[lo:hi])
        if sys.byteorder != "little":
            data.byteswap()
        return list(data)

    layers, units = column("layer", ("I64",)), column("unit", ("I64",))
    values = {c: column(f"value.{c}", ("I64", "F64")) for c in names}
    return [{"layer": layers[i], "unit": units[i], **{c: v[i] for c, v in values.items()}} for i in range(n)]


__all__ = [
    "BODY_KEYS",
    "check_members",
    "Column",
    "DTYPES",
    "EvidenceTable",
    "INLINE_THRESHOLD",
    "read_evidence_rows",
    "SCHEMA_VERSION",
    "SCHEMA_WINDOW",
    "write_evidence_rows",
]
