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
Unit Tables
^^^^^^^^^^^

A **unit table** holds recorded values about units, kept as a document of its own (``kind: unit_table``): one row per
``(layer, unit)``, typed columns that state what their values mean, and the provenance of the values. A treatment's
value may come ``from`` one of its columns (:mod:`venngram.treatment`: ``value: {from: {table: <name>, column:
<column>}}``), and an address book names the tables its treatments read (``tables:``, :mod:`venngram.book`), so a book
is published with the values it reads and a table is checked against the book when it loads. A table says what its
values are, not why they were chosen: a measured statistic, a published setting and a random-unit null are all tables.

.. code-block:: yaml

    schema_version: 1
    kind: unit_table
    name: nf_anime
    model: {id: Qwen/Qwen3.5-4B, revision: pinned-model-commit}    # the checkpoint the values were measured on
    source: {name: neuron-finder, repo: example/neuron-finder, commit: 0f3a9c1, license: MIT}
    columns:
      dose: {description: signed steering dose at the best confirm hit rate, measure: dose_unit}
      confirm_hit_rate: {description: fraction of confirm prompts that hit the target, measure: fraction}
      confirm_n: {description: confirm prompts, dtype: int}
    rows:
      - {layer: 13, unit: 4810, dose: 1.5, confirm_hit_rate: 0.85, confirm_n: 20}

**A table declares its key**, the coordinates its rows are keyed by: ``key: [layer, unit]`` when the document does
not say, or ``key: [stack, layer, unit]``. The key's vocabulary (:data:`KEY_FIELDS`) is the coordinates a unit target
resolves to, which is what lets a book check a table's coverage without a model, and a reader refuses a key it does
not know by name rather than misread its rows. Each key appears at most once, and every row gives every column: a unit
the table says nothing about has no row. A table holds one finding per unit, so a unit that is a finding for two
targets (with different statistics) is a row in two tables, one per finding, and a book carries one table per finding.

**Stacks.** A unit target sits in one block stack (a composed unit set sits at one site, and the stack is part of the
site). A table whose key has no ``stack`` is a table of the primary stack, and it is refused for a target in another
stack (``stack: vision``) rather than read as if its rows were measured there. A table keyed by ``stack`` gives each
row's stack, a row without one being the primary stack's:

.. code-block:: yaml

    key: [stack, layer, unit]
    rows:
      - {layer: 13, unit: 4810, dose: 1.5}                  # the primary stack
      - {stack: vision, layer: 2, unit: 77, dose: -0.4}

**The key grows as consumers need it.** For a looped or shared-weight model the layer is the weight layer a visit
reads (:mod:`venngram.maps.execution`), so a table about one visit's activations needs that visit in its key, and
models that are not built from layers would need another key. Each such coordinate joins :data:`KEY_FIELDS`, and a
reader that does not know it refuses the table by name.

**Columns state their meaning**: each has a ``description``, optionally a ``measure`` (what its numbers measure, such
as ``fraction`` or a named statistic), and a ``dtype`` (``float``, the default, or ``int``). Values are finite numbers.

**Provenance is required**: ``model`` is the checkpoint the values were measured on, pinned by ``{id, revision}`` (it
may differ from the model a treatment acts on, for example a quantized variant; a consumer decides whether that
matters), and ``source`` names where the values came from, with its ``license`` and a ``version``, ``commit`` or
``revision``. Other ``source`` keys are free-form.

**Large tables** go to a sidecar (``rows: {file, sha256}``): a safetensors file of typed columns with integer
``layer`` and ``unit`` columns, and a ``stack`` index column in a table keyed by stack (:func:`write_table_rows`),
read and written with the standard library.

**Read-time checks.** In a book with a ``tables`` section, a treatment's ``from`` value names one of the book's tables
and one of its columns, and every member of its target has a row (where the target's members can be listed without a
model), or the book is refused by name (:func:`check_members`). A book without the section leaves the tables to its
caller: a consumer that applies a ``from`` value then needs the caller's table, and refuses without it.
:meth:`UnitTable.table` gives a table in the shape such a caller passes (``{column: {(layer, unit): value}}``), for
one stack at a time.

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

#: The unit table document schema this code writes, and the window it reads.
SCHEMA_VERSION = 1
SCHEMA_WINDOW = (1, 1)
#: The row sidecar schema.
ROWS_SCHEMA_VERSION = 1
DTYPES = ("float", "int")
#: The coordinates a table's rows may be keyed by, in the order a key lists them.
KEY_FIELDS = ("stack", "layer", "unit")
#: The key of a table whose document gives none.
DEFAULT_KEY = ("layer", "unit")
#: Row keys that are not value columns (no column takes one of these names).
KEYS = KEY_FIELDS
#: Below this many rows, :meth:`UnitTable.write` keeps the rows inline.
INLINE_THRESHOLD = 512
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")
_TOP_KEYS = ("schema_version", "kind", "name", "description", "model", "source", "key", "columns", "rows")
#: The keys of a table written inline in a book (the document without its header).
BODY_KEYS = ("description", "model", "source", "key", "columns", "rows")
_VERSION_KEYS = ("version", "commit", "revision")

Key = tuple[int, int]
#: A row's full key: ``(stack, layer, unit)``, the stack ``None`` for the primary stack.
RowKey = tuple[str | None, int, int]


@dataclass(frozen=True)
class Column:
    """One column of an unit table: what its values mean."""

    name: str
    description: str
    measure: str | None = None
    """What its numbers measure (``fraction``, a named statistic, ...), where it has one."""
    dtype: str = "float"

    @classmethod
    def from_mapping(cls, name: Any, raw: Any, where: str) -> "Column":
        if not isinstance(name, str) or not _NAME.fullmatch(name) or name in KEYS:
            raise AddressError(f"{where}: {name!r} is not a valid column name (letters, digits, `_`, `.` and `-`; "
                               f"not {list(KEYS)}, which key the rows).")
        if not isinstance(raw, Mapping) or not isinstance(raw.get("description"), str) or not raw["description"]:
            raise AddressError(f"{where}: column `{name}` is {{description, measure, dtype}}, with a `description` "
                               "stating what its values mean.")
        unknown = sorted(set(raw) - {"description", "measure", "dtype"})
        if unknown:
            raise AddressError(f"{where}: column `{name}` has unknown keys {unknown}.")
        measure = raw.get("measure")
        if measure is not None and (not isinstance(measure, str) or not measure):
            raise AddressError(f"{where}: column `{name}`: `measure` is a name, got {measure!r}.")
        dtype = raw.get("dtype", "float")
        if dtype not in DTYPES:
            raise AddressError(f"{where}: column `{name}`: `dtype` is one of {list(DTYPES)}, got {dtype!r}.")
        return cls(name, raw["description"], measure, dtype)

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {"description": self.description}
        if self.measure is not None:
            out["measure"] = self.measure
        if self.dtype != "float":
            out["dtype"] = self.dtype
        return out

    def read(self, value: Any, where: str) -> float | int:
        if self.dtype == "int":
            if isinstance(value, bool) or not isinstance(value, int) or not -(1 << 63) <= value < (1 << 63):
                raise AddressError(f"{where}: `{self.name}` is an integer column (64-bit), got {value!r}.")
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise AddressError(f"{where}: `{self.name}` is a finite number, got {value!r}.")
        return float(value)


@dataclass(frozen=True)
class UnitTable:
    """A loaded unit table (see the module docs). Use :meth:`from_mapping` or :meth:`from_file`."""

    name: str
    model: Mapping[str, str]
    """The checkpoint the values were measured on, ``{id, revision}``."""
    columns: Mapping[str, Column]
    rows: Mapping[RowKey, Mapping[str, float | int]]
    """Each row's values by column, keyed by ``(stack, layer, unit)`` (``stack`` ``None`` for the primary stack), in the
    document's order."""
    provenance: Mapping[str, Any] = field(default_factory=dict)
    """The ``source`` section: where the values came from, its version and its license."""
    description: str | None = None
    source: str = "<inline>"
    rows_ref: Mapping[str, str] | None = None
    """The ``{file, sha256}`` sidecar the rows were read from, kept so the document round-trips."""
    key: tuple[str, ...] = DEFAULT_KEY
    """The coordinates the rows are keyed by (:data:`KEY_FIELDS`); a table without ``stack`` is the primary stack's."""

    @property
    def stacks(self) -> list[str | None]:
        """The stacks the table has rows for, in order of first appearance (``None`` for the primary stack)."""
        return list(dict.fromkeys(k[0] for k in self.rows))

    def column(self, name: str, *, stack: str | None = None) -> dict[Key, float | int]:
        """``(layer, unit) -> value`` of one column, over the rows of ``stack`` (the primary stack when ``None``).

        Raises:
            AddressError: If the table has no such column, or ``stack`` is another stack and the table's key has none.
        """
        if name not in self.columns:
            raise AddressError(f"{self.source}: unit table `{self.name}` has no column `{name}` (columns: "
                               f"{sorted(self.columns)}).")
        self._check_stack(stack, self.source)
        return {(layer, unit): row[name] for (s, layer, unit), row in self.rows.items() if s == stack}

    def table(self, *, stack: str | None = None) -> dict[str, dict[Key, float | int]]:
        """Every column as ``{column: {(layer, unit): value}}`` over the rows of ``stack``: the shape a consumer that
        reads ``from`` values takes its tables in."""
        return {name: self.column(name, stack=stack) for name in self.columns}

    def _check_stack(self, stack: str | None, where: str) -> None:
        if stack is not None and "stack" not in self.key:
            raise AddressError(f"{where}: unit table `{self.name}` is keyed by {list(self.key)}, so its rows are the "
                               f"primary stack's, and the target is in stack `{stack}`; a table with rows for "
                               "another stack declares `key: [stack, layer, unit]`.")

    @classmethod
    def from_mapping(cls, doc: Any, *, source: str = "<inline>", base_dir: str | Path | None = None,
                     name: str | None = None) -> "UnitTable":
        """Read an unit table document, refusing a malformed one by name. With ``name``, ``doc`` is a table
        written inline in a book: the document's body (:data:`BODY_KEYS`) without its header.

        Raises:
            AddressError: If a key is missing or malformed, the table's ``key`` names a coordinate this venngram does
                not know, provenance is incomplete, a row repeats a key, lacks a column or carries one the table does
                not declare, a value is not of its column's type or not finite, or a sidecar differs from its hash.
        """
        if name is None:
            if not isinstance(doc, Mapping) or doc.get("kind") != "unit_table":
                raise AddressError(f"{source}: an unit table is a mapping with `kind: unit_table`.")
            version = doc.get("schema_version")
            lo, hi = SCHEMA_WINDOW
            if isinstance(version, int) and not isinstance(version, bool) and version > hi:
                raise NewerAddressSchemaError(f"{source}: unit table schema {version} was written by a newer "
                                              f"venngram (this one reads {lo} to {hi}); upgrade venngram.",
                                              kind="unit_table", version=version, newest=hi)
            if not isinstance(version, int) or isinstance(version, bool) or not lo <= version <= hi:
                raise AddressError(f"{source}: `schema_version` is an integer in {lo}..{hi}, got {version!r}.")
            known: Sequence[str] = _TOP_KEYS
        else:
            if not isinstance(doc, Mapping):
                raise AddressError(f"{source}: an inline unit table is a mapping of {list(BODY_KEYS)}.")
            known = BODY_KEYS
        unknown = sorted(set(doc) - set(known))
        if unknown:
            raise AddressError(f"{source}: unknown keys {unknown} (known: {list(known)}).")
        model = doc.get("model")
        if not isinstance(model, Mapping) or not all(isinstance(model.get(k), str) and model[k]
                                                     for k in ("id", "revision")):
            raise AddressError(f"{source}: `model` is {{id, revision}}: the checkpoint the values were measured on.")
        provenance = doc.get("source")
        if not isinstance(provenance, Mapping) or not isinstance(provenance.get("license"), str) \
                or not provenance["license"] or not any(provenance.get(k) for k in _VERSION_KEYS):
            raise AddressError(f"{source}: `source` names where the values came from, with its `license` and one of "
                               f"{list(_VERSION_KEYS)}, so the table can be published and reproduced.")
        key = _key(doc.get("key"), source)
        raw_columns = doc.get("columns")
        if not isinstance(raw_columns, Mapping) or not raw_columns:
            raise AddressError(f"{source}: `columns` maps each column name to {{description, unit, dtype}}.")
        columns = {str(c): Column.from_mapping(c, raw, source) for c, raw in raw_columns.items()}
        raw_rows = doc.get("rows")
        rows_ref = None
        if _is_sidecar(raw_rows):
            rows_ref = {"file": raw_rows["file"], "sha256": raw_rows["sha256"]}
            raw_rows = read_table_rows(Path(base_dir or ".") / raw_rows["file"], sha256=raw_rows["sha256"],
                                          dtypes={c: col.dtype for c, col in columns.items()})
        if not isinstance(raw_rows, list) or not raw_rows:
            raise AddressError(f"{source}: `rows` lists at least one {{layer, unit, <column>: ...}}, or names a "
                               "sidecar {file, sha256}.")
        rows: dict[RowKey, dict[str, float | int]] = {}
        for i, raw in enumerate(raw_rows):
            row_key, values = _row(raw, i, key, columns, source)
            if row_key in rows:
                raise AddressError(f"{source}: row {i} repeats ({_shown(row_key)}): a table holds one finding per "
                                   "unit, so a second finding for a unit belongs in a table of its own.")
            rows[row_key] = values
        return cls(name=str(name or doc.get("name") or source), model=dict(model), columns=columns, rows=rows,
                   provenance=dict(provenance), description=doc.get("description"), source=source,
                   rows_ref=rows_ref, key=key)

    @classmethod
    def from_file(cls, path: str | Path, *, sha256: str | None = None) -> "UnitTable":
        """Read an unit table document from YAML; its sidecar resolves against its directory.

        Raises:
            AddressError: As :meth:`from_mapping`, or if the file differs from ``sha256``.
        """
        import yaml

        path = Path(path)
        if not path.is_file():
            raise AddressError(f"Unit table {path} does not exist.")
        if sha256 is not None and _sha256(path) != sha256.lower():
            raise AddressError(f"{path}: its sha256 differs from the reference's; the table changed after the book "
                               "named it, so it is refused rather than re-read.")
        with open(path, encoding="utf-8") as fh:
            return cls.from_mapping(yaml.safe_load(fh), source=str(path), base_dir=path.parent)

    def to_mapping(self, *, inline: bool = False) -> dict[str, Any]:
        """The document; rows read from a sidecar are written as the same reference (see :func:`write_table_rows`
        for writing one). With ``inline``, the body a book writes under its ``tables`` section, rows always inline:
        a sidecar path is relative to the table's directory, not the book's."""
        doc: dict[str, Any] = {} if inline else {"schema_version": SCHEMA_VERSION, "kind": "unit_table",
                                                 "name": self.name}
        if self.description is not None:
            doc["description"] = self.description
        doc["model"] = dict(self.model)
        doc["source"] = dict(self.provenance)
        if self.key != DEFAULT_KEY:
            doc["key"] = list(self.key)
        doc["columns"] = {name: c.to_mapping() for name, c in self.columns.items()}
        doc["rows"] = dict(self.rows_ref) if self.rows_ref is not None and not inline else \
            [{**({"stack": stack} if stack is not None else {}), "layer": layer, "unit": unit, **values}
             for (stack, layer, unit), values in self.rows.items()]
        return doc

    def write(self, path: str | Path, *, threshold: int = INLINE_THRESHOLD) -> Path:
        """Write the document to ``path`` (YAML), its rows inline up to ``threshold`` rows and otherwise to a sidecar
        beside it (``<stem>.rows.safetensors``)."""
        import yaml

        path = Path(path)
        table = replace(self, rows_ref=None)  # rows read from a sidecar elsewhere are written afresh, never referenced
        if len(self.rows) > threshold:
            sidecar = path.with_name(f"{path.stem}.rows.safetensors")
            table = replace(self, rows_ref={"file": sidecar.name, "sha256": write_table_rows(sidecar, self)})
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(table.to_mapping(), fh, sort_keys=False)
        return path


def check_members(table: UnitTable, column: str, members: Iterable[Key], where: str, *,
                  stack: str | None = None) -> None:
    """Refuse, naming them, the ``(layer, unit)`` members of a target in ``stack`` (the primary stack when ``None``)
    that have no row in ``table`` (whose ``column`` a value reads).

    Raises:
        AddressError: If the table has no such column, ``stack`` is another stack and the table's key has none, or a
            member has no row.
    """
    table.column(column)
    table._check_stack(stack, where)
    missing = [(stack, layer, unit) for layer, unit in members if (stack, layer, unit) not in table.rows]
    if missing:
        shown = ", ".join(f"({_shown(m)})" for m in missing[:8])
        more = f" and {len(missing) - 8} more" if len(missing) > 8 else ""
        raise AddressError(f"{where}: unit table `{table.name}` has no row for {shown}{more}, so column "
                           f"`{column}` gives them no value.")


def _shown(key: RowKey) -> str:
    stack, layer, unit = key
    return (f"stack {stack}, " if stack is not None else "") + f"layer {layer}, unit {unit}"


def _key(raw: Any, source: str) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_KEY
    if not isinstance(raw, list) or not all(isinstance(k, str) for k in raw):
        raise AddressError(f"{source}: `key` lists the coordinates the rows are keyed by ({list(KEY_FIELDS)}), "
                           f"got {raw!r}.")
    unknown = [k for k in raw if k not in KEY_FIELDS]
    if unknown:
        raise AddressError(f"{source}: `key` names {unknown}, which this venngram does not know (it reads keys of "
                           f"{list(KEY_FIELDS)}); upgrade venngram to read this table.")
    if len(set(raw)) != len(raw) or not {"layer", "unit"} <= set(raw):
        raise AddressError(f"{source}: `key` lists each coordinate once and includes `layer` and `unit`, got {raw!r}.")
    return tuple(k for k in KEY_FIELDS if k in raw)


def _is_sidecar(raw: Any) -> TypeGuard[Mapping[str, str]]:
    return isinstance(raw, Mapping) and set(raw) == {"file", "sha256"} and all(isinstance(raw[k], str) and raw[k]
                                                                               for k in raw)


def _row(raw: Any, i: int, key: Sequence[str], columns: Mapping[str, Column],
         source: str) -> tuple[RowKey, dict[str, float | int]]:
    where = f"{source}: row {i}"
    if not isinstance(raw, Mapping):
        raise AddressError(f"{where}: a row is {{layer, unit, <column>: <value>, ...}}, got {raw!r}.")
    if "stack" in raw and "stack" not in key:
        raise AddressError(f"{where}: gives a `stack`, and the table's key is {list(key)}; a table with rows for "
                           "another stack declares `key: [stack, layer, unit]`.")
    stack = raw.get("stack")
    if stack is not None and (not isinstance(stack, str) or not _NAME.fullmatch(stack)):
        raise AddressError(f"{where}: `stack` is a block stack's name (or absent for the primary stack), got "
                           f"{stack!r}.")
    coords = []
    for k in ("layer", "unit"):
        v = raw.get(k)
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise AddressError(f"{where}: `{k}` is an integer >= 0, got {v!r}.")
        coords.append(v)
    given = set(raw) - set(KEYS)
    unknown, missing = sorted(given - set(columns)), sorted(set(columns) - given)
    if unknown:
        raise AddressError(f"{where}: {unknown} are not declared columns (columns: {sorted(columns)}).")
    if missing:
        raise AddressError(f"{where}: lacks {missing}: every row gives every column (a unit the table says nothing "
                           "about has no row).")
    return (stack, coords[0], coords[1]), {c: col.read(raw[c], where) for c, col in columns.items()}


# ---- the row sidecar ------------------------------------------------------------------------------------------------

_CODES = {"I64": ("q", 8), "F64": ("d", 8)}
_COLUMN_DTYPE = {"int": "I64", "float": "F64"}


def write_table_rows(path: str | Path, table: UnitTable) -> str:
    """Write ``table``'s rows to a sidecar (safetensors: ``layer`` and ``unit`` as I64, each column as ``value.<name>``,
    I64 or F64 by its dtype, and in a table keyed by stack a ``stack`` I64 column indexing the stack names its
    metadata lists, ``-1`` for the primary stack) and return its sha256. Reference it from the document as ``rows:
    {file, sha256}``."""
    n = len(table.rows)
    blobs: dict[str, tuple[str, bytes]] = {}

    def put(name: str, dtype: str, values: Iterable[Any]) -> None:
        data = array(_CODES[dtype][0], values)
        if sys.byteorder != "little":
            data.byteswap()
        blobs[name] = (dtype, data.tobytes())

    meta: dict[str, Any] = {"columns": list(table.columns), "rows": n}
    if "stack" in table.key:
        stacks = [s for s in table.stacks if s is not None]
        meta["stacks"] = stacks
        put("stack", "I64", (-1 if k[0] is None else stacks.index(k[0]) for k in table.rows))
    put("layer", "I64", (k[1] for k in table.rows))
    put("unit", "I64", (k[2] for k in table.rows))
    for name, col in table.columns.items():
        put(f"value.{name}", _COLUMN_DTYPE[col.dtype], (row[name] for row in table.rows.values()))
    header: dict[str, Any] = {}
    offset = 0
    for name, (dtype, blob) in blobs.items():
        header[name] = {"dtype": dtype, "shape": [n], "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
    header["__metadata__"] = {"venngram.schema_version": str(ROWS_SCHEMA_VERSION), "venngram.kind": "unit_table_rows",
                              "venngram.table": json.dumps(meta)}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    path = Path(path)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)) + raw + b"".join(blob for _, blob in blobs.values()))
    return _sha256(path)


def read_table_rows(path: str | Path, *, sha256: str | None = None,
                       dtypes: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """A sidecar's rows as document mappings (what :meth:`UnitTable.from_mapping` reads inline); the document's
    ``columns`` then check their names and types, and ``dtypes`` (a column's :data:`DTYPES` value) the stored type.

    Raises:
        AddressError: If the file differs from ``sha256``, is not a venngram unit table row sidecar, its columns do not
            tile its data exactly, or a column has the wrong dtype or length.
    """
    path = Path(path)
    if not path.is_file():
        raise AddressError(f"Unit table rows {path} do not exist.")
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
            if meta_raw.get("venngram.kind") != "unit_table_rows" or \
                    int(meta_raw.get("venngram.schema_version", 0)) != ROWS_SCHEMA_VERSION:
                raise ValueError("not table rows")
            meta = json.loads(meta_raw["venngram.table"])
            n, names = int(meta["rows"]), [str(c) for c in meta["columns"]]
            stacks = None if meta.get("stacks") is None else [str(s) for s in meta["stacks"]]
        except (ValueError, KeyError, TypeError):
            raise AddressError(f"{path}: not a venngram unit table row sidecar (schema {ROWS_SCHEMA_VERSION})."
                               ) from None
        body = fh.read()

    spans = []
    for name, entry in header.items():
        offsets = entry.get("data_offsets") if isinstance(entry, Mapping) else None
        if not (isinstance(offsets, list) and len(offsets) == 2 and all(type(o) is int for o in offsets)
                and 0 <= offsets[0] <= offsets[1] <= len(body)):
            raise AddressError(f"{path}: column `{name}` has data offsets {offsets!r} outside its {len(body)} bytes.")
        spans.append(tuple(offsets))
    spans.sort()
    if [lo for lo, _ in spans] != [0] + [hi for _, hi in spans[:-1]] or (spans and spans[-1][1] != len(body)):
        raise AddressError(f"{path}: its columns do not tile its data exactly (overlapping, gapped or trailing bytes).")

    def column(name: str, dtypes: Sequence[str]) -> list[Any]:
        entry = header.get(name)
        if not isinstance(entry, Mapping):
            raise AddressError(f"{path}: no column `{name}`.")
        dtype = str(entry.get("dtype"))
        lo, hi = entry["data_offsets"]
        if dtype not in dtypes or entry.get("shape") != [n] or hi - lo != n * _CODES[dtype][1]:
            raise AddressError(f"{path}: column `{name}` is {' or '.join(dtypes)} with {n} rows, got {dtype} "
                               f"{entry.get('shape')}.")
        data = array(_CODES[dtype][0])
        data.frombytes(body[lo:hi])
        if sys.byteorder != "little":
            data.byteswap()
        return list(data)

    layers, units = column("layer", ("I64",)), column("unit", ("I64",))
    values = {c: column(f"value.{c}", (_COLUMN_DTYPE[dtypes[c]],) if dtypes and c in dtypes else ("I64", "F64"))
              for c in names}
    stack_of: list[dict[str, str]] = [{} for _ in range(n)]
    if stacks is not None:
        for i, s in enumerate(column("stack", ("I64",))):
            if not -1 <= s < len(stacks):
                raise AddressError(f"{path}: row {i}'s stack index {s} is outside the {len(stacks)} stacks it lists.")
            if s >= 0:
                stack_of[i] = {"stack": stacks[s]}
    elif "stack" in header:
        raise AddressError(f"{path}: has a `stack` column but lists no stacks in its metadata.")
    return [{**stack_of[i], "layer": layers[i], "unit": units[i], **{c: v[i] for c, v in values.items()}}
            for i in range(n)]


__all__ = [
    "BODY_KEYS",
    "check_members",
    "Column",
    "DEFAULT_KEY",
    "DTYPES",
    "KEY_FIELDS",
    "UnitTable",
    "INLINE_THRESHOLD",
    "read_table_rows",
    "SCHEMA_VERSION",
    "SCHEMA_WINDOW",
    "write_table_rows",
]
