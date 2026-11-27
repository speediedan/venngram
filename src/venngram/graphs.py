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
Graph Subsets
^^^^^^^^^^^^^

A **graph subset** is a pruned piece of an attribution graph, kept as a document of its own (``kind: graph_subset``):
nodes with their coordinates and source values, optional edges with their attributions, and the provenance of the graph
they came from. An address book names the graphs its treatments use (``graphs:``, :mod:`venngram.book`), and a
treatment applies one in another context (:mod:`venngram.recipe`: ``target: {graph: <name>}``).

.. code-block:: yaml

    schema_version: 1
    kind: graph_subset
    name: reasoning
    model: {id: example/model, revision: pinned-model-commit}
    spaces:                                 # each feature space, and the coordinates its units have
      tc: {id: example/transcoders, revision: pinned, kind: transcoder}                    # index: [feature]
      bsf: {id: example/bsf, revision: pinned, kind: block_sparse, index: [block], value_shape: [16]}
    source:                                 # provenance (free-form beyond these keys)
      tokenizer: {id: example/tokenizer, revision: pinned}
      token_ids: [11, 22, 33]
      attribution_targets: [{kind: logit, token_id: 44, position: 2}]
      pruning: {method: threshold, threshold: 0.01}
    nodes:
      - {id: f0, kind: feature, site: mlp.out.hook_out, space: tc, layer: 1, position: 2, index: {feature: 7},
         activation: 3.0, baseline: 1.0, anchor: {kind: from_end, offset: 0}}
      - {id: b0, kind: feature, site: mlp.out.hook_out, space: bsf, layer: 1, position: 2, index: {block: 3},
         activation: [0.1, -0.4, ...], anchor: {kind: from_end, offset: 0}}
    edges: [{source: b0, target: f0, attribution: 0.2}]

**Coordinates are declared per space**, so no featurizer is assumed. ``index`` lists a space's integer identity axes:
one tier for a transcoder or SAE (``[feature]``, the default), the block index for a block-sparse featurizer, and a
second tier for a coordinate within a block where one is addressed. ``value_shape`` is the shape of one node's value:
``[]`` (a scalar, the default) or ``[d]`` for a block's intra-block state. A resolved coordinate is
``(site, space, layer, position, index)``.

**Node kinds** are ``feature``, ``error``, ``embedding`` and ``logit``. A feature node has a site, a space, a layer, a
source position and its full index; an error node a site, a layer and a position; an embedding or logit node a
``token_id`` and a position. Every node has a unique ``id`` and an ``activation``; other value columns (``baseline``,
...) are optional. Only feature nodes are applied by a treatment: the other kinds are context. Edges are context too,
never an instruction to route gradients.

**Anchors** (:class:`Anchor`) place a node in a new prompt: ``absolute`` (a position), ``from_end`` (an offset from the
last input token), ``token_match`` (an exact span of token ids, unique unless an ``occurrence`` is given, with an
``offset`` into the span), or ``all`` (every input position). Positions index one unpadded prompt's input tokens.
:meth:`GraphSubset.plan` resolves every selected node before returning anything: a node without an anchor, an anchor
that does not resolve, or two nodes landing on one coordinate refuses the whole plan, never dropping a node.
:meth:`GraphSubset.restriction` gives the complement of a plan within a finite universe, for ablating everything outside
the subset. venngram never mutates a model: applying a plan belongs to the consumer.

**Large tables** go to a sidecar (``nodes: {file, sha256}``, and ``edges`` likewise): a safetensors file of typed
columns, one row per node (:func:`write_graph_table`), read and written with the standard library. Integer identities
stay integer columns.

.. warning:: This is an experimental API and may change in a future release.
"""
import json
import math
import re
import struct
import sys
from array import array
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TypeGuard

from venngram.errors import AddressError, NewerAddressSchemaError
from venngram.units import _sha256

#: The graph subset document schema this code writes, and the window it reads.
SCHEMA_VERSION = 1
SCHEMA_WINDOW = (1, 1)
#: The graph table sidecar schema.
TABLE_SCHEMA_VERSION = 1
NODE_KINDS = ("feature", "error", "embedding", "logit")
#: The node kinds a treatment applies; the others are context.
APPLIED_KINDS = ("feature",)
ANCHORS = ("absolute", "from_end", "token_match", "all")
#: Node keys that are not value columns, and so cannot name an index axis or a value column.
RESERVED = ("id", "kind", "site", "space", "layer", "position", "token_id", "index", "anchor")
#: Below this many nodes, :meth:`GraphSubset.to_mapping` keeps the table inline.
INLINE_THRESHOLD = 512
#: Value transforms a graph value may pass through: they act on each value alone, so they mean the same on source rows
#: and on rows expanded by an ``all`` anchor.
GRAPH_TRANSFORMS = ("scale", "abs", "statistic")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")
_TOP_KEYS = ("schema_version", "kind", "name", "description", "model", "spaces", "source", "nodes", "edges")


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AddressError(f"{where}: expected an integer >= {minimum}, got {value!r}.")
    return value


def _finite(values: Sequence[float], where: str) -> tuple[float, ...]:
    if not all(math.isfinite(v) for v in values):
        raise AddressError(f"{where}: values are finite numbers, got {list(values)!r}.")
    return tuple(values)


@dataclass(frozen=True)
class Space:
    """A feature space a graph's feature nodes live in: its dictionary's identity, and its units' coordinates."""

    name: str
    id: str
    revision: str
    kind: str
    """The featurizer kind (``transcoder``, ``clt``, ``sae``, ``block_sparse``; any name)."""
    index: tuple[str, ...] = ("feature",)
    """The integer identity axes of a unit, outermost first (``[block]`` or ``[block, coordinate]`` for a block-sparse
    featurizer)."""
    value_shape: tuple[int, ...] = ()
    """The shape of one node's value: ``()`` for a scalar activation, ``(d,)`` for a block's intra-block state."""

    @property
    def numel(self) -> int:
        return math.prod(self.value_shape)

    @classmethod
    def from_mapping(cls, name: str, raw: Any, where: str) -> "Space":
        if not isinstance(raw, Mapping) or not all(isinstance(raw.get(k), str) and raw[k] for k in
                                                   ("id", "revision", "kind")):
            raise AddressError(f"{where}: space `{name}` is {{id, revision, kind, index, value_shape}}, with the "
                               "dictionary pinned by id and revision.")
        unknown = sorted(set(raw) - {"id", "revision", "kind", "index", "value_shape"})
        if unknown:
            raise AddressError(f"{where}: space `{name}` has unknown keys {unknown}.")
        index = raw.get("index", ["feature"])
        if not isinstance(index, list) or not index or not all(isinstance(a, str) and _NAME.fullmatch(a)
                                                               for a in index) or len(set(index)) != len(index):
            raise AddressError(f"{where}: space `{name}`: `index` lists distinct axis names, got {index!r}.")
        clash = sorted(set(index) & set(RESERVED))
        if clash:
            raise AddressError(f"{where}: space `{name}`: {clash} are node keys, so they cannot name an index axis.")
        shape = raw.get("value_shape", [])
        if not isinstance(shape, list) or len(shape) > 1 or not all(isinstance(d, int) and not isinstance(d, bool)
                                                                    and d > 0 for d in shape):
            raise AddressError(f"{where}: space `{name}`: `value_shape` is [] (a scalar) or [d], got {shape!r}.")
        return cls(name, raw["id"], raw["revision"], raw["kind"], tuple(index), tuple(shape))

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "revision": self.revision, "kind": self.kind}
        if self.index != ("feature",):
            out["index"] = list(self.index)
        if self.value_shape:
            out["value_shape"] = list(self.value_shape)
        return out


@dataclass(frozen=True)
class Anchor:
    """Where a node lands in a new prompt (see the module docs)."""

    kind: str
    position: int | None = None
    offset: int = 0
    token_ids: tuple[int, ...] = ()
    occurrence: int | None = None

    @classmethod
    def from_mapping(cls, raw: Any, where: str) -> "Anchor":
        if not isinstance(raw, Mapping) or raw.get("kind") not in ANCHORS:
            raise AddressError(f"{where}: an anchor is {{kind: ...}} with a kind in {list(ANCHORS)}, got {raw!r}.")
        kind = raw["kind"]
        allowed = {"absolute": {"position"}, "from_end": {"offset"}, "all": set(),
                   "token_match": {"token_ids", "occurrence", "offset"}}[kind]
        unknown = sorted(set(raw) - {"kind"} - allowed)
        if unknown:
            raise AddressError(f"{where}: a `{kind}` anchor has no keys {unknown}.")
        if kind == "absolute":
            if "position" not in raw:
                raise AddressError(f"{where}: an `absolute` anchor gives its `position`.")
            return cls(kind, position=_integer(raw["position"], f"{where}: position"))
        if kind == "from_end":
            return cls(kind, offset=_integer(raw.get("offset", 0), f"{where}: offset"))
        if kind == "all":
            return cls(kind)
        span = raw.get("token_ids")
        if not isinstance(span, list) or not span:
            raise AddressError(f"{where}: a `token_match` anchor gives a nonempty exact span of `token_ids`.")
        ids = tuple(_integer(t, f"{where}: token id") for t in span)
        offset = _integer(raw.get("offset", 0), f"{where}: offset")
        if offset >= len(ids):
            raise AddressError(f"{where}: offset {offset} lies outside the matched span of {len(ids)} tokens.")
        occurrence = _integer(raw["occurrence"], f"{where}: occurrence") if "occurrence" in raw else None
        return cls(kind, offset=offset, token_ids=ids, occurrence=occurrence)

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind}
        if self.kind == "absolute":
            out["position"] = self.position
        elif self.kind == "from_end":
            out["offset"] = self.offset
        elif self.kind == "token_match":
            out["token_ids"] = list(self.token_ids)
            if self.offset:
                out["offset"] = self.offset
            if self.occurrence is not None:
                out["occurrence"] = self.occurrence
        return out

    def resolve(self, token_ids: Sequence[int]) -> tuple[int, ...]:
        """The positions this anchor gives in a prompt of ``token_ids`` (its input tokens, unpadded).

        Raises:
            AddressError: If it resolves to no position or outside the prompt, or a token span is missing or matches
                more than once without an ``occurrence``.
        """
        n = len(token_ids)
        if self.kind == "absolute":
            positions: tuple[int, ...] = (int(self.position or 0),)
        elif self.kind == "from_end":
            positions = (n - 1 - self.offset,)
        elif self.kind == "all":
            positions = tuple(range(n))
        else:
            span, tokens = list(self.token_ids), list(token_ids)
            matches = [i for i in range(n - len(span) + 1) if tokens[i:i + len(span)] == span]
            if self.occurrence is not None:
                if self.occurrence >= len(matches):
                    raise AddressError(f"token span {span} occurs {len(matches)} times, so it has no occurrence "
                                       f"{self.occurrence}.")
                positions = (matches[self.occurrence] + self.offset,)
            elif len(matches) == 1:
                positions = (matches[0] + self.offset,)
            else:
                state = "is missing" if not matches else f"is ambiguous ({len(matches)} matches; give an occurrence)"
                raise AddressError(f"token span {span} {state} in the prompt.")
        if not positions or any(p < 0 or p >= n for p in positions):
            raise AddressError(f"the `{self.kind}` anchor resolves outside the prompt of {n} tokens.")
        return positions


@dataclass(frozen=True)
class Node:
    """One node of a graph subset. ``values`` holds each value column, flattened row-major to the space's
    ``value_shape`` (one number for a scalar)."""

    id: str
    kind: str
    values: Mapping[str, tuple[float, ...]]
    site: str | None = None
    space: str | None = None
    layer: int | None = None
    position: int | None = None
    index: tuple[int, ...] | None = None
    token_id: int | None = None
    anchor: Anchor | None = None

    def to_mapping(self, spaces: Mapping[str, Space]) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "kind": self.kind}
        for key in ("site", "space", "layer", "position", "token_id"):
            if getattr(self, key) is not None:
                out[key] = getattr(self, key)
        space = spaces.get(self.space) if self.space is not None else None
        if self.index is not None and space is not None:
            out["index"] = dict(zip(space.index, self.index))
        for column, values in self.values.items():
            out[column] = list(values) if space is not None and space.value_shape else values[0]
        if self.anchor is not None:
            out["anchor"] = self.anchor.to_mapping()
        return out


@dataclass(frozen=True)
class Edge:
    """An attribution from one node to another (context, never applied)."""

    source: str
    target: str
    attribution: float


@dataclass(frozen=True)
class Placement:
    """One node at one resolved destination coordinate, with its value (flattened to ``shape``)."""

    node: str
    site: str
    space: str
    layer: int
    position: int
    index: tuple[int, ...]
    value: tuple[float, ...]
    shape: tuple[int, ...] = ()

    @property
    def coordinate(self) -> tuple[str, str, int, int, tuple[int, ...]]:
        return (self.site, self.space, self.layer, self.position, self.index)


@dataclass(frozen=True)
class Complement:
    """The coordinates of a finite universe that a plan does not place (:meth:`GraphSubset.restriction`), kept as the
    universe and the placed coordinates rather than enumerated."""

    universe: Mapping[tuple[str, str, int], frozenset[tuple[int, ...]]]
    """``(site, space, layer) -> unit indices`` the universe holds, at every prompt position."""
    positions: int
    kept: frozenset[tuple[str, str, int, int, tuple[int, ...]]]

    def __contains__(self, coordinate: object) -> bool:
        if not isinstance(coordinate, tuple) or len(coordinate) != 5:
            return False
        site, space, layer, position, index = coordinate
        return 0 <= position < self.positions and index in self.universe.get((site, space, layer), frozenset()) \
            and coordinate not in self.kept

    def __len__(self) -> int:
        return sum(len(v) for v in self.universe.values()) * self.positions - len(self.kept)

    def __iter__(self) -> Iterator[tuple[str, str, int, int, tuple[int, ...]]]:
        for (site, space, layer), indices in sorted(self.universe.items()):
            for position in range(self.positions):
                for index in sorted(indices):
                    if (site, space, layer, position, index) not in self.kept:
                        yield (site, space, layer, position, index)


@dataclass(frozen=True)
class GraphSubset:
    """A loaded graph subset document (see the module docs). Use :meth:`from_mapping` or :meth:`from_file`."""

    name: str
    model: Mapping[str, str]
    spaces: Mapping[str, Space]
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    """The ``source`` section: the source prompt's tokenizer and token ids, attribution targets, pruning."""
    description: str | None = None
    source: str = "<inline>"
    nodes_ref: Mapping[str, str] | None = None
    """The ``{file, sha256}`` sidecar the nodes were read from, kept so the document round-trips."""
    edges_ref: Mapping[str, str] | None = None

    @property
    def columns(self) -> tuple[str, ...]:
        """The value columns every feature node carries (every node's, in a graph without feature nodes)."""
        nodes = [n for n in self.nodes if n.kind in APPLIED_KINDS] or list(self.nodes)
        common = set(nodes[0].values) if nodes else set()
        for node in nodes[1:]:
            common &= set(node.values)
        return tuple(sorted(common))

    def node(self, node_id: str) -> Node:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise AddressError(f"{self.source}: no node `{node_id}`.")

    def sites(self, kinds: Sequence[str] = APPLIED_KINDS) -> tuple[str, ...]:
        """The sites of the nodes of ``kinds`` (feature nodes by default), in first-seen order."""
        return tuple(dict.fromkeys(n.site for n in self.nodes if n.kind in kinds and n.site is not None))

    @classmethod
    def from_mapping(cls, doc: Any, *, source: str = "<inline>", base_dir: str | Path | None = None) -> "GraphSubset":
        """Read a graph subset document, refusing a malformed one by name.

        Raises:
            AddressError: If a key is missing or malformed, a node lacks what its kind needs, ids or source coordinates
                repeat, an edge names no node, a value is not finite or not of its space's shape, or a sidecar differs
                from its hash.
        """
        if not isinstance(doc, Mapping) or doc.get("kind") != "graph_subset":
            raise AddressError(f"{source}: a graph subset is a mapping with `kind: graph_subset`.")
        version = doc.get("schema_version")
        lo, hi = SCHEMA_WINDOW
        if isinstance(version, int) and not isinstance(version, bool) and version > hi:
            raise NewerAddressSchemaError(f"{source}: graph subset schema {version} was written by a newer venngram "
                                          f"(this one reads {lo} to {hi}); upgrade venngram.", kind="graph_subset",
                                          version=version, newest=hi)
        if not isinstance(version, int) or isinstance(version, bool) or not lo <= version <= hi:
            raise AddressError(f"{source}: `schema_version` is an integer in {lo}..{hi}, got {version!r}.")
        unknown = sorted(set(doc) - set(_TOP_KEYS))
        if unknown:
            raise AddressError(f"{source}: unknown keys {unknown} (known: {list(_TOP_KEYS)}).")
        model = doc.get("model")
        if not isinstance(model, Mapping) or not model.get("id") or not model.get("revision"):
            raise AddressError(f"{source}: `model` is {{id, revision}}: a graph's coordinates mean something in one "
                               "revision only.")
        raw_spaces = doc.get("spaces") or {}
        if not isinstance(raw_spaces, Mapping):
            raise AddressError(f"{source}: `spaces` maps a space name to {{id, revision, kind, ...}}.")
        spaces = {}
        for name, raw in raw_spaces.items():
            if not isinstance(name, str) or not _NAME.fullmatch(name):
                raise AddressError(f"{source}: {name!r} is not a valid space name.")
            spaces[name] = Space.from_mapping(name, raw, source)
        provenance = doc.get("source") or {}
        if not isinstance(provenance, Mapping):
            raise AddressError(f"{source}: `source` is a mapping (tokenizer, token_ids, attribution_targets, ...).")
        _check_provenance(provenance, source)
        base = Path(base_dir) if base_dir is not None else Path(".")
        nodes_ref = edges_ref = None
        raw_nodes = doc.get("nodes")
        if _is_sidecar(raw_nodes):
            nodes_ref = {"file": raw_nodes["file"], "sha256": raw_nodes["sha256"]}
            table = read_graph_table(base / raw_nodes["file"], sha256=raw_nodes["sha256"])
            raw_nodes = table["nodes"]
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise AddressError(f"{source}: `nodes` lists at least one node, or names a sidecar {{file, sha256}}.")
        nodes = tuple(_node(raw, i, spaces, provenance, source) for i, raw in enumerate(raw_nodes))
        _check_nodes(nodes, source)
        raw_edges: Any = doc.get("edges") or []
        if _is_sidecar(raw_edges):
            edges_ref = {"file": raw_edges["file"], "sha256": raw_edges["sha256"]}
            raw_edges = read_graph_table(base / raw_edges["file"], sha256=raw_edges["sha256"])["edges"]
        if not isinstance(raw_edges, list):
            raise AddressError(f"{source}: `edges` lists {{source, target, attribution}}, or names a sidecar.")
        ids = {n.id for n in nodes}
        edges = []
        for j, raw in enumerate(raw_edges):
            where = f"{source}: edge {j}"
            if not isinstance(raw, Mapping) or set(raw) != {"source", "target", "attribution"}:
                raise AddressError(f"{where}: an edge is {{source, target, attribution}}.")
            missing = [raw[k] for k in ("source", "target") if raw[k] not in ids]
            if missing:
                raise AddressError(f"{where}: names {missing}, which are not node ids.")
            attribution = raw["attribution"]
            if isinstance(attribution, bool) or not isinstance(attribution, (int, float)) \
                    or not math.isfinite(attribution):
                raise AddressError(f"{where}: `attribution` is a finite number, got {attribution!r}.")
            edges.append(Edge(str(raw["source"]), str(raw["target"]), float(attribution)))
        return cls(name=str(doc.get("name") or source), model=dict(model), spaces=spaces, nodes=nodes,
                   edges=tuple(edges), provenance=dict(provenance), description=doc.get("description"),
                   source=source, nodes_ref=nodes_ref, edges_ref=edges_ref)

    @classmethod
    def from_file(cls, path: str | Path, *, sha256: str | None = None) -> "GraphSubset":
        """Read a graph subset document from YAML; its sidecars resolve against its directory.

        Raises:
            AddressError: As :meth:`from_mapping`, or if the file differs from ``sha256``.
        """
        import yaml

        path = Path(path)
        if not path.is_file():
            raise AddressError(f"Graph subset {path} does not exist.")
        if sha256 is not None and _sha256(path) != sha256.lower():
            raise AddressError(f"{path}: its sha256 differs from the reference's; the graph changed after the book "
                               "named it, so it is refused rather than re-read.")
        with open(path, encoding="utf-8") as fh:
            return cls.from_mapping(yaml.safe_load(fh), source=str(path), base_dir=path.parent)

    def to_mapping(self) -> dict[str, Any]:
        """The document; a table read from a sidecar is written as the same reference (see :func:`write_graph_table`
        for writing one)."""
        doc: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "kind": "graph_subset", "name": self.name}
        if self.description is not None:
            doc["description"] = self.description
        doc["model"] = dict(self.model)
        if self.spaces:
            doc["spaces"] = {k: s.to_mapping() for k, s in self.spaces.items()}
        if self.provenance:
            doc["source"] = dict(self.provenance)
        doc["nodes"] = dict(self.nodes_ref) if self.nodes_ref is not None else \
            [n.to_mapping(self.spaces) for n in self.nodes]
        if self.edges_ref is not None:
            doc["edges"] = dict(self.edges_ref)
        elif self.edges:
            doc["edges"] = [{"source": e.source, "target": e.target, "attribution": e.attribution} for e in self.edges]
        return doc

    def write(self, path: str | Path, *, threshold: int = INLINE_THRESHOLD) -> Path:
        """Write the document to ``path`` (YAML), its nodes and edges inline up to ``threshold`` nodes and otherwise to
        a sidecar table beside it (``<stem>.table.safetensors``)."""
        import yaml

        path = Path(path)
        graph = self
        if len(self.nodes) > threshold:
            table = path.with_name(f"{path.stem}.table.safetensors")
            ref = {"file": table.name, "sha256": write_graph_table(table, self)}
            graph = replace(self, nodes_ref=ref, edges_ref=ref if self.edges else None)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(graph.to_mapping(), fh, sort_keys=False)
        return path

    def plan(self, token_ids: Sequence[int], *, column: str = "activation", baseline: str | None = None,
             sites: Sequence[str] | None = None, tokenizer: Mapping[str, str] | None = None) -> tuple[Placement, ...]:
        """Resolve the graph's feature nodes in a prompt of ``token_ids``: each node's anchored positions and its value
        (``column``, minus ``baseline`` when given), in node order. All nodes resolve before anything is returned.

        Args:
            token_ids: The destination prompt's input token ids, unpadded.
            column: The value column to read.
            baseline: A column to subtract, node by node, for a delta.
            sites: Only nodes at these sites (all of them when ``None``).
            tokenizer: ``{id, revision}`` of the destination tokenizer; a ``token_match`` anchor needs it, and it must
                be the one the graph's source pins.

        Raises:
            AddressError: Naming the node, if a selected node has no anchor or its anchor does not resolve, a column is
                missing, a delta is not finite, two nodes land on one coordinate, the tokenizer differs from the
                source's, or no node is selected.
        """
        selected = [n for n in self.nodes if n.kind in APPLIED_KINDS and (sites is None or n.site in sites)]
        if not selected:
            raise AddressError(f"{self.source}: no feature node lies at {list(sites or ())}.")
        if any(n.anchor is not None and n.anchor.kind == "token_match" for n in selected):
            self.check_tokenizer(tokenizer)
        out: list[Placement] = []
        placed: dict[tuple[str, str, int, int, tuple[int, ...]], str] = {}
        for node in selected:
            where = f"{self.source}: node `{node.id}`"
            if node.anchor is None:
                raise AddressError(f"{where} has no anchor, so it cannot be placed in another prompt.")
            value = _column(node, column, where)
            if baseline is not None:
                base = _column(node, baseline, where)
                value = tuple(a - b for a, b in zip(value, base))
                if not all(math.isfinite(v) for v in value):
                    raise AddressError(f"{where}: `{column}` - `{baseline}` is not finite.")
            try:
                positions = node.anchor.resolve(token_ids)
            except AddressError as err:
                raise AddressError(f"{where}: {err}") from None
            space = self.spaces[str(node.space)]
            for position in positions:
                p = Placement(node.id, str(node.site), space.name, int(node.layer or 0), position,
                              tuple(node.index or ()), value, space.value_shape)
                if p.coordinate in placed:
                    raise AddressError(f"{where} lands on the coordinate of node `{placed[p.coordinate]}` "
                                       f"({p.coordinate}); a collision is refused rather than resolved.")
                placed[p.coordinate] = node.id
                out.append(p)
        return tuple(out)

    def restriction(self, plan: Iterable[Placement], universe: Mapping[tuple[str, str, int], Iterable[Sequence[int]]],
                    positions: int) -> Complement:
        """Everything in a finite ``universe`` that ``plan`` does not place: what a restriction ablates.

        Args:
            plan: The placements of :meth:`plan`.
            universe: ``(site, space, layer) -> unit indices`` (each a tuple over the space's index axes), resolved by
                the consumer from a finite book set and its dictionary's widths; every prompt position is included.
            positions: The destination prompt's length.

        Raises:
            AddressError: If a placement lies outside the universe (the subset must be inside what is restricted), or
                an index has the wrong number of axes.
        """
        sets = {}
        for key, indices in universe.items():
            site, space, layer = key
            axes = len(self.spaces[space].index) if space in self.spaces else None
            members = frozenset(tuple(int(i) for i in index) for index in indices)
            if axes is not None and any(len(m) != axes for m in members):
                raise AddressError(f"{self.source}: universe {key} gives indices without the {axes} axes of space "
                                   f"`{space}`.")
            sets[(site, space, layer)] = members
        kept = frozenset(p.coordinate for p in plan)
        for coordinate in kept:
            site, space, layer, position, index = coordinate
            if position >= positions or index not in sets.get((site, space, layer), frozenset()):
                raise AddressError(f"{self.source}: the placement {coordinate} lies outside the universe, so the "
                                   "subset is not inside what is restricted.")
        return Complement(sets, positions, kept)

    def check_tokenizer(self, tokenizer: Mapping[str, str] | None) -> None:
        """Refuse a destination tokenizer other than the one the source pins (token ids mean nothing across two).

        Raises:
            AddressError: If the source pins no tokenizer, none is given, or they differ.
        """
        pinned = self.provenance.get("tokenizer")
        if not isinstance(pinned, Mapping):
            raise AddressError(f"{self.source}: a `token_match` anchor needs the source's `tokenizer` {{id, "
                               "revision}}, which the graph does not pin.")
        if tokenizer is None or (tokenizer.get("id"), str(tokenizer.get("revision"))) != (
                pinned.get("id"), str(pinned.get("revision"))):
            raise AddressError(f"{self.source}: token ids are matched only with the pinned tokenizer "
                               f"{pinned.get('id')}@{pinned.get('revision')}, not {tokenizer!r}.")


def _is_sidecar(raw: Any) -> TypeGuard[Mapping[str, str]]:
    return isinstance(raw, Mapping) and set(raw) == {"file", "sha256"} and all(isinstance(raw[k], str) and raw[k]
                                                                               for k in raw)


def _check_provenance(provenance: Mapping[str, Any], source: str) -> None:
    tokenizer = provenance.get("tokenizer")
    if tokenizer is not None and not (isinstance(tokenizer, Mapping) and tokenizer.get("id")
                                      and tokenizer.get("revision")):
        raise AddressError(f"{source}: `source.tokenizer` is {{id, revision}}.")
    token_ids = provenance.get("token_ids")
    if token_ids is not None and not (isinstance(token_ids, list)
                                      and all(isinstance(t, int) and not isinstance(t, bool) and t >= 0
                                              for t in token_ids)):
        raise AddressError(f"{source}: `source.token_ids` lists the source prompt's token ids.")


def _values(raw: Any, shape: tuple[int, ...], where: str) -> tuple[float, ...]:
    flat = raw if shape else [raw]
    if not isinstance(flat, list) or len(flat) != math.prod(shape) or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in flat):
        what = f"a list of {shape[0]} numbers" if shape else "a number"
        raise AddressError(f"{where}: is {what} (the space's value_shape {list(shape)}), got {raw!r}.")
    return _finite([float(v) for v in flat], where)


def _node(raw: Any, i: int, spaces: Mapping[str, Space], provenance: Mapping[str, Any], source: str) -> Node:
    where = f"{source}: node {i}"
    if not isinstance(raw, Mapping) or not isinstance(raw.get("id"), str) or not raw["id"]:
        raise AddressError(f"{where}: a node is a mapping with a string `id`.")
    node_id = raw["id"]
    where = f"{source}: node `{node_id}`"
    kind = raw.get("kind")
    if kind not in NODE_KINDS:
        raise AddressError(f"{where}: `kind` is one of {list(NODE_KINDS)}, got {kind!r}.")
    need = {"feature": ("site", "space", "layer", "position", "index"), "error": ("site", "layer", "position"),
            "embedding": ("token_id", "position"), "logit": ("token_id", "position")}[kind]
    missing = [k for k in need if raw.get(k) is None]
    if missing:
        raise AddressError(f"{where}: a `{kind}` node gives {list(need)}; it lacks {missing}.")
    space = None
    if raw.get("space") is not None:
        if raw["space"] not in spaces:
            raise AddressError(f"{where}: space `{raw['space']}` is not declared (spaces: {sorted(spaces)}).")
        space = spaces[raw["space"]]
    index = None
    if raw.get("index") is not None:
        if space is None:
            raise AddressError(f"{where}: an `index` is read through a space, and the node names none.")
        rindex = raw["index"]
        if not isinstance(rindex, Mapping) or list(rindex) != list(space.index):
            raise AddressError(f"{where}: `index` gives the axes {list(space.index)} of space `{space.name}`, in order,"
                               f" got {rindex!r}.")
        index = tuple(_integer(rindex[a], f"{where}: index `{a}`") for a in space.index)
    shape = space.value_shape if space is not None else ()
    columns = [k for k in raw if k not in RESERVED]
    if "activation" not in columns:
        raise AddressError(f"{where}: every node carries its source `activation`.")
    for column in columns:
        if not _NAME.fullmatch(str(column)):
            raise AddressError(f"{where}: {column!r} is not a valid value column name.")
    values = {str(c): _values(raw[c], shape, f"{where}: `{c}`") for c in columns}
    position = _integer(raw["position"], f"{where}: position") if raw.get("position") is not None else None
    token_ids = provenance.get("token_ids")
    if position is not None and isinstance(token_ids, list) and position >= len(token_ids):
        raise AddressError(f"{where}: source position {position} lies outside the source prompt of {len(token_ids)} "
                           "tokens.")
    site = raw.get("site")
    if site is not None and (not isinstance(site, str) or not site):
        raise AddressError(f"{where}: `site` is a site name, got {site!r}.")
    return Node(node_id, kind, values, site=site, space=space.name if space is not None else None,
                layer=_integer(raw["layer"], f"{where}: layer") if raw.get("layer") is not None else None,
                position=position, index=index,
                token_id=_integer(raw["token_id"], f"{where}: token_id") if raw.get("token_id") is not None else None,
                anchor=Anchor.from_mapping(raw["anchor"], f"{where}: anchor") if raw.get("anchor") is not None
                else None)


def _check_nodes(nodes: Sequence[Node], source: str) -> None:
    seen: dict[str, Node] = {}
    coordinates: dict[tuple[Any, ...], str] = {}
    for node in nodes:
        if node.id in seen:
            raise AddressError(f"{source}: node id `{node.id}` is repeated.")
        seen[node.id] = node
        key = (node.kind, node.site, node.space, node.layer, node.position, node.index, node.token_id)
        if key in coordinates:
            raise AddressError(f"{source}: nodes `{coordinates[key]}` and `{node.id}` have one source coordinate.")
        coordinates[key] = node.id


def _column(node: Node, column: str, where: str) -> tuple[float, ...]:
    if column not in node.values:
        raise AddressError(f"{where} has no `{column}` column (it has {sorted(node.values)}).")
    return node.values[column]


# ---- the table sidecar ----------------------------------------------------------------------------------------------

_DTYPES = {"U8": ("B", 1), "I32": ("i", 4), "I64": ("q", 8), "F32": ("f", 4)}


def _blob(code: str, values: Iterable[Any]) -> bytes:
    data = array(code, values)
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()


def _codes(values: Sequence[str | None]) -> tuple[list[str], list[int]]:
    table = list(dict.fromkeys(v for v in values if v is not None))
    return table, [table.index(v) if v is not None else -1 for v in values]


def write_graph_table(path: str | Path, graph: GraphSubset, *, edges: bool = True) -> str:
    """Write ``graph``'s nodes (and its edges, unless ``edges`` is false) to a sidecar table and return its sha256.

    Reference it from the document as ``nodes: {file, sha256}`` (and ``edges`` likewise). Node ids are kept in the
    file's metadata; edges refer to node rows.
    """
    nodes = graph.nodes
    n = len(nodes)
    axes = list(dict.fromkeys(a for s in graph.spaces.values() for a in s.index))
    kinds_t, kind_codes = list(NODE_KINDS), [NODE_KINDS.index(x.kind) for x in nodes]
    sites_t, site_codes = _codes([x.site for x in nodes])
    spaces_t, space_codes = _codes([x.space for x in nodes])
    columns: dict[str, tuple[str, list[int], bytes]] = {}

    def put(name: str, dtype: str, shape: list[int], values: Iterable[Any]) -> None:
        columns[name] = (dtype, shape, _blob(_DTYPES[dtype][0], values))

    def opt(v: int | None) -> int:
        return -1 if v is None else int(v)

    put("kind", "U8", [n], kind_codes)
    put("site", "I32", [n], site_codes)
    put("space", "I32", [n], space_codes)
    for key in ("layer", "position", "token_id"):
        put(key, "I64", [n], (opt(getattr(x, key)) for x in nodes))
    for axis in axes:
        vals = []
        for x in nodes:
            space = graph.spaces.get(x.space) if x.space is not None else None
            vals.append(x.index[space.index.index(axis)] if space is not None and x.index is not None
                        and axis in space.index else -1)
        put(f"index.{axis}", "I64", [n], vals)
    anchors = [x.anchor for x in nodes]
    put("anchor.kind", "U8", [n], (255 if a is None else ANCHORS.index(a.kind) for a in anchors))
    put("anchor.position", "I64", [n], (-1 if a is None or a.position is None else a.position for a in anchors))
    put("anchor.offset", "I64", [n], (-1 if a is None else a.offset for a in anchors))
    put("anchor.occurrence", "I64", [n], (-1 if a is None or a.occurrence is None else a.occurrence for a in anchors))
    span: list[int] = []
    offsets = [0]
    for a in anchors:
        span += list(a.token_ids) if a is not None else []
        offsets.append(len(span))
    put("anchor.span", "I64", [len(span)], span)
    put("anchor.span_offsets", "I64", [n + 1], offsets)
    value_columns = sorted({c for x in nodes for c in x.values})
    shapes = {graph.spaces[x.space].value_shape if x.space is not None else () for x in nodes}
    if len(shapes) > 1:
        raise AddressError(f"{graph.source}: a table holds one value shape; its nodes' spaces have shapes "
                           f"{sorted(list(s) for s in shapes)}. Write one table per value shape.")
    (shape,) = shapes
    numel = math.prod(shape)
    for c in value_columns:
        put(f"value.{c}", "F32", [n, *shape], (v for x in nodes for v in x.values.get(c, (0.0,) * numel)))
        if any(c not in x.values for x in nodes):  # a column some nodes lack carries a mask of the rows that have it
            put(f"present.{c}", "U8", [n], (int(c in x.values) for x in nodes))
    meta: dict[str, Any] = {"kinds": kinds_t, "sites": sites_t, "spaces": spaces_t, "axes": axes,
                            "anchors": list(ANCHORS), "ids": [x.id for x in nodes], "columns": list(value_columns),
                            "rows": n}
    if edges and graph.edges:
        row = {x.id: i for i, x in enumerate(nodes)}
        put("edge.source", "I64", [len(graph.edges)], (row[e.source] for e in graph.edges))
        put("edge.target", "I64", [len(graph.edges)], (row[e.target] for e in graph.edges))
        put("edge.attribution", "F32", [len(graph.edges)], (e.attribution for e in graph.edges))
        meta["edges"] = len(graph.edges)
    header: dict[str, Any] = {}
    blobs, offset = [], 0
    for name, (dtype, shape, blob) in columns.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    header["__metadata__"] = {"venngram.schema_version": str(TABLE_SCHEMA_VERSION), "venngram.kind": "graph_table",
                              "venngram.table": json.dumps(meta, sort_keys=True)}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    path = Path(path)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))
    return _sha256(path)


def read_graph_table(path: str | Path, *, sha256: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """A sidecar table's nodes and edges as document mappings (what :meth:`GraphSubset.from_mapping` reads inline).

    Raises:
        AddressError: If the file differs from ``sha256``, is not a venngram graph table, or a column has the wrong
            dtype or length, a code lies outside its table, or an edge names no row.
    """
    path = Path(path)
    if not path.is_file():
        raise AddressError(f"Graph table {path} does not exist.")
    if sha256 is not None and _sha256(path) != sha256.lower():
        raise AddressError(f"{path}: its sha256 differs from the reference's; the table changed after the graph named "
                           "it.")
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
            if meta_raw.get("venngram.kind") != "graph_table" or \
                    int(meta_raw.get("venngram.schema_version", 0)) != TABLE_SCHEMA_VERSION:
                raise ValueError("not a graph table")
            meta = json.loads(meta_raw["venngram.table"])
        except (ValueError, KeyError):
            raise AddressError(f"{path}: not a venngram graph table (schema {TABLE_SCHEMA_VERSION}).") from None
        body = fh.read()

    def column(name: str, dtype: str, length: int | None = None) -> list[Any]:
        if name not in header:
            raise AddressError(f"{path}: no column `{name}`.")
        entry = header[name]
        code, size_of = _DTYPES[dtype]
        lo, hi = entry["data_offsets"]
        if entry["dtype"] != dtype or (hi - lo) % size_of or (length is not None and entry["shape"][0] != length) \
                or math.prod(entry["shape"]) * size_of != hi - lo:
            raise AddressError(f"{path}: column `{name}` is {dtype} with {length} rows, got {entry['dtype']} "
                               f"{entry['shape']}.")
        data = array(code)
        data.frombytes(body[lo:hi])
        if sys.byteorder != "little":
            data.byteswap()
        return list(data)

    n = int(meta["rows"])

    def coded(name: str, table: list[str], dtype: str, missing: int | None = -1) -> list[str | None]:
        out = []
        for c in column(name, dtype, n):
            if c == missing:
                out.append(None)
            elif not 0 <= c < len(table):
                raise AddressError(f"{path}: column `{name}` has code {c} outside its table of {len(table)}.")
            else:
                out.append(table[c])
        return out

    kinds = coded("kind", meta["kinds"], "U8", None)
    sites = coded("site", meta["sites"], "I32")
    spaces = coded("space", meta["spaces"], "I32")
    ints = {k: column(k, "I64", n) for k in ("layer", "position", "token_id")}
    axes = {a: column(f"index.{a}", "I64", n) for a in meta["axes"]}
    anchor_kinds = column("anchor.kind", "U8", n)
    anchor = {k: column(f"anchor.{k}", "I64", n) for k in ("position", "offset", "occurrence")}
    span = column("anchor.span", "I64")
    offsets = column("anchor.span_offsets", "I64", n + 1)
    if offsets[0] != 0 or offsets[-1] != len(span) or any(b < a for a, b in zip(offsets, offsets[1:])):
        raise AddressError(f"{path}: `anchor.span_offsets` does not index `anchor.span`.")
    values = {c: column(f"value.{c}", "F32") for c in meta["columns"]}
    present = {c: column(f"present.{c}", "U8", n) for c in meta["columns"] if f"present.{c}" in header}
    ids = meta["ids"]
    if len(ids) != n or len(set(ids)) != n:
        raise AddressError(f"{path}: the metadata gives {len(ids)} ids for {n} rows, or repeats one.")
    width = {c: len(v) // n if n else 1 for c, v in values.items()}
    nodes = []
    for i in range(n):
        node: dict[str, Any] = {"id": ids[i], "kind": kinds[i]}
        for key, vals in (("site", sites), ("space", spaces)):
            if vals[i] is not None:
                node[key] = vals[i]
        for key, vals in ints.items():
            if vals[i] >= 0:
                node[key] = vals[i]
        axis_values = {a: v[i] for a, v in axes.items() if v[i] >= 0}
        if axis_values:
            node["index"] = axis_values
        for c, vals in values.items():
            if c in present and not present[c][i]:
                continue
            row = vals[i * width[c]:(i + 1) * width[c]]
            node[c] = row if header[f"value.{c}"]["shape"][1:] else row[0]
        k = anchor_kinds[i]
        if k != 255:
            if k >= len(meta["anchors"]):
                raise AddressError(f"{path}: row {i} has anchor code {k} outside its table.")
            a: dict[str, Any] = {"kind": meta["anchors"][k]}
            if a["kind"] == "absolute":
                a["position"] = anchor["position"][i]
            elif a["kind"] == "from_end":
                a["offset"] = anchor["offset"][i]
            elif a["kind"] == "token_match":
                a["token_ids"] = span[offsets[i]:offsets[i + 1]]
                a["offset"] = anchor["offset"][i]
                if anchor["occurrence"][i] >= 0:
                    a["occurrence"] = anchor["occurrence"][i]
            node["anchor"] = a
        nodes.append(node)
    edges = []
    if "edges" in meta:
        m = int(meta["edges"])
        src, dst = column("edge.source", "I64", m), column("edge.target", "I64", m)
        attribution = column("edge.attribution", "F32", m)
        for j in range(m):
            if not (0 <= src[j] < n and 0 <= dst[j] < n):
                raise AddressError(f"{path}: edge {j} names a row outside the {n} nodes.")
            edges.append({"source": ids[src[j]], "target": ids[dst[j]], "attribution": attribution[j]})
    return {"nodes": nodes, "edges": edges}


def anchor_cases() -> list[dict[str, Any]]:
    """Conformance vectors for anchoring: each ``{anchor, token_ids}`` with the ``positions`` it resolves to, or a
    ``refuses`` substring of the refusal. Two consumers that pass them place a subset's nodes alike."""
    cases = [
        ({"kind": "absolute", "position": 2}, [11, 22, 33], [2], None),
        ({"kind": "absolute", "position": 3}, [11, 22, 33], None, "outside"),
        ({"kind": "from_end", "offset": 0}, [11, 22, 33], [2], None),
        ({"kind": "from_end", "offset": 2}, [11, 22, 33], [0], None),
        ({"kind": "from_end", "offset": 3}, [11, 22, 33], None, "outside"),
        ({"kind": "token_match", "token_ids": [22]}, [11, 22, 33], [1], None),
        ({"kind": "token_match", "token_ids": [22, 33], "offset": 1}, [11, 22, 33], [2], None),
        ({"kind": "token_match", "token_ids": [11]}, [11, 22, 11], None, "ambiguous"),
        ({"kind": "token_match", "token_ids": [11], "occurrence": 1}, [11, 22, 11], [2], None),
        ({"kind": "token_match", "token_ids": [44]}, [11, 22, 33], None, "missing"),
        ({"kind": "token_match", "token_ids": [11], "occurrence": 2}, [11, 22, 11], None, "occurrence"),
        ({"kind": "token_match", "token_ids": [11, 11], "occurrence": 1}, [11, 11, 11], [1], None),
        ({"kind": "all"}, [11, 22, 33], [0, 1, 2], None),
        ({"kind": "all"}, [], None, "outside"),
    ]
    return [{"anchor": a, "token_ids": t, **({"positions": p} if p is not None else {"refuses": r})}
            for a, t, p, r in cases]


__all__ = [
    "Anchor",
    "anchor_cases",
    "ANCHORS",
    "APPLIED_KINDS",
    "Complement",
    "Edge",
    "GRAPH_TRANSFORMS",
    "GraphSubset",
    "INLINE_THRESHOLD",
    "Node",
    "NODE_KINDS",
    "Placement",
    "read_graph_table",
    "SCHEMA_VERSION",
    "SCHEMA_WINDOW",
    "Space",
    "TABLE_SCHEMA_VERSION",
    "write_graph_table",
]
