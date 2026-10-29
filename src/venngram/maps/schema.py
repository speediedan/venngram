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
Component Maps
^^^^^^^^^^^^^^

Where each component lives in one architecture's module tree: the data a vocabulary-backed structure provider
(:class:`~venngram.maps.structure.MapStructure`) resolves ``component`` and ``stack`` through, and the schema those
documents are written in.

**Any architecture is expressible.** A map is a plain document, written by hand or derived from any source, for any
PyTorch module tree: a decoder, an encoder-decoder, a state-space or convolutional mixer, a mixture of experts, a
vision tower, or something no library anticipated. Its rows name whatever components the model has
(``mixer.conv``, ``router``, ``projector``); a component with behaviour the bundled kinds do not describe gets a kind
of its own, declared in the document (``kinds:``); a heterogeneous stack restricts a row to the layers that have it
(``layers:``); and a model with several block lists declares each further one (``stacks:``).

One document per architecture: a component path, the module path it names (with ``{i}`` for the layer), and the
component's *kind*. Block-relative rows are keyed ``blocks.{i}.<component>``; a further block stack is keyed
``<stack>.blocks.{i}.<component>``; anything outside every block is keyed by its component alone.

:data:`VOCABULARY` is a convention, not a closed set: the names a component plays a familiar part under, so that one
address (``component:mlp.out``) selects the same part on every architecture that has it. Use a vocabulary name when a
component plays that part, and a name of your own otherwise.

Sources, in precedence order: a runtime registration (:func:`register`), then the documents bundled under ``data/``.
The bundled documents are seeds for common architectures, several of them cross-checked against TransformerLens's
architecture adapters (:mod:`~venngram.maps.transformer_lens` derives a map from one); nothing here depends on
TransformerLens. The schema is the one interpretune's activation points were first written in, so a document either
package wrote reads in the other.

.. warning:: This is an experimental API and may change in a future release.
"""
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_DATA_DIR = Path(__file__).parent / "data"

#: The shared component names: a convention for the parts most architectures have, so an address ports across them. A
#: map may name any other component. The names follow the ones TransformerLens's bridge and interpretune's activation
#: points already emit, so a selection reads the same in all three. Block-relative components first, then the ones
#: outside every block.
VOCABULARY: tuple[str, ...] = (
    "ln1", "attn", "attn.q", "attn.k", "attn.v", "attn.qkv", "attn.o", "attn.q_norm", "attn.k_norm", "ln1_post",
    "ln2", "mlp", "mlp.gate", "mlp.in", "mlp.out", "ln2_post",
    "embed", "pos_embed", "ln_final", "unembed",
)

#: The schema this code writes and reads by default. An additive descriptive key (a property, an annotation) never
#: bumps it, because readers ignore unknown keys of that class; an applicability key (one that changes which rows apply
#: or how a row resolves, such as a per-row layer predicate or a second block stack) always does, because a reader that
#: skipped one would resolve the wrong rows silently.
COMPONENT_MAP_SCHEMA_VERSION = 2

#: The oldest schema this code reads. Published maps outlive the code that wrote them, so the reader accepts a window.
COMPONENT_MAP_SCHEMA_MIN_READABLE = 1

#: The bundled kinds: whether the module returns a tuple whose element 0 is the tensor, and whether it writes to the
#: residual stream (a sublayer). A schema-2 document may declare further kinds under ``kinds:``.
KIND_TUPLE_OUTPUT: dict[str, bool] = {
    "block": True, "attn": True, "mlp": False, "norm": False, "linear": False, "embed": False, "unembed": False,
}

#: The declared property vocabulary (schema 2 ``properties:``; schema 1 spelled it ``facts:``): name -> (rule, type).
#: A ``declared`` property's source is the document; a ``derived`` one's source is the model, and a document value is
#: a cross-check that :func:`~venngram.maps.check.check_map_against_model` refuses when it disagrees. Unknown names are
#: ignored: properties describe the architecture and never select rows.
KNOWN_PROPERTIES: dict[str, tuple[str, type]] = {
    "sandwich_norms": ("declared", bool),
    "sublayers": ("declared", list),
    "rmsnorm_offset": ("derived", bool),
}

#: Keys that change which rows apply or how a row resolves; they exist only from schema 2.
APPLICABILITY_KEYS = ("kinds", "stacks")

#: The primary block stack's name. Its rows keep the bare ``blocks.{i}`` spelling, and it is the stack an address
#: without a ``stack`` field means.
PRIMARY_STACK = "blocks"


@dataclass(frozen=True)
class KindSpec:
    """What a consumer needs to know about a component kind: its output rule, and whether it is a sublayer."""

    output: str = "tensor"
    """``"tensor"`` when the module returns the tensor, ``"tuple0"`` when it returns a tuple whose element 0 is."""
    sublayer: bool = False

    @property
    def tuple_output(self) -> bool:
        """Whether the module returns a tuple whose element 0 is the tensor."""
        return self.output == "tuple0"


BUNDLED_KINDS: dict[str, KindSpec] = {
    name: KindSpec("tuple0" if tuple_out else "tensor", sublayer=name in ("attn", "mlp"))
    for name, tuple_out in KIND_TUPLE_OUTPUT.items()
}


@dataclass(frozen=True)
class RowLayers:
    """Which layers a row covers: a list, an inclusive range ``"a-b"``, or a parity (``"odd"``, ``"even"``).

    >>> RowLayers.parse("odd", "x").covers(3), RowLayers.parse("2-4", "x").covers(5)
    (True, False)
    """

    spec: tuple[int, ...] | str
    form: str  # "list" | "range" | "parity"

    @classmethod
    def parse(cls, raw: Any, where: str) -> "RowLayers":
        """Read a ``layers`` value, refusing anything else."""
        if isinstance(raw, str):
            if raw in ("odd", "even"):
                return cls(raw, "parity")
            lo, sep, hi = raw.partition("-")
            if sep and lo.strip().isdigit() and hi.strip().isdigit():
                return cls((int(lo), int(hi)), "range")
        elif isinstance(raw, list) and raw and all(isinstance(i, int) and not isinstance(i, bool) for i in raw):
            return cls(tuple(sorted(set(raw))), "list")
        raise ValueError(f"{where}: `layers` must be a list of ints, an inclusive range 'a-b', 'odd' or 'even'; got "
                         f"{raw!r}")

    def covers(self, layer: int) -> bool:
        """Whether the row applies at ``layer``."""
        if isinstance(self.spec, str):
            return layer % 2 == (1 if self.spec == "odd" else 0)
        if self.form == "range":
            return self.spec[0] <= layer <= self.spec[1]
        return layer in self.spec

    def describe(self) -> str:
        """The predicate in words, for a refusal."""
        if isinstance(self.spec, str):
            return f"{self.spec} layers"
        if self.form == "range":
            return f"layers {self.spec[0]}-{self.spec[1]}"
        return f"layers {list(self.spec)}"


@dataclass(frozen=True)
class ComponentEntry:
    """One row: a module path template (``{i}`` for the layer), a kind, and the layers it covers (``None``: all)."""

    module: str
    kind: str
    layers: RowLayers | None = None


@dataclass(frozen=True)
class ComponentKey:
    """A row key split into its block stack (``None`` for the primary one), whether it is block-relative, and its
    component (``""`` for the block itself).

    >>> ComponentKey.parse("vision.blocks.{i}.attn.o", {"vision"})
    ComponentKey(stack='vision', in_block=True, component='attn.o')
    >>> ComponentKey.parse("unembed", set())
    ComponentKey(stack=None, in_block=False, component='unembed')
    """

    stack: str | None
    in_block: bool
    component: str

    @classmethod
    def parse(cls, key: str, stacks: Any) -> "ComponentKey":
        head, sep, rest = key.partition("blocks.{i}")
        if sep and head in ("", *(f"{s}." for s in stacks)):
            return cls(head[:-1] or None, True, rest[1:])
        first, dot, tail = key.partition(".")
        if dot and first in stacks:
            return cls(first, False, tail)
        return cls(None, False, key)


@dataclass(frozen=True)
class ComponentMap:
    """The component table for one architecture, plus the structural properties a consumer needs."""

    architecture: str
    components: dict[str, ComponentEntry]
    properties: dict[str, Any] = field(default_factory=dict)
    source: str = "bundled"
    kinds: dict[str, KindSpec] = field(default_factory=lambda: dict(BUNDLED_KINDS))
    """The bundled kinds merged with the document's own declarations."""
    stacks: dict[str, str] = field(default_factory=dict)
    """Further block stacks: name -> module template (the primary stack is implied)."""
    schema_version: int = COMPONENT_MAP_SCHEMA_VERSION
    deprecated_since: str | None = None
    """Set when the map is retired: loading it warns, strict lookup refuses, and ``replacement`` names what to use."""
    replacement: str | None = None

    @property
    def sandwich_norms(self) -> bool:
        """Whether the block has post-sublayer norms (``ln1_post`` / ``ln2_post``)."""
        return bool(self.properties.get("sandwich_norms", "blocks.{i}.ln2_post" in self.components))

    @property
    def sublayers(self) -> tuple[str, ...]:
        """The order in which sublayer kinds write to the residual: declared, else ``("attn", "mlp")``."""
        declared = self.properties.get("sublayers")
        return tuple(declared) if declared else ("attn", "mlp")

    def tuple_output(self, kind: str) -> bool:
        """Whether a kind's module returns a tuple whose element 0 is the tensor."""
        return self.kinds[kind].tuple_output

    @staticmethod
    def _key(component: str, layer: int | None, stack: str | None) -> str:
        prefix = "" if stack in (None, PRIMARY_STACK) else f"{stack}."
        if layer is None:
            return f"{prefix}{component}"
        return f"{prefix}blocks.{{i}}.{component}" if component else f"{prefix}blocks.{{i}}"

    def _entry(self, component: str, layer: int | None, stack: str | None) -> ComponentEntry | None:
        entry = self.components.get(self._key(component, layer, stack))
        if entry is None or layer is None or entry.layers is None or entry.layers.covers(layer):
            return entry
        return None

    def module_for(self, component: str, layer: int | None, stack: str | None = None) -> str | None:
        """The concrete module path of a component (``""`` for the block itself) at ``layer`` (``None`` outside
        every block), or ``None`` if unmapped there."""
        entry = self._entry(component, layer, stack)
        if entry is None:
            return None
        return entry.module.replace("{i}", str(layer)) if layer is not None else entry.module

    def kind_of(self, component: str, layer: int | None, stack: str | None = None) -> str | None:
        """The component's kind, or ``None`` if unmapped there."""
        entry = self._entry(component, layer, stack)
        return None if entry is None else entry.kind

    def unmapped_reason(self, component: str, layer: int | None, stack: str | None = None) -> str | None:
        """Why a component is unmapped when a row for it exists but does not cover ``layer``, else ``None``."""
        entry = self.components.get(self._key(component, layer, stack))
        if entry is None or layer is None or entry.layers is None or entry.layers.covers(layer):
            return None
        return f"the {component or 'block'!r} row covers {entry.layers.describe()}, not layer {layer}"

    def block_components(self, stack: str | None = None) -> list[str]:
        """Block-relative components in one stack (``""`` for the block itself), the primary one by default."""
        head = self._key("", 0, stack)
        return [("" if key == head else key[len(head) + 1:]) for key in self.components
                if key == head or key.startswith(head + ".")]

    def global_components(self) -> list[str]:
        """Row keys outside every block stack (embeddings, the final norm, the unembed, a projector)."""
        return [k for k in self.components if "blocks.{i}" not in k]

    def stack_names(self) -> tuple[str, ...]:
        """The primary stack first, then the declared ones."""
        return (PRIMARY_STACK, *sorted(self.stacks))

    def block_template(self, stack: str | None = None) -> str | None:
        """The module template of a stack's blocks (``model.layers.{i}``), or ``None`` if the map has none."""
        if stack not in (None, PRIMARY_STACK):
            return self.stacks.get(stack)
        entry = self.components.get("blocks.{i}")
        return None if entry is None else entry.module

    def key(self, key: str) -> ComponentKey:
        """A row key split into stack, block-relativity and component."""
        return ComponentKey.parse(key, self.stacks)


class ComponentMapDeprecationWarning(UserWarning):
    """A retired component map was loaded; its ``replacement`` names what to use instead."""


def _schema_version(doc: dict[str, Any], source: str) -> int:
    version = doc.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError(f"{source}: component map document lacks an integer `schema_version` (current: "
                         f"{COMPONENT_MAP_SCHEMA_VERSION}); a map must say which schema it was written against.")
    if version > COMPONENT_MAP_SCHEMA_VERSION:
        raise ValueError(f"{source}: component map schema {version} is newer than this venngram reads "
                         f"({COMPONENT_MAP_SCHEMA_MIN_READABLE}-{COMPONENT_MAP_SCHEMA_VERSION}); upgrade venngram. A "
                         "newer schema may carry keys that change which rows apply, so it is not guessed at.")
    if version < COMPONENT_MAP_SCHEMA_MIN_READABLE:
        raise ValueError(f"{source}: component map schema {version} is older than the oldest readable one "
                         f"({COMPONENT_MAP_SCHEMA_MIN_READABLE}); re-publish the map against a current schema.")
    return version


def _properties(raw: Any, source: str, key: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: `{key}` must be a mapping, got {type(raw).__name__}")
    for name, (_, expected) in KNOWN_PROPERTIES.items():
        if name in raw and not (isinstance(raw[name], expected) and isinstance(raw[name], bool) == (expected is bool)):
            raise ValueError(f"{source}: property {name!r} must be {expected.__name__}, got {raw[name]!r}; a mistyped "
                             "property would otherwise degrade silently to its default.")
    sublayers = raw.get("sublayers")
    if sublayers is not None and not all(isinstance(x, str) and x for x in sublayers):
        raise ValueError(f"{source}: property 'sublayers' must be a list of kind names, got {sublayers!r}")
    return dict(raw)


def _kinds(raw: Any, source: str) -> dict[str, KindSpec]:
    kinds = dict(BUNDLED_KINDS)
    if raw is None:
        return kinds
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: `kinds` must be a mapping of kind name -> {{output, sublayer}}")
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"{source}: kind {name!r} must be a mapping with `output` (tensor|tuple0) and `sublayer`")
        output, sublayer = spec.get("output", "tensor"), spec.get("sublayer", False)
        if output not in ("tensor", "tuple0"):
            raise ValueError(f"{source}: kind {name!r} has output {output!r}; expected 'tensor' or 'tuple0'")
        if not isinstance(sublayer, bool):
            raise ValueError(f"{source}: kind {name!r} has a non-boolean `sublayer`: {sublayer!r}")
        kinds[name] = KindSpec(output, sublayer)
    return kinds


def _stacks(raw: Any, source: str) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: `stacks` must be a mapping of stack name -> {{module}}")
    stacks: dict[str, str] = {}
    for name, spec in raw.items():
        if name == PRIMARY_STACK or not isinstance(name, str) or not name.isidentifier() or not name.islower():
            raise ValueError(f"{source}: stack {name!r} is not a valid stack name (a lowercase identifier other than "
                             f"{PRIMARY_STACK!r}, the implied primary stack)")
        module = spec.get("module") if isinstance(spec, dict) else None
        if not isinstance(module, str) or "{i}" not in module:
            raise ValueError(f"{source}: stack {name!r} needs a `module` template containing '{{i}}'")
        stacks[name] = module
    return stacks


def _rows(rows: Any, source: str, kinds: dict[str, KindSpec], stacks: dict[str, str],
          version: int) -> dict[str, ComponentEntry]:
    if not isinstance(rows, dict):
        raise ValueError(f"{source}: `components` must be a mapping of component path -> {{module, kind}}")
    components: dict[str, ComponentEntry] = {}
    for name, row in rows.items():
        where = f"{source}: component {name!r}"
        if not isinstance(row, dict) or "module" not in row or "kind" not in row:
            raise ValueError(f"{where} must be a mapping with `module` and `kind`")
        if row["kind"] not in kinds:
            raise ValueError(f"{where}: unknown component kind {row['kind']!r}; expected one of {sorted(kinds)}. A "
                             "kind is refused rather than defaulted, because a guess resolves to a plausible wrong "
                             "tensor; a schema-2 document may declare a new kind under `kinds:`.")
        layers = None
        if "layers" in row:
            if version < 2:
                raise ValueError(f"{where}: `layers` is a schema-2 key (a reader that skipped it would resolve the row "
                                 "for every layer); declare schema_version: 2")
            layers = RowLayers.parse(row["layers"], where)
        head, sep, _ = name.partition(".blocks.{i}")
        if sep and head not in stacks:
            raise ValueError(f"{where}: names the block stack {head!r}, which the document does not declare under "
                             "`stacks:`")
        # only `module`, `kind` and `layers` are read: a later schema may annotate rows, at no cost to this reader
        components[name] = ComponentEntry(module=row["module"], kind=row["kind"], layers=layers)
    return components


def from_document(doc: Any, source: str = "<document>") -> ComponentMap:
    """Read a parsed component map document, refusing a malformed one by name."""
    if not isinstance(doc, dict):
        raise ValueError(f"{source}: a component map document must be a mapping, got {type(doc).__name__}")
    version = _schema_version(doc, source)
    try:
        architecture, rows = doc["architecture"], doc["components"]
    except KeyError as err:
        raise ValueError(f"{source}: component map is missing the {err.args[0]!r} key") from None
    if version < 2:
        for key in APPLICABILITY_KEYS:
            if key in doc:
                raise ValueError(f"{source}: `{key}` is a schema-2 key (a reader that skipped it would resolve the "
                                 "wrong rows); declare schema_version: 2")
        if "properties" in doc:
            raise ValueError(f"{source}: `properties` is the schema-2 spelling of `facts`; declare schema_version: 2")
        properties = _properties(doc.get("facts"), source, "facts")
    else:
        if "facts" in doc:
            raise ValueError(f"{source}: `facts` was renamed to `properties` in schema 2")
        properties = _properties(doc.get("properties"), source, "properties")
    kinds, stacks = _kinds(doc.get("kinds"), source), _stacks(doc.get("stacks"), source)
    cmap = ComponentMap(architecture=architecture, components=_rows(rows, source, kinds, stacks, version),
                        properties=properties, source=source, kinds=kinds, stacks=stacks, schema_version=version,
                        deprecated_since=doc.get("deprecated_since"), replacement=doc.get("replacement"))
    if cmap.deprecated_since is not None:
        warnings.warn(f"{source}: the component map for {architecture!r} is deprecated since {cmap.deprecated_since}"
                      + (f"; use {cmap.replacement!r}" if cmap.replacement else "")
                      + ". It still loads; strict lookup refuses it.", ComponentMapDeprecationWarning, stacklevel=3)
    return cmap


def load_component_map(path: str | Path) -> ComponentMap:
    """Read one YAML component map document. Loading runs no code (``yaml.safe_load``)."""
    with open(path, encoding="utf-8") as fh:
        return from_document(yaml.safe_load(fh), source=str(path))


_REGISTRY: dict[str, ComponentMap] = {}
_BUNDLED: dict[str, ComponentMap] = {}


def _bundled() -> dict[str, ComponentMap]:
    if not _BUNDLED:
        for path in sorted(_DATA_DIR.glob("*.yaml")):
            cmap = load_component_map(path)
            _BUNDLED.setdefault(cmap.architecture, cmap)
    return _BUNDLED


def register(cmap: ComponentMap) -> None:
    """Register (or replace) the map for an architecture at runtime; it takes precedence over a bundled one."""
    _REGISTRY[cmap.architecture] = cmap


def component_map_for(architecture: str, *, strict: bool = False) -> ComponentMap:
    """The map for an architecture class name (``LlamaForCausalLM``).

    Args:
        architecture: The architecture class name.
        strict: Refuse a retired map (one carrying ``deprecated_since``) rather than serving it with a warning.

    Raises:
        KeyError: If no map is registered or bundled for ``architecture``, naming the known ones.
        ValueError: If ``strict`` and the map is retired.
    """
    cmap = _REGISTRY.get(architecture) or _bundled().get(architecture)
    if cmap is None:
        raise KeyError(f"no component map for architecture {architecture!r}; known: {known_architectures()}. Register "
                       "one with `venngram.maps.register`.")
    if strict and cmap.deprecated_since is not None:
        raise ValueError(f"the component map for {architecture!r} is deprecated since {cmap.deprecated_since}"
                         + (f"; use {cmap.replacement!r}" if cmap.replacement else "") + " (refused by strict lookup)")
    return cmap


def known_architectures() -> list[str]:
    """Every architecture with a registered or bundled map."""
    return sorted({*_bundled(), *_REGISTRY})


__all__ = [
    "BUNDLED_KINDS",
    "COMPONENT_MAP_SCHEMA_MIN_READABLE",
    "COMPONENT_MAP_SCHEMA_VERSION",
    "component_map_for",
    "ComponentEntry",
    "ComponentKey",
    "ComponentMap",
    "ComponentMapDeprecationWarning",
    "from_document",
    "KindSpec",
    "known_architectures",
    "KNOWN_PROPERTIES",
    "load_component_map",
    "PRIMARY_STACK",
    "register",
    "RowLayers",
    "VOCABULARY",
]
