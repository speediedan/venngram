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
Recipes
^^^^^^^

A recipe is a *treatment* over an address book's sets: what is done to the units a book selects. It is kept apart
from the selection (the book) and from the evidence (generations, graphs, scores, which it names), so one selection can
carry several treatments and a treatment can be re-run against a revised selection.

.. code-block:: yaml

    schema_version: 1
    kind: recipe
    name: chess
    model: {id: Qwen/Qwen3.5-4B, revision: 3f1e..., revision_source: pinned}
    book: ./chess.yaml                       # a path relative to the recipe, or the book inline
    method: {op: add, site: mlp.out.hook_in, normalization: raw, apply_to: all_tokens}
    treatments:
      - {set: chess, dose: 0.8}                         # one dose for every unit of the set
      - {set: openings, doses: {15: {4041: 0.5}}}       # or a dose per unit, by layer
    sampling: {temperature: 0.7}
    evidence: {examples: [...]}
    metadata: {...}

Applying a recipe to a model belongs to the consumer (interpretune steers, finetuning-scheduler tunes a unit adapter);
this module reads, writes and composes the documents. :func:`compose` unites several recipes' selections and merges
their doses per unit: doses add under ``add`` and multiply under ``mul`` (:data:`OPS`, extensible with
:func:`register_op`), and mixing ops on one unit is refused unless the composition states an order, since they do not
commute. :func:`from_neuronary` imports a `neuronary <https://github.com/lyramakesmusic/neuronary>`_ recipe, the first
concrete format the schema reads.

.. warning:: This is an experimental API and may change in a future release.
"""
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from venngram.address import Term
from venngram.book import AddressBook
from venngram.errors import AddressError

#: The recipe document schema this code writes and reads.
RECIPE_SCHEMA_VERSION = 1

#: Ops by name, each with how two doses on one unit combine.
OPS: dict[str, Callable[[float, float], float]] = {"add": operator.add, "mul": operator.mul}

#: neuronary sites -> venngram activation points. neuronary has one site today.
NEURONARY_SITES = {"mlp.down_proj.input": "mlp.out.hook_in"}

_NEURONARY_METADATA = ("id", "description", "tags", "author", "date", "aliases", "source", "notes", "conditional")


def register_op(name: str, combine: Callable[[float, float], float], *, replace: bool = False) -> None:
    """Register an op and how two of its doses on one unit combine (``add`` sums, ``mul`` multiplies)."""
    if name in OPS and not replace:
        raise ValueError(f"op {name!r} is already registered; pass replace=True to replace it")
    OPS[name] = combine


@dataclass(frozen=True)
class Method:
    """How doses are applied: the op, the activation point, the dose normalization and which tokens."""

    op: str
    site: str
    normalization: str = "raw"
    apply_to: str = "all_tokens"


@dataclass(frozen=True)
class Treatment:
    """A dose for every unit of a set, or a dose per unit (``{layer: {unit: dose}}``)."""

    set: str
    dose: float | None = None
    doses: Mapping[int, Mapping[int, float]] | None = None


@dataclass(frozen=True)
class Stage:
    """Doses applied together: one op over a unit table (``(layer, unit) -> dose``)."""

    method: Method
    doses: Mapping[tuple[int, int], float]


@dataclass(frozen=True)
class Recipe:
    """A treatment document (see the module docstring). Use :meth:`from_mapping` or :meth:`from_file`."""

    name: str
    model_id: str
    revision: str
    book: AddressBook
    method: Method
    treatments: tuple[Treatment, ...]
    book_ref: Any = None
    """What the document says for ``book`` (a path, or the inline mapping), kept so it round-trips."""
    revision_source: str = "pinned"
    sampling: Mapping[str, Any] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source: str = "<inline>"

    @classmethod
    def from_mapping(cls, doc: Mapping[str, Any], *, source: str = "<inline>",
                     base_dir: str | Path | None = None) -> "Recipe":
        """Read a recipe document, refusing a malformed one by name.

        Raises:
            AddressError: If a key is missing or malformed, the op is unknown, the book cannot load, a treatment names
                a set the book lacks, or the book pins another revision.
        """
        _check_header(doc, source)
        model = doc.get("model")
        if not isinstance(model, Mapping) or not model.get("id") or not model.get("revision"):
            raise AddressError(f"{source}: `model` is {{id, revision}}: a recipe's units mean something in one "
                               "revision only.")
        raw_book = doc.get("book")
        if isinstance(raw_book, str):
            book = AddressBook.from_file(Path(base_dir or ".") / raw_book)
        elif isinstance(raw_book, Mapping):
            book = AddressBook.from_mapping(raw_book, source=f"{source}: book", base_dir=base_dir, inline=True)
        else:
            raise AddressError(f"{source}: `book` is a path to an address book or the book inline.")
        pinned = [m for m in (book.applies_to.get("models") or []) if isinstance(m, Mapping)]
        if pinned and not any(m.get("id") == model["id"] and m.get("revision") == model["revision"] for m in pinned):
            raise AddressError(f"{source}: the book pins {[(m.get('id'), m.get('revision')) for m in pinned]}, not the "
                               f"recipe's {(model['id'], model['revision'])}.")
        method = _method(doc.get("method"), source)
        sets = book.sets()
        treatments = tuple(_treatment(t, sets, source) for t in doc.get("treatments") or ())
        if not treatments:
            raise AddressError(f"{source}: `treatments` lists at least one {{set, dose}} or {{set, doses}}.")
        return cls(name=str(doc.get("name") or source), model_id=model["id"], revision=model["revision"], book=book,
                   method=method, treatments=treatments, book_ref=raw_book,
                   revision_source=model.get("revision_source", "pinned"), sampling=dict(doc.get("sampling") or {}),
                   evidence=dict(doc.get("evidence") or {}), metadata=dict(doc.get("metadata") or {}), source=source)

    @classmethod
    def from_file(cls, path: str | Path) -> "Recipe":
        """Read a recipe document from YAML (loading runs no code)."""
        path = Path(path)
        with open(path, encoding="utf-8") as fh:
            return cls.from_mapping(yaml.safe_load(fh), source=str(path), base_dir=path.parent)

    def to_mapping(self) -> dict[str, Any]:
        """The recipe as a document stores it."""
        treatments = [{"set": t.set, **({"dose": t.dose} if t.dose is not None else {}),
                       **({"doses": {k: dict(v) for k, v in t.doses.items()}} if t.doses is not None else {})}
                      for t in self.treatments]
        doc: dict[str, Any] = {
            "schema_version": RECIPE_SCHEMA_VERSION, "kind": "recipe", "name": self.name,
            "model": {"id": self.model_id, "revision": self.revision, "revision_source": self.revision_source},
            "book": self.book_ref, "method": vars(self.method).copy(), "treatments": treatments}
        for key in ("sampling", "evidence", "metadata"):
            if getattr(self, key):
                doc[key] = dict(getattr(self, key))
        return doc

    def stage(self) -> Stage:
        """The recipe's doses per unit, from its book's sets."""
        doses: dict[tuple[int, int], float] = {}
        sets = self.book.sets()
        combine = OPS[self.method.op]
        for t in self.treatments:
            for layer, unit in _units(sets[t.set], t.set, self.source):
                dose = t.dose if t.doses is None else (t.doses.get(layer) or {}).get(unit)
                if dose is None:
                    continue  # a per-unit table may leave units of the set at no dose
                doses[(layer, unit)] = combine(doses[(layer, unit)], dose) if (layer, unit) in doses else dose
        return Stage(self.method, doses)


def _check_header(doc: Any, source: str) -> None:
    if not isinstance(doc, Mapping):
        raise AddressError(f"{source}: a recipe is a mapping, got {type(doc).__name__}.")
    version = doc.get("schema_version")
    if version != RECIPE_SCHEMA_VERSION:
        raise AddressError(f"{source}: recipe `schema_version` must be {RECIPE_SCHEMA_VERSION} (this venngram reads "
                           f"1), got {version!r}.")
    if doc.get("kind", "recipe") != "recipe":
        raise AddressError(f"{source}: `kind` is `recipe`, got {doc.get('kind')!r}.")


def _method(raw: Any, source: str) -> Method:
    if not isinstance(raw, Mapping) or not raw.get("op") or not raw.get("site"):
        raise AddressError(f"{source}: `method` names at least an `op` and a `site`.")
    if raw["op"] not in OPS:
        raise AddressError(f"{source}: unknown op {raw['op']!r}; known: {sorted(OPS)} (register one with "
                           "`venngram.recipe.register_op`).")
    unknown = set(raw) - {"op", "site", "normalization", "apply_to"}
    if unknown:
        raise AddressError(f"{source}: unknown `method` keys {sorted(unknown)}.")
    return Method(**{k: str(v) for k, v in raw.items()})


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _treatment(raw: Any, sets: Mapping[str, Any], source: str) -> Treatment:
    if not isinstance(raw, Mapping) or raw.get("set") not in sets:
        raise AddressError(f"{source}: a treatment names a set of the book ({sorted(sets)}), got {raw!r}.")
    dose, doses = raw.get("dose"), raw.get("doses")
    if (dose is None) == (doses is None):
        raise AddressError(f"{source}: treatment of `{raw['set']}` gives a `dose` or per-unit `doses`, not both or "
                           "neither.")
    if dose is not None and not _number(dose):
        raise AddressError(f"{source}: treatment of `{raw['set']}`: `dose` is a number, got {dose!r}.")
    if doses is not None:
        ok = isinstance(doses, Mapping) and all(
            isinstance(layer, int) and isinstance(units, Mapping)
            and all(isinstance(u, int) and _number(d) for u, d in units.items()) for layer, units in doses.items())
        if not ok:
            raise AddressError(f"{source}: treatment of `{raw['set']}`: `doses` is {{layer: {{unit: dose}}}}.")
    return Treatment(raw["set"], float(dose) if dose is not None else None,
                     {int(k): {int(u): float(d) for u, d in v.items()} for k, v in doses.items()} if doses else None)


def _units(expr: Any, name: str, source: str) -> list[tuple[int, int]]:
    """The ``(layer, unit)`` pairs a set names, for a set whose layers and units are explicit."""
    if not isinstance(expr, Term) or expr.units is None:
        raise AddressError(f"{source}: set `{name}` must be one address with `units` to carry doses.")
    units = expr.units
    if units.per_layer is not None:
        layers = [k for k, _ in units.per_layer]
    elif expr.layers is not None and expr.layers.indices is not None:
        layers = list(expr.layers.indices)
    else:
        raise AddressError(f"{source}: set `{name}` names its units for every layer of a slice or parity, so its "
                           "doses depend on the model's depth; name the layers explicitly.")
    out = []
    for layer in layers:
        idx = (dict(units.per_layer)[layer] if units.per_layer is not None else units.flat).indices  # type: ignore
        if idx is None:
            raise AddressError(f"{source}: set `{name}` names a slice of units at layer {layer}; a recipe needs "
                               "explicit units.")
        out += [(layer, u) for u in idx]
    return out


@dataclass(frozen=True)
class Composition:
    """Several recipes applied together, in order when ``ordered``."""

    recipes: tuple[Recipe, ...]
    ordered: bool = False

    def stages(self) -> list[Stage]:
        """The stages to apply, in order (see :func:`compose`)."""
        return compose(self.recipes, ordered=self.ordered)


def compose(recipes: Sequence[Recipe], *, ordered: bool = False) -> list[Stage]:
    """The stages several recipes on one model amount to.

    Recipes with the same op merge into one stage, their doses on a shared unit combining by the op (``add`` sums,
    ``mul`` multiplies). Ops that differ make separate stages: when ``ordered``, in the recipes' order (consecutive
    recipes of one op merging); otherwise only when no unit carries doses of two ops, since ``add`` then ``mul``
    differs from ``mul`` then ``add``.

    Raises:
        AddressError: If the recipes pin different models or revisions, apply at different sites, normalizations or
            tokens, or mix ops on a unit without an order.
    """
    if not recipes:
        return []
    first = recipes[0]
    for r in recipes[1:]:
        if (r.model_id, r.revision) != (first.model_id, first.revision):
            raise AddressError(f"{r.source}: composes {r.model_id}@{r.revision} with {first.model_id}@"
                               f"{first.revision}; a composition is over one revision.")
        m, f = r.method, first.method
        if (m.site, m.normalization, m.apply_to) != (f.site, f.normalization, f.apply_to):
            raise AddressError(f"{r.source}: applies at {(m.site, m.normalization, m.apply_to)}, the first recipe "
                               f"at {(f.site, f.normalization, f.apply_to)}; only the op may differ.")
    staged = [r.stage() for r in recipes]
    if not ordered:
        by_op: dict[str, dict[tuple[int, int], float]] = {}
        for s in staged:
            table = by_op.setdefault(s.method.op, {})
            for unit, dose in s.doses.items():
                table[unit] = OPS[s.method.op](table[unit], dose) if unit in table else dose
        if len(by_op) > 1:
            seen: dict[tuple[int, int], str] = {}
            for op, table in by_op.items():
                for unit in table:
                    if unit in seen:
                        raise AddressError(f"layer {unit[0]} unit {unit[1]} has `{seen[unit]}` and `{op}` doses, which "
                                           "do not commute; compose with an order.")
                    seen[unit] = op
        return [Stage(Method(op, first.method.site, first.method.normalization, first.method.apply_to), table)
                for op, table in by_op.items()]
    out: list[Stage] = []
    for s in staged:
        if out and out[-1].method.op == s.method.op:
            merged = dict(out[-1].doses)
            for unit, dose in s.doses.items():
                merged[unit] = OPS[s.method.op](merged[unit], dose) if unit in merged else dose
            out[-1] = Stage(out[-1].method, merged)
        else:
            out.append(s)
    return out


def load_recipe(path: str | Path) -> Recipe | Composition:
    """A recipe document, or a composition (``compose: [a.yaml, b.yaml]``, optional ``order: listed``)."""
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if isinstance(doc, Mapping) and "compose" in doc:
        _check_header({**doc, "kind": doc.get("kind", "recipe")}, str(path))
        parts = doc["compose"]
        if not isinstance(parts, list) or not parts or not all(isinstance(p, str) for p in parts):
            raise AddressError(f"{path}: `compose` lists recipe paths.")
        order = doc.get("order")
        if order not in (None, "listed"):
            raise AddressError(f"{path}: `order` is `listed` or absent, got {order!r}.")
        return Composition(tuple(Recipe.from_file(path.parent / p) for p in parts), ordered=order == "listed")
    return Recipe.from_mapping(doc, source=str(path), base_dir=path.parent)


def _resolve_revision(model_id: str) -> str:
    from huggingface_hub import HfApi  # an optional dependency: only resolving a revision at import needs it

    sha = HfApi().model_info(model_id).sha
    if not sha:
        raise AddressError(f"could not resolve a revision for {model_id}; pass `revision`.")
    return sha


def from_neuronary(doc: Mapping[str, Any], *, revision: str | None = None,
                   resolve_revision: Callable[[str], str] = _resolve_revision) -> tuple[dict, dict]:
    """A neuronary recipe as an address book and a venngram recipe (both mappings, for writing as documents).

    The neurons become a checkpoint book's per-layer units, their doses one dose (when they agree) or a per-unit table,
    the examples the evidence, and the descriptive fields metadata. neuronary pins no revision: unless ``revision`` is
    given, the model's current commit is resolved now and recorded as ``revision_source: resolved-at-import``.

    Raises:
        AddressError: If a neuron names an MoE expert (no venngram address field for one yet) or the site is unknown.
    """
    model_id, neurons = doc["model"], doc["neurons"]
    if any("expert" in n for n in neurons):
        raise AddressError(f"{doc.get('id')}: a neuron names an MoE expert, which venngram does not address yet.")
    method = {"op": "add", "site": "mlp.down_proj.input", "normalization": "layer_residual_median_over_down_column_l2",
              "apply_to": "all_tokens", **(doc.get("method") or {})}
    if method["site"] not in NEURONARY_SITES:
        raise AddressError(f"{doc.get('id')}: unknown neuronary site {method['site']!r}; known: "
                           f"{sorted(NEURONARY_SITES)}.")
    method["site"] = NEURONARY_SITES[method["site"]]
    source = "given" if revision is not None else "resolved-at-import"
    revision = revision if revision is not None else resolve_revision(model_id)
    per_layer: dict[int, list[int]] = {}
    for n in neurons:
        per_layer.setdefault(int(n["layer"]), []).append(int(n["neuron"]))
    name = str(doc["id"])
    book = {"schema_version": 1, "name": name, "portability": "checkpoint",
            "applies_to": {"models": [{"id": model_id, "revision": revision}]},
            "sets": {name: {"point": method["site"], "layers": sorted(per_layer),
                            "units": {k: sorted(set(v)) for k, v in sorted(per_layer.items())}}}}
    doses = {float(n["dose"]) for n in neurons}
    treatment: dict[str, Any] = {"set": name}
    if len(doses) == 1:
        treatment["dose"] = doses.pop()
    else:
        table: dict[int, dict[int, float]] = {}
        for n in neurons:
            table.setdefault(int(n["layer"]), {})[int(n["neuron"])] = float(n["dose"])
        treatment["doses"] = table
    recipe = {"schema_version": RECIPE_SCHEMA_VERSION, "kind": "recipe", "name": doc.get("name", name),
              "model": {"id": model_id, "revision": revision, "revision_source": source}, "book": book,
              "method": method, "treatments": [treatment], "sampling": dict(doc.get("sampling") or {}),
              "evidence": {"examples": list(doc.get("examples") or [])},
              "metadata": {"imported_from": "neuronary", **{k: doc[k] for k in _NEURONARY_METADATA if k in doc}}}
    return book, recipe


__all__ = [
    "compose",
    "Composition",
    "from_neuronary",
    "load_recipe",
    "Method",
    "NEURONARY_SITES",
    "OPS",
    "Recipe",
    "RECIPE_SCHEMA_VERSION",
    "register_op",
    "Stage",
    "Treatment",
]
