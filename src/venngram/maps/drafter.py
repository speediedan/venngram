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
Drafting a Map From a Module Tree
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A map source that needs no library's knowledge of an architecture: it drafts a component map from a model's own module
tree, by **drafting rules** that are data, from as many sources as are given:

- **rules mined from the registry** (:func:`mine_rules`): every registered map already says which module names became
  which components, so the maps teach the rules, and the rules grow as the registry does;
- **rule documents** (``kind: drafting_rules``): the bundled ``transformer`` and ``ssm`` sets, or any a family's
  maintainers publish (:func:`load_drafting_rules`, or through :mod:`venngram.store`).

None is privileged: :func:`draft_map` tries the sources in the order given (by default, mined rules first, then the
bundled documents) and the first rule that matches names a module. A rule document declares **containers** (a module
whose children are named under a component, such as ``self_attn`` -> ``attn`` or ``mixer`` -> ``mixer``), **leaves**
per scope (a container's component, ``block`` for modules directly in a block, ``outer`` for modules outside the block
stack), conditional rules (``when_sibling``: sandwich norms rename the post-attention norm), and any **kinds** it
introduces.

The drafter reads every block of the stack: a module present in only some blocks (a hybrid stack's alternating mixers)
becomes an ``optional`` row. Every module with parameters that no rule names becomes a ``draft.<path>`` row. The draft
says ``draft: true`` and lists its ``unconfirmed`` rows and any other stacks it left ``unmapped_stacks``. Each row
carries a ``confidence`` (a mined rule's share of the maps' votes; 1 for a declared rule; 0 for an unnamed module) and,
where maps disagreed on the name, the ``alternatives`` they chose, so a reviewer starts with the least certain rows.

**Model-assisted drafting** (:func:`draft_with_model`) is the fallback for an architecture no rule describes well: the
caller supplies a completion function for whatever model or service it uses, and venngram gives it the module tree, the
rule-based draft and a digest of the registry's patterns, reads back a map document, and checks it deterministically
against the model (every row's module exists) before returning it with any problems. venngram depends on no model
service.

.. warning:: This is an experimental API and may change in a future release.
"""
import json
import re
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from torch import nn

from venngram.maps.check import check_map_against_model, MapProblem
from venngram.maps.schema import (BUNDLED_KINDS, component_map_for, ComponentEntry, ComponentKey, ComponentMap,
                                  from_document, KindSpec, registered_architectures, VOCABULARY)
from venngram.sources import SourceError

#: The drafting-rules document schema this code reads.
DRAFTING_RULES_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Rule:
    """A module-name pattern (a full-match regex) and the component it names, optionally only beside a sibling."""

    match: str
    component: str
    when_sibling: str | None = None
    share: float | None = None
    """For a mined rule, the share of the maps' votes for this reading (its confidence); ``None`` for a declared one."""
    alternatives: tuple[str, ...] = ()
    """Other components maps have given the same name in the same scope."""

    def matches(self, name: str, siblings: set[str]) -> bool:
        return bool(re.fullmatch(self.match, name)) and (self.when_sibling is None or self.when_sibling in siblings)


@dataclass(frozen=True)
class ContainerRule:
    match: str
    component: str
    kind: str


@dataclass(frozen=True)
class DraftingRules:
    """One source's rules (see the module docstring)."""

    name: str
    containers: tuple[ContainerRule, ...] = ()
    leaves: Mapping[str, tuple[Rule, ...]] = field(default_factory=dict)
    kinds: Mapping[str, KindSpec] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, doc: Any, *, source: str = "<inline>") -> "DraftingRules":
        """Read a ``drafting_rules`` document, refusing a malformed one by name."""
        if not isinstance(doc, Mapping) or doc.get("kind") != "drafting_rules" \
                or doc.get("schema_version") != DRAFTING_RULES_SCHEMA_VERSION:
            raise SourceError(f"{source}: a drafting rules document is `kind: drafting_rules`, `schema_version: 1`.")
        try:
            containers = tuple(ContainerRule(str(c["match"]), str(c["component"]), str(c["kind"]))
                               for c in doc.get("containers") or ())
            leaves = {str(scope): tuple(Rule(str(r["match"]), str(r["component"]), r.get("when_sibling"),
                                             float(r["share"]) if r.get("share") is not None else None,
                                             tuple(str(a) for a in r.get("alternatives") or ()))
                                        for r in rules)
                      for scope, rules in (doc.get("leaves") or {}).items()}
            kinds = {str(k): KindSpec(**v) for k, v in (doc.get("kinds") or {}).items()}
            for pattern in [c.match for c in containers] + [r.match for rs in leaves.values() for r in rs]:
                re.compile(pattern)
        except (KeyError, TypeError, re.error) as e:
            raise SourceError(f"{source}: malformed drafting rule: {e}") from e
        return cls(str(doc.get("name") or source), containers, leaves, kinds)

    def to_mapping(self) -> dict[str, Any]:
        """The ``drafting_rules`` document (a mined rule keeps its share and alternatives)."""
        def leaf(r: Rule) -> dict[str, Any]:
            out: dict[str, Any] = {"match": r.match, "component": r.component}
            if r.when_sibling is not None:
                out["when_sibling"] = r.when_sibling
            if r.share is not None:
                out["share"] = r.share
            if r.alternatives:
                out["alternatives"] = list(r.alternatives)
            return out

        doc: dict[str, Any] = {"schema_version": DRAFTING_RULES_SCHEMA_VERSION, "kind": "drafting_rules",
                               "name": self.name}
        if self.containers:
            doc["containers"] = [{"match": c.match, "component": c.component, "kind": c.kind} for c in self.containers]
        if self.leaves:
            doc["leaves"] = {scope: [leaf(r) for r in rules] for scope, rules in self.leaves.items()}
        if self.kinds:
            doc["kinds"] = {k: {"output": v.output, "sublayer": v.sublayer} for k, v in self.kinds.items()}
        return doc


def load_drafting_rules(path: str | Path) -> DraftingRules:
    """Read a ``drafting_rules`` document from YAML (loading runs no code)."""
    with open(path, encoding="utf-8") as fh:
        return DraftingRules.from_mapping(yaml.safe_load(fh), source=str(path))


def bundled_drafting_rules() -> list[DraftingRules]:
    """The bundled rule documents (``transformer``, ``ssm``)."""
    root = resources.files("venngram.maps") / "data" / "drafting"
    return [DraftingRules.from_mapping(yaml.safe_load(p.read_text(encoding="utf-8")), source=p.name)
            for p in sorted(root.iterdir(), key=lambda p: p.name) if p.name.endswith(".yaml")]


def mine_rules(maps: Iterable[ComponentMap] | None = None) -> DraftingRules:
    """Drafting rules learned from component maps (by default the maps registered in this process; the registries
    publish the rules mined from all of theirs, :func:`venngram.registry.mined_rules`).

    Each primary-stack row teaches a leaf rule (its module's last name, within its container's component or the block)
    and each container row a container rule; rows outside the stack teach ``outer`` rules. Where maps disagree on one
    name in one scope, the majority's reading is the plain rule and each other reading becomes a conditional rule on a
    sibling only its maps have (sandwich norms are learned this way), when one exists.
    """
    maps = list(maps) if maps is not None else [component_map_for(a) for a in registered_architectures()]
    votes: dict[tuple[str, str], Counter] = defaultdict(Counter)
    siblings: dict[tuple[str, str, str], list[set[str]]] = defaultdict(list)
    containers: dict[str, tuple[str, str]] = {}
    for cmap in maps:
        rows = {k: v for k, v in cmap.components.items() if ComponentKey.parse(k, cmap.stacks).stack is None}
        block = rows.get("blocks.{i}")
        container_modules = {}
        for key, entry in rows.items():
            parsed = ComponentKey.parse(key, cmap.stacks)
            if parsed.in_block and parsed.component and "." not in parsed.component and entry.kind not in (
                    "linear", "norm", "embed", "unembed"):
                name = entry.module.rsplit(".", 1)[-1]
                containers.setdefault(name, (parsed.component, entry.kind))
                container_modules[parsed.component] = entry.module
        block_names = {e.module.rsplit(".", 1)[-1] for k, e in rows.items()
                       if block is not None and e.module.startswith(block.module + ".")}
        for key, entry in rows.items():
            parsed = ComponentKey.parse(key, cmap.stacks)
            if not parsed.component or (parsed.in_block and parsed.component in container_modules):
                continue
            leaf = entry.module.rsplit(".", 1)[-1]
            if not parsed.in_block:
                scope = "outer"
            else:
                head = parsed.component.split(".", 1)[0]
                scope = head if head in container_modules and entry.module.startswith(
                    container_modules[head] + ".") else "block"
            votes[(scope, leaf)][parsed.component] += 1
            siblings[(scope, leaf, parsed.component)].append(block_names)
    leaves: dict[str, list[Rule]] = defaultdict(list)
    for (scope, leaf), counter in votes.items():
        ranked = [c for c, _ in counter.most_common()]
        markers = {}
        for reading in ranked:
            mine = set.intersection(*siblings[(scope, leaf, reading)])
            others = set().union(*(s for c in ranked if c != reading for s in siblings[(scope, leaf, c)]))
            markers[reading] = sorted(mine - others)
        # a reading its maps mark with a sibling of their own is conditional on it; the plain rule is the one reading
        # without a marker (else the majority), and a marked reading never shadows it
        plain = [c for c in ranked if not markers[c]]
        default = plain[0] if len(plain) == 1 else ranked[0]
        total = sum(counter.values())

        def rule(c: str, marker: str | None = None) -> Rule:
            return Rule(re.escape(leaf), c, marker, counter[c] / total, tuple(o for o in ranked if o != c))

        leaves[scope] += [rule(c, markers[c][0]) for c in ranked if c != default and markers[c]]
        leaves[scope].append(rule(default))
    return DraftingRules("registry", tuple(ContainerRule(re.escape(n), c, k) for n, (c, k) in containers.items()),
                         {s: tuple(rs) for s, rs in leaves.items()})


def default_rules() -> list[DraftingRules]:
    """Rules mined from the maps registered in this process, then those the configured registries mined from theirs,
    then the bundled rule documents. A registry that cannot be read is skipped with a warning."""
    from venngram.registry import mined_rules, RegistryError

    try:
        published = mined_rules()
    except RegistryError as err:
        warnings.warn(f"drafting without registry-mined rules: {err}", stacklevel=2)
        published = []
    local = [mine_rules()] if registered_architectures() else []
    return [*local, *published, *bundled_drafting_rules()]


def _block_stack(model: nn.Module) -> tuple[str, nn.ModuleList]:
    stacks = [(path, m) for path, m in model.named_modules() if isinstance(m, nn.ModuleList) and len(m) >= 2
              and len({type(c) for c in m}) <= 2]  # a hybrid stack may alternate two block classes
    if not stacks:
        raise SourceError(f"{type(model).__name__}: no ModuleList of blocks to draft from.")
    return max(stacks, key=lambda s: (len(s[1]), -s[0].count(".")))


def _kind(module: nn.Module, component: str) -> str:
    if component == "unembed":
        return "unembed"
    name = type(module).__name__
    return "norm" if "Norm" in name else ("embed" if "Embedding" in name else "linear")


def _has_params(module: nn.Module) -> bool:
    return any(True for _ in module.parameters(recurse=False))


def draft_map(model: nn.Module, architecture: str | None = None,
              rules: Sequence[DraftingRules] | None = None) -> ComponentMap:
    """A draft component map from ``model``'s module tree, named by ``rules`` (default :func:`default_rules`)."""
    architecture = architecture or type(model).__name__
    rules = list(rules) if rules is not None else default_rules()
    blocks_path, blocks = _block_stack(model)
    rows: dict[str, ComponentEntry] = {"blocks.{i}": ComponentEntry(f"{blocks_path}.{{i}}", "block")}
    kinds = dict(BUNDLED_KINDS)
    unconfirmed: list[str] = []
    confidence: dict[str, float] = {}
    alternatives: dict[str, list[str]] = {}
    # every block's modules: a module in only some blocks is an optional row
    seen: dict[str, nn.Module] = {}
    count: Counter = Counter()
    for block in blocks:
        for rel, module in block.named_modules():
            if rel:
                seen.setdefault(rel, module)
                count[rel] += 1
    containers: dict[str, str] = {}  # relative path -> component
    taken: set[str] = set()
    for rel, module in seen.items():
        if "." in rel or not any(True for _ in module.children()):
            continue
        for rs in rules:
            hit = next((c for c in rs.containers if re.fullmatch(c.match, rel) and c.component not in taken), None)
            if hit:
                containers[rel] = hit.component
                taken.add(hit.component)
                kinds.update({k: v for k, v in rs.kinds.items() if k not in kinds})
                rows[f"blocks.{{i}}.{hit.component}"] = ComponentEntry(
                    f"{blocks_path}.{{i}}.{rel}", hit.kind, optional=count[rel] < len(blocks))
                break
    names_by_parent: dict[str, set[str]] = defaultdict(set)
    for rel in seen:
        parent, _, leaf = rel.rpartition(".")
        names_by_parent[parent].add(leaf)
    for rel, module in seen.items():
        if rel in containers or not _has_params(module):
            continue
        parent, _, leaf = rel.rpartition(".")
        top = rel.split(".", 1)[0]
        scope = containers.get(top, "block") if parent else "block"
        hit = _first(rules, scope, leaf, names_by_parent[parent], taken)
        component = hit.component if hit else f"draft.{rel}"
        _annotate(f"blocks.{{i}}.{component}", hit, confidence, alternatives, unconfirmed)
        taken.add(component)
        rows[f"blocks.{{i}}.{component}"] = ComponentEntry(f"{blocks_path}.{{i}}.{rel}", _kind(module, component),
                                                           optional=count[rel] < len(blocks))
    others = [p for p, m in model.named_modules() if isinstance(m, nn.ModuleList) and p != blocks_path
              and not p.startswith(f"{blocks_path}.")]
    outer_names = {p.rsplit(".", 1)[-1] for p, _ in model.named_modules() if p}
    for path, module in model.named_modules():
        if not path or path == blocks_path or path.startswith(f"{blocks_path}.") \
                or any(path == o or path.startswith(f"{o}.") for o in others) or not _has_params(module):
            continue
        hit = _first(rules, "outer", path.rsplit(".", 1)[-1], outer_names, taken)
        component = hit.component if hit else f"draft.{path}"
        _annotate(component, hit, confidence, alternatives, unconfirmed)
        taken.add(component)
        rows[component] = ComponentEntry(path, _kind(module, component))
    unmapped = sorted(o for o in others if not any(o.startswith(f"{p}.") for p in others if p != o))
    properties: dict[str, Any] = {"draft": True, "unconfirmed": unconfirmed, "confidence": confidence,
                                  "alternatives": alternatives, "drafting_rules": [r.name for r in rules]}
    if unmapped:
        properties["unmapped_stacks"] = unmapped
    return ComponentMap(architecture=architecture, components=rows, source="module_tree", kinds=kinds,
                        properties=properties)


def _annotate(row: str, hit: Rule | None, confidence: dict[str, float], alternatives: dict[str, list[str]],
              unconfirmed: list[str]) -> None:
    """Record how sure the draft is of ``row``: a mined rule's vote share (a declared rule's is 1), its alternatives,
    and no confidence at all for an unnamed module, which is unconfirmed."""
    if hit is None:
        unconfirmed.append(row)
        confidence[row] = 0.0
        return
    confidence[row] = hit.share if hit.share is not None else 1.0
    if hit.alternatives:
        alternatives[row] = list(hit.alternatives)


def _first(rules: Sequence[DraftingRules], scope: str, name: str, siblings: set[str], taken: set[str]) -> Rule | None:
    for rs in rules:
        for rule in rs.leaves.get(scope, ()):
            if rule.component not in taken and rule.matches(name, siblings):
                return rule
    return None


def map_document(cmap: ComponentMap) -> dict[str, Any]:
    """A component map as a schema 3 document (``from_document`` reads it back)."""
    declared = {k: {"output": v.output, "sublayer": v.sublayer} for k, v in cmap.kinds.items()
                if k not in BUNDLED_KINDS}
    doc: dict[str, Any] = {"schema_version": 3, "architecture": cmap.architecture}
    if declared:
        doc["kinds"] = declared
    doc["components"] = {k: {"module": e.module, "kind": e.kind, **({"optional": True} if e.optional else {})}
                         for k, e in cmap.components.items()}
    return doc


def registry_digest(rules: Sequence[DraftingRules] | None = None) -> dict[str, Any]:
    """A compact view of what the registry knows, for a model drafting a map: the vocabulary, and per component the
    module names maps have used for it."""
    rules = list(rules) if rules is not None else default_rules()
    names: dict[str, set[str]] = defaultdict(set)
    for rs in rules:
        for rule in [r for scoped in rs.leaves.values() for r in scoped]:
            names[rule.component].add(rule.match.replace("\\", ""))
        for c in rs.containers:
            names[c.component].add(c.match.replace("\\", ""))
    return {"vocabulary": list(VOCABULARY),
            "module_names_by_component": {k: sorted(v) for k, v in sorted(names.items())}}


_INSTRUCTIONS = """Draft a venngram component map (YAML, schema_version 3) for the model below.
- `components` maps a key to {module, kind}: `blocks.{i}` is the block stack, `blocks.{i}.<component>` a part of a
  block, other keys parts outside it. `{i}` stands for the layer index in module paths.
- Use the vocabulary's names where the part is that part; otherwise name it yourself (any lowercase dotted name).
- `kind` is one of block, attn, mlp, norm, linear, embed, unembed, or a kind you declare under `kinds`.
- A part present in only some blocks is `optional: true`. Every module path must exist in the tree.
- Start from the rule-based draft; rename its `draft.*` rows and fix any row it got wrong.
Return only the YAML document."""


def draft_prompt(model: nn.Module, draft: ComponentMap, digest: Mapping[str, Any]) -> str:
    """The prompt :func:`draft_with_model` sends: instructions, the registry digest, the module tree and the draft."""
    tree = [{"path": p, "class": type(m).__name__,
             "params": {n: list(t.shape) for n, t in m.named_parameters(recurse=False)}}
            for p, m in model.named_modules() if p and _has_params(m) and ".1." not in f".{p}."]
    return "\n\n".join([_INSTRUCTIONS, "Registry digest:\n" + json.dumps(digest),
                        "Module tree (block 0 and modules outside the stack):\n" + json.dumps(tree),
                        "Rule-based draft:\n" + yaml.safe_dump(map_document(draft), sort_keys=False)])


def draft_with_model(model: nn.Module, complete: Callable[[str], str], *, architecture: str | None = None,
                     rules: Sequence[DraftingRules] | None = None) -> tuple[ComponentMap, list[MapProblem]]:
    """Draft a map with a model's help, then check it deterministically.

    ``complete`` sends a prompt to whatever model or service the caller uses and returns its text; venngram builds the
    prompt (:func:`draft_prompt`) from the rule-based draft and :func:`registry_digest`, reads the reply as a map
    document, and checks every row's module against ``model``. Verification against the architecture
    (:func:`venngram.maps.verify`) is the next step, as for any map.

    Returns:
        The drafted map, and every problem the check found (empty when it describes the model).

    Raises:
        SourceError: If the reply is not a readable map document.
    """
    rules = list(rules) if rules is not None else default_rules()
    draft = draft_map(model, architecture, rules)
    reply = complete(draft_prompt(model, draft, registry_digest(rules)))
    text = re.sub(r"^```(?:yaml)?\s*|\s*```\s*$", "", reply.strip())
    try:
        cmap = from_document(yaml.safe_load(text), source="<model draft>")
    except (yaml.YAMLError, ValueError) as e:
        raise SourceError(f"the model's reply is not a readable component map: {e}") from e
    cmap = ComponentMap(cmap.architecture, cmap.components, {**cmap.properties, "draft": True, "drafted_by": "model"},
                        "model_assisted", cmap.kinds, cmap.stacks, cmap.schema_version)
    return cmap, check_map_against_model(cmap, model)


class ModuleTreeSource:
    """:func:`draft_map` as a :class:`~venngram.sources.MapSource`. It needs the model (a meta-device one will do)."""

    name = "module_tree"
    version = "2"
    unexpressed = ("named stacks other than the block stack",)

    def __init__(self, rules: Sequence[DraftingRules] | None = None) -> None:
        self._rules = list(rules) if rules is not None else None

    def architectures(self) -> tuple[str, ...]:
        return ()  # any: it reads whatever model it is given

    def derive(self, architecture: str, *, config: Any = None, model: Any = None) -> ComponentMap:
        if model is None:
            raise SourceError("module_tree: drafting needs the model (`model=`); a meta-device instance is enough.")
        return draft_map(model, architecture, self._rules)


__all__ = [
    "bundled_drafting_rules",
    "ContainerRule",
    "default_rules",
    "draft_map",
    "draft_prompt",
    "draft_with_model",
    "DraftingRules",
    "load_drafting_rules",
    "map_document",
    "mine_rules",
    "ModuleTreeSource",
    "registry_digest",
    "Rule",
]
