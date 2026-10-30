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
Verifying a Map
^^^^^^^^^^^^^^^

Deterministic checks that a component map describes an architecture, run without downloading weights: a meta-device
model built from the configuration carries every check but the last, which runs a forward on a small random-init model
of the same architecture.

- **exists**: every row's module exists for every layer it covers
  (:func:`~venngram.maps.check.check_map_against_model`);
- **partition**: every parameter is located at a component, and no two rows name the same module. A coarse map passes
  this trivially, which is why it is not the only check;
- **shapes**: the widths each row's module must have, as expressions over meanings read from the configuration (under
  grouped-query attention ``attn.q`` and ``attn.k`` differ, so a swap fails);
- **order**: forward pre-hooks on the modules no other mapped module sits under record when each component starts,
  and every order chain is checked where its components fired, which catches swapped labels that widths cannot
  (``ln1`` for ``ln2``).

What the shapes and order checks expect is data, not code: a rule set (:mod:`~venngram.maps.rules`), layered from the
bundled ``core`` set, the map's own ``verification:`` section and any the caller passes. So a map for an architecture
nobody anticipated states its own widths and orders, or removes a core rule that does not hold for it. A rule applies
only where the map names its component and the configuration states what it needs; a map is verified as far as its
rules reach and reported ``partial``, never failed for what nothing claims.

A report produces a verification record (:meth:`VerificationReport.record`): every check, the rule sets applied, the
order observed in the first block of each stack that ran, the configuration's fingerprint, venngram's version and the
date. The observed order is evidence a later run is compared against without any rule (``verify(..., against=record)``,
the **against_record** check).

Building a model from a configuration needs ``transformers`` (the ``hf`` extra); a model passed in directly does not.

.. warning:: This is an experimental API and may change in a future release.
"""
import argparse
import copy
import datetime
import hashlib
import json
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import yaml
from torch import nn

from venngram.__about__ import __version__
from venngram.maps.check import check_map_against_model, MapProblem, stack_length, submodule
from venngram.maps.rules import config_field, core_rules, evaluate, layered, RuleSet, Unstated
from venngram.maps.schema import component_map_for, ComponentMap, load_component_map, PRIMARY_STACK
from venngram.maps.structure import map_for, MapStructure
from venngram.resolve import component_matches

#: The schema of a verification record.
RECORD_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CheckResult:
    """One check's outcome: ``passed``, ``failed`` or ``skipped``, its problems, and what it covered or why it was
    skipped."""

    name: str
    status: str
    problems: tuple[MapProblem, ...] = ()
    note: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    """Evidence beyond pass or fail (the order check's observed order)."""


@dataclass(frozen=True)
class VerificationReport:
    """Every check's result for one map against one architecture."""

    architecture: str
    checks: tuple[CheckResult, ...]
    config: dict[str, Any] = field(default_factory=dict)
    """Where the configuration came from (``source``) and its fingerprint (``sha256``)."""
    rules: tuple[str, ...] = ()
    """The rule sets applied, in layering order."""

    @property
    def status(self) -> str:
        """``failed`` if any check failed, ``partial`` if any was skipped, else ``verified``."""
        statuses = {c.status for c in self.checks}
        return "failed" if "failed" in statuses else "partial" if "skipped" in statuses else "verified"

    @property
    def passed(self) -> bool:
        """Whether no check failed."""
        return self.status != "failed"

    def check(self, name: str) -> CheckResult:
        """The result of the check ``name``."""
        return next(c for c in self.checks if c.name == name)

    def problems(self) -> list[MapProblem]:
        """Every problem of every check."""
        return [p for c in self.checks for p in c.problems]

    def record(self) -> dict[str, Any]:
        """The verification record: a plain mapping, safe to write as YAML or JSON."""
        return {
            "schema_version": RECORD_SCHEMA_VERSION,
            "architecture": self.architecture,
            "status": self.status,
            "checks": {c.name: {"status": c.status, "note": c.note, "problems": [str(p) for p in c.problems],
                                **c.detail} for c in self.checks},
            "rules": list(self.rules),
            "config": dict(self.config),
            "venngram_version": __version__,
            "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    def __str__(self) -> str:
        lines = [f"{self.architecture}: {self.status}"]
        for c in self.checks:
            lines.append(f"  {c.name}: {c.status}" + (f" ({c.note})" if c.note else ""))
            lines += [f"    {p}" for p in c.problems]
        return "\n".join(lines)


def rules_for(cmap: ComponentMap, rules: Iterable[Any] = (), *, core: bool = True) -> RuleSet:
    """The rule set verifying ``cmap``: the core set (unless ``core`` is false), the map's own ``verification:``
    section, then ``rules`` (rule sets, documents or paths), each layered over the last."""
    return layered(core_rules() if core else None, cmap.verification, *rules)


def _stack_config(config: Any, stack: str) -> Any | None:
    if stack == PRIMARY_STACK:
        return getattr(config, "text_config", None) or config
    return getattr(config, f"{stack}_config", None)


def _measure(module: Any, what: str) -> int | None:
    """A module's width: ``in``/``out`` of a projection (``nn.Linear``'s, or a transposed ``Conv1D``'s, as GPT-2's), or
    its weight's ``rows``, ``numel`` or ``ndim``; ``None`` when the module has no such width."""
    weight = getattr(module, "weight", None)
    if what in ("in", "out"):
        if isinstance(getattr(module, "in_features", None), int) and isinstance(getattr(module, "out_features", None),
                                                                               int):
            return module.in_features if what == "in" else module.out_features
        if isinstance(getattr(module, "nf", None), int) and weight is not None and weight.dim() == 2:
            return int(weight.shape[0 if what == "in" else 1])
        return None
    if not isinstance(weight, torch.Tensor):
        return None
    if what == "rows":
        return int(weight.shape[0]) if weight.dim() >= 1 else None
    return weight.numel() if what == "numel" else weight.dim()


def check_shapes(cmap: ComponentMap, model: nn.Module, config: Any, rules: RuleSet | None = None) -> CheckResult:
    """The **shapes** check: every width ``rules`` expects of a row's module, wherever the configuration states it."""
    rules = rules if rules is not None else rules_for(cmap)
    if config is None:
        return CheckResult("shapes", "skipped", note="no configuration to read widths from")
    problems: list[MapProblem] = []
    checked = unstated = 0
    for stack in cmap.stack_names():
        cfg, template = _stack_config(config, stack), cmap.block_template(stack)
        if cfg is None:
            continue
        value, depth = rules.resolver(cfg), stack_length(model, template) if template else 0
        for key, entry in cmap.components.items():
            k = cmap.key(key)
            spec = rules.widths_for(k.component, entry.kind) if (k.stack or PRIMARY_STACK) == stack else {}
            layers = [None] if not k.in_block else [i for i in range(depth)
                                                     if entry.layers is None or entry.layers.covers(i)]
            for layer in layers if spec else ():
                module = submodule(model, entry.module if layer is None else entry.module.replace("{i}", str(layer)))
                for what, expr in spec.items() if module is not None else ():  # the exists check names a missing one
                    try:
                        expected = evaluate(expr, value)
                    except Unstated:
                        unstated += 1
                        continue
                    actual = _measure(module, what)
                    if actual is None:
                        continue
                    checked += 1
                    if actual != expected:
                        problems.append(MapProblem(key, layer, f"{what} is {actual}, the rules expect {expected} "
                                                   f"({expr})"))
    note = f"{checked} widths checked" + (f", {unstated} not stated by the configuration" if unstated else "")
    if not checked:
        return CheckResult("shapes", "skipped", note=f"no width the rules expect could be measured; {note}")
    return CheckResult("shapes", "failed" if problems else "passed", tuple(problems), note)


def check_partition(cmap: ComponentMap, model: nn.Module) -> CheckResult:
    """The **partition** check: every parameter is located at a component, and no two rows name one module."""
    structure = MapStructure(model, cmap)
    if structure.refusal is not None:
        return CheckResult("partition", "failed", (MapProblem("map", None, structure.refusal.reason),))
    problems = [MapProblem(name, structure.locate(name).layer, "is located at no component")
                for name, _ in model.named_parameters() if structure.locate(name).component is None]
    owners: dict[str, str] = {}
    for key, entry in cmap.components.items():
        k = cmap.key(key)
        depth = structure.depth(k.stack) if k.in_block else None
        for layer in (range(depth) if isinstance(depth, int) else ()) if k.in_block else (None,):
            if entry.layers is not None and layer is not None and not entry.layers.covers(layer):
                continue
            path = entry.module if layer is None else entry.module.replace("{i}", str(layer))
            if path in owners:
                problems.append(MapProblem(key, layer, f"names {path!r}, which {owners[path]!r} already names"))
            owners[path] = key
    total = sum(1 for _ in model.parameters())
    return CheckResult("partition", "failed" if problems else "passed", tuple(problems), f"{total} parameters")


def _leaves(placed: dict[str, Any]) -> dict[str, Any]:
    """The placed modules inside a block that no other placed module sits under (a container's pre-hook fires before
    its children, so only leaves say when a component's own computation starts)."""
    containers = {p for q in placed for p in placed if p != q and q.startswith(f"{p}.")}
    return {p: loc for p, loc in placed.items() if p not in containers and loc.layer is not None and loc.component}


def check_order(cmap: ComponentMap, model: nn.Module, run: Callable[[nn.Module], Any],
                rules: RuleSet | None = None) -> CheckResult:
    """The **order** check: ``run(model)`` performs one forward while pre-hooks record when each mapped leaf component
    of every block first starts; each consecutive pair of groups in each order chain is checked where both fired in
    the same block."""
    rules = rules if rules is not None else rules_for(cmap)
    started: dict[tuple[str | None, int], dict[str, int]] = {}
    handles = []
    tick = iter(range(1 << 30))
    for path, loc in _leaves(MapStructure(model, cmap).placed()).items():
        module = submodule(model, path)
        if not isinstance(module, nn.Module):
            continue
        block, component = (loc.stack, int(loc.layer)), str(loc.component)

        def hook(_m, _a, block=block, component=component):
            started.setdefault(block, {}).setdefault(component, next(tick))
        handles.append(module.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            run(model)
    except Exception as err:  # a model the default inputs cannot drive is a skipped check, not a failed map
        return CheckResult("order", "skipped", note=f"the forward failed: {type(err).__name__}: {err}"[:300])
    finally:
        for h in handles:
            h.remove()
    observed: dict[str, list[str]] = {}
    for (stack, _layer), times in sorted(started.items(), key=lambda kv: (kv[0][0] or "", kv[0][1])):
        observed.setdefault(stack or PRIMARY_STACK, sorted(times, key=times.__getitem__))
    problems, compared = [], 0
    for (stack, layer), times in sorted(started.items(), key=lambda kv: (kv[0][0] or "", kv[0][1])):
        for name, chain in rules.order.items():
            for earlier, later in zip(chain, chain[1:]):
                a = [c for c in times if any(component_matches(p, c) for p in earlier)]
                b = [c for c in times if any(component_matches(p, c) for p in later)]
                if not (a and b):
                    continue
                compared += 1
                last_a, first_b = max(a, key=times.__getitem__), min(b, key=times.__getitem__)
                if times[last_a] > times[first_b]:
                    where = f"{stack}.blocks.{{i}}" if stack else "blocks.{i}"
                    problems.append(MapProblem(f"{where}.{last_a}", layer, f"started after {first_b!r}, against "
                                               f"order chain {name!r} ({'/'.join(earlier)} before {'/'.join(later)})"))
    detail = {"observed_order": observed}
    if not compared:
        return CheckResult("order", "skipped", note="no order chain had both sides fire in one block", detail=detail)
    return CheckResult("order", "failed" if problems else "passed", tuple(problems),
                       f"{compared} ordered pairs over {len(started)} blocks", detail)


#: The small sizes :func:`tiny_config` sets, by meaning; the key-value head count keeps grouped queries grouped.
TINY: dict[str, int] = {"heads": 4, "head_dim": 8, "hidden": 32, "intermediate": 64}


def tiny_config(config: Any, layers: int = 4, rules: RuleSet | None = None) -> Any:
    """A copy of ``config`` small enough to run on a CPU, for the order check: the configuration fields ``rules`` (the
    core set by default) names for each :data:`TINY` meaning are set to its size, the key-value head count to 2 where
    the original groups its queries (4 otherwise), and the layer count to at most ``layers``. Nested ``*_config``
    sections are shrunk too; a field the configuration does not have is left alone."""
    rules = rules if rules is not None else core_rules()
    config = copy.deepcopy(config)
    for section in [config, *(v for k, v in vars(config).items() if k.endswith("_config") and v is not None)]:
        value = rules.resolver(section)
        try:
            grouped = value("kv_heads") < value("heads")
        except Unstated:
            grouped = False
        sizes = {**TINY, "kv_heads": 2 if grouped else TINY["heads"]}
        for meaning, size in sizes.items():
            for source in rules.config.get(meaning, ()):
                if isinstance(source, str) and config_field(section, source) is not None:
                    setattr(section, source, size)
        for name in ("num_hidden_layers", "n_layer", "num_layers"):
            n = config_field(section, name)
            if n is not None and n > layers:
                setattr(section, name, layers)
                if isinstance(getattr(section, "layer_types", None), list):
                    section.layer_types = section.layer_types[:layers]
    return config


def _build(config: Any, architecture: str, device: str) -> nn.Module:
    import transformers  # an optional dependency: building from a configuration needs it

    cls = getattr(transformers, architecture, None)
    if cls is None:
        raise KeyError(f"transformers has no class {architecture!r} to build the model from")
    with torch.device(device):
        return cls(config)


def default_inputs(model: nn.Module) -> None:
    """A forward of four tokens, for a model whose configuration states a vocabulary size."""
    config = getattr(model, "config", None)
    cfg = getattr(config, "text_config", None) or config
    vocab = config_field(cfg, "vocab_size") if cfg is not None else None
    if not vocab:
        raise ValueError("the model states no vocabulary size; pass `inputs`")
    model(input_ids=torch.randint(0, min(vocab, 32), (1, 4)))


def check_against(report_order: CheckResult, record: dict[str, Any], architecture: str) -> CheckResult:
    """The **against_record** check: the order this run observed equals the one a stored verification record observed
    (a later checkpoint or library release that reorders a block is caught without any order rule)."""
    if record.get("architecture") != architecture:
        return CheckResult("against_record", "failed", (MapProblem("record", None, f"is for "
                           f"{record.get('architecture')!r}, not {architecture!r}"),))
    expected = ((record.get("checks") or {}).get("order") or {}).get("observed_order")
    observed = report_order.detail.get("observed_order")
    if not expected or not observed:
        return CheckResult("against_record", "skipped", note="the record or this run observed no order")
    problems = tuple(MapProblem(f"stack {stack!r}", None, f"observed {observed.get(stack)}, the record observed "
                                f"{expected.get(stack)}")
                     for stack in sorted({*expected, *observed}) if expected.get(stack) != observed.get(stack))
    return CheckResult("against_record", "failed" if problems else "passed", problems,
                       f"record of {record.get('date', '?')}")


def _fingerprint(config: Any) -> str | None:
    to_dict = getattr(config, "to_dict", None)
    if to_dict is None:
        return None
    return hashlib.sha256(json.dumps(to_dict(), sort_keys=True, default=str).encode()).hexdigest()


def verify(target: Any, cmap: ComponentMap | None = None, *, architecture: str | None = None, config: Any = None,
           rules: Iterable[Any] = (), core: bool = True, inputs: Callable[[nn.Module], Any] | None = None,
           order: bool = True, against: dict[str, Any] | None = None) -> VerificationReport:
    """Verify a component map against an architecture.

    Args:
        target: A Hugging Face model id (only its configuration is fetched), a ``transformers`` configuration, or a
            model. A model is checked as it is; a configuration is built on the meta device.
        cmap: The map; looked up from ``architecture`` or the target (:func:`~venngram.maps.map_for`) when ``None``.
        architecture: The architecture class name, when the target does not state it.
        config: The configuration widths are read from, for a model without a ``config`` of its own (an object or a
            mapping).
        rules: Further rule sets (:class:`~venngram.maps.rules.RuleSet`, documents or paths), layered over the core set
            and the map's own section.
        core: Start from the bundled core rule set; ``False`` verifies against the map's and ``rules`` alone.
        inputs: Runs one forward of a model for the order check (``inputs(model)``); the default feeds four tokens.
            The order check runs on ``target`` itself when it is a model and ``inputs`` is given, and otherwise on a
            small random-init model built from the configuration (:func:`tiny_config`).
        order: Run the order check.
        against: A stored verification record (:meth:`VerificationReport.record`) whose observed order this run's must
            equal (the ``against_record`` check; needs ``order``).

    Returns:
        The report; :attr:`VerificationReport.passed` is false when any check failed.
    """
    model, source = None, "module"
    if isinstance(target, str):
        from transformers import AutoConfig  # an optional dependency

        config, source = AutoConfig.from_pretrained(target), f"hf:{target}"
    elif isinstance(target, nn.Module):
        model, config = target, config if config is not None else getattr(target, "config", None)
    else:
        config, source = target, "config"
    if architecture is None and model is None:
        architecture = cmap.architecture if cmap is not None else (getattr(config, "architectures", None) or [None])[0]
    if cmap is None:
        cmap = map_for(model, architecture) if model is not None else component_map_for(str(architecture))
    arch = architecture or cmap.architecture
    ruleset = rules_for(cmap, rules, core=core)
    if model is None:
        model = _build(config, arch, "meta")
    checks = [CheckResult("exists", *_status(check_map_against_model(cmap, model))), check_partition(cmap, model),
              check_shapes(cmap, model, config, ruleset)]
    if order:
        checks.append(_order(cmap, target, config, arch, inputs, ruleset))
        if against is not None:
            checks.append(check_against(checks[-1], against, cmap.architecture))
    return VerificationReport(cmap.architecture, tuple(checks), {"source": source, "sha256": _fingerprint(config)},
                              ruleset.sources)


def _status(problems: list[MapProblem]) -> tuple[str, tuple[MapProblem, ...]]:
    return ("failed" if problems else "passed"), tuple(problems)


def _order(cmap: ComponentMap, target: Any, config: Any, arch: str, inputs: Callable[[nn.Module], Any] | None,
           rules: RuleSet) -> CheckResult:
    if isinstance(target, nn.Module) and inputs is not None:
        return check_order(cmap, target, inputs, rules)
    if config is None or not hasattr(config, "to_dict"):
        return CheckResult("order", "skipped", note="no configuration to build a small model from; pass `inputs`")
    try:
        small = _build(tiny_config(config, rules=rules), arch, "cpu").eval()
    except Exception as err:
        return CheckResult("order", "skipped", note=f"could not build a small model: {type(err).__name__}: {err}"[:300])
    return check_order(cmap, small, inputs or default_inputs, rules)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m venngram.maps.verification <model id> [--map FILE] [--rules FILE ...] [--record FILE]
    [--against RECORD]``: exit 1 when a check fails, 2 when there is no map to verify."""
    parser = argparse.ArgumentParser(prog="python -m venngram.maps.verification", description=main.__doc__)
    parser.add_argument("model", help="a Hugging Face model id (only its configuration is fetched)")
    parser.add_argument("--map", help="a component map document (default: the registered or bundled one)")
    parser.add_argument("--architecture", help="the architecture class name, when the configuration lacks it")
    parser.add_argument("--rules", action="append", default=[], help="a rule set document to layer on (repeatable)")
    parser.add_argument("--no-core", action="store_true", help="do not start from the bundled core rule set")
    parser.add_argument("--record", help="write the verification record to this YAML file")
    parser.add_argument("--no-order", action="store_true", help="skip the forward-order check")
    parser.add_argument("--against", help="a stored verification record whose observed order this run's must equal")
    args = parser.parse_args(argv)
    cmap = load_component_map(args.map) if args.map else None
    against = None
    if args.against:
        with open(args.against, encoding="utf-8") as fh:
            against = yaml.safe_load(fh)
    try:
        report = verify(args.model, cmap, architecture=args.architecture, rules=args.rules, core=not args.no_core,
                        order=not args.no_order, against=against)
    except KeyError as err:
        print(f"{args.model}: {err.args[0]}", file=sys.stderr)
        return 2
    print(report)
    if args.record:
        with open(args.record, "w", encoding="utf-8") as fh:
            yaml.safe_dump(report.record(), fh, sort_keys=False)
    return 0 if report.passed else 1


__all__ = [
    "check_against",
    "check_order",
    "check_partition",
    "check_shapes",
    "CheckResult",
    "default_inputs",
    "RECORD_SCHEMA_VERSION",
    "rules_for",
    "TINY",
    "tiny_config",
    "verify",
    "VerificationReport",
]

if __name__ == "__main__":
    sys.exit(main())
