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
- **shapes**: widths a vocabulary component must have, read from the configuration: ``attn.q`` from the head count
  and head size, ``attn.k`` and ``attn.v`` from the key-value head count (so a swapped ``q`` and ``k`` fails under
  grouped-query attention), the MLP's from its intermediate size, norms one-dimensional;
- **order**: forward pre-hooks confirm the conventional order within a block, every fired component of a rule's
  earlier side starting before every one of its later side (attention norm, the q/k/v projections,
  ``attn.o``, the MLP norm, ``mlp.gate``/``mlp.in``, ``mlp.out``), which catches swapped labels that shapes cannot
  (``ln1`` for ``ln2``).

A rule applies only where the map names its component and the configuration states what it needs, so a map for an
architecture outside the vocabulary's conventions is verified as far as its rows allow and reported ``partial``, never
failed for what it does not claim. A report produces a verification record (:meth:`VerificationReport.record`): checks,
configuration fingerprint, venngram version and date, written only by this verifier.

Building a model from a configuration needs ``transformers`` (the ``hf`` extra); a model passed in directly does not.

.. warning:: This is an experimental API and may change in a future release.
"""
import argparse
import copy
import datetime
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import yaml
from torch import nn

from venngram.__about__ import __version__
from venngram.maps.check import check_map_against_model, MapProblem, stack_length, submodule
from venngram.maps.schema import component_map_for, ComponentMap, load_component_map, PRIMARY_STACK
from venngram.maps.structure import map_for, MapStructure

#: The schema of a verification record.
RECORD_SCHEMA_VERSION = 1

#: Configuration fields by meaning, under the names architectures spell them; the first one present wins.
CONFIG_FIELDS: dict[str, tuple[str, ...]] = {
    "hidden": ("hidden_size", "n_embd", "d_model"),
    "heads": ("num_attention_heads", "n_head", "num_heads"),
    "kv_heads": ("num_key_value_heads", "n_head_kv", "num_kv_heads"),
    "head_dim": ("head_dim",),
    "intermediate": ("intermediate_size", "ffn_dim", "n_inner", "d_ff"),
    "vocab": ("vocab_size",),
}

#: Pairs ``(earlier, later)``: within one block, a component of ``earlier`` starts before one of ``later`` (the
#: conventional order of a sequential block). A pair is checked only where both fired in the same block.
ORDER_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("ln1",), ("attn.q", "attn.k", "attn.v", "attn.qkv")),
    (("attn.q", "attn.k", "attn.v", "attn.qkv"), ("attn.o",)),
    (("attn.q",), ("attn.q_norm",)),
    (("attn.k",), ("attn.k_norm",)),
    (("attn.o",), ("ln1_post", "ln2")),
    (("ln1_post",), ("ln2",)),
    (("ln2",), ("mlp.gate", "mlp.in")),
    (("mlp.gate", "mlp.in"), ("mlp.out",)),
    (("mlp.out",), ("ln2_post",)),
)

_CONTAINER_KINDS = ("block", "attn", "mlp")


@dataclass(frozen=True)
class CheckResult:
    """One check's outcome: ``passed``, ``failed`` or ``skipped``, its problems, and what it covered or why it was
    skipped."""

    name: str
    status: str
    problems: tuple[MapProblem, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class VerificationReport:
    """Every check's result for one map against one architecture."""

    architecture: str
    checks: tuple[CheckResult, ...]
    config: dict[str, Any] = field(default_factory=dict)
    """Where the configuration came from (``source``) and its fingerprint (``sha256``)."""

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
            "checks": {c.name: {"status": c.status, "note": c.note, "problems": [str(p) for p in c.problems]}
                       for c in self.checks},
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


def config_value(config: Any, meaning: str) -> int | None:
    """The configuration's value for a :data:`CONFIG_FIELDS` meaning, or ``None`` if it states none."""
    for name in CONFIG_FIELDS[meaning]:
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _dims(module: Any) -> tuple[int, int] | None:
    """A projection's ``(in, out)`` widths: ``nn.Linear``'s, or a transposed ``Conv1D``'s (GPT-2's)."""
    if isinstance(getattr(module, "in_features", None), int) and isinstance(getattr(module, "out_features", None), int):
        return module.in_features, module.out_features
    weight = getattr(module, "weight", None)
    if isinstance(getattr(module, "nf", None), int) and weight is not None and weight.dim() == 2:
        return int(weight.shape[0]), int(weight.shape[1])
    return None


def _stack_config(config: Any, stack: str) -> Any | None:
    if stack == PRIMARY_STACK:
        return getattr(config, "text_config", None) or config
    return getattr(config, f"{stack}_config", None)


def _shape_rules(cfg: Any) -> dict[str, list[tuple[str, int]]]:
    """Component -> the widths it must have (``"in"``, ``"out"``, ``"numel"``), from what ``cfg`` states."""
    hidden, heads = config_value(cfg, "hidden"), config_value(cfg, "heads")
    kv = config_value(cfg, "kv_heads") or heads
    head_dim = config_value(cfg, "head_dim") or (hidden // heads if hidden and heads else None)
    inter, rules = config_value(cfg, "intermediate"), {}
    if heads and head_dim:
        rules["attn.q"] = [("out", heads * head_dim)]
        rules["attn.k"] = rules["attn.v"] = [("out", kv * head_dim)]  # type: ignore[operator]
        rules["attn.qkv"] = [("out", (heads + 2 * kv) * head_dim)]  # type: ignore[operator]
        rules["attn.o"] = [("in", heads * head_dim)] + ([("out", hidden)] if hidden else [])
        rules["attn.q_norm"] = rules["attn.k_norm"] = [("numel", head_dim)]
    if inter:
        rules["mlp.gate"] = rules["mlp.in"] = [("out", inter)] + ([("in", hidden)] if hidden else [])
        rules["mlp.out"] = [("in", inter)] + ([("out", hidden)] if hidden else [])
    if hidden:
        for norm in ("ln1", "ln2", "ln1_post", "ln2_post", "ln_final"):
            rules[norm] = [("numel", hidden)]
    return rules


def check_shapes(cmap: ComponentMap, model: nn.Module, config: Any) -> CheckResult:
    """The **shapes** check (see the module docstring)."""
    if config is None:
        return CheckResult("shapes", "skipped", note="no configuration to read widths from")
    problems: list[MapProblem] = []
    checked = 0
    for stack in cmap.stack_names():
        cfg, template = _stack_config(config, stack), cmap.block_template(stack)
        if cfg is None:
            continue
        rules = _shape_rules(cfg)
        depth = stack_length(model, template) if template else 0
        for key, entry in cmap.components.items():
            k = cmap.key(key)
            if (k.stack or PRIMARY_STACK) != stack or k.component not in rules:
                continue
            layers = [None] if not k.in_block else [i for i in range(depth)
                                                     if entry.layers is None or entry.layers.covers(i)]
            for layer in layers:
                module = submodule(model, entry.module if layer is None else entry.module.replace("{i}", str(layer)))
                if module is None:
                    continue  # the exists check names it
                checked += 1
                problems += _shape_problems(key, layer, entry.kind, module, rules[k.component])
        for key in ("embed", "unembed"):
            entry, vocab = cmap.components.get(key), config_value(cfg, "vocab")
            module = submodule(model, entry.module) if entry is not None and stack == PRIMARY_STACK else None
            weight = getattr(module, "weight", None)
            if weight is not None and vocab and weight.dim() == 2:
                checked += 1
                if int(weight.shape[0]) != vocab:
                    problems.append(MapProblem(key, None, f"has {int(weight.shape[0])} rows, the vocabulary is "
                                               f"{vocab}"))
    if not checked:
        return CheckResult("shapes", "skipped", note="no row the shape rules cover")
    return CheckResult("shapes", "failed" if problems else "passed", tuple(problems), f"{checked} modules checked")


def _shape_problems(key: str, layer: int | None, kind: str, module: Any, rules: list[tuple[str, int]]):
    weight = getattr(module, "weight", None)
    if kind == "norm" and weight is not None and weight.dim() != 1:
        yield MapProblem(key, layer, f"is a norm, but its weight is {weight.dim()}-dimensional")
    dims = _dims(module)
    for what, expected in rules:
        if what == "numel":
            actual = None if weight is None else weight.numel()
        else:
            actual = None if dims is None else dims[0 if what == "in" else 1]
        if actual is not None and actual != expected:
            yield MapProblem(key, layer, f"{what} width is {actual}, the configuration says {expected}")


def check_partition(cmap: ComponentMap, model: nn.Module) -> CheckResult:
    """The **partition** check (see the module docstring)."""
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


def check_order(cmap: ComponentMap, model: nn.Module, run: Callable[[nn.Module], Any],
                rules: Sequence[tuple[tuple[str, ...], tuple[str, ...]]] = ORDER_RULES) -> CheckResult:
    """The **order** check: ``run(model)`` performs one forward while pre-hooks record when each mapped component of
    every block first starts; each rule is checked where both its sides fired in the same block."""
    structure = MapStructure(model, cmap)
    started: dict[tuple[str | None, int], dict[str, int]] = {}
    handles = []
    tick = iter(range(1 << 30))
    for path, loc in structure.placed().items():
        if loc.layer is None or not loc.component or cmap.kind_of(loc.component, loc.layer, loc.stack) in (
                None, *_CONTAINER_KINDS):
            continue
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
    problems, compared = [], 0
    for (stack, layer), times in sorted(started.items(), key=lambda kv: (kv[0][0] or "", kv[0][1])):
        for earlier, later in rules:
            a = [c for c in earlier if c in times]
            b = [c for c in later if c in times]
            if a and b:
                compared += 1
                last_a, first_b = max(a, key=times.__getitem__), min(b, key=times.__getitem__)
                if times[last_a] > times[first_b]:
                    where = f"{stack}.blocks.{{i}}" if stack else "blocks.{i}"
                    problems.append(MapProblem(f"{where}.{last_a}", layer, f"started after {first_b!r}; "
                                               f"{'/'.join(earlier)} come before {'/'.join(later)} in a block"))
    if not compared:
        return CheckResult("order", "skipped", note="no order rule had both sides fire in one block")
    return CheckResult("order", "failed" if problems else "passed", tuple(problems),
                       f"{compared} ordered pairs over {len(started)} blocks")


def tiny_config(config: Any, layers: int = 4) -> Any:
    """A copy of ``config`` small enough to run on a CPU: four heads of size 8 (two key-value heads where the original
    groups its queries), an intermediate size of 64, and at most ``layers`` layers. Nested ``*_config`` sections are
    shrunk too; a field the configuration does not have is left alone."""
    config = copy.deepcopy(config)
    for section in [config, *(v for k, v in vars(config).items() if k.endswith("_config") and v is not None)]:
        heads, kv = config_value(section, "heads"), config_value(section, "kv_heads")
        values = {"heads": 4, "kv_heads": 2 if heads and kv and kv < heads else 4, "head_dim": 8, "hidden": 32,
                  "intermediate": 64}
        for meaning, value in values.items():
            for name in CONFIG_FIELDS[meaning]:
                if isinstance(getattr(section, name, None), int):
                    setattr(section, name, value)
        for name in ("num_hidden_layers", "n_layer", "num_layers"):
            n = getattr(section, name, None)
            if isinstance(n, int) and n > layers:
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
    """A forward of four tokens, for a model whose configuration states a vocabulary."""
    config = getattr(model, "config", None)
    cfg = getattr(config, "text_config", None) or config
    vocab = config_value(cfg, "vocab") if cfg is not None else None
    if not vocab:
        raise ValueError("the model states no vocabulary size; pass `inputs`")
    model(input_ids=torch.randint(0, min(vocab, 32), (1, 4)))


def _fingerprint(config: Any) -> str | None:
    to_dict = getattr(config, "to_dict", None)
    if to_dict is None:
        return None
    return hashlib.sha256(json.dumps(to_dict(), sort_keys=True, default=str).encode()).hexdigest()


def verify(target: Any, cmap: ComponentMap | None = None, *, architecture: str | None = None,
           inputs: Callable[[nn.Module], Any] | None = None, order: bool = True) -> VerificationReport:
    """Verify a component map against an architecture.

    Args:
        target: A Hugging Face model id (only its configuration is fetched), a ``transformers`` configuration, or a
            model. A model is checked as it is; a configuration is built on the meta device.
        cmap: The map; looked up from ``architecture`` or the target (:func:`~venngram.maps.map_for`) when ``None``.
        architecture: The architecture class name, when the target does not state it.
        inputs: Runs one forward of a model for the order check (``inputs(model)``); the default feeds four tokens.
            The order check runs on ``target`` itself when it is a model and ``inputs`` is given, and otherwise on a
            small random-init model built from the configuration (:func:`tiny_config`).
        order: Run the order check.

    Returns:
        The report; :attr:`VerificationReport.passed` is false when any check failed.
    """
    model, config, source = None, None, "module"
    if isinstance(target, str):
        from transformers import AutoConfig  # an optional dependency

        config, source = AutoConfig.from_pretrained(target), f"hf:{target}"
    elif isinstance(target, nn.Module):
        model, config = target, getattr(target, "config", None)
    else:
        config, source = target, "config"
    if architecture is None and model is None:
        architecture = cmap.architecture if cmap is not None else (getattr(config, "architectures", None) or [None])[0]
    if cmap is None:
        cmap = map_for(model, architecture) if model is not None else component_map_for(str(architecture))
    arch = architecture or cmap.architecture
    if model is None:
        model = _build(config, arch, "meta")
    checks = [CheckResult("exists", *_status(check_map_against_model(cmap, model))), check_partition(cmap, model),
              check_shapes(cmap, model, config)]
    if order:
        checks.append(_order(cmap, target, model, config, arch, inputs))
    return VerificationReport(cmap.architecture, tuple(checks), {"source": source, "sha256": _fingerprint(config)})


def _status(problems: list[MapProblem]) -> tuple[str, tuple[MapProblem, ...]]:
    return ("failed" if problems else "passed"), tuple(problems)


def _order(cmap: ComponentMap, target: Any, model: nn.Module, config: Any, arch: str,
           inputs: Callable[[nn.Module], Any] | None) -> CheckResult:
    if isinstance(target, nn.Module) and inputs is not None:
        return check_order(cmap, target, inputs)
    if config is None:
        return CheckResult("order", "skipped", note="no configuration to build a small model from; pass `inputs`")
    try:
        small = _build(tiny_config(config), arch, "cpu").eval()
    except Exception as err:
        return CheckResult("order", "skipped", note=f"could not build a small model: {type(err).__name__}: {err}"[:300])
    return check_order(cmap, small, inputs or default_inputs)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m venngram.maps.verification <model id> [--map FILE] [--record FILE]``: exit 1 when a check fails,
    2 when there is no map to verify."""
    parser = argparse.ArgumentParser(prog="python -m venngram.maps.verification", description=main.__doc__)
    parser.add_argument("model", help="a Hugging Face model id (only its configuration is fetched)")
    parser.add_argument("--map", help="a component map document (default: the registered or bundled one)")
    parser.add_argument("--architecture", help="the architecture class name, when the configuration lacks it")
    parser.add_argument("--record", help="write the verification record to this YAML file")
    parser.add_argument("--no-order", action="store_true", help="skip the forward-order check")
    args = parser.parse_args(argv)
    cmap = load_component_map(args.map) if args.map else None
    try:
        report = verify(args.model, cmap, architecture=args.architecture, order=not args.no_order)
    except KeyError as err:
        print(f"{args.model}: {err.args[0]}", file=sys.stderr)
        return 2
    print(report)
    if args.record:
        with open(args.record, "w", encoding="utf-8") as fh:
            yaml.safe_dump(report.record(), fh, sort_keys=False)
    return 0 if report.passed else 1


__all__ = [
    "check_order",
    "check_partition",
    "check_shapes",
    "CheckResult",
    "CONFIG_FIELDS",
    "config_value",
    "default_inputs",
    "ORDER_RULES",
    "RECORD_SCHEMA_VERSION",
    "tiny_config",
    "verify",
    "VerificationReport",
]

if __name__ == "__main__":
    sys.exit(main())
