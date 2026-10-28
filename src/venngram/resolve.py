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
Resolution
^^^^^^^^^^

Resolving an address is the only place where knowledge of an architecture or a backend enters, and it enters only
through providers:

- a :class:`StructureProvider` places each tensor in a block stack, at a layer, as a component. The physical default
  (:class:`~venngram.physical.PhysicalStructure`) finds the block list of any PyTorch model; a provider backed by an
  activation-point vocabulary (interpretune supplies one) also reports architecture-neutral components;
- a site resolver turns an activation point at a layer into a hookable position in a given backend, and a latent
  resolver turns a ``features`` or ``subspace`` address into a loaded latent model or basis. Both are a consumer's
  (interpretune's), so this version resolves only ``param`` targets and refuses the others, naming what is missing.

:func:`select` evaluates an address over a :class:`ParamSpace` (the tensors a model or a view exposes, with their
roles), following references into an address book's sets.

.. warning:: This is an experimental API and may change in a future release.
"""
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from venngram.address import And, Expr, Guarded, Minus, parse, Ref, Term, to_string
from venngram.errors import AddressUnresolvable, Refusal
from venngram.roles import selector_matches

_WRAPPER_SEGMENTS = ("_orig_mod.", "_checkpoint_wrapped_module.", "_fsdp_wrapped_module.")


def strip_wrappers(name: str) -> str:
    """``name`` without wrapper segments (``torch.compile``'s ``_orig_mod.``, activation checkpointing's, FSDP1's).

    >>> strip_wrappers("_orig_mod.layers.3._checkpoint_wrapped_module.mlp.weight")
    'layers.3.mlp.weight'
    """
    for seg in _WRAPPER_SEGMENTS:
        name = name.replace(seg, "")
    return name


@dataclass(frozen=True)
class Location:
    """Where a tensor sits in a model's block structure.

    Attributes:
        stack: The block stack (``None`` for the primary stack).
        layer: The block index, or ``None`` for a tensor outside every block.
        component: The architecture-neutral component (``attn.q``), when a vocabulary provider reports one.
    """

    stack: str | None = None
    layer: int | None = None
    component: str | None = None


@runtime_checkable
class StructureProvider(Protocol):
    """Places tensors in a model's block structure."""

    @property
    def vocabulary(self) -> bool:
        """Whether :attr:`Location.component` holds architecture-neutral vocabulary names."""
        ...

    def locate(self, name: str) -> Location:
        """Where the tensor ``name`` sits."""
        ...

    def depth(self, stack: str | None = None) -> int | Refusal:
        """The number of blocks in ``stack`` (the primary stack when ``None``), or why it is unknown."""
        ...


def _glob(pattern: str) -> str:
    # `*` is one path segment, `**` one or more
    parts = []
    for seg in pattern.split("."):
        if seg == "**":
            parts.append(r"[^.]+(?:\.[^.]+)*")
        elif seg == "*":
            parts.append(r"[^.]+")
        else:
            parts.append("[^.]*".join(re.escape(p) for p in seg.split("*")))
    return r"\.".join(parts)


def module_matches(pattern: str, name: str) -> bool:
    """Whether the tensor ``name`` is under a module whose path ends with ``pattern``, on whole segments.

    >>> module_matches("q_proj", "model.layers.3.self_attn.q_proj.lora_A.default.weight")
    True
    >>> module_matches("q_proj", "model.layers.3.self_attn.xq_proj.weight")
    False
    >>> module_matches("layers.*.mlp", "layers.3.mlp.weight")
    True
    """
    owner = strip_wrappers(name).rpartition(".")[0]
    return re.search(rf"(?:^|\.){_glob(pattern)}(?:\.|$)", owner) is not None


def leaf_matches(pattern: str, name: str) -> bool:
    """Whether the tensor ``name`` ends with ``pattern``, on whole segments (``weight``, ``lora_A.*``)."""
    return re.search(rf"(?:^|\.){_glob(pattern)}$", strip_wrappers(name)) is not None


def component_matches(pattern: str, component: str | None) -> bool:
    """Whether a vocabulary ``component`` matches ``pattern`` (globbed per segment, matched in full)."""
    return component is not None and re.fullmatch(_glob(pattern), component) is not None


@dataclass
class ParamSpace:
    """The tensors an address selects from.

    Args:
        names: The tensors' names, in the order results are reported.
        roles: Roles by name (anything unlisted is ``base``).
        aliases: Other names each tensor is known by (a view's logical names); ``name`` and ``module`` match them too.
        structure: Places tensors in the block structure; ``layers``, ``stack`` and ``component`` need one.
    """

    names: Sequence[str]
    roles: Mapping[str, str] = field(default_factory=dict)
    aliases: Mapping[str, Sequence[str]] = field(default_factory=dict)
    structure: StructureProvider | None = None

    def spellings(self, name: str) -> tuple[str, ...]:
        """``name`` and its aliases."""
        return (name, *self.aliases.get(name, ()))


_TARGET_PROVIDER = {"site": "a site resolver", "feature": "a latent resolver", "subspace": "a latent resolver"}


class _Evaluator:
    def __init__(self, space: ParamSpace, sets: Mapping[str, Any]) -> None:
        self.space = space
        self.sets = sets
        self.refusals: list[Refusal] = []
        self._cache: dict[str, list[str]] = {}
        self._stack: list[str] = []

    def refuse(self, reason: str, alternatives: Sequence[str] = ()) -> list[str]:
        refusal = Refusal(reason, tuple(alternatives))
        if refusal not in self.refusals:
            self.refusals.append(refusal)
        return []

    def run(self, expr: Expr) -> list[str]:
        if isinstance(expr, Term):
            return self.term(expr)
        if isinstance(expr, Ref):
            return self.ref(expr.name)
        if isinstance(expr, Guarded):
            return self.guarded(expr, label=to_string(expr.expr))
        if isinstance(expr, Minus):
            excluded = set(self.run(expr.right))
            return [n for n in self.run(expr.left) if n not in excluded]
        results = [set(self.run(i)) for i in expr.items]
        keep = set.intersection(*results) if isinstance(expr, And) else set.union(*results)
        return [n for n in self.space.names if n in keep]

    def guarded(self, expr: Guarded, label: str) -> list[str]:
        out = self.run(expr.expr)
        if expr.expect is not None and len(out) != expr.expect:
            self.refuse(f"`{label}` selects {len(out)} tensors where `expect` requires {expr.expect}")
        return out

    def ref(self, name: str) -> list[str]:
        if name in self._cache:
            return self._cache[name]
        if name in self._stack:
            return self.refuse(f"the set references form a cycle: {' -> '.join('@' + s for s in self._stack + [name])}")
        if name not in self.sets:
            return self.refuse(f"no set is named `@{name}`", sorted(f"@{s}" for s in self.sets))
        self._stack.append(name)
        expr = parse(self.sets[name])
        out = self.run(expr)
        self._stack.pop()
        if not out and not (isinstance(expr, Guarded) and expr.optional):
            self.refuse(f"the set `@{name}` matches nothing (mark it `optional: true` to allow that)")
        self._cache[name] = out
        return out

    def term(self, t: Term) -> list[str]:
        try:
            t.check_target()
        except ValueError as err:
            return self.refuse(str(err))
        if t.target != "param":
            return self.refuse(f"`{to_string(t)}` addresses a {t.target}, which resolves through "
                               f"{_TARGET_PROVIDER[t.target]} (interpretune supplies them), not over tensors")
        if t.units is not None:
            return self.refuse(f"`units:{t.units.spec}` on tensors: tuning granularity is the whole tensor, so a "
                               "tensor's units are tuned through a unit adapter, a tensor of its own with an `adapter` "
                               "role")
        checks = self.checks(t)
        if checks is None:
            return []
        return [n for n in self.space.names if all(check(n) for check in checks)]

    def checks(self, t: Term) -> list[Callable[[str], bool]] | None:
        space = self.space
        checks: list[Callable[[str], bool]] = []
        if t.role is not None:
            checks.append(lambda n: selector_matches(f"role:{t.role}", space.roles.get(n, "base")))
        if t.name is not None:
            rx = re.compile(t.name)
            checks.append(lambda n: any(rx.fullmatch(s) for s in space.spellings(n)))
        if t.module is not None:
            checks.append(lambda n: any(module_matches(t.module, s) for s in space.spellings(n)))  # type: ignore
        if t.leaf is not None:
            checks.append(lambda n: any(leaf_matches(t.leaf, s) for s in space.spellings(n)))  # type: ignore
        structural = self.structural(t)
        if structural is None:
            return None
        return checks + structural

    def structural(self, t: Term) -> list[Callable[[str], bool]] | None:
        if t.layers is None and t.stack is None and t.component is None:
            return []
        structure = self.space.structure
        if structure is None:
            self.refuse(f"`{to_string(t)}` needs a structure provider (for `layers`, `stack` or `component`), and none"
                        " was given")
            return None
        if t.component is not None and not structure.vocabulary:
            self.refuse(f"`component:{t.component}` needs a vocabulary structure provider; this one reports physical "
                        "structure only", ("module:<pattern>",))
            return None
        checks: list[Callable[[str], bool]] = [lambda n: structure.locate(n).stack == t.stack]
        if t.layers is not None:
            depth = structure.depth(t.stack)
            if isinstance(depth, Refusal):
                self.refuse(depth.reason, depth.alternatives)
                return None
            try:
                wanted = set(t.layers.select(depth))
            except ValueError as err:
                self.refuse(str(err))
                return None
            checks.append(lambda n: structure.locate(n).layer in wanted)
        if t.component is not None:
            checks.append(lambda n: component_matches(t.component, structure.locate(n).component))  # type: ignore
        return checks


def select(address: Any, space: ParamSpace, sets: Mapping[str, Any] | None = None) -> list[str]:
    """The tensors of ``space`` that ``address`` selects, in ``space`` order.

    Args:
        address: An address, in either form.
        space: The tensors to select from.
        sets: Named sets that references resolve against (an address book's :meth:`~venngram.book.AddressBook.sets`).

    Raises:
        AddressUnresolvable: With every refusal, if any field does not resolve, a referenced set is missing or matches
            nothing, references form a cycle, or the address matches nothing (and is not ``optional``).
    """
    expr = parse(address)
    ev = _Evaluator(space, sets or {})
    out = ev.run(expr)
    if not out and not ev.refusals and not (isinstance(expr, Guarded) and expr.optional):
        ev.refuse("it matches no tensor")
    if ev.refusals:
        raise AddressUnresolvable(to_string(expr), ev.refusals)
    return out


__all__ = [
    "component_matches",
    "leaf_matches",
    "Location",
    "module_matches",
    "ParamSpace",
    "select",
    "strip_wrappers",
    "StructureProvider",
]
