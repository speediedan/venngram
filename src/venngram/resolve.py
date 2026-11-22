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

from venngram.address import And, Expr, Guarded, Minus, Or, parse, Path, Ref, references, Term, terms, to_string
from venngram.errors import AddressError, AddressUnresolvable, Refusal
from venngram.roles import selector_matches
from venngram.units import ElementMask

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


def segment_after(pattern: str, name: str) -> tuple[str, str] | None:
    """Where ``pattern`` matches the tensor ``name`` on whole segments: the path of the matched module and the segment
    that follows it, or ``None``.

    >>> segment_after("lora_A", "model.layers.3.self_attn.q_proj.lora_A.default.weight")
    ('model.layers.3.self_attn.q_proj.lora_A', 'default')
    >>> segment_after("lora_A", "lora_A") is None
    True
    """
    found = re.search(rf"(?:^|\.)({_glob(pattern)})\.([^.]+)", name)
    return (name[:found.end(1)], found.group(2)) if found else None


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


#: The gradient readings a consumer may honour beyond one ``of`` per set and one mask per conjunction: a union uniting
#: a value part and a gradient part (:func:`split_of`), and a union of gradient conjunctions with unit sets
#: (:func:`gradient_masks`).
#: ``unit-axis`` and ``physical-axis`` are a unit set's explicit ``axis`` (:func:`gradient_selections`).
#: ``mask-algebra`` is an intersection or a difference of masks (two unit sets in one conjunction, a unit set under a
#: difference, or under a union inside an intersection), and ``element-masks`` an element mask
#: (``units: {elements: ...}``), both resolved by :func:`gradient_selections`.
GRADIENT_FORMS = ("mixed-sets", "union-masks", "unit-axis", "physical-axis", "mask-algebra", "element-masks")

_TARGET_PROVIDER = {"site": "a site resolver", "feature": "a latent resolver", "subspace": "a latent resolver"}


class _Evaluator:
    def __init__(self, space: ParamSpace, sets: Mapping[str, Any], *, masks: bool = False) -> None:
        self.space = space
        self.sets = sets
        self.masks = masks
        """Whether ``units`` on tensors is a gradient mask (the address is ``of: grad``), so it does not filter."""
        self.refusals: list[Refusal] = []
        self._cache: dict[str, list[str]] = {}
        self._stack: list[str] = []

    def refuse(self, reason: str, alternatives: Sequence[str] = ()) -> list[str]:
        refusal = Refusal(reason, tuple(alternatives))
        if refusal not in self.refusals:
            self.refusals.append(refusal)
        return []

    def run(self, expr: Expr) -> list[str]:
        if isinstance(expr, Path):
            return self.refuse(f"`{to_string(expr)}` is a path, which a consumer's route resolver places "
                               "(`venngram.paths`), not a set of tensors")
        if isinstance(expr, Term):
            return self.term(expr)
        if isinstance(expr, Ref):
            return self.ref(expr.name)
        if isinstance(expr, Guarded):
            return self.guarded(expr, label=to_string(expr.expr))
        if isinstance(expr, Minus):
            excluded = set(self.run(expr.right))
            if self.masks:  # a masked right side subtracts from a tensor's mask; only a whole one drops the tensor
                excluded = self.whole(expr.right)
            return [n for n in self.run(expr.left) if n not in excluded]
        results = [set(self.run(i)) for i in expr.items]
        keep = set.intersection(*results) if isinstance(expr, And) else set.union(*results)
        return [n for n in self.space.names if n in keep]

    def whole(self, expr: Expr, stack: tuple[str, ...] = ()) -> set[str]:
        """The tensors whose whole gradient ``expr`` keeps (selected by no unit set), for a masked evaluation."""
        if isinstance(expr, Path):
            return set()
        if isinstance(expr, Term):
            return set(self.term(expr)) if expr.units is None else set()
        if isinstance(expr, Ref):
            if expr.name not in self.sets or expr.name in stack:
                return set()  # run reports it
            return self.whole(parse(self.sets[expr.name]), (*stack, expr.name))
        if isinstance(expr, Guarded):
            return self.whole(expr.expr, stack)
        if isinstance(expr, Minus):
            return self.whole(expr.left, stack) - set(self.run(expr.right))
        parts = [self.whole(i, stack) for i in expr.items]
        return set.intersection(*parts) if isinstance(expr, And) else set.union(*parts)

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
        if t.units is not None and not self.masks:
            return self.refuse(f"`units:{t.units.spec}` on tensors: tuning granularity is the whole tensor, so a "
                               "tensor's units are tuned through a unit adapter, a tensor of its own with an `adapter` "
                               "role, or masked as gradients with `of: grad`")
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


def gradient_of(address: Any, sets: Mapping[str, Any] | None = None) -> str:
    """What a consumer acts on for ``address``: ``value``, or ``grad`` when its terms say ``of: grad``, following
    references into ``sets``. ``of`` does not filter, so :func:`select` gives the same tensors either way.

    >>> gradient_of("role:adapter & of:grad"), gradient_of("role:adapter")
    ('grad', 'value')

    Raises:
        AddressUnresolvable: If its terms disagree on ``of`` (a set mixing values and gradients has no one thing to act
            on; :func:`split_of` separates a union's value and gradient parts), or a referenced set is missing.
    """
    expr = parse(address)
    sets = sets or {}
    seen, refusals = _of_values(expr, sets)
    if len(seen) > 1:
        _, _, split = _split(expr, sets)
        refusals.append(_disagreement(seen, "" if split else " (it unites a value part and a gradient part, which "
                                                                  "`split_of` separates)"))
    if refusals:
        raise AddressUnresolvable(to_string(expr), refusals)
    return next(iter(seen), "value")


def split_of(address: Any, sets: Mapping[str, Any] | None = None) -> tuple[Expr | None, Expr | None]:
    """``address`` as its value part and its gradient part, either ``None`` when it has none.

    A set's terms usually agree on ``of``, and terms that do not state it follow those that do, so the whole set is one
    part. A union's disjuncts may differ: then every disjunct states its ``of`` (following references), and each part
    is the union of the disjuncts that say it. An intersection or a difference of values and gradients has no meaning
    and stays refused.

    >>> value, grad = split_of("module:k_proj | module:q_proj & units:[1] & of:grad")
    >>> value, to_string(grad)
    (None, 'module:k_proj | module:q_proj & units:[1] & of:grad')
    >>> [to_string(p) for p in split_of("module:k_proj & of:value | module:q_proj & units:[1] & of:grad")]
    ['module:k_proj & of:value', 'module:q_proj & units:[1] & of:grad']

    Raises:
        AddressUnresolvable: If a referenced set is missing, a disjunct mixes values and gradients, or a disjunct
            beside disjuncts that differ on ``of`` states none.
    """
    expr = parse(address)
    value, grad, refusals = _split(expr, sets or {})
    if refusals:
        raise AddressUnresolvable(to_string(expr), refusals)
    return _union(value), _union(grad)


def _of_values(expr: Expr, sets: Mapping[str, Any]) -> tuple[dict[str, list[str]], list[Refusal]]:
    """The ``of`` values ``expr``'s terms set (each with where it came from), and a refusal for each missing set."""
    seen: dict[str, list[str]] = {}
    refusals: list[Refusal] = []

    def walk(e: Expr, via: str, stack: tuple[str, ...]) -> None:
        for t in terms(e):
            if t.of is not None:
                seen.setdefault(t.of, []).append(via or to_string(t))
        for name in references(e):
            if name in stack:
                continue  # a cycle is select's to report
            if name not in sets:
                refusals.append(Refusal(f"no set is named `@{name}`", tuple(sorted(f"@{s}" for s in sets))))
                continue
            walk(parse(sets[name]), f"@{name}", (*stack, name))

    walk(expr, "", ())
    return seen, refusals


def _disagreement(seen: Mapping[str, list[str]], note: str = "") -> Refusal:
    return Refusal("its terms disagree on `of` (" + "; ".join(
        f"`{v}` in {', '.join(dict.fromkeys(w))}" for v, w in seen.items()) + "): a set is either values or gradients"
        + note)


def _disjuncts(expr: Expr, sets: Mapping[str, Any], opens: Callable[[Expr], bool],
               stack: tuple[str, ...] = ()) -> list[Expr]:
    """``expr``'s union disjuncts, flattening nested unions and opening each reference (or ``optional`` mapping) for
    which ``opens`` holds; ``expect`` counts the whole set, so a mapping that carries it stays one disjunct."""
    if isinstance(expr, Or):
        return [d for i in expr.items for d in _disjuncts(i, sets, opens, stack)]
    if isinstance(expr, Ref) and expr.name in sets and expr.name not in stack and opens(expr):
        inner = _disjuncts(parse(sets[expr.name]), sets, opens, (*stack, expr.name))
        return inner if len(inner) > 1 else [expr]
    if isinstance(expr, Guarded) and expr.expect is None and opens(expr):
        inner = _disjuncts(expr.expr, sets, opens, stack)
        if len(inner) > 1:
            return [Guarded(d, optional=True) for d in inner] if expr.optional else inner
    return [expr]


def _split(expr: Expr, sets: Mapping[str, Any]) -> tuple[list[Expr], list[Expr], list[Refusal]]:
    """The value and gradient disjuncts of ``expr`` (see :func:`split_of`), and why it cannot be split."""
    seen, refusals = _of_values(expr, sets)
    if refusals or len(seen) < 2:
        return ([], [expr], refusals) if list(seen) == ["grad"] else ([expr], [], refusals)
    parts: dict[str, list[Expr]] = {"value": [], "grad": []}
    for d in _disjuncts(expr, sets, lambda e: len(_of_values(e, sets)[0]) > 1):
        own = _of_values(d, sets)[0]
        if len(own) > 1:
            refusals.append(_disagreement(own, f", and `{to_string(d)}` mixes them in an intersection or a difference,"
                                               " where only a union's disjuncts may differ"))
        elif not own:
            refusals.append(Refusal(f"`{to_string(d)}` states no `of` beside disjuncts that differ on it: say "
                                    "`of:value` or `of:grad`"))
        else:
            parts[next(iter(own))].append(d)
    return parts["value"], parts["grad"], refusals


def _union(parts: Sequence[Expr]) -> Expr | None:
    return None if not parts else parts[0] if len(parts) == 1 else Or(tuple(parts))


def _mask_terms(expr: Expr, sets: Mapping[str, Any]) -> tuple[list[Term], list[str]]:
    """The terms of ``expr`` that set ``units``, following references, and the ways its masks need mask algebra rather
    than one mask from one conjunction: a unit set under a union or a difference, or two unit sets intersected."""
    found: list[Term] = []
    problems: list[str] = []

    def walk(e: Expr, conjunctive: bool, stack: tuple[str, ...]) -> None:
        if isinstance(e, Term):
            if e.units is not None:
                found.append(e)
                if not conjunctive:
                    problems.append(f"`units:{e.units.spec}` sits under a union or a difference")
        elif isinstance(e, Ref):
            if e.name in sets and e.name not in stack:
                walk(parse(sets[e.name]), conjunctive, (*stack, e.name))
        elif isinstance(e, Guarded):
            walk(e.expr, conjunctive, stack)
        elif isinstance(e, And):
            for i in e.items:
                walk(i, conjunctive, stack)
        elif isinstance(e, Minus):
            walk(e.left, conjunctive, stack)
            walk(e.right, False, stack)
        elif isinstance(e, Or):
            for i in e.items:
                walk(i, False, stack)

    walk(expr, True, ())
    if len(found) > 1:
        problems.append("it intersects " + " and ".join(f"`units:{t.units.spec}`" for t in found if t.units))
    return found, problems


def gradient_masks(address: Any, space: ParamSpace, sets: Mapping[str, Any] | None = None, *,
                   widths: Mapping[str, int]) -> dict[str, tuple[int, ...]]:
    """The gradient mask of each tensor an ``of: grad`` address selects: the indices along its first axis (a weight's
    output rows, a vector's entries) whose gradients are kept. A tensor the address selects without a unit set keeps
    all of them; a per-layer unit set gives a tensor at a layer it does not name an empty mask. Masks combine as their
    sets do, exactly (:func:`gradient_selections`): a union unites them, an intersection intersects them and a
    difference subtracts them.

    Args:
        address: An address whose terms say ``of: grad`` (the gradient part of a mixed set, from :func:`split_of`).
        space: The tensors to select from.
        sets: Named sets that references resolve against.
        widths: Each tensor's first-axis length, by name.

    Raises:
        AddressUnresolvable: As :func:`select`, if the address is not ``of: grad``, a unit set names an ``axis`` or is
            an element mask (which a first-axis mask cannot carry: :func:`gradient_selections` resolves both), or as
            :func:`gradient_selections`.
    """
    expr = _grad_address(address, sets or {})
    masks = [t for d in _disjuncts(expr, sets or {}, lambda e: True) for t in _mask_terms(d, sets or {})[0]]
    axes = {str(t.axis) for t in masks if t.axis is not None}
    if any(t.units is not None and t.units.heads is not None for t in masks):
        axes.add("unit")
    if axes:
        raise AddressUnresolvable(to_string(expr), [Refusal(
            f"`axis:{', '.join(sorted(axes))}` names the dimension a mask runs along, which a first-axis mask "
            "cannot carry: resolve it with `venngram.resolve.gradient_selections`")])
    if "element-masks" in gradient_forms(expr, sets):
        raise AddressUnresolvable(to_string(expr), [Refusal(
            "an element mask keeps individual elements, which a first-axis mask cannot carry: resolve it with "
            "`venngram.resolve.gradient_selections`")])
    out = gradient_selections(expr, space, sets, shapes={n: (int(w),) for n, w in widths.items()})
    return {n: tuple(range(widths[n])) if sel.whole else sel.indices or () for n, sel in out.items()}


def _grad_address(address: Any, sets: Mapping[str, Any]) -> Expr:
    expr = parse(address)
    if gradient_of(expr, sets) != "grad":
        raise AddressUnresolvable(to_string(expr), [Refusal("a gradient mask needs an `of: grad` address")])
    return expr


@dataclass(frozen=True)
class GradientSelection:
    """The part of one tensor's gradient an ``of: grad`` address keeps: ``indices`` along dimension ``axis``, an element
    mask (``elements``), or the whole tensor (all ``None``)."""

    axis: int | None = None
    indices: tuple[int, ...] | None = None
    elements: ElementMask | None = None

    @property
    def whole(self) -> bool:
        return self.indices is None and self.elements is None


_WHOLE = "whole"
_DROP = "drop"


@dataclass(frozen=True)
class _Axis:
    dim: int
    indices: frozenset[int]


class _Masks:
    """Each tensor's kept gradient under an ``of: grad`` expression, combined exactly as the expression's sets are:
    the whole tensor, indices along one dimension (:class:`_Axis`), or an element mask."""

    def __init__(self, space: ParamSpace, sets: Mapping[str, Any], shapes: Mapping[str, Sequence[int]]) -> None:
        self.space, self.sets = space, sets
        self.shapes = {n: tuple(int(d) for d in shape) for n, shape in shapes.items()}
        self.ev = _Evaluator(space, sets, masks=True)  # select has reported every refusal
        self.problems: list[str] = []

    def problem(self, text: str) -> str:
        self.problems.append(text)
        return _WHOLE  # keep going, so every problem is reported at once

    def run(self, e: Expr, within: set[str], stack: tuple[str, ...] = ()) -> dict[str, Any]:
        """Each tensor of ``within`` that ``e`` selects, with what ``e`` keeps of its gradient. ``within`` is what the
        enclosing expression selects, so a lone ``units`` term (which selects every tensor) is resolved only on the
        tensors its conjunction narrows it to."""
        if isinstance(e, Path):
            return {}
        if isinstance(e, Term):
            names = [n for n in self.ev.term(e) if n in within]
            return {n: _WHOLE if e.units is None else self.term(e, n) for n in names}
        if isinstance(e, Ref):
            if e.name not in self.sets or e.name in stack:
                return {}
            return self.run(parse(self.sets[e.name]), within, (*stack, e.name))
        if isinstance(e, Guarded):
            return self.run(e.expr, within, stack)
        if isinstance(e, Minus):
            left = self.run(e.left, within, stack)
            right = self.run(e.right, set(left), stack)
            out = {}
            for n, kept in left.items():
                rest = kept if n not in right else self.less(n, kept, right[n])
                if rest is not _DROP:
                    out[n] = rest
            return out
        if isinstance(e, And):
            within = within & set(self.ev.run(e))
            combine = self.meet
        else:
            combine = self.join
        parts = [self.run(i, within, stack) for i in e.items]
        out = {}
        for n in set.union(*(set(p) for p in parts)):
            present = [p[n] for p in parts if n in p]
            if isinstance(e, And) and len(present) < len(parts):
                continue
            kept = present[0]
            for other in present[1:]:
                kept = combine(n, kept, other)
            out[n] = kept
        return out

    def term(self, t: Term, n: str) -> Any:
        units, shape, structure = t.units, self.shapes[n], self.space.structure
        assert units is not None
        layer = structure.locate(n).layer if structure is not None and units.layers is not None else None
        try:
            if units.is_elements:
                return units.elements_for_layer(layer, shape)
            if t.axis == "unit" or units.heads is not None:
                unit_dim = getattr(structure, "unit_dim", None)
                found = unit_dim(n, len(shape)) if unit_dim is not None else Refusal(
                    "the structure provides no unit axes (a map's structure does)")
                if isinstance(found, Refusal):
                    return self.problem(f"`{n}`: `axis:unit` has no unit axis: {found.reason}")
                dim = int(found)
            else:
                dim = int(t.axis) if t.axis is not None else 0
            if not 0 <= dim < len(shape):
                return self.problem(f"`{n}` has {len(shape)} dimensions, so it has no dimension {dim}")
            block_of = getattr(structure, "unit_block", None)
            block = block_of(n) if units.heads is not None and block_of is not None else None
            return _Axis(dim, frozenset(units.for_layer(layer, shape[dim], unit_block=block)))
        except AddressError as err:
            return self.problem(f"`{n}`: {err}")

    def _mixed(self, n: str, a: Any, b: Any, verb: str) -> Any:
        if isinstance(a, _Axis) and isinstance(b, _Axis):
            dims = sorted({a.dim, b.dim})
            if verb == "join":
                return self.problem(f"`{n}` is masked along dimensions {dims} by different disjuncts; one axis and its "
                                    "indices cannot carry their union")
            return self.problem(f"`{n}`: masks along dimensions {dims} {verb} to a block (rows crossed with columns), "
                                "which one axis and its indices cannot carry")
        return self.problem(f"`{n}`: an element mask and a mask along one dimension do not combine; write them as one "
                            "element mask")

    def meet(self, n: str, a: Any, b: Any) -> Any:
        if a is _WHOLE or b is _WHOLE:
            return b if a is _WHOLE else a
        if isinstance(a, _Axis) and isinstance(b, _Axis) and a.dim == b.dim:
            return _Axis(a.dim, a.indices & b.indices)
        if isinstance(a, ElementMask) and isinstance(b, ElementMask):
            return a & b
        return self._mixed(n, a, b, "intersect")

    def join(self, n: str, a: Any, b: Any) -> Any:
        if a is _WHOLE or b is _WHOLE:
            return _WHOLE
        if isinstance(a, _Axis) and isinstance(b, _Axis) and a.dim == b.dim:
            return _Axis(a.dim, a.indices | b.indices)
        if isinstance(a, ElementMask) and isinstance(b, ElementMask):
            return a | b
        return self._mixed(n, a, b, "join")

    def less(self, n: str, a: Any, b: Any) -> Any:
        if b is _WHOLE:
            return _DROP
        if a is _WHOLE:
            if isinstance(b, _Axis):
                return _Axis(b.dim, frozenset(range(self.shapes[n][b.dim])) - b.indices)
            return ElementMask.full(self.shapes[n]) - b
        if isinstance(a, _Axis) and isinstance(b, _Axis) and a.dim == b.dim:
            return _Axis(a.dim, a.indices - b.indices)
        if isinstance(a, ElementMask) and isinstance(b, ElementMask):
            return a - b
        return self._mixed(n, a, b, "subtract")


def gradient_selections(address: Any, space: ParamSpace, sets: Mapping[str, Any] | None = None, *,
                        shapes: Mapping[str, Sequence[int]]) -> dict[str, GradientSelection]:
    """What each tensor an ``of: grad`` address selects keeps of its gradient, along the dimension its unit set names
    or element by element.

    A unit set's ``axis`` says which dimension it indexes: ``unit`` is the component's unit axis, from the structure's
    measured layout (:meth:`venngram.maps.structure.MapStructure.unit_dim`), so one address selects an MLP's neurons on
    a linear and on a transposed weight alike; an integer is that physical dimension; unset is the first dimension, as
    :func:`gradient_masks` reads it. An element mask (``units: {elements: ...}``) keeps individual elements of a tensor
    of its exact shape. A head selection (``units: {heads: [2, 5]}``) expands over the recorded unit axis using the
    target row's ``unit_block``. A tensor selected without a unit set keeps its whole gradient.

    Masks combine exactly as their sets do: a union unites them, an intersection intersects them, and a difference
    subtracts the right side's mask from the left's (a right side keeping a tensor's whole gradient drops the tensor;
    a left side keeping it all keeps the complement). Masks along one dimension combine along it, and element masks
    of one shape element by element. A combination one axis and its indices cannot carry is refused: masks along two
    dimensions (rows crossed with columns), or an element mask with a mask along one dimension. Every term is checked
    before anything is returned, so a whole-tensor branch does not hide a malformed one.

    Args:
        address: An address whose terms say ``of: grad``.
        space: The tensors to select from.
        sets: Named sets that references resolve against.
        shapes: Each tensor's shape, by name.

    Raises:
        AddressUnresolvable: As :func:`select`, if the address is not ``of: grad``, a unit axis has no layout (the
            structure provides none, or names why the tensor has no unit axis), a dimension lies outside a tensor's
            rank, an element mask was written for another shape, or masks combine in a way one axis and its indices
            cannot carry.
    """
    sets = sets or {}
    expr = _grad_address(address, sets)
    names = select(expr, space, sets)
    masks = _Masks(space, sets, shapes)
    kept = masks.run(expr, set(names))
    if masks.problems:
        raise AddressUnresolvable(to_string(expr), [Refusal(p) for p in dict.fromkeys(masks.problems)])
    out: dict[str, GradientSelection] = {}
    for n in names:
        k = kept.get(n, _WHOLE)
        if isinstance(k, _Axis):
            out[n] = GradientSelection(k.dim, tuple(sorted(k.indices)))
        elif isinstance(k, ElementMask):
            out[n] = GradientSelection(elements=k)
        else:
            out[n] = GradientSelection()
    return out


def gradient_forms(address: Any, sets: Mapping[str, Any] | None = None) -> frozenset[str]:
    """The gradient readings ``address`` needs beyond one ``of`` and one mask (:data:`GRADIENT_FORMS`): ``mixed-sets``
    when a union unites a value part and a gradient part, ``union-masks`` when its gradient part unites disjuncts and
    one of them sets ``units``. An address that cannot be split needs nothing here; resolving it reports why.

    >>> sorted(gradient_forms("module:k_proj & of:value | module:q_proj & units:[1] & of:grad"))
    ['mixed-sets']
    >>> sorted(gradient_forms("module:k_proj & units:[0] & of:grad | module:q_proj & units:[1] & of:grad"))
    ['union-masks']
    """
    sets = sets or {}
    value, grad, refusals = _split(parse(address), sets)
    if refusals:
        return frozenset()
    out = {"mixed-sets"} if value and grad else set()
    disjuncts = _disjuncts(_union(grad), sets, lambda e: True) if grad else []  # type: ignore[arg-type]
    masks = [_mask_terms(d, sets)[0] for d in disjuncts]
    if len(disjuncts) > 1 and any(masks):
        out.add("union-masks")
    if any(_mask_terms(d, sets)[1] for d in disjuncts):
        out.add("mask-algebra")
    if any(t.units is not None and t.units.is_elements for found in masks for t in found):
        out.add("element-masks")
    axes = {t.axis for found in masks for t in found if t.axis is not None}
    if "unit" in axes or any(t.units is not None and t.units.heads is not None for found in masks for t in found):
        out.add("unit-axis")
    if any(isinstance(a, int) for a in axes):
        out.add("physical-axis")
    return frozenset(out)


def select(address: Any, space: ParamSpace, sets: Mapping[str, Any] | None = None) -> list[str]:
    """The tensors of ``space`` that ``address`` selects, in ``space`` order.

    Args:
        address: An address, in either form.
        space: The tensors to select from.
        sets: Named sets that references resolve against (an address book's :meth:`~venngram.book.AddressBook.sets`).

    Raises:
        AddressUnresolvable: With every refusal, if any field does not resolve, a referenced set is missing or matches
            nothing, references form a cycle, the address matches nothing (and is not ``optional``), or its terms
            disagree on ``of`` other than across a union's disjuncts (:func:`split_of`). ``units`` on tensors is
            refused unless it is in the address's gradient part, where it is a gradient mask (:func:`gradient_masks`)
            and does not filter.
    """
    expr = parse(address)
    sets = sets or {}
    value, grad, of_refusals = _split(expr, sets)
    keep: set[str] = set()
    refusals: list[Refusal] = []
    for parts, masks in ((value, False), (grad, True)):
        if parts:
            ev = _Evaluator(space, sets, masks=masks)
            keep.update(ev.run(_union(parts)))  # type: ignore[arg-type]
            refusals += [r for r in ev.refusals if r not in refusals]
    refusals += [r for r in of_refusals if r not in refusals]  # a missing set is the evaluator's to report as well
    out = [n for n in space.names if n in keep]
    if not out and not refusals and not (isinstance(expr, Guarded) and expr.optional):
        refusals.append(Refusal("it matches no tensor"))
    if refusals:
        raise AddressUnresolvable(to_string(expr), refusals)
    return out


__all__ = [
    "component_matches",
    "GRADIENT_FORMS",
    "gradient_forms",
    "gradient_masks",
    "gradient_selections",
    "GradientSelection",
    "gradient_of",
    "leaf_matches",
    "Location",
    "module_matches",
    "ParamSpace",
    "segment_after",
    "select",
    "split_of",
    "strip_wrappers",
    "StructureProvider",
]
