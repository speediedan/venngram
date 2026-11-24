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
Tunable Views
^^^^^^^^^^^^^

A :class:`TunableView` is the contract between a model's *logical* parameter names (the names a schedule uses) and its
*physical* parameters (the tensors the optimizer, checkpoints and sharding see). It lets a schedule address a model
through names other than ``named_parameters()``, for example the names an interpretability library gives a wrapped
model, where one logical name may stand for several physical tensors.

A view needs only four members (:class:`TunableView`). The optional members (:class:`TunableViewExtras`) have defaults
that are part of the protocol: :class:`BoundView` applies them to a view that does not define them, and
:class:`ViewDefaults` provides every default for implementers who prefer to inherit. :class:`CanonicalView` is the
identity view every plain model gets.

Views are discovered, in order, from an explicit ``view`` argument, then a view attached to the model as its
``tunable_view`` attribute, then :class:`CanonicalView` (see :func:`resolve_view`).

**Cooperative roles.** A view may assign roles itself (a view over a model with spliced transcoders reports them as
``artifact``, say), and a consumer that attaches modules of its own (a trainable intervention, a latent model) may
declare roles for their tensors through :meth:`BoundView.declare_roles` (or ``resolve_view(..., roles=...)``). The rule
holds for every view, because the bound view applies it: a declared role on a tensor the view does not assign is
merged and reported like the view's own, selectable by ``role:``, and its kind joins the support record's declared
kinds; a declared role equal to the one the view assigns is accepted; a declared role that differs from the view's
own is a conflict, refused by name with the tensor and both roles, never resolved by precedence. A tensor the view
assigns is one in its own :meth:`~TunableViewExtras.roles` (a ``base`` it lists explicitly counts), or one promoted
since.

**Roles by address.** A view's ``roles`` maps a tensor name to its role, or a role to an address
(``{"adapter:lora": 'component:attn & name:".*lora_[AB].*"'}``) resolved against the view's own names and structure
when the view is built (:func:`expand_roles`), so one declaration carries to every view that places the same
components. The address states no ``role`` term, since roles are what it assigns, and may reference a book's sets.
A tensor two role addresses give different roles is refused by name unless a name entry settles it, and a name
entry always wins over an address.

.. warning:: This is an experimental API and may change in a future release.
"""
import itertools
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from torch import nn, Tensor

from .errors import RoleKeyError, TunableViewError, Unresolvable
from venngram.physical import PhysicalStructure
from venngram.resolve import StructureProvider
from venngram.deprecation import deprecated
from venngram.roles import (canonical_kind, declared_kinds, parse_role, role_kind, UNCHECKPOINTED_ROLE_KINDS,
                            validate_roles)

# the module attribute a view is attached under for discovery
VIEW_ATTR = "tunable_view"
REQUIRED_MEMBERS = ("root", "support", "to_physical", "to_logical")
OPTIONAL_MEMBERS = ("logical_named_parameters", "coupled", "depth_groups", "roles", "checkpoint_state_dict",
                    "load_checkpoint_state_dict", "promote", "structure")


@dataclass(frozen=True)
class TunableSupport:
    """What a view declares about its model.

    Attributes:
        trainable: Whether the model can be fine-tuned through this view. When ``False``, a consumer refuses the model,
            naming ``reason``.
        reason: Why the model cannot be fine-tuned (required when ``trainable`` is ``False``).
        naming_scheme: The scheme the view's logical names follow. ``"physical"`` means logical names are the root's
            own parameter names; any other value (for example ``"tl"``) requires the view to define
            ``logical_named_parameters``.
        logical_names_are_views: Whether logical names stand for reshaped or sliced views of physical tensors.
        declared_roles: The role kinds the view's tensors use (see :mod:`~venngram.tuning.roles`).
    """

    trainable: bool
    reason: str | None = None
    naming_scheme: str = "physical"
    logical_names_are_views: bool = False
    declared_roles: frozenset[str] = field(default_factory=lambda: frozenset({"base"}))

    def __post_init__(self) -> None:
        canonical = frozenset(canonical_kind(k) for k in self.declared_roles)  # `instrument` is `artifact`
        if canonical != self.declared_roles:
            object.__setattr__(self, "declared_roles", canonical)
        if not self.trainable and not self.reason:
            raise TunableViewError("A `TunableSupport` with `trainable=False` must give a `reason`.")


@runtime_checkable
class TunableView(Protocol):
    """The required members of a tunable view.

    Any object with these members is a view; inheriting from a protocol class is not required.
    """

    @property
    def root(self) -> nn.Module:
        """The module whose ``named_parameters()`` are the physical names the optimizer and checkpoints use."""
        ...

    def support(self) -> TunableSupport:
        """The view's support record.

        It is read once, at setup.
        """
        ...

    def to_physical(self, logical: Sequence[str]) -> Mapping[str, tuple[str, ...]]:
        """Map each logical name to the physical parameter names it stands for (one or more).

        Raises:
            Unresolvable: If any name does not resolve, naming every such name.
        """
        ...

    def to_logical(self, physical: Sequence[str]) -> list[str]:
        """Map physical parameter names to logical names, without duplicates and in order; physical names without a
        logical name pass through unchanged."""
        ...


@runtime_checkable
class TunableViewExtras(Protocol):
    """The optional members of a tunable view.

    The protocol defines a default for each one a view lacks (see :class:`ViewDefaults`).
    """

    def logical_named_parameters(self) -> Iterator[tuple[str, Tensor]]:
        """Logical names with a representative tensor, in registration order (for schedule generation and
        display)."""
        ...

    def coupled(self, physical: Sequence[str]) -> list[str]:
        """Physical names thawed implicitly with ``physical`` (for example a block's norms)."""
        ...

    def depth_groups(self) -> list[list[str]]:
        """Logical names grouped into phases for a generated schedule, the first group thawed first."""
        ...

    def roles(self) -> Mapping[str, str]:
        """Physical parameter or buffer names mapped to roles; anything unlisted is ``base``."""
        ...

    def checkpoint_state_dict(self) -> dict[str, Any]:
        """The state to checkpoint."""
        ...

    def load_checkpoint_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:
        """Load state produced by :meth:`checkpoint_state_dict`."""
        ...

    def promote(self, physical: Sequence[str], role: str) -> Mapping[str, str]:
        """Promote ``physical`` tensors to ``role``, returning each promoted tensor's old name mapped to its new
        name."""
        ...

    def structure(self) -> StructureProvider:
        """Places the root's physical tensors in its block structure, for addresses' ``layers``, ``stack`` and
        ``component`` fields."""
        ...


def expand_roles(roles: Mapping[str, Any], names: Sequence[str], structure: StructureProvider | None, *,
                 known: Collection[str] | None = None, sets: Any = None) -> dict[str, str]:
    """A view's ``roles`` as tensor names mapped to roles: name entries as they are, and each role-to-address entry
    expanded to the tensors its address selects among ``names`` (placed by ``structure``).

    Args:
        roles: Tensor names mapped to roles, and roles mapped to addresses. A key that is a tensor name (one of
            ``known``) is a name entry; any other key is a role.
        names: The tensors role addresses select from (the root's parameters and buffers).
        structure: What places them in blocks and components, for ``layers``, ``stack`` and ``component``.
        known: The names a name entry may use (``names`` when ``None``).
        sets: Named sets an address may reference (``@<set>``): a mapping, or an
            :class:`~venngram.book.AddressBook` (its :meth:`~venngram.book.AddressBook.sets`).

    Raises:
        RoleKeyError: If a key is neither a tensor name nor a valid role (a :class:`~venngram.errors.RoleError` and a
            :class:`TunableViewError` both).
        TunableViewError: If an address states a ``role`` term, does not resolve or selects nothing, or two role
            addresses give one tensor different roles and no name entry settles it.
    """
    from venngram.address import parse, terms, to_string
    from venngram.resolve import ParamSpace, select

    known_names = set(names if known is None else known)
    by_name = {k: v for k, v in roles.items() if k in known_names}
    by_address = {k: v for k, v in roles.items() if k not in known_names}
    if not by_address:
        return dict(by_name)
    book_sets = sets.sets() if callable(getattr(sets, "sets", None)) else dict(sets or {})
    space = ParamSpace(list(names), structure=structure)
    claimed: dict[str, list[str]] = {}
    for role, address in by_address.items():
        try:
            parse_role(role)
        except ValueError as err:
            raise RoleKeyError(f"The role key {role!r} names neither a parameter nor a buffer of the module, and "
                                   f"is not a role either: {err}") from None
        try:
            expr = parse(address)
            stated = [to_string(t) for t in terms(expr) if t.role is not None]
            if stated:
                raise TunableViewError(f"Role `{role}`: its address states a role ({stated[0]}); a role address "
                                       "selects by name and structure, since roles are what it assigns.")
            for name in select(expr, space, book_sets):
                claimed.setdefault(name, []).append(role)
        except TunableViewError:
            raise
        except ValueError as err:
            raise TunableViewError(f"Role `{role}`: its address does not resolve on this view: {err}") from None
    conflicts = [f"{n!r} ({' and '.join(sorted(set(rs)))})" for n, rs in claimed.items()
                 if len(set(rs)) > 1 and n not in by_name]
    if conflicts:
        raise TunableViewError(f"Role addresses give tensors different roles: {', '.join(conflicts)}; name each one's "
                               "role explicitly to settle it.")
    return {**{n: rs[0] for n, rs in claimed.items()}, **by_name}


def uncheckpointed_keys(roles: Mapping[str, str]) -> set[str]:
    """The tensor names whose role is never checkpointed (``artifact``)."""
    return {k for k, r in roles.items() if role_kind(r) in UNCHECKPOINTED_ROLE_KINDS}


# module-name segments that wrappers insert and that state dicts (DSD, activation checkpointing) leave out
_WRAPPER_SEGMENTS = ("_orig_mod.", "_checkpoint_wrapped_module.", "_fsdp_wrapped_module.")


def canonical_fqn(name: str) -> str:
    """``name`` without wrapper segments (``torch.compile``'s ``_orig_mod.``, activation checkpointing's).

    >>> canonical_fqn("_orig_mod.block._checkpoint_wrapped_module.weight")
    'block.weight'
    """
    for seg in _WRAPPER_SEGMENTS:
        name = name.replace(seg, "")
    return name


def artifact_state_keys(root: nn.Module, roles: Mapping[str, str], keys: Collection[str]) -> set[str]:
    """The state-dict ``keys`` that hold ``artifact`` tensors, matched by tensor identity (so every alias of a
    tied tensor counts) and by wrapper-free name (so keys from modules wrapped by ``torch.compile`` or by
    activation checkpointing match)."""
    artifact = uncheckpointed_keys(roles)
    if not artifact:
        return set()
    named = list(itertools.chain(root.named_parameters(remove_duplicate=False),
                                 root.named_buffers(remove_duplicate=False)))
    aliases: dict[int, list[str]] = {}
    for n, t in named:
        aliases.setdefault(id(t), []).append(n)
    ids = {id(t) for n, t in named if n in artifact}
    names = set(artifact) | {a for i in ids for a in aliases[i]}
    canonical = {canonical_fqn(n) for n in names}
    return {k for k in keys if canonical_fqn(k) in canonical}


def instrument_state_keys(root: nn.Module, roles: Mapping[str, str], keys: Collection[str]) -> set[str]:
    """Deprecated: :func:`artifact_state_keys` (the ``instrument`` role kind is now ``artifact``)."""
    deprecated("function", "instrument_state_keys", since="0.1", removed_in="0.2", replacement="`artifact_state_keys`")
    return artifact_state_keys(root, roles, keys)


def pair_groups(names: Sequence[str]) -> list[list[str]]:
    """Group ``names`` in pairs starting from the last, the order default schedules use.

    >>> pair_groups(["a", "b", "c"])
    [['c', 'b'], ['a']]
    """
    reversed_names = list(names)[::-1]
    return [reversed_names[i:i + 2] for i in range(0, len(reversed_names), 2)]


def _storage_key(t: Tensor) -> int:
    """What two tensors that share their storage have in common: the storage's address, or the tensor's identity for
    one with no storage of its own (a ``DTensor``, whose shards are local tensors inside it, or another wrapper)."""
    try:
        return t.untyped_storage().data_ptr()
    except RuntimeError:
        return id(t)


class ViewDefaults:
    """Every optional view member's default, for views that prefer to inherit them.

    Subclasses provide :attr:`root`, :meth:`support`, :meth:`to_physical` and :meth:`to_logical`.
    """

    root: nn.Module

    def support(self) -> TunableSupport:
        raise NotImplementedError

    def to_physical(self, logical: Sequence[str]) -> Mapping[str, tuple[str, ...]]:
        raise NotImplementedError

    def to_logical(self, physical: Sequence[str]) -> list[str]:
        raise NotImplementedError

    def logical_named_parameters(self) -> Iterator[tuple[str, Tensor]]:
        """The root's ``named_parameters()`` when the naming scheme is ``"physical"``.

        Raises:
            TunableViewError: For any other naming scheme, which must define this member.
        """
        scheme = self.support().naming_scheme
        if scheme != "physical":
            raise TunableViewError(
                f"A view whose naming scheme is {scheme!r} must define `logical_named_parameters`; the default serves"
                " only the 'physical' scheme."
            )
        return self.root.named_parameters()

    def coupled(self, physical: Sequence[str]) -> list[str]:
        """No implicit co-thaws."""
        return []

    def depth_groups(self) -> list[list[str]]:
        """Logical names in pairs, starting from the last registered one."""
        return pair_groups([n for n, _ in self.logical_named_parameters()])

    def roles(self) -> Mapping[str, str]:
        """Everything is ``base``."""
        return {}

    def checkpoint_state_dict(self) -> dict[str, Any]:
        """The root's ``state_dict()``, without ``artifact`` tensors (which are never checkpointed)."""
        state = self.root.state_dict()
        artifact = artifact_state_keys(self.root, self.roles(), state.keys())
        return {k: v for k, v in state.items() if k not in artifact}

    def load_checkpoint_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:
        """Load into the root. A strict load is strict by role: the keys missing from ``state_dict`` must be
        exactly the ``artifact`` keys, and it may hold no key the root lacks. It is checked before anything is
        loaded.

        Raises:
            RuntimeError: If a strict load's keys do not match, naming the keys (as ``load_state_dict`` does).
        """
        expected = set(self.root.state_dict())
        artifact = artifact_state_keys(self.root, self.roles(), expected)
        if not strict or not artifact:
            self.root.load_state_dict(state_dict, strict=strict)
            return
        missing = sorted(expected - set(state_dict) - artifact)
        unexpected = sorted(set(state_dict) - expected)
        if missing or unexpected:
            raise RuntimeError(f"Error(s) in loading state_dict for {type(self.root).__name__}: missing keys"
                               f" {missing}, unexpected keys {unexpected}.")
        self.root.load_state_dict(state_dict, strict=False)

    def structure(self) -> StructureProvider:
        """The physical structure of the root (:class:`~venngram.physical.PhysicalStructure`): its block list, and no
        components."""
        return PhysicalStructure(self.root)

    def promote(self, physical: Sequence[str], role: str) -> Mapping[str, str]:
        """Keep every name.

        A promoted parameter that shares its storage with another tensor (a tied weight, for example) is replaced by a
        frozen copy of itself, so training it leaves the other tensor unchanged; otherwise only the role changes.

        Raises:
            TunableViewError: If a promoted parameter is a tied ``DTensor``. The placement (FSDP2, tensor parallelism)
                holds the parameter it registered, so a copy in its place would be outside the placement.
        """
        holders: dict[int, set[str]] = {}
        for name, t in itertools.chain(self.root.named_parameters(remove_duplicate=False),
                                       self.root.named_buffers(remove_duplicate=False)):
            if t.device.type != "meta":
                holders.setdefault(_storage_key(t), set()).add(name)
        params = dict(self.root.named_parameters(remove_duplicate=False))
        tied = {n: sorted(others) for n in physical
                if params[n].device.type != "meta" and (others := holders[_storage_key(params[n])] - {n})}
        sharded = {n: others for n, others in tied.items() if type(params[n]).__name__ == "DTensor"}
        if sharded:
            raise TunableViewError(
                "A tied `DTensor` cannot be promoted, since the placement holds the parameter a copy would replace:"
                f" {sharded} (each with the names it shares its tensor with). Untie them before sharding the model.")
        for name in tied:
            owner, _, attr = name.rpartition(".")
            copy = nn.Parameter(params[name].detach().clone(), requires_grad=False)
            setattr(self.root.get_submodule(owner), attr, copy)
        return {n: n for n in physical}


class BoundView:
    """A view with every optional member filled in: the view's own member when it defines one, the protocol's
    default otherwise. Its support record is read once and cached.

    Args:
        view: The view to bind.
    """

    def __init__(self, view: Any) -> None:
        missing = [m for m in REQUIRED_MEMBERS if not hasattr(view, m)]
        if missing:
            raise TunableViewError(
                f"{type(view).__name__} is not a tunable view: it lacks {', '.join(f'`{m}`' for m in missing)}."
            )
        # a member set to a value (`roles = {...}`) would otherwise be silently replaced by its default
        not_callable = [m for m in (*REQUIRED_MEMBERS[1:], *OPTIONAL_MEMBERS)
                        if getattr(view, m, None) is not None and not callable(getattr(view, m))]
        if not_callable:
            raise TunableViewError(
                f"{type(view).__name__}'s {', '.join(f'`{m}`' for m in not_callable)} must be methods (or callables),"
                " not values.")
        self.view = view
        self._support: TunableSupport | None = None
        self._promoted: dict[str, str] = {}
        self._declared: dict[str, str] = {}

    @property
    def root(self) -> nn.Module:
        return self.view.root

    def support(self) -> TunableSupport:
        if self._support is None:
            support = self.view.support()
            if not isinstance(support, TunableSupport):
                raise TunableViewError(f"{type(self.view).__name__}.support() must return a `TunableSupport`, got"
                                            f" {type(support).__name__}.")
            self._support = support
        if self._declared:  # a cooperative declaration's role kinds are declared like the view's own
            return replace(self._support,
                           declared_roles=self._support.declared_roles | declared_kinds(self._declared))
        return self._support

    def to_physical(self, logical: Sequence[str]) -> Mapping[str, tuple[str, ...]]:
        return self.view.to_physical(logical)

    def to_logical(self, physical: Sequence[str]) -> list[str]:
        return self.view.to_logical(physical)

    def _member(self, name: str) -> Any:
        # the view's own member, or else the default bound to this view, so the defaults see the promoted roles
        if self.overrides(name):
            return getattr(self.view, name)
        return getattr(ViewDefaults, name).__get__(self)

    def logical_named_parameters(self) -> Iterator[tuple[str, Tensor]]:
        return self._member("logical_named_parameters")()

    def coupled(self, physical: Sequence[str]) -> list[str]:
        return list(self._member("coupled")(physical))

    def depth_groups(self) -> list[list[str]]:
        return [list(g) for g in self._member("depth_groups")()]

    def roles(self) -> Mapping[str, str]:
        """The view's roles, merged with the cooperatively declared ones (see :meth:`declare_roles`) and the roles of
        tensors promoted since (see :meth:`promote`)."""
        return {**self._declared, **self.view_roles(), **self._promoted}

    def declare_roles(self, roles: Mapping[str, str]) -> dict[str, str]:
        """Declare roles for tensors the view does not assign itself (the module docstring states the rule).

        Returns:
            The roles declared on this view so far.

        Raises:
            RoleError: If a role is invalid or a key names neither a parameter nor a buffer of the root.
            TunableViewError: If a declared role differs from the role the view assigns that tensor (every conflict
                is named, with both roles), or from one declared earlier; nothing is declared then.
        """
        roles = validate_roles(roles, (n for n, _ in self.root.named_parameters()),
                               (n for n, _ in self.root.named_buffers()))
        assigned = {**self.view_roles(), **self._promoted}
        conflicts = [f"{n!r}: the view assigns {assigned[n]!r}, declared {r!r}" for n, r in roles.items()
                     if n in assigned and assigned[n] != r]
        conflicts += [f"{n!r}: declared {self._declared[n]!r} earlier, now {r!r}" for n, r in roles.items()
                      if n in self._declared and self._declared[n] != r]
        if conflicts:
            raise TunableViewError(f"{type(self.view).__name__} refuses these role declarations, which conflict with "
                                   f"roles it already has: {'; '.join(conflicts)}.")
        self._declared.update({n: r for n, r in roles.items() if n not in assigned})
        return dict(self._declared)

    @property
    def declared(self) -> dict[str, str]:
        """The roles declared cooperatively (see :meth:`declare_roles`) on tensors the view does not assign."""
        return dict(self._declared)

    def view_roles(self) -> dict[str, str]:
        """The roles the view itself reports, without promotions."""
        return dict(self._member("roles")())

    @property
    def promoted(self) -> dict[str, str]:
        """The tensors promoted through :meth:`promote`, mapped to their new roles."""
        return dict(self._promoted)

    def checkpoint_state_dict(self) -> dict[str, Any]:
        return self._member("checkpoint_state_dict")()

    def load_checkpoint_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:
        self._member("load_checkpoint_state_dict")(state_dict, strict=strict)

    def structure(self) -> StructureProvider:
        return self._member("structure")()

    def promote(self, physical: Sequence[str], role: str) -> Mapping[str, str]:
        """Promote ``physical`` to ``role`` through the view (or the default), and record the new role.

        Raises:
            TunableViewError: If ``role`` is not a ``latent:<name>`` role, or the view renamed a promoted tensor (a
                promotion keeps names in this version of the protocol).
        """
        if not isinstance(role, str) or parse_role(role)[0] != "latent":
            raise TunableViewError(f"Tensors can be promoted only to a `latent:<name>` role, got {role!r}.")
        mapping = dict(self._member("promote")(physical, role))
        renamed = {old: new for old, new in mapping.items() if old != new}
        if renamed:
            raise TunableViewError(
                f"{type(self.view).__name__}.promote renamed the tensors it promoted ({renamed}); a promotion must keep"
                " each tensor's name in this version of the protocol.")
        self._promoted.update(dict.fromkeys(physical, role))
        return mapping

    def defines(self, member: str) -> bool:
        """Whether the bound view has ``member`` (its own or inherited from :class:`ViewDefaults`)."""
        return callable(getattr(self.view, member, None))

    def overrides(self, member: str) -> bool:
        """Whether the bound view implements ``member`` differently from the protocol's default (on its class or on
        the instance), so that its own implementation is the one used."""
        fn = getattr(self.view, member, None)
        return callable(fn) and getattr(fn, "__func__", None) is not getattr(ViewDefaults, member)


def as_view(view: Any) -> BoundView:
    """Bind ``view`` (returning it unchanged if it is already bound)."""
    return view if isinstance(view, BoundView) else BoundView(view)


def resolve_view(model: nn.Module, view: Any | None = None, *, roles: Mapping[str, str] | None = None) -> BoundView:
    """Find the view for ``model``: ``view`` when given, else a view attached as ``model.tunable_view``, else
    :class:`CanonicalView` over ``model``. ``roles`` are declared cooperatively on it (:meth:`BoundView.declare_roles`).

    Raises:
        TunableViewError: If the chosen object is not a tunable view, or a declared role conflicts with the view's.
    """
    if view is None:
        view = getattr(model, VIEW_ATTR, None)
    if view is None:
        view = CanonicalView(model)
    bound = as_view(view)
    if roles:
        bound.declare_roles(roles)
    return bound


class CanonicalView(ViewDefaults):
    """The identity view over ``root.named_parameters()``.

    A parametrized tensor (``torch.nn.utils.parametrize``) is also reachable by its unparametrized name:
    ``layer.weight`` resolves to ``layer.parametrizations.weight.original`` (or to every ``original<i>`` of a
    multi-tensor parametrization).

    Args:
        root: The module to view.
        roles: Optional roles for the root's parameters and buffers (anything unlisted is ``base``): tensor names
            mapped to roles, or roles mapped to addresses resolved against this view's names and structure (see
            "Roles by address" above, and :func:`expand_roles`).
        sets: Named sets the role addresses may reference: a mapping, or an address book.

    Raises:
        RoleKeyError: If a role key names neither a tensor nor a role (a :class:`~venngram.errors.RoleError` and a
            :class:`TunableViewError` both).
        TunableViewError: If a role is invalid, or a role address does not resolve or gives a tensor two roles.
    """

    def __init__(self, root: nn.Module, roles: Mapping[str, Any] | None = None, *, sets: Any = None,
                 _base_roles: Mapping[str, str] | None = None) -> None:
        self.root = root
        names = [n for n, _ in root.named_parameters()] + [n for n, _ in root.named_buffers()]
        expanded = expand_roles(roles or {}, names, self.structure() if roles else None, sets=sets)
        self._roles = validate_roles({**(_base_roles or {}), **expanded}, (n for n, _ in root.named_parameters()),
                                     (n for n, _ in root.named_buffers()))

    def support(self) -> TunableSupport:
        return TunableSupport(trainable=True, naming_scheme="physical", declared_roles=declared_kinds(self._roles))

    def _parametrized(self) -> dict[str, tuple[str, ...]]:
        """Unparametrized names of parametrized tensors mapped to their underlying physical names."""
        out: dict[str, list[str]] = {}
        for name, _ in self.root.named_parameters():
            if name.startswith("parametrizations."):  # a parametrization on the root module itself
                head, tail = "", name[len("parametrizations."):]
            else:
                head, sep, tail = name.partition(".parametrizations.")
                if not sep:
                    continue
            tensor_name, _, original = tail.partition(".")
            if original.startswith("original"):
                logical = f"{head}.{tensor_name}" if head else tensor_name
                out.setdefault(logical, []).append(name)
        return {k: tuple(v) for k, v in out.items()}

    def to_physical(self, logical: Sequence[str]) -> Mapping[str, tuple[str, ...]]:
        physical = {n for n, _ in self.root.named_parameters()}
        parametrized = self._parametrized()
        resolved: dict[str, tuple[str, ...]] = {}
        unresolved = []
        for name in logical:
            if name in physical:
                resolved[name] = (name,)
            elif name in parametrized:
                resolved[name] = parametrized[name]
            else:
                unresolved.append(name)
        if unresolved:
            raise Unresolvable(unresolved)
        return resolved

    def to_logical(self, physical: Sequence[str]) -> list[str]:
        return list(dict.fromkeys(physical))

    def roles(self) -> Mapping[str, str]:
        return dict(self._roles)


__all__ = [
    "as_view",
    "BoundView",
    "canonical_fqn",
    "CanonicalView",
    "expand_roles",
    "artifact_state_keys",
    "instrument_state_keys",
    "OPTIONAL_MEMBERS",
    "pair_groups",
    "REQUIRED_MEMBERS",
    "resolve_view",
    "TunableSupport",
    "TunableView",
    "TunableViewExtras",
    "uncheckpointed_keys",
    "VIEW_ATTR",
    "ViewDefaults",
]
