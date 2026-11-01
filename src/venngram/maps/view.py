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
The Mapped View
^^^^^^^^^^^^^^^

A tunable view for any architecture with a component map, built from data alone: the identity view
(:class:`~venngram.tuning.CanonicalView`), the map's structure (:class:`~venngram.maps.structure.MapStructure`), and
roles from role rules. So ``component:mlp.* & layers:-4:`` means the same tensors on a Llama, a Gemma and an OPT model
although their modules are named differently, and a mapped architecture needs no view class of its own.

Roles come from role rules (:mod:`venngram.maps.role_rules`), data documents naming an adapter library's conventions
rather than an architecture's. By default the view applies every known rule set whose ``applies_when`` holds for the
model, so ``MappedView(model)`` finds both the map (from the model's class) and the roles (from the adapter modules
present); ``role_rules`` names a rule set, or gives one inline, instead.

.. warning:: This is an experimental API and may change in a future release.
"""
from collections.abc import Mapping

from torch import nn

from venngram.maps.role_rules import detect_role_rules, role_rules_for, RoleRules, rule_roles
from venngram.maps.schema import ComponentMap
from venngram.maps.structure import map_for, MapStructure
from venngram.tuning.errors import TunableViewError
from venngram.tuning.views import CanonicalView

def resolve_roles(root: nn.Module, role_rules: str | Mapping[str, str] | RoleRules | None = None) -> dict[str, str]:
    """The roles role rules give ``root``'s tensors.

    Args:
        root: The model.
        role_rules: A rule set's name (looked up with :func:`~venngram.maps.role_rules.role_rules_for`), a rule set,
            or a module pattern -> role mapping; ``None`` applies every known rule set that applies to ``root``
            (:func:`~venngram.maps.role_rules.detect_role_rules`). An empty mapping gives no roles.

    Raises:
        KeyError: If a named rule set is not found.
        TunableViewError: If two detected rule sets give one tensor different roles.
    """
    if isinstance(role_rules, str):
        return role_rules_for(role_rules).roles(root)
    if isinstance(role_rules, RoleRules):
        return role_rules.roles(root)
    if role_rules is not None:
        return rule_roles(root, role_rules)
    roles: dict[str, str] = {}
    origin: dict[str, str] = {}
    for rules in detect_role_rules(root):
        for name, role in rules.roles(root).items():
            if name in roles and roles[name] != role:
                raise TunableViewError(f"role rules {origin[name]!r} and {rules.name!r} both apply to this model and "
                                       f"give {name!r} different roles ({roles[name]!r}, {role!r}); name the rule set "
                                       "to use with `role_rules=`.")
            roles.setdefault(name, role)
            origin.setdefault(name, rules.name)
    return roles


class MappedView(CanonicalView):
    """The identity view of ``root`` with the component structure of its map and roles from role rules.

    Args:
        root: The model.
        cmap: Its component map; looked up from ``architecture`` or the model (:func:`~venngram.maps.map_for`) when
            ``None``.
        architecture: The architecture whose map to use, when the model's own class does not name it.
        roles: Explicit roles, applied over the ones ``role_rules`` give.
        role_rules: A rule set's name (``"peft.lora"``), a rule set, or a module pattern -> role mapping; ``None``
            (the default) applies every known rule set that applies to the model (:func:`resolve_roles`).
        prefix: The leading module path ``root`` lacks relative to the map; found from the blocks when ``None``.

    Raises:
        KeyError: If no map is found for the model, or a named rule set is not found.
        TunableViewError: If a role is invalid, or two detected rule sets disagree on a tensor's role.
    """

    def __init__(self, root: nn.Module, cmap: ComponentMap | None = None, *, architecture: str | None = None,
                 roles: Mapping[str, str] | None = None,
                 role_rules: str | Mapping[str, str] | RoleRules | None = None, prefix: str | None = None) -> None:
        super().__init__(root, roles={**resolve_roles(root, role_rules), **(roles or {})})
        self.cmap = cmap if cmap is not None else map_for(root, architecture)
        self._structure = MapStructure(root, self.cmap, prefix)

    def structure(self) -> MapStructure:
        return self._structure


__all__ = ["MappedView", "resolve_roles", "rule_roles"]
