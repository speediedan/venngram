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

A tunable view composed from maps, built from data alone: the identity view (:class:`~venngram.tuning.CanonicalView`),
the structure of the model's base map (:class:`~venngram.maps.structure.MapStructure`), and the roles of every overlay
map that applies to it (:mod:`~venngram.maps.overlay`). So ``role:adapter & component:mlp.* & layers:-4:`` means the
same tensors on a Llama, a Gemma and an OPT model with PEFT's LoRA, although their modules are named differently, and
neither the architecture nor the adapter library needs code of its own.

:func:`venngram.view` is the one call: ``view(model)`` finds the base map from the model's class and the overlays from
the modules it carries. ``MappedView(model, maps=[...])`` names the maps to use instead.

.. warning:: This is an experimental API and may change in a future release.
"""
from collections.abc import Sequence

from torch import nn

from venngram.maps.check import submodule
from venngram.maps.overlay import OverlayMap, overlays_for
from venngram.maps.schema import ComponentMap, known_architectures
from venngram.maps.structure import find_mounts, map_for, MapStructure
from venngram.tuning.errors import TunableViewError
from venngram.tuning.views import CanonicalView


def overlay_roles(root: nn.Module, overlays: Sequence[OverlayMap]) -> dict[str, str]:
    """The roles ``overlays`` give ``root``'s tensors.

    Raises:
        TunableViewError: If two overlays give one tensor different roles.
    """
    roles: dict[str, str] = {}
    origin: dict[str, str] = {}
    for overlay in overlays:
        for name, role in overlay.roles_for(root).items():
            if name in roles and roles[name] != role:
                raise TunableViewError(f"overlays {origin[name]!r} and {overlay.name!r} both apply to this model and "
                                       f"give {name!r} different roles ({roles[name]!r}, {role!r}); name the maps to "
                                       "use with `maps=`.")
            roles.setdefault(name, role)
            origin.setdefault(name, overlay.name)
    return roles


def _base_map(root: nn.Module, mount: str | None) -> tuple[ComponentMap, str | None]:
    """The base map for ``root``, or for the submodule it is mounted at: named by ``mount``, else ``root`` itself, else
    the one outermost descendant some known map is written for."""
    if mount:
        target = submodule(root, mount)
        if target is None:
            raise TunableViewError(f"{type(root).__name__} has no module {mount!r} to mount a base map at.")
        return map_for(target), mount
    try:
        return map_for(root), mount
    except KeyError:
        if mount == "":
            raise
        mounts = find_mounts(root, known_architectures())
        if len(mounts) > 1:
            raise TunableViewError(f"{type(root).__name__} carries {len(mounts)} mapped models ({mounts}); name the "
                                   "one this view describes with `mount=`.") from None
        if not mounts:
            raise
        return map_for(root.get_submodule(mounts[0])), mounts[0]


def _split(root: nn.Module, maps: ComponentMap | OverlayMap | Sequence[ComponentMap | OverlayMap] | None,
           mount: str | None = None) -> tuple[ComponentMap, list[OverlayMap], str | None]:
    if maps is None:
        cmap, mount = _base_map(root, mount)
        return cmap, overlays_for(root), mount
    listed = [maps] if isinstance(maps, (ComponentMap, OverlayMap)) else list(maps)
    bases = [m for m in listed if isinstance(m, ComponentMap)]
    if len(bases) > 1:
        raise TunableViewError(f"`maps` names {len(bases)} base maps ({[b.architecture for b in bases]}); a view has "
                               "one base map and any number of overlays.")
    overlays = [m for m in listed if isinstance(m, OverlayMap)]
    if bases:
        return bases[0], overlays, mount
    cmap, mount = _base_map(root, mount)
    return cmap, overlays, mount


class MappedView(CanonicalView):
    """The identity view of ``root`` with the structure of its base map and the roles of its overlay maps.

    Args:
        root: The model.
        maps: The maps to use instead of lookup: a base map, overlays, or both. ``None`` (the default) looks the base
            map up from the model (:func:`~venngram.maps.structure.map_for`) and applies every known overlay that
            applies to it (:func:`~venngram.maps.overlay.overlays_for`). When only overlays are given, the base map
            is still looked up, and exactly those overlays apply.
        roles: Explicit roles, applied over the ones the overlays give.
        prefix: The leading module path the mapped model lacks relative to the base map; found from the blocks when
            ``None``.
        mount: The path of the submodule of ``root`` holding the mapped model, for a root that wraps it (a framework
            module carrying a Hugging Face model under ``model.``, with its own modules beside it). ``None`` (the
            default) uses ``root`` when a map describes it, and otherwise the one outermost descendant one does; ``""``
            uses ``root`` only. The view's names stay ``root``'s, overlays and ``roles`` apply across all of ``root``,
            and tensors outside the mount have no layer or component.

    Raises:
        KeyError: If no base map is found for the model or a descendant of it.
        TunableViewError: If a role is invalid, two overlays disagree on a tensor's role, or ``maps`` names two base
            maps, or ``root`` carries several mapped models and ``mount`` names none.
    """

    def __init__(self, root: nn.Module, maps: ComponentMap | OverlayMap | Sequence[ComponentMap | OverlayMap] |
                 None = None, *, roles: dict[str, str] | None = None, prefix: str | None = None,
                 mount: str | None = None) -> None:
        cmap, overlays, mount = _split(root, maps, mount)
        super().__init__(root, roles={**overlay_roles(root, overlays), **(roles or {})})
        self.cmap = cmap
        self.overlays = tuple(overlays)
        self._structure = MapStructure(root, cmap, prefix, mount)

    def structure(self) -> MapStructure:
        return self._structure


__all__ = ["MappedView", "overlay_roles"]
