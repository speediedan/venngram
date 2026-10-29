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

Role rules name an adapter library's conventions rather than an architecture's: a rule maps a module pattern (matched
as :func:`~venngram.resolve.module_matches` does, on whole segments) to a role. :data:`PEFT_LORA_RULES` holds PEFT's
LoRA names.

.. warning:: This is an experimental API and may change in a future release.
"""
import itertools
from collections.abc import Mapping

from torch import nn

from venngram.maps.schema import ComponentMap
from venngram.maps.structure import map_for, MapStructure
from venngram.resolve import leaf_matches, module_matches
from venngram.tuning.views import CanonicalView

#: PEFT's LoRA tensors (``lora_A``, ``lora_B``, the embedding variants, and DoRA's magnitude vector) as adapters.
PEFT_LORA_RULES: Mapping[str, str] = {
    "lora_A": "adapter", "lora_B": "adapter", "lora_embedding_A": "adapter", "lora_embedding_B": "adapter",
    "lora_magnitude_vector": "adapter",
}


def rule_roles(root: nn.Module, rules: Mapping[str, str]) -> dict[str, str]:
    """The role of each of ``root``'s parameters and buffers that a rule matches (the first matching rule wins).

    A pattern matches the tensors under a module of that name and, as a leaf, a tensor registered under that name
    itself (DoRA's ``lora_magnitude_vector`` is a parameter, not a module, in some PEFT versions).
    """
    roles: dict[str, str] = {}
    for name, _ in itertools.chain(root.named_parameters(), root.named_buffers()):
        for pattern, role in rules.items():
            if module_matches(pattern, name) or leaf_matches(pattern, name):
                roles[name] = role
                break
    return roles


class MappedView(CanonicalView):
    """The identity view of ``root`` with the component structure of its map and roles from role rules.

    Args:
        root: The model.
        cmap: Its component map; looked up from ``architecture`` or the model (:func:`~venngram.maps.map_for`) when
            ``None``.
        architecture: The architecture whose map to use, when the model's own class does not name it.
        roles: Explicit roles, applied over the ones ``role_rules`` give.
        role_rules: Module pattern -> role (:data:`PEFT_LORA_RULES`, for example).
        prefix: The leading module path ``root`` lacks relative to the map; found from the blocks when ``None``.

    Raises:
        KeyError: If no map is found for the model.
        TunableViewError: If a role is invalid.
    """

    def __init__(self, root: nn.Module, cmap: ComponentMap | None = None, *, architecture: str | None = None,
                 roles: Mapping[str, str] | None = None, role_rules: Mapping[str, str] | None = None,
                 prefix: str | None = None) -> None:
        super().__init__(root, roles={**rule_roles(root, role_rules or {}), **(roles or {})})
        self.cmap = cmap if cmap is not None else map_for(root, architecture)
        self._structure = MapStructure(root, self.cmap, prefix)

    def structure(self) -> MapStructure:
        return self._structure


__all__ = ["MappedView", "PEFT_LORA_RULES", "rule_roles"]
