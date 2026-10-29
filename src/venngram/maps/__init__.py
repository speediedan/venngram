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
Component Maps
^^^^^^^^^^^^^^

Per-architecture component maps, and what is built from them: a vocabulary structure provider, so ``component`` and
``stack`` resolve, and a tunable view that needs no per-model code.

- :mod:`~venngram.maps.schema`: the document schema, its reader, the vocabulary and the registry of maps;
- :mod:`~venngram.maps.check`: checking a map against a model (every row's module exists where it says);
- :mod:`~venngram.maps.structure`: :class:`~venngram.maps.structure.MapStructure`, the map-backed structure provider;
- :mod:`~venngram.maps.view`: :class:`~venngram.maps.view.MappedView`, the generic view over it;
- :mod:`~venngram.maps.transformer_lens`: deriving a map from a TransformerLens architecture adapter.

Maps are data: loading one runs no code.

.. warning:: This is an experimental API and may change in a future release.
"""
from venngram.maps.check import (check_map_against_model, ComponentMapModelMismatch, MapProblem,
                                 validate_map_against_model)
from venngram.maps.schema import (component_map_for, ComponentMap, from_document, known_architectures,
                                  load_component_map, register, VOCABULARY)
from venngram.maps.structure import map_for, map_structure, MapStructure
from venngram.maps.view import MappedView, PEFT_LORA_RULES

__all__ = [
    "check_map_against_model",
    "component_map_for",
    "ComponentMap",
    "ComponentMapModelMismatch",
    "from_document",
    "known_architectures",
    "load_component_map",
    "map_for",
    "map_structure",
    "MappedView",
    "MapProblem",
    "MapStructure",
    "PEFT_LORA_RULES",
    "register",
    "validate_map_against_model",
    "VOCABULARY",
]
