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
``stack`` resolve, and a tunable view that needs no per-model code. A map is a data document that can describe any
PyTorch module tree. Maps are found through the registries (:mod:`venngram.registry`), which index seeds for common
architectures and maps from any other source; a map for any other model is written by hand (or derived from another
source) and registered in process, or published and indexed.

- :mod:`~venngram.maps.schema`: the document schema, its reader, the vocabulary and the registry of maps;
- :mod:`~venngram.maps.check`: checking a map against a model (every row's module exists where it says);
- :mod:`~venngram.maps.structure`: :class:`~venngram.maps.structure.MapStructure`, the map-backed structure provider;
- :mod:`~venngram.maps.rules`: what verification expects of an architecture, as layered data documents;
- :mod:`~venngram.maps.verification`: verifying a map against an architecture without weights (partition, widths
  from the configuration, forward order), and its verification record;
- :mod:`~venngram.maps.role_rules`: role rules, the data documents naming an adapter library's tensors and their roles,
  found like maps and detected from a model;
- :mod:`~venngram.maps.view`: :class:`~venngram.maps.view.MappedView`, the generic view over a map and role rules;
- :mod:`~venngram.maps.transformer_lens`: one source of seed maps, deriving a map from a TransformerLens architecture
  adapter (read duck-typed: venngram does not depend on TransformerLens);
- :mod:`~venngram.maps.drafter`: a source that needs no library, drafting a map from a model's own module tree for a
  person or an agent to confirm.

Maps are data: loading one runs no code.

.. warning:: This is an experimental API and may change in a future release.
"""
from venngram.maps.check import (check_map_against_model, ComponentMapModelMismatch, MapProblem,
                                 validate_map_against_model)
from venngram.maps.schema import (component_map_for, ComponentMap, from_document, known_architectures,
                                  load_component_map, register, registered_architectures, unregister, VOCABULARY)
from venngram.maps.structure import map_for, map_structure, MapStructure
from venngram.maps.verification import VerificationReport, verify
from venngram.maps.role_rules import (detect_role_rules, known_role_rules, load_role_rules, register_role_rules,
                                      registered_role_rules, role_rules_for, role_rules_from_document, RoleRules,
                                      unregister_role_rules)
from venngram.maps.view import MappedView, resolve_roles

__all__ = [
    "check_map_against_model",
    "component_map_for",
    "detect_role_rules",
    "ComponentMap",
    "ComponentMapModelMismatch",
    "from_document",
    "known_architectures",
    "known_role_rules",
    "load_component_map",
    "load_role_rules",
    "map_for",
    "map_structure",
    "MappedView",
    "MapProblem",
    "MapStructure",
    "register",
    "register_role_rules",
    "registered_architectures",
    "registered_role_rules",
    "resolve_roles",
    "role_rules_for",
    "role_rules_from_document",
    "RoleRules",
    "unregister",
    "unregister_role_rules",
    "validate_map_against_model",
    "VerificationReport",
    "verify",
    "VOCABULARY",
]
