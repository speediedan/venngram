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

The tunable view protocol: the contract between the logical parameter names a consumer uses (a fine-tuning schedule,
for example) and the physical parameters an optimizer, checkpoint and sharding see, with a lifecycle role
(:mod:`venngram.roles`) for each tensor. Fine-Tuning Scheduler is one consumer of it.

- :mod:`~venngram.tuning.views`: the :class:`~venngram.tuning.views.TunableView` protocol, its support record, the
  defaults of its optional members and the identity view;
- :mod:`~venngram.tuning.errors`: the errors the protocol raises;
- :mod:`~venngram.tuning.conformance`: the conformance cases every implementation passes, whatever consumes it.

The subpackage imports ``torch``, the standard library and :mod:`venngram`'s own modules only.

.. warning:: This is an experimental API and may change in a future release.
"""
from .conformance import ConformanceCase, ConformanceError, ConformanceReport, ViewConformance
from .errors import TunableViewError, Unresolvable
from venngram.roles import ROLE_KINDS
from .views import (as_view, BoundView, CanonicalView, resolve_view, TunableSupport, TunableView, TunableViewExtras,
                    VIEW_ATTR, ViewDefaults)

__all__ = [
    "as_view",
    "BoundView",
    "CanonicalView",
    "ConformanceCase",
    "ConformanceError",
    "ConformanceReport",
    "resolve_view",
    "ROLE_KINDS",
    "TunableSupport",
    "TunableView",
    "TunableViewError",
    "TunableViewExtras",
    "Unresolvable",
    "VIEW_ATTR",
    "ViewConformance",
    "ViewDefaults",
]
