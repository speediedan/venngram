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
venngram
^^^^^^^^

A set-algebraic protocol for addressing model slices: tensors (for tuning), activation sites (for steering and
capture), and units inside either (neurons, latent features, featurizer blocks, subspaces).

The name joins *Venn*, for the set algebra (unions, intersections, differences, overlapping regions), and *engram*,
neuroscience's term for the physical trace of a memory or function in the brain: a venngram is a set-algebraic
specification of the neural traces it targets.

- :mod:`venngram.address`: the address record, its string form and its algebra;
- :mod:`venngram.book`: address books, the shareable documents of named sets;
- :mod:`venngram.units`: large unit sets as sidecar files an address book names;
- :mod:`venngram.recipe`: recipes, treatments over a book's sets, and their composition;
- :mod:`venngram.manifest`: the manifest (``venngram.yaml``) that lists a directory's documents;
- :mod:`venngram.store`: document references, and the stores (a directory, Hub repositories and buckets) they name;
- :mod:`venngram.sources`: map sources and formats, which turn something outside venngram into venngram documents;
- :mod:`venngram.resolve`: resolution over a model's tensors, and the provider interfaces;
- :mod:`venngram.physical`: the structure provider every PyTorch model gets;
- :mod:`venngram.maps`: per-architecture component maps, the structure provider and the generic view built from them;
- :mod:`venngram.roles`: the role vocabulary (what a tensor *is*, where an address says *which*);
- :mod:`venngram.tuning`: the tunable view protocol, through which a fine-tuning consumer sees a model.

.. warning:: This is an experimental API and may change in a future release.
"""
from venngram.__about__ import __version__
from venngram.address import is_address, LayerSet, parse, to_mapping, to_string, UnitSet
from venngram.book import AddressBook, diff_lock, make_lock, register_uri_resolver
from venngram.errors import AddressError, AddressUnresolvable, Refusal, RoleError, VenngramError
from venngram.resolve import Location, ParamSpace, select, StructureProvider

__all__ = [
    "__version__",
    "AddressBook",
    "AddressError",
    "AddressUnresolvable",
    "diff_lock",
    "is_address",
    "LayerSet",
    "Location",
    "make_lock",
    "ParamSpace",
    "parse",
    "Refusal",
    "register_uri_resolver",
    "RoleError",
    "select",
    "StructureProvider",
    "to_mapping",
    "to_string",
    "UnitSet",
    "VenngramError",
]
