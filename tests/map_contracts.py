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
"""Document-kind contracts shared by synthetic cases and sampled registry implementations.

Callers supply independent expected locations or roles. A successful self-round-trip alone cannot show that a map
labels the right tensors. These contracts need no library-specific names, network access, weights or forward pass.
"""
from collections.abc import Mapping

from torch import nn

from venngram import Location
from venngram.maps import check_map_against_model, ComponentMap, MapStructure, overlay_roles, OverlayMap
from venngram.maps.verification import check_partition


def base_map_contract(cmap: ComponentMap, model: nn.Module, *, expected: Mapping[str, Location]) -> None:
    """Rows exist, parameters are covered without duplicate module claims, and independent locations agree."""
    assert expected, "The base-map contract needs independent expected locations"
    assert list(model.named_parameters()), "An empty model checks nothing"
    problems = check_map_against_model(cmap, model)
    assert not problems, "\n".join(str(p) for p in problems)
    partition = check_partition(cmap, model)
    assert partition.status == "passed", "\n".join(str(p) for p in partition.problems)
    structure = MapStructure(model, cmap)
    params = dict(model.named_parameters())
    for name, location in expected.items():
        assert name in params, f"Expected tensor {name!r} does not exist"
        assert structure.locate(name) == location, f"Wrong location for {name!r}: {structure.locate(name)}"


def overlay_contract(overlay: OverlayMap, model: nn.Module, *, applies: bool,
                     expected: Mapping[str, str]) -> None:
    """Detection agrees with an independent oracle, and exactly the admitted tensors get each keyed role.

    Also checks a plain model as a negative control; an always-applicable overlay must not pass.
    """
    assert overlay.applies_to(model) is applies, f"Wrong applies_when result for {overlay.name}"
    assert not overlay.applies_to(nn.Linear(3, 2)), "Overlay admits the negative control"
    assert not overlay_roles(nn.Linear(3, 2), [overlay]), "Overlay assigns roles on the negative control"
    if applies:
        assert expected, "An applicable sample must exercise some roles"
    else:
        assert not expected, "A non-applicable sample cannot expect roles"
    assert overlay_roles(model, [overlay]) == dict(expected), f"Wrong roles for {overlay.name}"
