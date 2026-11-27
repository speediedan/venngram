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
"""The public API is what each module's ``__all__`` lists, and every error venngram raises is a VenngramError."""
import importlib
import pkgutil

import pytest

import venngram
from venngram.errors import AddressUnresolvable, VenngramError
from venngram.maps.check import ComponentMapModelMismatch
from venngram.tuning import TunableViewError, Unresolvable

#: Modules that hold no API (the version module is read by the build).
NO_API = ("venngram.__about__", "venngram.__main__")
MODULES = sorted(m.name for m in pkgutil.walk_packages(venngram.__path__, "venngram.") if m.name not in NO_API)


@pytest.mark.parametrize("name", MODULES)
def test_every_module_declares_its_public_names_and_each_exists(name):
    module = importlib.import_module(name)
    public = getattr(module, "__all__", None)
    assert isinstance(public, list) and public, f"{name} declares no `__all__`: its public names are unstated"
    missing = [n for n in public if not hasattr(module, n)]
    assert not missing, f"{name}.__all__ lists names it does not define: {missing}"
    assert len(set(public)) == len(public), f"{name}.__all__ repeats a name"


def test_the_retired_recipe_module_is_gone_and_inline_limit_stays_in_operands():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("venngram.recipe")
    from venngram.operands import INLINE_LIMIT

    assert INLINE_LIMIT > 0


def test_a_map_that_does_not_describe_the_model_is_a_venngram_error():
    assert issubclass(ComponentMapModelMismatch, VenngramError) and issubclass(ComponentMapModelMismatch, ValueError)


def test_a_view_that_cannot_resolve_names_is_also_an_unresolvable_address():
    err = Unresolvable(["blocks.0.missing", "head.w"], detail="through the view")
    assert isinstance(err, TunableViewError) and isinstance(err, AddressUnresolvable)
    assert err.names == ["blocks.0.missing", "head.w"] and err.address == "blocks.0.missing, head.w"
    assert [r.reason for r in err.refusals] == ["`blocks.0.missing` does not resolve to any parameter",
                                                "`head.w` does not resolve to any parameter"]
    assert str(err) == ("These names do not resolve to any parameter: ['blocks.0.missing', 'head.w']. "
                        "through the view")
    with pytest.raises(AddressUnresolvable):
        raise err
