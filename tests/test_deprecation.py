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
"""One deprecation cycle and strict mode for everything venngram retires."""
import os
import subprocess
import sys
import warnings

import pytest

from venngram.deprecation import deprecated, STRICT_ENV, strict, VenngramDeprecationWarning
from venngram.maps.schema import ComponentMapDeprecationWarning
from venngram.recipe import RecipeSchemaWarning


def test_a_deprecation_warns_once_naming_its_replacement():
    with pytest.warns(VenngramDeprecationWarning, match=r"the function `old_fn` is deprecated since venngram 0\.1 "
                                                        r"and is refused from 0\.2; use new_fn"):
        deprecated("function", "old_fn", since="0.1", removed_in="0.2", replacement="new_fn")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        deprecated("function", "old_fn", since="0.1", removed_in="0.2", replacement="new_fn")  # once per process


def test_strict_mode_turns_deprecations_into_errors():
    with strict(), pytest.raises(VenngramDeprecationWarning, match="old_name"):
        deprecated("component name", "old_name", since="0.1", removed_in="0.2")
    with strict(), pytest.raises(ComponentMapDeprecationWarning):
        warnings.warn("a retired map", ComponentMapDeprecationWarning)


def test_existing_warnings_join_the_cycle_without_changing_who_sees_them():
    assert issubclass(ComponentMapDeprecationWarning, VenngramDeprecationWarning)
    assert issubclass(ComponentMapDeprecationWarning, UserWarning)  # still shown by default
    assert issubclass(RecipeSchemaWarning, VenngramDeprecationWarning)
    assert issubclass(RecipeSchemaWarning, DeprecationWarning)  # still hidden by default outside __main__


def test_the_environment_switch_applies_to_the_whole_process():
    code = ("from venngram.deprecation import deprecated\n"
            "deprecated('function', 'f', since='0.1', removed_in='0.2')\n")
    env = {**os.environ, STRICT_ENV: "1"}
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "VenngramDeprecationWarning" in result.stderr
    env[STRICT_ENV] = "0"
    assert subprocess.run([sys.executable, "-W", "ignore", "-c", code], env=env).returncode == 0
