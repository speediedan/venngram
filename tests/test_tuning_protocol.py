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
"""venngram imports only the standard library, ``torch`` and ``yaml``, and its core (addresses, books, roles,
resolution) imports no ``torch`` at all."""
import ast
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import venngram
from venngram import tuning as tv
from venngram.errors import RoleError, VenngramError

PKG_DIR = Path(venngram.__file__).parent
ALLOWED = {"torch", "yaml", "venngram"}
TORCH_FREE = ("__init__.py", "__about__.py", "address.py", "book.py", "errors.py", "resolve.py", "roles.py")


def _top_level_imports(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            found.add((node.module or "").split(".")[0])
    return found


@pytest.mark.parametrize("module", sorted(str(p.relative_to(PKG_DIR)) for p in PKG_DIR.rglob("*.py")))
def test_imports_only_stdlib_torch_and_yaml(module):
    outside = sorted(m for m in _top_level_imports(PKG_DIR / module) if m not in ALLOWED | set(sys.stdlib_module_names))
    assert not outside, f"{module} imports {outside}"


def test_the_core_imports_without_torch():
    script = ("import sys; sys.modules['torch'] = None\n"
              "import venngram; from venngram import select, ParamSpace, AddressBook\n"
              "assert venngram.select('role:adapter', ParamSpace(['a.w'], roles={'a.w': 'adapter'})) == ['a.w']\n"
              "print('CORE-OK')")
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
                          cwd=str(PKG_DIR.parent))
    assert proc.returncode == 0 and "CORE-OK" in proc.stdout, proc.stderr
    for module in TORCH_FREE:
        assert "torch" not in _top_level_imports(PKG_DIR / module), module


def test_the_tunable_view_protocol_conforms():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    setattr(model, tv.VIEW_ATTR, tv.CanonicalView(model, roles={"1.bias": "fitted"}))
    bound = tv.resolve_view(model)
    assert bound.roles() == {"1.bias": "fitted"}
    with pytest.raises(tv.Unresolvable) as excinfo:
        bound.to_physical(["nope"])
    assert isinstance(excinfo.value, VenngramError) and excinfo.value.names == ["nope"]
    report = tv.ViewConformance(lambda: tv.CanonicalView(torch.nn.Linear(2, 2), roles={"bias": "instrument"})).run()
    assert report.ok and "promotion" in report.ran, report


def test_role_errors_are_venngram_errors():
    with pytest.raises(RoleError, match="unknown role kind"):
        venngram.roles.parse_selector("role:weights")
    assert issubclass(RoleError, VenngramError) and issubclass(tv.TunableViewError, VenngramError)


def test_a_views_structure_defaults_to_the_physical_structure_of_its_root():
    from conftest import TinyDecoder

    from venngram.physical import PhysicalStructure

    model = TinyDecoder(depth=3)
    structure = tv.as_view(tv.CanonicalView(model)).structure()
    assert isinstance(structure, PhysicalStructure) and structure.depth() == 3
