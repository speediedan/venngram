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
"""venngram imports only the standard library, ``torch`` and ``yaml`` at module level, and its core (addresses, books,
roles, resolution) imports no ``torch`` at all. A declared optional dependency may be imported inside a function, by the
feature that needs it."""
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
#: Optional dependencies (each an extra in ``pyproject.toml``), importable inside a function of the named modules only.
OPTIONAL = {"maps/verification.py": {"transformers"}, "recipe.py": {"huggingface_hub"}}
TORCH_FREE = ("__init__.py", "__about__.py", "address.py", "book.py", "errors.py", "resolve.py", "roles.py", "units.py",
              "recipe.py")


def _names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name.split(".")[0] for alias in node.names}
    if isinstance(node, ast.ImportFrom) and node.level == 0:
        return {(node.module or "").split(".")[0]}
    return set()


def _top_level_imports(path: Path, *, nested: bool = True) -> set[str]:
    """Every package ``path`` imports; with ``nested=False``, only the ones imported outside a function."""
    tree = ast.parse(path.read_text())
    inner = {id(n) for f in ast.walk(tree) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
             for n in ast.walk(f)}
    return {m for node in ast.walk(tree) if nested or id(node) not in inner for m in _names(node)}


@pytest.mark.parametrize("module", sorted(p.relative_to(PKG_DIR).as_posix() for p in PKG_DIR.rglob("*.py")))
def test_imports_only_stdlib_torch_and_yaml(module):
    allowed = ALLOWED | set(sys.stdlib_module_names)
    outside = sorted(_top_level_imports(PKG_DIR / module, nested=False) - allowed)
    assert not outside, f"{module} imports {outside}"
    lazy = sorted(_top_level_imports(PKG_DIR / module) - allowed - OPTIONAL.get(module, set()))
    assert not lazy, f"{module} imports {lazy}, which is not a declared optional dependency of it"


def test_optional_dependencies_are_declared_extras():
    import tomllib

    extras = tomllib.loads((PKG_DIR.parents[1] / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    declared = {req.split(">")[0].split("=")[0].split("[")[0].strip() for reqs in extras.values() for req in reqs}
    assert set().union(*OPTIONAL.values()) <= declared


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
