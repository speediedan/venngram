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
#: Dependencies importable inside a function of the named modules only (so ``import venngram`` stays light): each an
#: extra in ``pyproject.toml`` or, for the registry's readers, a core dependency.
OPTIONAL = {"maps/verification.py": {"transformers"}, "store.py": {"huggingface_hub"},
            "registry.py": {"huggingface_hub", "pyarrow"}}
TORCH_FREE = ("__init__.py", "__about__.py", "address.py", "book.py", "errors.py", "resolve.py", "roles.py", "units.py",
              "recipe.py", "manifest.py", "store.py", "sources.py",
              "ops.py", "operands.py")


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

    project = tomllib.loads((PKG_DIR.parents[1] / "pyproject.toml").read_text())["project"]
    reqs = [r for rs in project["optional-dependencies"].values() for r in rs] + project["dependencies"]
    declared = {req.split(">")[0].split("=")[0].split("[")[0].strip() for req in reqs}
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
    report = tv.ViewConformance(lambda: tv.CanonicalView(torch.nn.Linear(2, 2), roles={"bias": "artifact"})).run()
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


class SplicedView:
    """A view that assigns roles itself, as a view over a model with spliced transcoders does: every transcoder
    tensor is ``artifact``, and it says nothing about any other tensor."""

    def __init__(self) -> None:
        self.root = torch.nn.ModuleDict({"block": torch.nn.Linear(4, 4), "transcoder": torch.nn.Linear(4, 8),
                                         "steer": torch.nn.Linear(4, 4)})

    def support(self):
        return tv.TunableSupport(trainable=True, declared_roles=frozenset({"base", "artifact"}))

    def to_physical(self, logical):
        return {n: (n,) for n in logical}

    def to_logical(self, physical):
        return list(physical)

    def roles(self):
        return {n: "artifact" for n, _ in self.root.named_parameters() if n.startswith("transcoder.")}


def test_a_self_assigning_view_merges_cooperative_roles():
    view = tv.as_view(SplicedView())
    view.declare_roles({"steer.weight": "intervention:steer", "steer.bias": "intervention:steer",
                        "transcoder.weight": "artifact"})  # equal to the view's own: accepted
    roles = view.roles()
    assert roles["steer.weight"] == "intervention:steer" and roles["transcoder.weight"] == "artifact"
    assert view.declared == {"steer.weight": "intervention:steer", "steer.bias": "intervention:steer"}
    assert view.support().declared_roles == {"base", "artifact", "intervention"}
    from venngram.roles import select

    assert select("role:intervention:steer", list(roles), roles) == ["steer.weight", "steer.bias"]
    report = tv.ViewConformance(SplicedView).run()
    assert report.ok and {"cooperative_roles_merge", "cooperative_roles_conflict"} <= set(report.ran), report


class InterveningView(SplicedView):
    """A self-assigning view that already assigns a role of the kind the merge case declares."""

    def support(self):
        return tv.TunableSupport(trainable=True, declared_roles=frozenset({"base", "artifact", "intervention"}))

    def roles(self):
        return {**super().roles(), "steer.bias": "intervention:steer"}


def test_cooperative_merge_conformance_holds_for_a_view_with_its_own_role_of_that_kind():
    report = tv.ViewConformance(InterveningView).run()
    assert report.ok and "cooperative_roles_merge" in report.ran, report


@pytest.mark.parametrize("promoted,declared", [(True, False), (False, True), (True, True)])
def test_cooperative_conformance_preserves_promotions_and_earlier_declarations(promoted, declared):
    def make_view():
        view = tv.as_view(SplicedView())
        if promoted:
            view.promote(["transcoder.weight"], "latent:tc")
        if declared:
            view.declare_roles({"steer.weight": "intervention:steer"})
        return view

    cases = {"cooperative_roles_merge", "cooperative_roles_conflict"}
    report = tv.ViewConformance(make_view).run(select=cases)
    assert report.ok and set(report.ran) == cases, report


def test_cooperative_conflict_conformance_applies_to_a_promotion_without_own_roles():
    def make_view():
        view = tv.as_view(tv.CanonicalView(torch.nn.Linear(2, 2)))
        view.promote(["weight"], "latent:tc")
        return view

    report = tv.ViewConformance(make_view).run(select=["cooperative_roles_conflict"])
    assert report.ok and report.ran == ["cooperative_roles_conflict"], report


def test_cooperative_conflict_conformance_detects_lost_earlier_declarations(monkeypatch):
    declare = tv.BoundView.declare_roles

    def make_view():
        view = tv.as_view(SplicedView())
        declare(view, {"steer.weight": "intervention:steer"})
        return view

    def forget_on_refusal(view, roles):
        try:
            return declare(view, roles)
        except tv.TunableViewError:
            view._declared.clear()
            raise

    monkeypatch.setattr(tv.BoundView, "declare_roles", forget_on_refusal)
    report = tv.ViewConformance(make_view).run(select=["cooperative_roles_conflict"])
    assert not report.ok and "changed the roles" in report.failures["cooperative_roles_conflict"]


def test_a_conflicting_declaration_is_refused_by_name_and_changes_nothing():
    view = tv.as_view(SplicedView())
    with pytest.raises(tv.TunableViewError, match=r"'transcoder.bias': the view assigns 'artifact', declared "
                                                  r"'latent:tc'"):
        view.declare_roles({"steer.weight": "intervention", "transcoder.bias": "latent:tc"})
    assert view.declared == {} and "steer.weight" not in view.roles()
    view.declare_roles({"steer.weight": "intervention"})
    with pytest.raises(tv.TunableViewError, match="declared 'intervention' earlier, now 'adapter'"):
        view.declare_roles({"steer.weight": "adapter"})
    with pytest.raises(RoleError, match="name neither a parameter nor a buffer"):
        view.declare_roles({"nowhere.weight": "adapter"})


def test_promoted_roles_count_as_assigned_and_resolve_view_declares():
    view = tv.as_view(SplicedView())
    view.promote(["transcoder.weight"], "latent:tc")
    with pytest.raises(tv.TunableViewError, match="the view assigns 'latent:tc'"):
        view.declare_roles({"transcoder.weight": "artifact"})
    model = torch.nn.Linear(2, 2)
    bound = tv.resolve_view(model, roles={"bias": "adapter"})
    assert bound.roles() == {"bias": "adapter"} and "adapter" in bound.support().declared_roles


@pytest.fixture
def one_rank_mesh(tmp_path):
    """A one-rank gloo mesh: enough for parameters to be ``DTensor`` with no storage of their own."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("needs torch.distributed with gloo")
    dist.init_process_group("gloo", store=dist.FileStore(str(tmp_path / "store"), 1), rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


class _LensModel(torch.nn.Module):
    def __init__(self, tie: bool = False):
        super().__init__()
        self.base, self.lens = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
        if tie:
            self.lens.weight = self.base.weight

    def forward(self, x):
        return self.lens(self.base(x))


def _fsdp2(model, mesh):
    from torch.distributed.fsdp import fully_shard

    return fully_shard(model, mesh=mesh)


def _distributed(model, mesh):
    from torch.distributed.tensor import distribute_module

    return distribute_module(model, mesh)


@pytest.mark.parametrize("place", [_fsdp2, _distributed])
def test_the_default_promote_keeps_a_sharded_models_parameters(one_rank_mesh, place):
    """A `DTensor` has no storage address to compare; promoting one changes its role and keeps the parameter."""
    model = place(_LensModel(), one_rank_mesh)
    before = dict(model.named_parameters())
    assert all(type(p).__name__ == "DTensor" for p in before.values())
    view = tv.as_view(tv.CanonicalView(model, roles={"lens.weight": "artifact"}))
    assert view.promote(["lens.weight"], "latent:lens") == {"lens.weight": "lens.weight"}
    assert view.roles()["lens.weight"] == "latent:lens"
    after = dict(model.named_parameters())
    assert all(after[n] is before[n] for n in before)


def test_the_default_promote_refuses_a_tied_dtensor_and_unties_a_plain_one(one_rank_mesh):
    # `base.weight`, the name under which a tied tensor is listed
    plain = _LensModel(tie=True)
    tv.as_view(tv.CanonicalView(plain, roles={"base.weight": "artifact"})).promote(["base.weight"], "latent:lens")
    assert plain.lens.weight is not plain.base.weight and torch.equal(plain.lens.weight, plain.base.weight)
    model = _fsdp2(_LensModel(tie=True), one_rank_mesh)
    before = dict(model.named_parameters(remove_duplicate=False))
    view = tv.as_view(tv.CanonicalView(model, roles={"base.weight": "artifact"}))
    with pytest.raises(tv.TunableViewError, match=r"tied `DTensor` cannot .*'base.weight': \['lens.weight'\]"):
        view.promote(["base.weight"], "latent:lens")
    after = dict(model.named_parameters(remove_duplicate=False))
    assert all(after[n] is before[n] for n in before)


def test_the_retired_instrument_role_reads_as_artifact_with_one_warning_and_strict_mode_refuses_it():
    import warnings

    from venngram.deprecation import strict, VenngramDeprecationWarning
    from venngram.roles import canonical_role, parse_role, selector_matches, validate_roles
    from venngram.tuning.views import instrument_state_keys

    with pytest.warns(VenngramDeprecationWarning, match="role kind `instrument` is deprecated.*use `artifact`"):
        assert parse_role("instrument") == ("artifact", None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # warned once per process: later reads are quiet
        assert canonical_role("instrument") == "artifact" and selector_matches("role:instrument", "artifact")
        assert validate_roles({"bias": "instrument"}, ["weight", "bias"], []) == {"bias": "artifact"}
        assert tv.TunableSupport(trainable=True, declared_roles=frozenset({"base", "instrument"})).declared_roles == {
            "base", "artifact"}
    view = tv.CanonicalView(torch.nn.Linear(2, 2), roles={"bias": "artifact"})
    assert "bias" not in tv.as_view(view).checkpoint_state_dict()  # an artifact is never checkpointed
    with pytest.warns(VenngramDeprecationWarning, match="instrument_state_keys"):
        assert instrument_state_keys(view.root, {"bias": "artifact"}, ["weight", "bias"]) == {"bias"}
    with strict(), pytest.raises(VenngramDeprecationWarning):
        parse_role("instrument")
