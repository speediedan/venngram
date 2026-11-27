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
"""Sphinx configuration for venngram's documentation (built on Read the Docs with warnings as errors)."""
import os
import re
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
GENERATED = HERE / "generated"
REPO = "https://github.com/speediedan/venngram"

_spec = spec_from_file_location("venngram_about", SRC / "venngram" / "__about__.py")
_about = module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_about)  # type: ignore[union-attr]

project = "venngram"
author = "Daniel Dale"
copyright = "2025-2026, Daniel Dale"  # noqa: A001
version = release = _about.__version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

# venngram's heavy dependencies are imported for their types and lazily for their work; the docs build reads only the
# docstrings and signatures, so it mocks them rather than installing torch.
autodoc_mock_imports = ["torch", "transformers", "peft", "huggingface_hub", "pyarrow", "safetensors"]
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_dmath_allow_space = False
myst_dmath_allow_digits = False
myst_heading_anchors = 3

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

html_theme = "pydata_sphinx_theme"
html_title = f"venngram {release}"
html_theme_options = {"github_url": REPO, "show_toc_level": 2, "navigation_with_keys": False}
html_static_path: list[str] = []
exclude_patterns = ["_build"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"

#: Modules that hold no API.
NO_API = ("venngram.__about__", "venngram.__main__")


def _write_if_changed(path: Path, text: str) -> None:
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")


def _readme() -> None:
    """The README is the guide: copied for the index with links outside the docs made absolute."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    text = re.sub(r"\]\((?!https?://|#)([^)]+)\)", lambda m: f"]({REPO}/blob/main/{m.group(1)})", text)
    _write_if_changed(GENERATED / "README.md", text)


def _api() -> None:
    """One page per module, from each module's docstring and its public names (``__all__``)."""
    pages = GENERATED / "api"
    pages.mkdir(parents=True, exist_ok=True)
    # walk the files, not the packages: importing a subpackage here would need its mocked dependencies
    found = [".".join(p.relative_to(SRC).with_suffix("").parts) for p in (SRC / "venngram").rglob("*.py")]
    packages = {n.removesuffix(".__init__") for n in found if n.endswith(".__init__")}
    names = sorted(n.removesuffix(".__init__") for n in found if n.removesuffix(".__init__") not in NO_API)
    for name in names:
        # a package re-exports its modules' names; each is documented once, where it is defined
        extra = ("   :no-members:\n\nIts public names are documented in the modules that define them.\n"
                 if name in packages else "")
        _write_if_changed(pages / f"{name}.rst", f"{name}\n{'=' * len(name)}\n\n.. automodule:: {name}\n{extra}")
    toc = "\n".join(f"   generated/api/{n}" for n in names)
    _write_if_changed(HERE / "api.rst", "API reference\n=============\n\nEvery module, with its public names (its "
                      "``__all__``). A name not listed is internal.\n\n.. toctree::\n   :maxdepth: 1\n\n" + toc + "\n")


GENERATED.mkdir(exist_ok=True)
_readme()
_api()
os.environ.setdefault("VENNGRAM_CACHE", str(HERE / "_build" / "cache"))  # an import never touches the user's cache
