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
"""Every module docstring is valid reStructuredText, since consumers' Sphinx builds document venngram's modules."""
import importlib
import io
import pkgutil

import pytest

import venngram

docutils_core = pytest.importorskip("docutils.core")
from docutils import nodes  # noqa: E402
from docutils.parsers.rst import roles  # noqa: E402

# the Sphinx roles the docstrings use; plain docutils knows none of them, and only their markup is checked here
SPHINX_ROLES = ("class", "func", "meth", "mod", "attr", "data", "exc", "obj", "ref", "doc")


def _literal(name, rawtext, text, lineno, inliner, options=None, content=None):
    return [nodes.literal(rawtext, text)], []


for _role in SPHINX_ROLES:
    roles.register_local_role(_role, _literal)

MODULES = sorted(m.name for m in pkgutil.walk_packages(venngram.__path__, "venngram."))


def rst_problems(text: str) -> str:
    """docutils' errors and worse for ``text``, empty when it parses cleanly."""
    stream = io.StringIO()
    docutils_core.publish_doctree(text, settings_overrides={"report_level": 3, "halt_level": 5,
                                                            "warning_stream": stream})
    return stream.getvalue()


def test_a_malformed_table_is_reported():
    # the middle column's text runs into the gap before the last column, as the ``axis`` row once did
    table = "=====  =====  =====\nfield  value  ex\n=====  =====  =====\nrole   valueXX  e\n=====  =====  =====\n"
    assert "Malformed table" in rst_problems(table)


@pytest.mark.parametrize("name", MODULES)
def test_module_docstring_is_valid_rst(name):
    doc = importlib.import_module(name).__doc__
    if doc:
        assert not rst_problems(doc), f"{name}'s docstring is not valid reStructuredText:\n{rst_problems(doc)}"
