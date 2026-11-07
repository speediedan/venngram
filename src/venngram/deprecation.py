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
Deprecation
^^^^^^^^^^^

One cycle for everything venngram retires: a document schema it still reads, an API name kept as an alias, or a
vocabulary name read through an alias table. A name deprecated in ``0.N`` warns through ``0.N`` and is refused in
``0.N+1``, by name and with its replacement; from 1.0, it warns for one full minor version.

Every such warning is a :class:`VenngramDeprecationWarning` (a :class:`FutureWarning`, so a user sees it by default).
**Strict mode** turns them all into errors: set ``VENNGRAM_STRICT=1`` before venngram is imported, or run code under
:func:`strict`. A registry ``strict`` lookup refuses a retired map either way.

.. warning:: This is an experimental API and may change in a future release.
"""
import contextlib
import os
import warnings
from collections.abc import Iterator

#: Set to ``1`` to turn every venngram deprecation warning into an error for the whole process.
STRICT_ENV = "VENNGRAM_STRICT"


class VenngramDeprecationWarning(FutureWarning):
    """Something venngram has retired was used: it still works in this minor version and is refused in the next."""


_WARNED: set[tuple[str, str]] = set()


def deprecated(what: str, name: str, *, since: str, removed_in: str, replacement: str | None = None,
               stacklevel: int = 3) -> None:
    """Warn, once per process for each ``(what, name)``, that ``name`` is deprecated.

    Args:
        what: The sort of thing (``recipe schema``, ``function``, ``component name``).
        name: Its name.
        since: The version that deprecated it.
        removed_in: The version that refuses it.
        replacement: What to use instead, if anything.
        stacklevel: Passed to :func:`warnings.warn` (the default points at the caller's caller).
    """
    if (what, name) in _WARNED:
        return
    _WARNED.add((what, name))
    use = f"; use {replacement}" if replacement else ""
    warnings.warn(f"the {what} `{name}` is deprecated since venngram {since} and is refused from {removed_in}{use}.",
                  VenngramDeprecationWarning, stacklevel=stacklevel)


@contextlib.contextmanager
def strict() -> Iterator[None]:
    """Turn venngram deprecation warnings into errors inside the block (and warn again for names already warned)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", VenngramDeprecationWarning)
        warned = set(_WARNED)
        _WARNED.clear()
        try:
            yield
        finally:
            _WARNED.update(warned)


if os.environ.get(STRICT_ENV, "") == "1":
    warnings.filterwarnings("error", category=VenngramDeprecationWarning)


__all__ = ["deprecated", "STRICT_ENV", "strict", "VenngramDeprecationWarning"]
