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
Tunable View Errors
^^^^^^^^^^^^^^^^^^^

The errors the tunable view protocol raises. Each is a :class:`~venngram.errors.VenngramError` (so ``ValueError``)
subclass. A consumer may translate them into its
own error types at its boundary (Fine-Tuning Scheduler raises its ``FTSConfigurationError`` for them).

.. warning:: This is an experimental API and may change in a future release.
"""
from collections.abc import Sequence

from venngram.errors import AddressUnresolvable, Refusal, RoleError, VenngramError


class TunableViewError(VenngramError):
    """Raised when a view, a role or a role selector does not satisfy the protocol."""


class RoleKeyError(RoleError, TunableViewError):
    """Raised when a role key names neither a parameter or buffer of the module nor a role.

    It is both a :class:`~venngram.errors.RoleError`, what a key naming no tensor has always raised, and a
    :class:`TunableViewError`, what a malformed role address raises, so a consumer catching either still catches it.
    """


class Unresolvable(TunableViewError, AddressUnresolvable):
    """Raised when logical names cannot be resolved to physical parameters; names every unresolved input.

    It is both a :class:`TunableViewError` and an :class:`~venngram.errors.AddressUnresolvable` (the same outcome seen
    through a view), so a consumer catches either through one class. Its ``address`` lists the names, and its
    ``refusals`` give one per name.

    Args:
        names: Every name that did not resolve.
        detail: Optional context appended to the message.
    """

    def __init__(self, names: Sequence[str], detail: str | None = None) -> None:
        self.names = list(names)
        self.detail = detail
        self.address = ", ".join(self.names)
        self.refusals = tuple(Refusal(f"`{n}` does not resolve to any parameter") for n in self.names)
        msg = f"These names do not resolve to any parameter: {self.names}."
        VenngramError.__init__(self, f"{msg} {detail}" if detail else msg)


__all__ = ["RoleKeyError", "TunableViewError", "Unresolvable"]
