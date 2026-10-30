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
Errors
^^^^^^

Every error venngram raises is a :class:`VenngramError`, itself a ``ValueError``, so a consumer can translate them into
its own error types at one boundary.

A refusal is also a *value*: :class:`Refusal` says why something did not resolve and what would, so a consumer that
enumerates support can render it and a consumer that needs the result raises :class:`AddressUnresolvable` with it.

.. warning:: This is an experimental API and may change in a future release.
"""
from collections.abc import Sequence
from dataclasses import dataclass, field


class VenngramError(ValueError):
    """The base of every error venngram raises."""


class RoleError(VenngramError):
    """Raised when a role or a role selector is invalid."""


class AddressError(VenngramError):
    """Raised when an address, an address book or a reference is malformed."""


class StoreError(VenngramError):
    """Raised when a manifest, a document reference or a store refuses: a malformed or unlisted document, a content
    hash that does not match, or a store that cannot provide what a reference names."""


@dataclass(frozen=True)
class Refusal:
    """Why an address (or one of its fields) did not resolve, as a value.

    Attributes:
        reason: What failed, naming the field and the value.
        alternatives: What would resolve instead, when something would.
    """

    reason: str
    alternatives: tuple[str, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        if not self.alternatives:
            return self.reason
        return f"{self.reason} (alternatives: {', '.join(self.alternatives)})"


class AddressUnresolvable(VenngramError):
    """Raised when an address does not resolve, with every refusal that caused it.

    Args:
        address: The address, in its string form.
        refusals: Why it did not resolve.
    """

    def __init__(self, address: str, refusals: Sequence[Refusal]) -> None:
        self.address = address
        self.refusals = tuple(refusals)
        detail = "; ".join(str(r) for r in self.refusals)
        super().__init__(f"The address `{address}` does not resolve: {detail}")


__all__ = ["AddressError", "AddressUnresolvable", "Refusal", "RoleError", "StoreError", "VenngramError"]
