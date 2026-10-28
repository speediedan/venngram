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
Physical Structure
^^^^^^^^^^^^^^^^^^

The structure provider every PyTorch model gets: it finds the model's block list and places each tensor at a layer, so
``layers`` resolves on any model with no table of architectures. It reports no components (``component`` needs a
vocabulary-backed provider) and one stack, the primary one.

The block list is the model's one top-level ``nn.ModuleList``: one not nested inside another ``ModuleList``'s elements,
so a mixture-of-experts layer's expert list is never mistaken for it. A model with several (an encoder and a decoder, a
vision tower beside a language model) is refused, naming each, rather than guessed: name the one meant with
``layers_pattern``, matched on whole path segments from the end (PEFT's ``layers_pattern`` plays the same role).

.. warning:: This is an experimental API and may change in a future release.
"""
import re

from torch import nn

from venngram.errors import Refusal
from venngram.resolve import Location, strip_wrappers


def block_lists(root: nn.Module) -> list[tuple[str, int]]:
    """The top-level ``nn.ModuleList`` paths of ``root`` (wrapper segments removed) with their lengths, in module
    order."""
    found = [(strip_wrappers(path), len(m)) for path, m in root.named_modules()
             if isinstance(m, nn.ModuleList) and len(m)]
    paths = [p for p, _ in found]

    def nested(path: str) -> bool:
        return any(other != path and (path.startswith(f"{other}.") if other else bool(path)) for other in paths)

    return [(p, n) for p, n in found if not nested(p)]


class PhysicalStructure:
    """Places ``root``'s tensors in its block list.

    Args:
        root: The module whose tensor names are located.
        layers_pattern: The block list's path (or its ending, on whole segments) when ``root`` has several.
    """

    vocabulary = False

    def __init__(self, root: nn.Module, layers_pattern: str | None = None) -> None:
        candidates = block_lists(root)
        if layers_pattern is not None:
            rx = re.compile(rf"(?:^|\.){re.escape(layers_pattern)}$")
            candidates = [(p, n) for p, n in candidates if rx.search(p)]
        self.prefix: str | None = None
        self._depth = 0
        self.refusal: Refusal | None = None
        if len(candidates) == 1:
            self.prefix, self._depth = candidates[0]
        elif not candidates:
            where = f" matching `layers_pattern={layers_pattern!r}`" if layers_pattern else ""
            self.refusal = Refusal(f"{type(root).__name__} has no top-level block list (`nn.ModuleList`){where}, so "
                                   "`layers` cannot resolve")
        else:
            self.refusal = Refusal(
                f"{type(root).__name__} has several top-level block lists, so which one `layers` counts is ambiguous: "
                "name it with `layers_pattern`", tuple(p or "<root>" for p, _ in candidates))
        self._rx = None if self.prefix is None else re.compile(
            (rf"^{re.escape(self.prefix)}\." if self.prefix else "^") + r"(\d+)\.")

    def locate(self, name: str) -> Location:
        """The layer of the tensor ``name`` (``None`` outside the block list)."""
        if self._rx is None:
            return Location()
        m = self._rx.match(strip_wrappers(name))
        return Location(layer=int(m.group(1))) if m else Location()

    def depth(self, stack: str | None = None) -> int | Refusal:
        """The block list's length, or why it is unknown."""
        if stack is not None:
            return Refusal(f"the physical structure has only the primary stack, not `{stack}`; a vocabulary provider "
                           "names further stacks")
        return self.refusal if self.refusal is not None else self._depth


__all__ = ["block_lists", "PhysicalStructure"]
