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
"""The venngram-addressing skill's examples, run by ``tests/test_skill_examples.py`` so the skill cannot drift from
the API.

A small model: four blocks, each an attention projection and an MLP projection with a LoRA pair under the
attention one, plus an embedding and a head outside the blocks.
"""
from torch import nn

from venngram import AddressBook, parse, ParamSpace, select, to_string
from venngram.errors import AddressUnresolvable
from venngram.physical import PhysicalStructure


class Block(nn.Module):
    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.attn = nn.ModuleDict({"q_proj": nn.ModuleDict({"base": nn.Linear(d, d),
                                                            "lora_A": nn.Linear(d, 2, bias=False),
                                                            "lora_B": nn.Linear(2, d, bias=False)})})
        self.mlp = nn.ModuleDict({"up_proj": nn.Linear(d, d)})


class Model(nn.Module):
    def __init__(self, depth: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.layers = nn.ModuleList(Block() for _ in range(depth))
        self.head = nn.Linear(8, 16)


def space(model: nn.Module | None = None) -> ParamSpace:
    """The model's tensors, with LoRA tensors as adapters and the default (physical) structure for ``layers``."""
    model = model or Model()
    names = [n for n, _ in model.named_parameters()]
    roles = {n: "adapter" for n in names if ".lora_" in n}
    return ParamSpace(names, roles=roles, structure=PhysicalStructure(model))


def examples() -> dict[str, object]:
    """Each example the skill shows, mapped to its result."""
    s = space()
    book = AddressBook.from_mapping({"schema_version": 1, "name": "example-sets", "portability": "family",
                                     "sets": {"top": {"role": "adapter", "layers": "-2:"},
                                              "rest": {"role": "adapter", "exclude": "@top"}}},
                                    source="examples", inline=False)
    out: dict[str, object] = {
        "top adapters": select("role:adapter & layers:-2:", s),
        "every q_proj tensor": select("module:q_proj", s),
        "base weights of the last block": select("role:base & layers:-1: & leaf:weight", s),
        "adapters minus the top two blocks": select("role:adapter - layers:-2:", s),
        "MLP or head": select("module:mlp.* | module:head", s),
        "odd blocks' MLPs": select("module:up_proj & layers:odd", s),
        "a regex, as an address": select('name:"layers\\.0\\..*\\.lora_A\\..*"', s),
        "book @rest": select("@rest", s, book.sets()),
        "parse": parse("role:adapter & layers:-4:"),
        "mapping to string": to_string(parse({"role": "adapter", "exclude": "@top"})),
    }
    try:
        select("role:adapter & layers:9", s)
    except AddressUnresolvable as err:
        out["refusal"] = str(err)
    return out


if __name__ == "__main__":
    for key, value in examples().items():
        print(f"{key}: {value}")
