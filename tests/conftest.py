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
from pathlib import Path

import pytest
import torch
from torch import nn

FIXTURE_MAPS = Path(__file__).parent / "fixtures" / "maps"


@pytest.fixture(scope="session")
def seed_registry(tmp_path_factory) -> str:
    """The seed maps (``fixtures/maps``) published to a local store and indexed: what the Hub registry serves."""
    import os

    from venngram import registry
    from venngram.store import publish

    root = tmp_path_factory.mktemp("registry")
    os.environ.setdefault("VENNGRAM_CACHE", str(root / "cache"))
    maps = root / "maps"
    maps.mkdir()
    for path in sorted(FIXTURE_MAPS.glob("*.yaml")):
        (maps / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    publish("local", str(maps), maps, documents=[("component_map", p.name) for p in sorted(maps.glob("*.yaml"))])
    registry.write_index(registry.crawl([f"local:{maps}"]), root / "index", name="test-seeds")
    return f"local:{root / 'index'}"


@pytest.fixture(autouse=True)
def _registries(seed_registry):
    """Every test looks maps up in the local seed registry, never the Hub."""
    from venngram import registry

    registry.set_registries([seed_registry])
    yield
    registry.set_registries([seed_registry])


class Attn(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.q_proj, self.k_proj, self.o_proj = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)


class LoRA(nn.Module):
    """A LoRA-shaped child: adapters live *under* the module they adapt, as PEFT's do."""

    def __init__(self, d: int, r: int = 2) -> None:
        super().__init__()
        self.lora_A = nn.Linear(d, r, bias=False)
        self.lora_B = nn.Linear(r, d, bias=False)


class Block(nn.Module):
    def __init__(self, d: int, experts: int = 0) -> None:
        super().__init__()
        self.attn = Attn(d)
        self.attn.q_proj.add_module("adapter", LoRA(d))
        self.attn.k_proj.add_module("adapter", LoRA(d))
        self.mlp = nn.Linear(d, d)
        self.experts = nn.ModuleList(nn.Linear(d, d) for _ in range(experts)) if experts else None


class TinyDecoder(nn.Module):
    """Six blocks with q/k adapters, an embedding and a head; optionally a mixture of experts per block."""

    def __init__(self, depth: int = 6, d: int = 4, experts: int = 0) -> None:
        super().__init__()
        self.embed = nn.Embedding(10, d)
        self.layers = nn.ModuleList(Block(d, experts) for _ in range(depth))
        self.head = nn.Linear(d, 10)


def adapter_roles(model: nn.Module) -> dict[str, str]:
    return {n: "adapter" for n, _ in model.named_parameters() if ".adapter." in n}


@pytest.fixture
def model() -> TinyDecoder:
    torch.manual_seed(0)
    return TinyDecoder()
