"""Component maps describe any module tree, not only the architectures a library already adapts.

Each model here is one no library ships, mapped by a hand-written document: a hybrid stack whose layers alternate an
attention block with a convolutional mixer of a declared kind, an encoder-decoder with two named stacks and no primary
one, and a model with no block list at all.
"""
import pytest
import yaml
from torch import nn

from venngram import ParamSpace, select
from venngram.errors import AddressUnresolvable
from venngram.maps import check_map_against_model, from_document, map_for, MappedView, register
from venngram.maps.schema import _REGISTRY
from venngram.tuning import BoundView


class Attention(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d, d) for _ in range(4))


class Mixer(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(d, d, 3, padding=1, groups=d)
        self.gate = nn.Linear(d, d)


class HybridLayer(nn.Module):
    """Attention on odd layers, a convolutional mixer on even ones, a routed feed-forward on every layer."""

    def __init__(self, i: int, d: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        if i % 2:
            self.attention = Attention(d)
        else:
            self.mixer = Mixer(d)
        self.router = nn.Linear(d, 2)
        self.experts = nn.ModuleList(nn.Linear(d, d) for _ in range(2))


class Hybrid(nn.Module):
    def __init__(self, depth: int = 4, d: int = 4) -> None:
        super().__init__()
        self.tok = nn.Embedding(10, d)
        self.trunk = nn.ModuleList(HybridLayer(i, d) for i in range(depth))
        self.readout = nn.Linear(d, 10)


HYBRID = """
schema_version: 2
architecture: Hybrid
kinds:
  mixer: {output: tensor, sublayer: true}
  router: {output: tensor}
components:
  embed:               {module: "tok",                          kind: embed}
  blocks.{i}:          {module: "trunk.{i}",                    kind: block}
  blocks.{i}.ln1:      {module: "trunk.{i}.norm",               kind: norm}
  blocks.{i}.attn:     {module: "trunk.{i}.attention",          kind: attn,   layers: odd}
  blocks.{i}.attn.q:   {module: "trunk.{i}.attention.wq",       kind: linear, layers: odd}
  blocks.{i}.attn.k:   {module: "trunk.{i}.attention.wk",       kind: linear, layers: odd}
  blocks.{i}.attn.v:   {module: "trunk.{i}.attention.wv",       kind: linear, layers: odd}
  blocks.{i}.attn.o:   {module: "trunk.{i}.attention.wo",       kind: linear, layers: odd}
  blocks.{i}.mixer:    {module: "trunk.{i}.mixer",              kind: mixer,  layers: even}
  blocks.{i}.mixer.conv: {module: "trunk.{i}.mixer.conv",       kind: linear, layers: even}
  blocks.{i}.mixer.gate: {module: "trunk.{i}.mixer.gate",       kind: linear, layers: even}
  blocks.{i}.moe.router: {module: "trunk.{i}.router",           kind: router}
  blocks.{i}.moe.experts: {module: "trunk.{i}.experts",         kind: linear}
  unembed:             {module: "readout",                      kind: unembed}
"""


def space(view) -> ParamSpace:
    bound = BoundView(view)
    return ParamSpace([n for n, _ in bound.root.named_parameters()], roles=bound.roles(), structure=bound.structure())


def test_a_hybrid_stack_with_a_declared_kind_and_its_own_names():
    model, cmap = Hybrid(), from_document(yaml.safe_load(HYBRID), source="hybrid")
    assert check_map_against_model(cmap, model) == []
    s = space(MappedView(model, cmap))
    assert {n.split(".")[1] for n in select("component:attn.q", s)} == {"1", "3"}  # attention lives on odd layers
    assert {n.split(".")[1] for n in select("component:mixer.*", s)} == {"0", "2"}
    assert select("component:moe.router & layers:-1:", s) == ["trunk.3.router.weight", "trunk.3.router.bias"]
    assert len(select("component:moe.experts", s)) == 4 * 2 * 2  # four layers, two experts, weight and bias
    with pytest.raises(AddressUnresolvable, match="matches no tensor"):  # absent where the row excludes it
        select("component:attn.q & layers:0", s)


def test_a_layer_predicate_that_lies_about_the_model_is_caught():
    doc = yaml.safe_load(HYBRID)
    doc["components"]["blocks.{i}.mixer"]["layers"] = "odd"
    problems = check_map_against_model(from_document(doc), Hybrid())
    assert {(p.row, p.layer) for p in problems} >= {("blocks.{i}.mixer", 0), ("blocks.{i}.mixer", 1)}


class Seq2Seq(nn.Module):
    """Two block lists and nothing that is the primary one."""

    def __init__(self, d: int = 4) -> None:
        super().__init__()
        self.enc = nn.ModuleList(nn.ModuleDict({"ff": nn.Linear(d, d)}) for _ in range(3))
        self.dec = nn.ModuleList(nn.ModuleDict({"ff": nn.Linear(d, d), "cross": nn.Linear(d, d)}) for _ in range(2))


SEQ2SEQ = {
    "schema_version": 2, "architecture": "Seq2Seq",
    "stacks": {"encoder": {"module": "enc.{i}"}, "decoder": {"module": "dec.{i}"}},
    "kinds": {"cross_attn": {"output": "tensor", "sublayer": True}},
    "components": {
        "encoder.blocks.{i}": {"module": "enc.{i}", "kind": "block"},
        "encoder.blocks.{i}.mlp": {"module": "enc.{i}.ff", "kind": "mlp"},
        "decoder.blocks.{i}": {"module": "dec.{i}", "kind": "block"},
        "decoder.blocks.{i}.mlp": {"module": "dec.{i}.ff", "kind": "mlp"},
        "decoder.blocks.{i}.cross_attn": {"module": "dec.{i}.cross", "kind": "cross_attn"},
    },
}


def test_named_stacks_without_a_primary_one():
    model, cmap = Seq2Seq(), from_document(SEQ2SEQ)
    assert check_map_against_model(cmap, model) == []
    s = space(MappedView(model, cmap))
    assert select("stack:encoder & layers:-1: & component:mlp", s) == ["enc.2.ff.weight", "enc.2.ff.bias"]
    assert select("stack:decoder & component:cross_attn", s) == [f"dec.{i}.cross.{p}" for i in range(2)
                                                                  for p in ("weight", "bias")]
    with pytest.raises(AddressUnresolvable, match="no primary block stack") as err:
        select("layers:0", s)
    assert "stack:encoder" in str(err.value)


class Probe(nn.Module):
    """No block list at all."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.head = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))


def test_a_model_without_a_block_list_and_a_registered_map():
    register(from_document({"schema_version": 2, "architecture": "Probe", "components": {
        "trunk": {"module": "backbone", "kind": "linear"}, "unembed": {"module": "head", "kind": "unembed"}}}))
    try:
        model = Probe()
        assert map_for(model).architecture == "Probe"  # found by the model's class, as a bundled map is
        s = space(MappedView(model))
        assert select("component:unembed", s) == [n for n, _ in model.head.named_parameters(prefix="head")]
        assert select("component:trunk", s) == ["backbone.weight", "backbone.bias"]
    finally:
        _REGISTRY.pop("Probe")
