"""Verification rules are data: a map states what holds for its own architecture, and removes what does not."""
import dataclasses

import pytest
import torch
import yaml
from test_maps import CONFIGS
from test_maps_any_architecture import HYBRID, Hybrid
from test_verification import mutated
from torch import nn

from venngram.maps import from_document
from venngram.maps.rules import core_rules, evaluate, layered, rules_from_document, Unstated
from venngram.maps.verification import verify

pytest.importorskip("transformers")


def run_hybrid(m):
    x = m.tok(torch.tensor([[1, 2, 3]]))
    for i, layer in enumerate(m.trunk):
        x = layer.norm(x)
        if i % 2:
            a = layer.attention
            x = a.wo(a.wq(x) + a.wk(x) + a.wv(x))
        else:
            x = layer.mixer.gate(layer.mixer.conv(x.transpose(1, 2)).transpose(1, 2))
    return m.readout(x)


HYBRID_RULES = {"widths": {"mixer.*": {"in": "hidden"}, "moe.router": {"in": "hidden", "out": "experts"}},
                "config": {"experts": ["num_experts"]},
                "order": {"norm_then_mixer": ["ln1", "mixer.conv", "mixer.gate"]}}


def hybrid_map(rules=HYBRID_RULES):
    return from_document({**yaml.safe_load(HYBRID), "verification": rules})


def test_a_custom_map_states_its_own_widths_and_order():
    report = verify(Hybrid(), hybrid_map(), config={"hidden_size": 4, "num_experts": 2}, inputs=run_hybrid)
    assert report.status == "verified", str(report)
    assert "map:Hybrid" in report.rules
    record = report.record()
    assert record["checks"]["order"]["observed_order"]["blocks"] == ["ln1", "mixer.conv", "mixer.gate"]


def test_a_custom_rule_that_does_not_hold_fails_by_name():
    rules = {**HYBRID_RULES, "order": {"gate_first": ["mixer.gate", "mixer.conv"]}}
    order = verify(Hybrid(), hybrid_map(rules), config={"hidden_size": 4}, inputs=run_hybrid).check("order")
    assert order.status == "failed" and all("gate_first" in p.reason for p in order.problems)
    wide = {**HYBRID_RULES, "widths": {"mixer.gate": {"out": "2 * hidden"}}}
    shapes = verify(Hybrid(), hybrid_map(wide), config={"hidden_size": 4}, order=False).check("shapes")
    assert shapes.status == "failed" and "the rules expect 8 (2 * hidden)" in shapes.problems[0].reason


class ParallelBlock(nn.Module):
    """A parallel-residual block: the MLP reads the block input, so it starts before the attention output."""

    def __init__(self, d: int = 4) -> None:
        super().__init__()
        self.ln_attn, self.ln_mlp = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.o = nn.Linear(d, d), nn.Linear(d, d)
        self.up, self.down = nn.Linear(d, 8), nn.Linear(8, d)

    def forward(self, x):
        h = self.q(self.ln_attn(x))
        m = self.down(self.up(self.ln_mlp(x)))
        return x + self.o(h) + m


class Parallel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(ParallelBlock() for _ in range(2))

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def parallel_doc(verification=None):
    rows = {"blocks.{i}": ("blocks.{i}", "block"), "blocks.{i}.ln1": ("blocks.{i}.ln_attn", "norm"),
            "blocks.{i}.attn.q": ("blocks.{i}.q", "linear"), "blocks.{i}.attn.o": ("blocks.{i}.o", "linear"),
            "blocks.{i}.ln2": ("blocks.{i}.ln_mlp", "norm"), "blocks.{i}.mlp.in": ("blocks.{i}.up", "linear"),
            "blocks.{i}.mlp.out": ("blocks.{i}.down", "linear")}
    doc = {"schema_version": 2, "architecture": "Parallel",
           "components": {k: {"module": m, "kind": kind} for k, (m, kind) in rows.items()}}
    return from_document({**doc, "verification": verification} if verification else doc)


def test_a_parallel_residual_map_removes_the_core_chain_that_does_not_hold_for_it():
    run = lambda m: m(torch.zeros(1, 4))
    core_only = verify(Parallel(), parallel_doc(), inputs=run).check("order")
    assert core_only.status == "failed" and {p.reason.split("chain ")[1].split(" ")[0] for p in core_only.problems} \
        == {"'attn_then_mlp'"}
    relaxed = parallel_doc({"order": {"attn_then_mlp": None,
                                      "parallel_attention": ["ln1", "attn.q", "attn.o"],
                                      "parallel_mlp": ["ln2", "mlp.in", "mlp.out"]}})
    assert verify(Parallel(), relaxed, inputs=run).check("order").status == "passed"


def test_gpt2s_own_section_makes_its_mlp_widths_checkable():
    cmap = mutated("GPT2LMHeadModel", blocks__mlp__in="transformer.h.{i}.mlp.c_proj",
                   blocks__mlp__out="transformer.h.{i}.mlp.c_fc")
    shapes = verify(CONFIGS["GPT2LMHeadModel"](), cmap, order=False).check("shapes")
    assert any("(intermediate)" in p.reason for p in shapes.problems)  # `n_inner: null`, through the map's `4 * hidden`
    without = verify(CONFIGS["GPT2LMHeadModel"](), dataclasses.replace(cmap, verification=None), order=False)
    assert not any("intermediate" in p.reason for p in without.check("shapes").problems)
    assert "not stated by the configuration" in without.check("shapes").note


def test_layers_replace_by_key_and_null_removes():
    base = rules_from_document({"schema_version": 1, "name": "a", "widths": {"x": {"out": 1}, "y": {"out": 2}},
                                "order": {"c": ["x", "y"]}})
    top = rules_from_document({"name": "b", "widths": {"x": {"out": 3}, "y": None}, "order": {"c": None}},
                              section=True)
    merged = layered(base, top)
    assert merged.widths == {"x": {"out": 3}} and merged.order == {} and merged.sources == ("a", "b")
    assert set(core_rules().order) >= {"attn_then_mlp", "mlp_input_then_output"}


def test_meanings_resolve_through_fields_then_other_meanings():
    rules = rules_from_document({"schema_version": 1, "config": {
        "heads": ["n_head"], "kv": ["n_kv", "heads"], "loop": ["loop + 1"]}})
    value = rules.resolver({"n_head": 6})
    assert value("kv") == 6
    with pytest.raises(Unstated):
        value("loop")  # a cycle is unstated, not a recursion error


@pytest.mark.parametrize("expr", ["__import__('os')", "a.b", "f(1)", "2 ** 3", "[1]", "1 if a else 2"])
def test_expressions_admit_arithmetic_only(expr):
    with pytest.raises(ValueError, match="only integers"):
        evaluate(expr, lambda _n: 1)


def test_exact_division_is_checked():
    assert evaluate("hidden / heads", {"hidden": 12, "heads": 4}.__getitem__) == 3
    with pytest.raises(ValueError, match="does not divide exactly"):
        evaluate("hidden / heads", {"hidden": 10, "heads": 4}.__getitem__)


@pytest.mark.parametrize("doc, match", [
    ({"schema_version": 2}, "`schema_version` must be 1"),
    ({"schema_version": 1, "orders": {}}, "unknown rule set keys"),
    ({"schema_version": 1, "order": {"c": ["x"]}}, "at least two groups"),
    ({"schema_version": 1, "widths": {"x": {"wide": 1}}}, "must map some of"),
    ({"schema_version": 1, "widths": {"x": {"out": "a +"}}}, "malformed expression"),
])
def test_malformed_rule_documents_are_refused(doc, match):
    with pytest.raises(ValueError, match=match):
        rules_from_document(doc)


def test_a_malformed_map_section_is_refused_at_load():
    with pytest.raises(ValueError, match="verification"):
        hybrid_map({"order": {"c": "ln1"}})
