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
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from venngram import AddressBook, ParamSpace, select
from venngram.address import parse, to_mapping, to_string
from venngram.errors import AddressError, AddressUnresolvable
from venngram.maps import LayerVisit, MapStructure, from_document
from venngram.maps.drafter import map_document
from venngram.maps.verification import check_execution

MAP = Path(__file__).resolve().parents[1] / "registry/candidates/inkling2_mini.yaml"


@pytest.fixture
def document():
    return yaml.safe_load(MAP.read_text(encoding="utf-8"))


def test_inkling_map_resolves_layer_visits_in_forward_order(document):
    cmap = from_document(document)
    execution = cmap.execution
    visits = [(i, 1) for i in range(6)]
    visits += [(i, p) for p in (1, 2, 3) for i in range(6, 18)]
    visits += [(i, 1) for i in range(18, 24)]
    assert execution.visits == tuple(LayerVisit(i, p, step) for step, (i, p) in enumerate(visits))
    assert len(execution.resolve("point:state.active")) == 48
    active = execution.resolve("point:state.active & layers:12")
    assert [s.call_index for s in active] == [0, 1, 2]
    assert [s.visit.step for s in active] == [12, 24, 36]
    assert len({s.module for s in active}) == 1
    assert from_document(map_document(cmap)).execution == execution
    assert cmap.sublayers == ()


def test_executed_layers_keep_one_copy_of_their_parameters(document):
    cmap = from_document(document)
    model = nn.Module()
    model.model = nn.Module()
    model.model.language_model = nn.Module()
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = nn.Module()
    layer.self_attn.q_proj.linear = nn.Linear(4, 4, bias=False)
    model.model.language_model.layers = nn.ModuleList([nn.Module() for _ in range(24)])
    model.model.language_model.layers[12] = layer
    names = dict(model.named_parameters())
    space = ParamSpace(list(names), structure=MapStructure(model, cmap, prefix=""))
    (name,) = select("component:attn.q & layers:12", space)
    assert names[name] is layer.self_attn.q_proj.linear.weight
    assert len(cmap.execution.resolve("point:state.active & layers:12")) == 3
    with pytest.raises(AddressUnresolvable, match="Shared parameters"):
        select("component:attn.q & layers:12 & passes:2", space)
    assert check_execution(cmap, model).status == "failed"   # this test model has no hook modules beyond the layers


def test_messages_use_their_modules_own_call_indices(document):
    execution = from_document(document).execution
    gate, = execution.resolve("point:message.gated.q & layers:12 & passes:3")
    raw, = execution.resolve("point:message.raw.q & layers:12 & passes:3")
    bank, = execution.resolve("point:message.bank.1 & layers:12 & passes:3")
    assert gate.call_index == 1
    assert raw.call_index == bank.call_index == 18
    assert raw.module == bank.module == "model.language_model.message_input"
    assert raw.visit == gate.visit == bank.visit == LayerVisit(12, 3, 36)
    with pytest.raises(AddressUnresolvable, match="no pass 1"):
        execution.resolve("point:message.raw.q & layers:12 & passes:1")
    with pytest.raises(AddressUnresolvable, match="no pass 2"):
        execution.resolve("point:state.active & layers:[12, 20] & passes:2")
    with pytest.raises(AddressUnresolvable, match="no pass 2"):
        execution.resolve("point:state.completed.2 & layers:12 & passes:2")


def test_site_paths_select_the_named_state_and_token(document):
    execution = from_document(document).execution
    hidden = torch.arange(12).reshape(1, 4, 3)
    blocks = torch.arange(36).reshape(1, 4, 3, 3)
    args = hidden, blocks
    entering, = execution.resolve("point:state.enter & layers:13 & passes:2")
    completed, = execution.resolve("point:state.completed.2 & layers:13 & passes:2")
    mixed, = execution.resolve("point:attn.mix.k & layers:17 & passes:3")
    final, = execution.resolve("point:state.final_mix")
    assert entering.hook == completed.hook == "input"
    assert torch.equal(entering.read(args, token=1), hidden[:, 1])
    assert torch.equal(completed.read(args, token=-1), blocks[:, -1, 2])
    assert torch.equal(mixed.read((hidden, hidden + 1, hidden + 2), token=0), hidden[:, 0] + 1)
    assert final.visit is None and final.call_index == 0


def test_book_sets_keep_passes_and_allow_set_algebra(document):
    expr = parse("point:state.enter & layers:12 & passes:2")
    assert parse(to_mapping(expr)) == parse(to_string(expr)) == expr
    sets = {"entered": to_mapping(expr), "active": "point:state.active & layers:12"}
    with pytest.raises(AddressError, match="schema 12"):
        AddressBook.from_mapping({"schema_version": 11, "sets": sets})
    book = AddressBook.from_mapping({"schema_version": 12, "sets": sets})
    execution = from_document(document).execution
    resolved = execution.resolve("(@active & passes:2) | @entered", book.sets())
    assert {s.point for s in resolved} == {"state.enter", "state.active"}
    assert all(s.visit == LayerVisit(12, 2, 24) for s in resolved)
    with pytest.raises(AddressUnresolvable, match="expected 2"):
        execution.resolve({"point": "state.final_mix", "expect": 2})


def test_invalid_map_timeline_is_refused(document):
    with pytest.raises(ValueError, match="schema 7"):
        from_document({**document, "schema_version": 6})
    corrupt = deepcopy(document)
    corrupt["execution"]["loops"].append({"start": 10, "stop": 13, "passes": 2})
    with pytest.raises(ValueError, match="disjoint"):
        from_document(corrupt)
    corrupt = deepcopy(document)
    corrupt["execution"]["sites"]["message.raw.k"]["first_pass"] = 3
    with pytest.raises(ValueError, match="inconsistent call timeline"):
        from_document(corrupt)
