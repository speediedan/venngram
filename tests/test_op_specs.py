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
"""The straight-through substitution op, and op declarations read from documents."""
import copy

import pytest
import torch
import yaml
from test_recipe import recipe_doc

from venngram import NewerSchemaError
from venngram.errors import AddressError
from venngram.ops import (check_ops, check_support, CORE_OPS, op_cases, op_spec_from_mapping, op_spec_to_mapping,
                          OPS, OpSpec, ParamSpec, read_op_specs, register_op, Support, write_op_specs)
from venngram.recipe import Recipe
from venngram.store import load, publish


@pytest.fixture(autouse=True)
def restore_ops():
    saved = dict(OPS)
    yield
    OPS.clear()
    OPS.update(saved)


def straight_through(case):
    """``o + sg(x - o)`` in torch, as a consumer would apply it."""
    x = torch.tensor(case.inputs["x"], requires_grad=True)
    o = torch.tensor(case.inputs["o"], requires_grad=True)
    y = o + (x - o).detach()
    y.backward(torch.tensor(case.inputs["g"]))
    grad_x = x.grad if x.grad is not None else torch.zeros_like(x)
    return {"value": y.tolist(), "grad_o": o.grad.tolist(), "grad_x": grad_x.tolist()}


def test_replace_with_error_is_a_core_op_with_a_stated_numeric_meaning():
    spec = OPS["replace_with_error"]
    assert "replace_with_error" in CORE_OPS and spec.operand == ("none", "vector") and spec.of == ("value",)
    assert spec.composition == "ordered" and len(op_cases("replace_with_error")) == 3
    assert check_ops(straight_through) == []


def test_an_implementation_that_lets_the_gradient_reach_the_activation_fails_by_name():
    def plain_replace(case):  # substitutes o for x outright: wrong value, and wrong gradients
        zeros = [0.0] * len(case.inputs["g"])
        return {"value": list(case.inputs["o"]), "grad_o": list(case.inputs["g"]), "grad_x": zeros}

    def identity(case):  # keeps x and its gradient: right value, gradient to the wrong place
        zeros = [0.0] * len(case.inputs["g"])
        return {"value": list(case.inputs["x"]), "grad_o": zeros, "grad_x": list(case.inputs["g"])}

    assert {f.split(":")[0] for f in check_ops(plain_replace)} == {"replace-with-error-keeps-the-forward-value"}
    failing = check_ops(identity)
    assert {f.split(":")[0] for f in failing} == {"replace-with-error-routes-the-gradient-to-the-operand",
                                                  "replace-with-error-when-the-operand-equals-the-activation"}


def test_a_recipe_uses_it_and_a_consumer_must_honour_it():
    doc = recipe_doc([{"site": "mlp.out.hook_in", "ops": [{"op": "replace_with_error", "target": {"set": "chess"},
                                                            "value": {"dose": 1.0}}]}])
    recipe = Recipe.from_mapping(doc)
    assert check_support(recipe, Support(ops=frozenset({"replace_with_error"}), targets=frozenset({"units"}))) == []
    assert "not supported" in check_support(recipe, Support(ops=frozenset({"replace"})))[0]
    grad = copy.deepcopy(doc)
    grad["book"]["schema_version"] = 2
    grad["book"]["sets"]["chess"]["of"] = "grad"
    with pytest.raises(AddressError, match=r"addresses `of: grad`, and this op acts on \['value'\]"):
        Recipe.from_mapping(grad)


SOFT_CLIP = {"name": "soft_clip", "operand": ["none"], "params": {"bound": {"type": "float", "required": True,
                                                                             "range": [0.0, 100.0]}},
             "targets": ["activation", "units"], "value": "none", "composition": "ordered"}
DOC = {"schema_version": 1, "kind": "op_specs", "name": "example-tensor-ops", "ops": [SOFT_CLIP]}


def test_a_published_op_validates_a_recipe_without_its_code():
    doc = recipe_doc([{"site": "mlp.out.hook_in", "ops": [{"op": "soft_clip", "target": {"set": "chess"},
                                                            "params": {"bound": 3.0}}]}])
    with pytest.raises(AddressError, match="unknown op 'soft_clip'"):
        Recipe.from_mapping(doc)
    (spec,) = read_op_specs(DOC)
    assert spec.params["bound"] == ParamSpec(float, True, (0.0, 100.0)) and spec.of == ("value", "grad")
    recipe = Recipe.from_mapping(doc)
    assert check_support(recipe, Support(ops=frozenset({"soft_clip"}), targets=frozenset({"units"}))) == []
    bad = copy.deepcopy(doc)
    bad["stages"][0]["ops"][0]["params"]["bound"] = 500.0
    with pytest.raises(AddressError, match=r"lies in \[0.0, 100.0\]"):
        Recipe.from_mapping(bad)
    assert read_op_specs(DOC) == (spec,)  # reading the same declaration again is a no-op


def test_declarations_round_trip_through_the_writer():
    spec = OpSpec("nudge", ("none", "vector"), {"by": ParamSpec(float, False, (-1.0, 1.0)), "tag": ParamSpec(str)},
                  targets=("activation",), value="required", token_override=True, composition="sum",
                  at_least_one_of=("by",), of=("value",))
    doc = write_op_specs([spec], name="mine", description="what I registered")
    assert doc["kind"] == "op_specs" and doc["description"] == "what I registered"
    (back,) = read_op_specs(yaml.safe_load(yaml.safe_dump(doc)), register=False)
    assert back == spec and "nudge" not in OPS
    assert op_spec_from_mapping(op_spec_to_mapping(OPS["clamp"])) == OPS["clamp"]


@pytest.mark.parametrize("change, error, match", [
    (lambda d: d.update(schema_version=2), NewerSchemaError, "op_specs schema 2 is newer"),
    (lambda d: d.update(kind="ops"), ValueError, "is `kind: op_specs`"),
    (lambda d: d.update(ops=[]), ValueError, "non-empty list"),
    (lambda d: d.update(extra=1), ValueError, r"unknown keys \['extra'\]"),
    (lambda d: d["ops"].append(dict(SOFT_CLIP)), ValueError, "declares `soft_clip` twice"),
    (lambda d: d["ops"].append({"name": "scale"}), ValueError, "`scale` is a core op"),
    (lambda d: d["ops"][0].update(params={"bound": {"type": "tensor"}}), ValueError, "has type 'tensor'"),
    (lambda d: d["ops"][0].update(targets=["neurons"]), ValueError, r"unknown \['neurons'\]"),
    (lambda d: d["ops"][0].update(of=["hessian"]), ValueError, r"unknown \['hessian'\]"),
    (lambda d: d["ops"][0].update(bogus=True), ValueError, r"unknown keys \['bogus'\]"),
])
def test_a_malformed_or_clashing_declaration_is_refused_by_name_and_registers_nothing(change, error, match):
    doc = copy.deepcopy(DOC)
    change(doc)
    with pytest.raises(error, match=match):
        read_op_specs(doc)
    assert "soft_clip" not in OPS


def test_a_declaration_that_differs_from_a_registered_op_is_refused():
    register_op(OpSpec("soft_clip", ("none",), {"bound": ParamSpec(float, True)}))
    with pytest.raises(ValueError, match="already registered with a different declaration"):
        read_op_specs(DOC)


def test_a_store_publishes_and_loads_an_op_specs_document(tmp_path):
    (tmp_path / "ops.yaml").write_text(yaml.safe_dump(DOC))
    publish("local", str(tmp_path), tmp_path, documents=[("op_specs", "ops.yaml")])
    (spec,) = load(f"local:{tmp_path}:ops.yaml")
    assert spec.name == "soft_clip" and OPS["soft_clip"] == spec
