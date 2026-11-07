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
"""Value transform conformance vectors, checked by an evaluator written here from the stated meanings."""
import math

import pytest

from venngram import ops
from venngram.ops import (check_transforms, EVIDENCE_TRANSFORMS, ParamSpec, register_transform, TRANSFORM_MEANINGS,
                          transform_cases, TransformCase, TRANSFORMS)


def sign(x):
    return (x > 0) - (x < 0)


def reference(case: TransformCase, *, statistic=lambda v, t: v * t, sign_source="column"):
    """Each transform as the ``venngram.ops`` docstring states it; ``statistic`` and ``sign_source`` let a test build
    an evaluator that gets one meaning wrong."""
    values = [m.start for m in case.members]
    for t in case.transform:
        if t.name == "scale":
            values = [v * t.params["factor"] for v in values]
        elif t.name == "abs":
            values = [abs(v) for v in values]
        elif t.name == "normalize" and t.params["by"] == "max_abs":
            peak = max(abs(v) for v in values)
            values = [v / peak if peak else 0.0 for v in values]
        elif t.name == "normalize":
            values = [v / m.statistics[t.params["by"]] for v, m in zip(values, case.members)]
        elif t.name == "statistic":
            values = [statistic(v, m.statistics[t.params["name"]]) for v, m in zip(values, case.members)]
        elif t.name == "sign_of":
            values = [sign(m.columns[t.params["column"]]) * abs(v) for v, m in zip(values, case.members)]
        elif t.name == "sign_aware_scale":
            scores = [m.columns[case.column] for m in case.members] if sign_source == "column" else values
            values = [sign(s) * abs(v) * t.params["factor"] for v, s in zip(values, scores)]
        else:
            raise AssertionError(f"no reference meaning for {t.name}")
    return values


def test_the_vectors_cover_every_transform_and_both_normalize_forms():
    cases = transform_cases()
    used = {t.name for c in cases for t in c.transform}
    assert used == set(TRANSFORMS)
    assert {t.params["by"] == "max_abs" for c in cases for t in c.transform if t.name == "normalize"} == {True, False}
    assert all(c.source == "from" for c in cases if {t.name for t in c.transform} & EVIDENCE_TRANSFORMS)
    assert all(len(c.expect) == len(c.members) for c in cases)


def test_the_reference_evaluator_agrees_with_every_vector():
    assert check_transforms(reference) == []


def test_an_evaluator_that_divides_by_a_statistic_fails_by_name():
    failures = check_transforms(lambda c: reference(c, statistic=lambda v, t: v / t))
    assert [f.split(":")[0] for f in failures] == ["statistic-makes-a-dose-absolute", "statistic-inverts-normalize"]


def test_an_evaluator_that_takes_the_sign_from_the_value_fails_by_name():
    failures = check_transforms(lambda c: reference(c, sign_source="value"))
    assert [f.split(":")[0] for f in failures] == ["sign-aware-scale-reads-the-score-not-the-value"]


def test_a_refusal_or_a_wrong_count_is_reported_not_raised():
    def refuse_statistics(case):
        if any(t.name == "statistic" for t in case.transform):
            raise KeyError("no statistic supplied")
        return reference(case)

    failures = check_transforms(refuse_statistics)
    assert len(failures) == 2 and all("raised KeyError" in f for f in failures)
    assert len(check_transforms(lambda c: reference(c)[:-1])) == len(transform_cases())


def test_a_case_reads_its_starting_values_from_its_source():
    by_name = {c.name: c for c in transform_cases()}
    assert [m.start for m in by_name["sign-of-column"].members] == [2.0, -3.0, -1.5]
    assert [m.start for m in by_name["normalize-by-statistic"].members] == [2.0, 3.0]
    assert by_name["scale-dose"].column is None and math.isclose(by_name["scale-dose"].members[0].start, 0.5)


@pytest.fixture
def registered():
    """Restore the transform registry after a test registers one."""
    saved = (dict(TRANSFORMS), dict(TRANSFORM_MEANINGS), set(EVIDENCE_TRANSFORMS), dict(ops._REGISTERED_CASES))
    yield
    for live, old in zip((TRANSFORMS, TRANSFORM_MEANINGS, EVIDENCE_TRANSFORMS, ops._REGISTERED_CASES), saved):
        live.clear()
        live.update(old)


CLIP = {"name": "clip-at-one", "value": {"from": "score"}, "transform": [{"clip": {"bound": 1.0}}],
        "members": [{"columns": {"score": -3.0}}, {"columns": {"score": 0.5}}], "expect": [-1.0, 0.5]}


def test_a_registered_transform_brings_its_meaning_and_vectors(registered):
    register_transform("clip", {"bound": ParamSpec(float, True)}, meaning="v clamped to [-bound, bound]",
                       cases=[CLIP])
    assert "clip" in TRANSFORMS and TRANSFORM_MEANINGS["clip"].startswith("v clamped")
    assert transform_cases()[-1].name == "clip-at-one"

    def evaluate(case):
        if case.transform[0].name == "clip":
            b = case.transform[0].params["bound"]
            return [max(-b, min(b, m.start)) for m in case.members]
        return reference(case)

    assert check_transforms(evaluate) == []
    assert check_transforms(reference) == ["clip-at-one: raised AssertionError: no reference meaning for clip"]


@pytest.mark.parametrize("kwargs, match", [
    ({"name": "scale"}, "already registered"),
    ({"meaning": " "}, "state what it computes"),
    ({"cases": [{**CLIP, "transform": ["abs"]}]}, "no conformance case uses it"),
    ({"cases": [{**CLIP, "expect": [1.0]}]}, "expects 1 values for 2 members"),
    ({"cases": [{**CLIP, "transform": [{"clip": {"bound": 1.0}}, "bogus"]}]}, r"unknown transforms \['bogus'\]"),
    ({"cases": [{**CLIP, "value": {"dose": 1.0}}], "reads_evidence": True}, "needs a `from` value"),
    ({"cases": [{"name": "broken"}]}, "conformance case is malformed"),
])
def test_a_transform_without_a_stated_meaning_or_sound_vectors_is_refused(registered, kwargs, match):
    args = {"name": "clip", "params": {"bound": ParamSpec(float, True)}, "meaning": "v clamped", "cases": [CLIP],
            **kwargs}
    with pytest.raises(ValueError, match=match):
        register_transform(args.pop("name"), args.pop("params"), **args)
    assert "clip" not in TRANSFORMS


def test_a_registered_evidence_transform_is_refused_on_a_dose_when_read(registered):
    from test_recipe import add, recipe_doc

    from venngram.errors import AddressError
    from venngram.recipe import Recipe

    register_transform("clip", {"bound": ParamSpec(float, True)}, meaning="v clamped", cases=[CLIP],
                       reads_evidence=True)
    doc = recipe_doc([add("chess", {"dose": 1.0, "transform": [{"clip": {"bound": 1.0}}]})])
    with pytest.raises(AddressError, match=r"\['clip'\] read the evidence table"):
        Recipe.from_mapping(doc)
