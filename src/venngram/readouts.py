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
Readouts
^^^^^^^^

A **readout** says what a gradient is taken of: the scalar a backward pass differentiates when a treatment acts on
``of: grad`` targets. Its inputs are addresses and its form is a weighted sum of terms, so a document never carries an
expression or code:

.. code-block:: yaml

    readout:                                   # a logit difference at the last token
      terms:
        - {reduce: sum, set: answer_logit, tokens: last, weight: 1.0}
        - {reduce: sum, set: foil_logit, tokens: last, weight: -1.0}

Each term reads the activations its book set addresses (a site's units, such as ``unembed``'s output units, which are
the vocabulary, or a latent space's features), at its token scope, and reduces them to a number with a **registered
reduction** (:data:`READOUTS`); the readout is the weighted sum of its terms. With ``x_t`` the addressed values at
token ``t`` of the term's scope:

=========================  ==========================================================================================
reduction                  value
=========================  ==========================================================================================
``sum``                    ``sum_t sum_u x_t[u]``: the addressed values summed over their units and the tokens
``project`` (an operand)   ``sum_t <x_t, o>``: each token's values dotted with the operand ``o``, a vector in a
                           declared basis over the addressed units, summed over the tokens
``norm``                   ``sum_t ||x_t||_2``: each token's Euclidean norm, summed over the tokens
=========================  ==========================================================================================

:func:`register_readout` adds a reduction in code, with what it computes and conformance vectors, which is how a
transform function enters the vocabulary; :func:`check_readouts` runs every vector against a consumer's evaluator.
Computing a readout, and taking its gradient, is the consumer's: venngram declares and validates it.

.. warning:: This is an experimental API and may change in a future release.
"""
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import yaml

from venngram.ops import ParamSpec


@dataclass(frozen=True)
class ReadoutSpec:
    """A registered reduction: its name, whether it takes a vector operand, its parameters and what it computes."""

    name: str
    operand: bool = False
    params: Mapping[str, ParamSpec] = field(default_factory=dict)
    meaning: str = ""


#: The registered reductions, by name (the core ones: the module docstring).
READOUTS: dict[str, ReadoutSpec] = {spec.name: spec for spec in (
    ReadoutSpec("sum", meaning="sum_t sum_u x_t[u]: the addressed values summed over their units and the term's "
                               "tokens"),
    ReadoutSpec("project", operand=True, meaning="sum_t <x_t, o>: each token's addressed values dotted with the "
                                                 "operand o, summed over the term's tokens"),
    ReadoutSpec("norm", meaning="sum_t ||x_t||_2: the Euclidean norm of each token's addressed values, summed over the "
                                "term's tokens"),
)}

#: The reductions venngram defines; registering one of these names needs ``replace``.
CORE_READOUTS = frozenset(READOUTS)

_REGISTERED_CASES: dict[str, tuple["ReadoutCase", ...]] = {}


@dataclass(frozen=True)
class CaseTerm:
    """One term of a conformance vector: its reduction, its addressed values (one row per token), operand and weight."""

    reduce: str
    values: tuple[tuple[float, ...], ...]
    operand: tuple[float, ...] | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    weight: float = 1.0


@dataclass(frozen=True)
class ReadoutCase:
    """A readout conformance vector: its terms and the scalar the readout must be."""

    name: str
    terms: tuple[CaseTerm, ...]
    expect: float


def readout_cases() -> tuple[ReadoutCase, ...]:
    """The readout conformance vectors: venngram's own (``data/readout_vectors.yaml``, covering every core reduction
    and a weighted sum), then those each :func:`register_readout` supplied."""
    doc = yaml.safe_load((resources.files("venngram") / "data" / "readout_vectors.yaml").read_text("utf-8"))
    return tuple(_case(raw) for raw in doc["cases"]) + tuple(c for cs in _REGISTERED_CASES.values() for c in cs)


def register_readout(name: str, *, meaning: str, cases: Sequence[Mapping[str, Any]], operand: bool = False,
                     params: Mapping[str, ParamSpec] | None = None, replace: bool = False) -> None:
    """Register a reduction, so readouts may name it and consumers may declare it.

    A reduction is shared only with what it computes: ``meaning`` states it, and ``cases`` are conformance vectors in
    the ``data/readout_vectors.yaml`` form, which :func:`check_readouts` then runs against every consumer's evaluator.

    Raises:
        ValueError: If the name is taken and ``replace`` is not set, ``meaning`` is empty, no case uses the reduction, a
            case is malformed or names an unknown reduction, or a term's operand does not match its reduction.
    """
    if name in READOUTS and not replace:
        raise ValueError(f"readout reduction {name!r} is already registered; pass replace=True to replace it")
    if not isinstance(meaning, str) or not meaning.strip():
        raise ValueError(f"readout reduction {name!r}: state what it computes (`meaning`)")
    spec = ReadoutSpec(name, operand, dict(params or {}), meaning)
    known = {**READOUTS, name: spec}
    try:
        parsed = tuple(_case(c) for c in cases)
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError(f"readout reduction {name!r}: a conformance case is malformed ({type(err).__name__}: "
                         f"{err})") from err
    if not any(t.reduce == name for c in parsed for t in c.terms):
        raise ValueError(f"readout reduction {name!r}: no conformance case uses it")
    for c in parsed:
        for t in c.terms:
            if t.reduce not in known:
                raise ValueError(f"readout reduction {name!r}: case {c.name!r} uses unknown reduction {t.reduce!r}")
            if (t.operand is not None) != known[t.reduce].operand:
                raise ValueError(f"readout reduction {name!r}: case {c.name!r} gives `{t.reduce}` "
                                 f"{'an operand it does not take' if t.operand is not None else 'no operand'}")
    READOUTS[name] = spec
    _REGISTERED_CASES[name] = parsed


def _case(raw: Mapping[str, Any]) -> ReadoutCase:
    terms = []
    for t in raw["terms"]:
        rows = tuple(tuple(float(v) for v in row) for row in t["values"])
        operand = tuple(float(v) for v in t["operand"]) if "operand" in t else None
        if operand is not None and any(len(row) != len(operand) for row in rows):
            raise ValueError(f"case {raw['name']!r}: an operand of {len(operand)} values for rows of another width")
        terms.append(CaseTerm(str(t["reduce"]), rows, operand, dict(t.get("params") or {}),
                              float(t.get("weight", 1.0))))
    if not terms:
        raise ValueError(f"case {raw['name']!r} has no terms")
    return ReadoutCase(str(raw["name"]), tuple(terms), float(raw["expect"]))


def check_readouts(evaluate: Callable[[ReadoutCase], float], *, rel_tol: float = 1e-9,
                   abs_tol: float = 1e-12) -> list[str]:
    """Every conformance vector ``evaluate`` (a consumer's evaluator, adapted to take a :class:`ReadoutCase` and
    return the readout's value) gets wrong, by name; empty when it agrees with venngram's meanings."""
    out: list[str] = []
    for case in readout_cases():
        try:
            got = float(evaluate(case))
        except Exception as err:  # a consumer's refusal or bug, reported by case rather than raised
            out.append(f"{case.name}: raised {type(err).__name__}: {err}")
            continue
        if not math.isclose(got, case.expect, rel_tol=rel_tol, abs_tol=abs_tol):
            out.append(f"{case.name}: expected {case.expect}, got {got}")
    return out


__all__ = ["CaseTerm", "check_readouts", "CORE_READOUTS", "readout_cases", "READOUTS", "ReadoutCase", "ReadoutSpec",
           "register_readout"]
