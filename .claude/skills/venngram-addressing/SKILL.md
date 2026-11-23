---
name: venngram-addressing
description: Compose, resolve and debug venngram addresses, the set-algebraic way to name slices of a model (tensors to tune, activation sites to steer, units, latent features) by role, layer, module and component rather than by parameter names and regexes; and write shareable address books. Use when writing a fine-tuning schedule or steering configuration that names model parts, when an address does not resolve or selects the wrong tensors, when one selection must work across several models of a family, or when asked about the address grammar. Do NOT use when a single explicit name or a simple regex already says what is meant.
---

# Addressing model slices with venngram

An **address** selects a slice of a model. Its fields intersect, and addresses combine by set algebra. Consumers
such as finetuning-scheduler (schedules) and interpretune (steering and analysis) accept addresses wherever they
accept names. The tensor examples come from `references/examples.py`, which `tests/test_skill_examples.py` runs.

## First: does this need an address?

- **No.** For one tensor, or a pattern that is clearest as a regex over one model's names, use the name or regex.
  Consumers keep accepting them unchanged.
- **Yes**, when the selection is by **what tensors are** rather than what they are called: a role (`adapter`), a
  layer range counted from the depth (`layers:-4:`), a module kind (`module:q_proj`), an architecture-neutral
  component (`component:mlp.*`); when one selection must hold on several models; when sets overlap; or when sets are
  shared between people or runs.

## The grammar

A term is `<field>:<value>`. `&` intersects, `-` subtracts, `|` unites, `&` binds tighter than `-`, which binds
tighter than `|`, and parentheses group. Operators are delimited by spaces, so values keep their own `-` and `:`.
`@<set>` references a named set (`@<import>.<set>` within an imported book).

| field | selects | example |
| --- | --- | --- |
| `role` | tensors by lifecycle role (`base`, `adapter[:name]`, `intervention[:name]`, `latent:<name>`, `fitted`, `artifact`) | `role:adapter` |
| `layers` | blocks: an index, a list, a half-open slice (negatives count from the depth), `odd`, `even`; an index or slice end may be a fraction of the depth, `0.25d` (the floor of a quarter of it; book schema 7) | `layers:-4:`, `layers:8:-4`, `layers:0.25d:0.75d` |
| `module` | tensors under modules whose path ends with the pattern, on whole segments (`*` one segment, `**` any) | `module:q_proj`, `module:mlp.*` |
| `leaf` | tensor-name suffixes, on whole segments | `leaf:weight` |
| `name` | a regex over tensor names, matched in full: the only regex field, and quoted | `name:"layers\.0\..*"` |
| `component`, `stack` | architecture-neutral components and block stacks (need a vocabulary structure provider) | `component:attn.q` |
| `point` | an activation site (steering), not tensors | `point:resid_post` |
| `passes` | a one-based pass through a weight layer at a mapped activation site (book schema 12) | `passes:3` |
| `features`, `subspace`, `space` | latent feature spaces, named bases, linear maps a site is read through | `features:sae` |
| `units` | indices along a target's unit axis (neurons, features); on a tensor's gradient, an element mask instead: `{elements: {file, key, sha256}}`, a sidecar bitmask of the tensor's exact shape (book schema 9, checkpoint books only) | `units:[431, 1007]` |
| `of` | `grad` acts on the gradient of what the rest selects (never filters; book schema 2); with `units` on tensors, a gradient mask, and masks combine exactly as their sets do (union, intersection, difference) along one dimension or element by element; a union's disjuncts may differ (`split_of` gives the value and gradient parts) | `of:grad` |
| `axis` | which dimension of a gradient the term's `units` index: `unit` (the component's unit axis, from the map's layout) or a physical dimension; unset, the first dimension (book schema 6; resolve with `gradient_selections`) | `axis:unit` |
| `path` | routes between ordered waypoints (addresses, optionally anchored to tokens, or `{graph, nodes}`), acted on by a treatment's `stop` or `route` gradient op and placed by the consumer's route resolver (mapping form only; book schema 10; `venngram.paths`) | `{path: ["component:embed", "@heads", "component:unembed"]}` |

Examples, on a four-block model with a LoRA pair under each block's `q_proj`:

| address | selects |
| --- | --- |
| `role:adapter & layers:-2:` | the adapters of blocks 2 and 3 |
| `role:adapter - layers:-2:` | the other adapters (blocks 0 and 1) |
| `role:base & layers:-1: & leaf:weight` | the last block's base weights |
| `module:mlp.* \| module:head` | every MLP tensor, and the head |
| `module:up_proj & layers:odd` | the MLP of blocks 1 and 3 |
| `name:"layers\.0\..*\.lora_A\..*"` | a regex, inside an address |

The mapping form is the same record: the fields of one mapping intersect, `any: [...]` unites, `all: [...]`
intersects, `exclude: ...` subtracts, `optional: true` allows an empty match and `expect: N` requires exactly `N`
members. `{"role": "adapter", "exclude": "@top"}` is `role:adapter - @top`.

## Address books: define sets once

```yaml
schema_version: 1
name: qwen3-tuning-sets
portability: family          # family: roles, layers, modules, components; checkpoint: units of one model revision
applies_to: {architectures: [Qwen3ForCausalLM]}
imports: {base: ./common.yaml}
sets:
  top:  {role: adapter, layers: "-4:"}
  rest: {role: adapter, exclude: "@top"}
```

- A book is **data only**: loading it runs no code, so it is safe to share.
- `portability: checkpoint` books carry unit indices meaningful in one checkpoint, so they must pin
  `applies_to.models` with an `id` and a `revision`.
- A book whose `name`, `module` or `leaf` values are written in a view's own vocabulary (TransformerLens names, say)
  declares it: `requires: {naming_scheme: tl}`. A consumer calls `book.check_requires(naming_scheme=...)` with its
  view's scheme, and a mismatched book is refused by name instead of matching the wrong tensors. Books of `role`,
  `layers` and `component` sets are vocabulary-neutral and declare none.
- A book that names latent models pins what they were trained against in `requires` (schema 8): `base_model: {id,
  revision}` and, per latent model, its dictionary `release` (`{sae_lens: <release>, id: <id>}` or `{hub: <repo>,
  revision: <rev>}`) and the exact `frames` versions its units are read in. A consumer calls
  `book.check_pins(base_model=..., latent_models=...)` with what it loaded, and a mismatch is refused naming each entry.
- **Keep run parameters out of books.** Doses, learning rates and intervention strengths belong in the consumer's
  configuration. A book names sets, nothing else.
- In finetuning-scheduler, a schedule's top-level `addresses:` key names a book file (relative to the schedule) or
  holds one inline, and phases reference its sets as `@<set>`.

## Resolving and debugging

Resolve an address against a `ParamSpace`: the tensor names, their roles, and a structure provider.

```python
from venngram import ParamSpace, select
from venngram.physical import PhysicalStructure

names = [n for n, _ in model.named_parameters()]
roles = {n: "adapter" for n in names if ".lora_" in n}
space = ParamSpace(names, roles=roles, structure=PhysicalStructure(model))
select("role:adapter & layers:-2:", space)
select("@rest", space, book.sets())            # with an AddressBook's sets
```

- **Print the selection, and read it.** An address that resolves can still select the wrong tensors.
- **A refusal names its reason**, as `AddressUnresolvable`: `role:adapter & layers:9` on four blocks says
  "Layer 9 is outside a stack of 4 blocks". Other reasons: it matches nothing, a set is missing, `component:` without
  a vocabulary provider, `point:` where tensors are wanted. Add `optional: true` only when an empty match is
  genuinely acceptable.
- **`layers` needs one block list.** `PhysicalStructure` finds a model's single top-level `nn.ModuleList`. A model
  with several (an encoder and a decoder) is refused until one is named with `layers_pattern`.
- **`component` needs a vocabulary.** `venngram.view(model)` supplies one for any architecture with a base
  map (`venngram.maps.known_architectures()`), as can a consumer's own view. Without one, fall back to
  `module:`, which is tied to the model's module names.
- `parse(...)` and `to_string(...)` round-trip the two forms, which helps when composing an address in code.

## Looped models

Resolve `point:state.active & layers:12 & passes:3` with `cmap.execution.resolve(...)` on a verified schema 7 map.
`layers` still identifies a weight layer. `passes` chooses one forward visit. With `passes` omitted, a point resolves
to every call it has on the selected layers. Shared weights use the usual parameter address, such as
`component:attn.q & layers:12`. Freeze or thaw the weight once even when it serves several passes.

Use each result's `module`, `call_index`, `hook` and `path` to locate the activation. Call indices start at zero and
count calls of that module. A gate that starts on pass two has call index zero there. A module reused across layers
has a single call sequence. `ExecutionSite.read` follows the input or output path and can select a token on axis 1.
See the [Inkling 2 Mini map](../../../registry/candidates/README.md) for a model with active blocks, completed blocks,
and messages available only on later passes.

## Related

- In finetuning-scheduler, a tunable view supplies the roles and structure a schedule's addresses resolve against.
  Its `tunable-view-generator` skill writes one for a model that has none.
- interpretune resolves `point`, `features` and `subspace` addresses per interpretability backend.
