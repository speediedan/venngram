---
name: venngram-addressing
description: Compose, resolve and debug venngram addresses, the set-algebraic way to name slices of a model (tensors to tune, activation sites to steer, units, latent features) by role, layer, module and component rather than by parameter names and regexes; and write shareable address books. Use when writing a fine-tuning schedule or steering configuration that names model parts, when an address does not resolve or selects the wrong tensors, when one selection must work across several models of a family, or when asked about the address grammar. Do NOT use when a single explicit name or a simple regex already says what is meant.
---

# Addressing model slices with venngram

An **address** selects a slice of a model. Its fields intersect, and addresses combine by set algebra. Consumers
such as finetuning-scheduler (schedules) and interpretune (steering and analysis) accept addresses wherever they
accept names. Every result below comes from `references/examples.py`, which `tests/test_skill_examples.py` runs.

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
| `role` | tensors by lifecycle role (`base`, `adapter[:name]`, `intervention[:name]`, `latent:<name>`, `fitted`, `instrument`) | `role:adapter` |
| `layers` | blocks: an index, a list, a half-open slice (negatives count from the depth), `odd`, `even` | `layers:-4:`, `layers:8:-4` |
| `module` | tensors under modules whose path ends with the pattern, on whole segments (`*` one segment, `**` any) | `module:q_proj`, `module:mlp.*` |
| `leaf` | tensor-name suffixes, on whole segments | `leaf:weight` |
| `name` | a regex over tensor names, matched in full: the only regex field, and quoted | `name:"layers\.0\..*"` |
| `component`, `stack` | architecture-neutral components and block stacks (need a vocabulary structure provider) | `component:attn.q` |
| `point` | an activation site (steering), not tensors | `point:resid_post` |
| `features`, `subspace`, `space` | latent feature spaces, named bases, linear maps a site is read through | `features:sae` |
| `units` | indices along a target's unit axis (neurons, features) | `units:[431, 1007]` |

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
- **`component` needs a vocabulary.** A consumer's view, or interpretune's component maps, supplies one. Without it,
  fall back to `module:`, which is tied to the model's module names.
- `parse(...)` and `to_string(...)` round-trip the two forms, which helps when composing an address in code.

## Related

- In finetuning-scheduler, a tunable view supplies the roles and structure a schedule's addresses resolve against.
  Its `tunable-view-generator` skill writes one for a model that has none.
- interpretune resolves `point`, `features` and `subspace` addresses per interpretability backend.
