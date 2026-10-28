# venngram

A set-algebraic protocol for addressing model slices: tensors (for tuning), activation sites (for steering and capture),
and units inside either (neurons, latent features, featurizer blocks, subspaces). You name a slice once, combine slices
with set algebra, and share them as data documents.

**The name** joins *Venn*, for the set algebra (unions, intersections, differences, overlapping regions), and *engram*,
neuroscience's term for the physical trace of a memory or function in the brain. A venngram is a set-algebraic
specification of the neural traces it targets.

> **Status: experimental (0.x).** The API and the document schema may change. Design discussion:
> [finetuning-scheduler#153](https://github.com/speediedan/finetuning-scheduler/issues/153).

## Addresses

An address is a record whose fields intersect:

| field | selects | example |
| --- | --- | --- |
| `role` | tensors by lifecycle role | `adapter`, `adapter:top` |
| `layers` | blocks: an index, a list, a half-open slice (negatives count from the depth), `odd`, `even` | `-4:`, `8:-4` |
| `module` | tensors under modules whose path ends with the pattern, on whole segments | `q_proj` |
| `name` | tensor names by regex (`re.fullmatch`) | `"model\.layers\.2[4-7]\..*"` |
| `leaf` | tensor-name suffixes | `lora_A.weight` |
| `component`, `stack` | architecture-neutral components and block stacks (needs a vocabulary provider) | `attn.q` |
| `point` | an activation site | `resid_post` |
| `features`, `subspace`, `space` | latent feature spaces, named bases, and maps a site is read through | `sae`, `jlens` |
| `units` | indices along a target's unit axis | `[431, 1007]` |

The string form writes each field as `<field>:<value>`:

- `&` intersects, `-` subtracts and `|` unites;
- `@<set>` references a named set;
- operators are delimited by spaces.

```python
from venngram import ParamSpace, select
from venngram.physical import PhysicalStructure

model = ...  # any torch.nn.Module with a block list
roles = {n: "adapter" for n, _ in model.named_parameters() if "lora_" in n}
space = ParamSpace([n for n, _ in model.named_parameters()], roles=roles, structure=PhysicalStructure(model))

select("role:adapter & layers:-4:", space)          # the adapters of the top four blocks
select("role:adapter & module:q_proj", space)       # every q_proj adapter (overlapping the above)
select("role:adapter - layers:-4:", space)          # the rest
```

## Address books

Named sets live in versioned YAML documents. They are data only, so loading one runs no code.

```yaml
schema_version: 1
name: qwen3-tuning-sets
portability: family   # or checkpoint: unit indices of one pinned model revision
applies_to: {architectures: [Qwen3ForCausalLM]}
imports: {base: ./common.yaml}
sets:
  top:  {role: adapter, layers: "-4:"}
  rest: {role: adapter, exclude: "@top"}
```

```python
from venngram import AddressBook, select

book = AddressBook.from_file("qwen3.yaml")
select("@rest", space, book.sets())
```

## What resolves where

The grammar, the algebra and the book schema know no model library or interpretability package. Knowledge of an
architecture or a backend enters only through providers:

- **`PhysicalStructure`** (included) finds any PyTorch model's block list, so `layers` works everywhere.
- **Vocabulary-backed structure providers** add architecture-neutral components.
- **Site and latent resolvers** turn `point`, `features` and `subspace` addresses into hooks and loaded latent models
  per backend.

[interpretune](https://github.com/speediedan/interpretune) supplies both of the last two.

## Tunable views

`venngram.tuning` holds the tunable view protocol: the contract between the logical names a fine-tuning consumer uses
and a model's physical parameters, with a role for each tensor.
[Fine-Tuning Scheduler](https://github.com/speediedan/finetuning-scheduler) consumes it, and its schedules accept
addresses.

## License

Apache-2.0.
