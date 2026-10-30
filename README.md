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

### Large unit sets and recipes

A `units` value may name a sidecar instead of listing indices:
`units: {file: units.safetensors, key: L15, sha256: ...}`, flat or per layer. A sidecar is a safetensors file holding a
sorted index array or a packed bitmask per set, read with the standard library (no `torch`, no pickle). A changed file,
or a set written for another axis width, is refused. `venngram.units.units_value` keeps small sets inline and writes
large ones to a sidecar.

A **recipe** is a treatment over a book's sets: ordered **stages**, each applying ordered **ops** at one site under one
token scope. An op acts on the whole activation, a set's units or a latent space's features; it may take a named
**operand** (a direction, a pair, a subspace, each in a declared basis, inline or in a sidecar) and a **value** (a dose,
a per-unit table, or a column of recorded evidence through declared transforms). Ops are declared specs
(`venngram.ops`: `replace`, `add`, `scale`, `project`, `reject`, `patch`, `clamp`, and any registered); venngram
validates and composes them and never executes one. A token scope is `all`, `last`, `prompt` (the input, before any
generated token), `generated` (each decode step), explicit positions or a slice. A consumer declares what it honours as
a `Support` record, in code or published as a `support_record` document, and `check_support` names everything a recipe
asks for beyond it. `compose` joins recipes stage by stage, refusing ops that
do not commute unless the composition gives an order. Schema 1 recipes are read through `migrate`. Applying a recipe is
the consumer's; recipes written in another tool's format are read by an adapter outside venngram's core.

### Sharing documents: manifests and stores

A directory of venngram documents (a Hub repository, a bucket prefix, a local folder) lists them in `venngram.yaml`:
each document's kind, path, schema version, sha256 and the files it needs (a book's sidecars and imports, a recipe's
book). A **document reference** names one document wherever it lives:
`<store>:<ref>[@<revision>]:<path>[#<sha256>]`, for example
`hub-model:speediedan/venngram-maps@3f1e...:maps/Qwen3ForCausalLM.yaml`.

```python
from venngram.store import load, publish

publish("hub-dataset", "me/my-sets", "./sets", documents=[("recipe", "recipes/chess.yaml")])
book = load("hub-dataset:me/my-sets:books/chess.yaml")
```

A reference without `@<revision>` or `#<sha256>` follows the latest published snapshot; `venngram.store.pin` returns
the exact reference it resolves to (a commit and the content hash), for records that must be reproducible. On the Hub
stores the ref may name a directory of a repository or bucket (`me/my-sets/qwen3`), each with its own manifest, so one
repository can hold many small manifests. Fetching reads the manifest first, at one revision, and refuses a path it does
not list or a file whose content differs from what the manifest or the reference cites. A book can import another book
by reference. The Hub stores need the `hf` extra; a new place to keep
documents is a new store (`venngram.store.register_store`).

`python -m venngram.conformance <store>:<ref>[@<revision>] [--support <record>]` checks a published directory the way a
reader meets it: every listed document is fetched at one revision and loaded by its kind, and every recipe is checked
against each support record given. It exits non-zero when a document fails to load or a recipe asks for more than a
given record honours, so a format adapter's CI, or a scheduled job over published examples, can run it.

### Sources: map sources and formats

A **map source** derives a component map from something that knows an architecture (a model's own module tree through
`venngram.maps.drafter.ModuleTreeSource`; TransformerLens's adapters through
`venngram.maps.transformer_lens.TransformerLensSource`; or any package's own). The module-tree drafter names modules by
**drafting rules** that are data, from any number of sources: rules mined from the registry's own maps (so they grow
with it), bundled rule documents for transformer and state-space blocks, or any a family publishes
(`kind: drafting_rules`). What no rule names becomes a `draft.*` row for a person or an agent to confirm, and
`draft_with_model` asks a model the caller chooses to complete the draft, then checks it against the model; a **format** translates documents
another tool writes into books and recipes, for tools that do not integrate venngram themselves. Both are found by
entry point (`venngram.map_sources`, `venngram.formats`) or registered in process, and record their name and version as
provenance. A format's translation is written to a directory (`venngram.sources.write_translation`) and published like
any other documents. A manifest can list an installable format as data (`kind: format_adapter`), so a reader can say how
to read a foreign document without running anything.

## What resolves where

The grammar, the algebra and the book schema know no model library or interpretability package. Knowledge of an
architecture or a backend enters only through providers:

- **`PhysicalStructure`** (included) finds any PyTorch model's block list, so `layers` works everywhere.
- **Vocabulary-backed structure providers** add architecture-neutral components: `venngram.maps` builds one from a
  per-architecture component map (below).
- **Site and latent resolvers** turn `point`, `features` and `subspace` addresses into hooks and loaded latent models
  per backend.

[interpretune](https://github.com/speediedan/interpretune) supplies both of the last two.

## Component maps

`venngram.maps` reads component maps: one YAML document per architecture naming where each of its components lives in
its module tree. A map can describe any PyTorch model, not only the architectures some library already adapts:

- rows name whatever components the model has (`mixer.conv`, `moe.router`, `projector`);
- a component whose behaviour the bundled kinds do not describe gets a kind declared in the document (`kinds:`);
- a heterogeneous stack restricts a row to the layers that have it (`layers: odd`), or marks it `optional` so it
  applies wherever its module exists (a hybrid stack whose layout varies with the model's size);
- further block lists (an encoder and a decoder, a vision tower) are declared as named stacks (`stacks:`), and a model
  with no block list is mapped by its other rows.

```yaml
schema_version: 2
architecture: Hybrid          # matched against the model's class name
kinds:
  mixer: {output: tensor, sublayer: true}
components:
  blocks.{i}:            {module: "trunk.{i}",              kind: block}
  blocks.{i}.attn.q:     {module: "trunk.{i}.attention.wq", kind: linear, layers: odd}
  blocks.{i}.mixer.conv: {module: "trunk.{i}.mixer.conv",   kind: linear, layers: even}
  unembed:               {module: "readout",                kind: unembed}
```

A shared vocabulary (`embed`, `ln1`, `attn.q`, `attn.k`, `attn.v`, `attn.o`, `ln2`, `mlp.gate`, `mlp.in`, `mlp.out`,
`ln_final`, `unembed`, and a few more) is the convention for the parts many architectures have, so one address selects
the same part on all of them; its names match the ones TransformerLens's bridge and interpretune's activation points
emit. A component outside it simply uses a name of its own.

Seed maps are bundled for GPT-2, Llama, Qwen3, Qwen3.5 (hybrid linear and softmax attention, multimodal), OPT, Gemma 2,
Gemma 3 and Gemma 3 multimodal, each checked against a
model of its architecture and cross-checked against TransformerLens's adapter for it. TransformerLens is one source of
seeds (`venngram.maps.transformer_lens` derives a map from an adapter); venngram does not depend on it.

A map is verified against an architecture without downloading weights: on a meta-device model built from the
configuration, every row's module exists, every parameter is claimed by exactly one component, and the widths the
configuration implies hold (`attn.q` and `attn.k` differ under grouped-query attention, so a swap fails); a forward on
a small random-init model confirms the order within a block, which catches swapped labels that widths cannot. What the
width and order checks expect is data, not code: rule sets layered from a bundled `core` set (the conventional decoder
parts), the map's own `verification:` section, and any the caller passes, where a later layer replaces an entry by key
and `null` removes one. A map for an architecture nobody anticipated states its own widths and orders, and a
parallel-residual one drops the core chain that does not hold for it:

```yaml
verification:
  config:  {experts: [num_experts]}                       # a meaning: configuration fields, or arithmetic
  widths:  {moe.router: {in: hidden, out: experts}}       # expressions over the meanings
  order:   {attn_then_mlp: null,                           # remove a core chain
            parallel_mlp: [ln2, mlp.in, mlp.out]}          # add one of its own
```

A rule applies only where the map names its component and the configuration states what it needs, so a map is
reported `partial` rather than failed for what nothing claims. Each record keeps the order observed in the first block,
evidence a later checkpoint can be compared against without any rule.

```bash
python -m venngram.maps.verification Qwen/Qwen3-0.6B --record qwen3.yaml   # or venngram.maps.verify(config_or_model)
python -m venngram.maps.verification Qwen/Qwen3-0.6B --against qwen3.yaml  # a later run must observe the same order
```

A map is all a tunable view needs, so a mapped architecture needs no view code:

```python
from venngram.maps import from_document, MappedView, PEFT_LORA_RULES, register

register(from_document(yaml.safe_load(open("hybrid.yaml"))))  # a bundled architecture needs no registration
view = MappedView(model, role_rules=PEFT_LORA_RULES)          # the map is found from the model's class
# `component:mlp.out & layers:-4:` selects the same tensors on a Llama, an OPT or a Gemma model
```

## Tunable views

`venngram.tuning` holds the tunable view protocol: the contract between the logical names a fine-tuning consumer uses
and a model's physical parameters, with a role for each tensor.
[Fine-Tuning Scheduler](https://github.com/speediedan/finetuning-scheduler) consumes it, and its schedules accept
addresses.

## Agent skill

`.agents/skills/venngram-addressing/` (mirrored in `.claude/skills/`) is an agent skill, in the open skill format,
for composing, resolving and debugging addresses and writing address books. Its examples run in the test suite.

## License

Apache-2.0.
