# venngram

A set-algebraic protocol for addressing model slices: tensors (for tuning), activation sites (for steering and capture),
and units inside either (neurons, latent features, featurizer blocks, subspaces). You name a slice once, combine slices
with set algebra, and share them as data documents.

**The name** joins *Venn*, for the set algebra (unions, intersections, differences, overlapping regions), and *engram*,
neuroscience's term for the physical trace of a memory or function in the brain. A venngram is a set-algebraic
specification of the neural traces it targets.

> **Status: experimental (0.x).** The API and the document schema may change. Design discussion:
> [finetuning-scheduler#153](https://github.com/speediedan/finetuning-scheduler/issues/153).

## A model's view in one call

```python
from venngram import view

v = view(model)   # a Qwen3, Llama, OPT, Gemma ... model, with or without PEFT LoRA adapters
# every tensor now has a role (base, adapter, ...), a layer and an architecture-neutral component, so
# "role:adapter & component:mlp.* & layers:-4:" means the same tensors on every mapped architecture
```

`view(model)` composes the model's view from published data: the **base map** of its architecture (found from the
model's class) and the **overlay map** of each library whose modules it carries (PEFT's LoRA layers, a spliced SAE or
transcoder, a steering module). Neither the architecture nor the library needs code. [Maps](#maps-base-and-overlay)
describes both shapes and how they are found.

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

### Large unit sets and treatments

A `units` value may name a sidecar instead of listing indices:
`units: {file: units.safetensors, key: L15, sha256: ...}`, flat or per layer. A sidecar is a safetensors file holding a
sorted index array or a packed bitmask per set, read with the standard library (no `torch`, no pickle). A changed file,
or a set written for another axis width, is refused. `venngram.units.units_value` keeps small sets inline and writes
large ones to a sidecar.

A book may carry **treatments**: what is done to its sets for one model, by name, under its `treatments` section, so a
selection and what is done to it travel together in one document. A treatment (a **recipe**, `venngram.recipe.Recipe`)
is ordered **stages**, each applying ordered **ops** at one site under one
token scope. An op acts on the whole activation, a set's units or a latent space's features; it may take a named
**operand** (a direction, a pair, a subspace, each in a declared basis, inline or in a sidecar) and a **value** (a dose,
a per-unit table, or a column of recorded evidence through declared transforms). Ops are declared specs
(`venngram.ops`: `replace`, `add`, `scale`, `project`, `reject`, `patch`, `clamp`, and any registered); venngram
validates and composes them and never executes one. A token scope is `all`, `last`, `prompt` (the input, before any
generated token), `generated` (each decode step), explicit positions or a slice. A consumer declares what it honours as
a `Support` record, in code or as the `steering` section of a published `support_record` document (see "Support
records" below), and `check_support` names everything a recipe asks for beyond it. `compose` joins recipes stage by stage, refusing ops that
do not commute unless the composition gives an order. A `kind: recipe` document naming its book (the form written
before books and recipes merged) is still read, and `Recipe.as_book` turns it into its book with one treatment; schema 1
recipes are read through `migrate`. Applying a treatment is the consumer's; treatments written in another tool's format
are read by an adapter outside venngram's core.

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
by reference. The Hub stores use `huggingface_hub`; a new place to keep
documents is a new store (`venngram.store.register_store`).

`python -m venngram.conformance <store>:<ref>[@<revision>] [--support <record>]` checks a published directory the way a
reader meets it: every listed document is fetched at one revision and loaded by its kind, every book's sets are
checked against each support record's `tuning` section, and its treatments (and any recipe) against its `steering`
section. It exits non-zero when a
document fails to load or a recipe or book asks for more than a given record honours, so a format adapter's CI, or a
scheduled job over published examples, can run it.

### Support records

A consumer's **support record** (`venngram.support`) declares, section by section, each contract it offers, so a
document asking for anything else is refused by name before anything runs, and without the consumer installed:

```yaml
schema_version: 2
kind: support_record
name: finetuning-scheduler
sections:
  tuning: {fields: [role, name, module, layers, stack, component, leaf], targets: [param], units: none,
           naming_schemes: [physical]}
  steering: {ops: [add, scale]}
```

The core sections are `steering` (a book's treatments), `tuning` (address books whose sets select tensors to tune),
`circuits` (traced or validated circuits, attribution graphs in general; a prototype) and `formats` (the foreign formats
a package reads, with where to install it from). A section a record omits is a contract its
consumer does not offer. A section type the reader does not know is reported and skipped, never fatal: a section
defined outside venngram is namespaced (`acme.probes`), declared under `extensions` with the schema that defines it,
and parsed wherever code registers a parser for it (`register_support_section`). Schema 1 records read as a `steering`
section.

### Sources: map sources and formats

A **map source** derives a component map from something that knows an architecture (a model's own module tree through
`venngram.maps.drafter.ModuleTreeSource`; TransformerLens's adapters through
`venngram.maps.transformer_lens.TransformerLensSource`; or any package's own). The module-tree drafter names modules by
**drafting rules** that are data, from any number of sources: rules mined from the registry's own maps (so they grow
with it), bundled rule documents for transformer and state-space blocks, or any a family publishes (the `draft`
section of a `kind: rules` document). What no rule names becomes a `draft.*` row for a person or an agent to confirm, and
`draft_with_model` asks a model the caller chooses to complete the draft, then checks it against the model; a **format** translates documents
another tool writes into books and recipes, for tools that do not integrate venngram themselves. Both are found by
entry point (`venngram.map_sources`, `venngram.formats`) or registered in process, and record their name and version as
provenance. A format's translation is written to a directory (`venngram.sources.write_translation`) and published like
any other documents. A manifest can list an installable format as data (a support record's `formats` section, naming
the formats a package reads and where to install it from), so a reader can say how to read a foreign document without
running anything.

## What resolves where

The grammar, the algebra and the book schema know no model library or interpretability package. Knowledge of an
architecture or a backend enters only through providers:

- **`PhysicalStructure`** (included) finds any PyTorch model's block list, so `layers` works everywhere.
- **Vocabulary-backed structure providers** add architecture-neutral components: `venngram.maps` builds one from a
  per-architecture component map (below).
- **Site and latent resolvers** turn `point`, `features` and `subspace` addresses into hooks and loaded latent models
  per backend.

[interpretune](https://github.com/speediedan/interpretune) supplies both of the last two.

## Maps: base and overlay

`venngram.maps` reads maps, in two shapes. A **base map** (a component map) is one YAML document per architecture
naming where each of its components lives in its module tree; an **overlay map** describes what a library adds to a
tree and the role each of its tensors takes (below). A base map can describe any PyTorch model, not only the
architectures some library already adapts:

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

**Maps are found through registries, not bundled.** A map is looked up first among those registered in the running
process, then in the configured registries (`VENNGRAM_REGISTRIES`, a list of index locations; unset, it is the seed
registry, and empty, none). A registry is an index over published maps, rebuilt by a crawler from manifest locations
and from Hub repositories tagged `venngram`, so a map from any source (written by hand, drafted from a module tree, or
derived from TransformerLens or another package) is indexed and found like the seeds. Several maps may serve one
architecture: lookup serves the best verified one, and `venngram.registry.rows` lists every candidate. The index is a
parquet dataset partitioned by kind and subject (architecture, or model id for checkpoint documents), so a lookup reads
one small file however large the registry grows; the last revision read is served from the cache, with a warning, when
the store is unreachable (or `HF_HUB_OFFLINE` is set), and a registry never read before fails by name. A location may
pin a revision (`hub-dataset:<owner>/<repo>/index@<commit>`), so an example rendered against it cannot be moved by a
later registry generation. `venngram.maps.known_architectures()` and `known_overlays()` list what the process and the
registries know.

```bash
python -m venngram.registry build --source hub-dataset:<owner>/<repo>/maps --discover-hub --out ./index
python -m venngram.registry publish ./index hub-dataset:<owner>/<repo>/index
python -m venngram.registry lookup Qwen3ForCausalLM     # every candidate, and the one served
```

The seeds cover GPT-2, Llama, Qwen3, Qwen3.5 (hybrid linear and softmax attention, multimodal), OPT, Gemma 2, Gemma 3
and Gemma 3 multimodal, each checked against a model of its architecture and cross-checked against TransformerLens's
adapter for it. TransformerLens is one source of seeds (`venngram.maps.transformer_lens` derives a map from an
adapter); venngram does not depend on it.

A map is verified against an architecture without downloading weights: on a meta-device model built from the
configuration, every row's module exists, every parameter is claimed by exactly one component, and the widths the
configuration implies hold (`attn.q` and `attn.k` differ under grouped-query attention, so a swap fails); a forward on
a small random-init model confirms the order within a block, which catches swapped labels that widths cannot. What the
width and order checks expect is data, not code: rule sets (the `verify` section of a `kind: rules` document, which
holds a family's drafting rules too, in its `draft` section) layered from a bundled `core` set (the conventional decoder
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

**Overlay maps.** What a library adds to a model's module tree is data too, so venngram's code names no adapter
library. An overlay says which module patterns take which role, and when it applies:

```yaml
schema_version: 5
overlay: peft.lora
applies_when:            # any condition holding means the overlay applies
  module_classes: [peft.tuners.lora.layer.LoraLayer]   # matched against each module's class hierarchy
  module_names: [lora_A, lora_B, lora_E, lora_embedding_A, lora_embedding_B, lora_magnitude_vector]
roles:                   # module pattern -> role; the first matching pattern wins
  lora_A: adapter:{key}   # the entry of the keyed container matched: here, the adapter name
  lora_B: adapter:{key}
  lora_E: adapter:{key}
  lora_embedding_A: adapter:{key}
  lora_embedding_B: adapter:{key}
  lora_magnitude_vector: adapter:{key}
```

A tensor under an overlay's module keeps the layer and component of the module it is attached to, so one overlay serves
every architecture. A role ending in `:{key}` takes the entry of the keyed container (`nn.ModuleDict`,
`nn.ParameterDict`) the pattern matches, so each named adapter has a role of its own (`adapter:default`); where the
matched module is not a keyed container, the qualifier is dropped (`adapter`). Overlays are documents of the same kind
as base maps, found the same way (registered in process, then the registries, as subject `overlay:<name>`); the seed
overlays cover each PEFT tuner family (`peft.lora`, `peft.ia3`, `peft.loha`, `peft.lokr`, `peft.oft`,
`peft.boft`, `peft.vera`, `peft.fourierft`). A spliced SAE or transcoder
(`latent:<name>`, or `instrument` when frozen), a block-sparse featurizer or a steering module (`intervention:<kind>`)
is an overlay in the same way: a new library is a published document, not a venngram release.

**The view.** `view(model)` (a `venngram.maps.MappedView`) applies **every map that applies**: the model's base map and
every known overlay whose `applies_when` holds, refusing two overlays that give one tensor different roles. To choose
instead, name the maps: `view(model, [base_map, overlay])`, or `view(model, overlay)` to keep the looked-up base map with
exactly that overlay. `roles=` sets individual tensors' roles over the overlays'. A model held inside a module of a
framework's own (a wrapper carrying it under `model.`, with steering vectors or spliced dictionaries beside it) is
described at that submodule: `view(wrapper)` finds the one mapped descendant, or `view(wrapper, mount="model")` names
it. Names stay the wrapper's, and the wrapper's own tensors keep their roles with no layer or component.

```python
from venngram import view
from venngram.maps import load_map, register

register(load_map("hybrid.yaml"))    # only for an architecture no registry indexes
register(load_map("my_adapter.yaml"))  # only for a library no registry indexes
v = view(model)
```

## Tunable views

`venngram.tuning` holds the tunable view protocol: the contract between the logical names a fine-tuning consumer uses
and a model's physical parameters, with a role for each tensor.
[Fine-Tuning Scheduler](https://github.com/speediedan/finetuning-scheduler) consumes it, and its schedules accept
addresses.

## Versions and evolution

Published documents outlive the venngram that wrote them, so every document kind carries one integer
`schema_version`, and each reader accepts a window of them:

| Kind | Reads | Writes | Keys it does not know |
| --- | --- | --- | --- |
| component map | 1 to 4 | 4 | ignored when descriptive |
| overlay map | 4 to 5 | 5 | unknown `applies_when` conditions refused |
| rules | 2 | 2 | refused |
| address book | 1 | 1 | ignored at the top level |
| recipe | 1 to 2 | 2 | refused in `method` and in an op's parameters |
| support record | 1 to 2 | 2 | refused; an unknown *section* is skipped with a warning |
| manifest (`venngram.yaml`) | 1 | 1 | refused in a document entry |
| registry index header | 1 | 1 | |

**What bumps a version.** A key a reader may ignore without misreading the document (a description, an annotation)
never does, where the kind ignores unknown keys. A key that changes what a document selects or how a row resolves
always does, since a reader that skipped it would resolve the wrong thing silently. Where a kind refuses unknown keys,
any new key bumps it.

**Too new, too old, malformed.** Each is refused by name, with its own message. A document newer than the reader is a
`venngram.NewerSchemaError` (a `ValueError`; for books and recipes also an `AddressError`), so code that serves
documents from shared places can tell it apart from a broken one. Registry lookups do exactly that: a registry may hold
documents written for a newer venngram, and a lookup passes over them with a warning and serves the best one it reads,
while a malformed document of a schema it reads still fails. Each index row records its document's schema, so a newer
one is passed over without being fetched.

**Settled before 0.1, and not yet.** The windows above, the newer-schema refusal and the registry rule are what 0.1
commits to. Still open (speediedan/venngram#18): a `venngram migrate` that rewrites an old-schema document to the
current one, golden fixtures of every published schema, a deprecation cycle (warn, then refuse, with a strict mode) for
document kinds, API names and vocabulary names (maps already carry `deprecated_since` and `replacement`), a version
check when reading verification records, an audit of the places a new architecture or tool still needs a venngram
change, and a stated compatibility surface for each consumer.

## Agent skill

`.agents/skills/venngram-addressing/` (mirrored in `.claude/skills/`) is an agent skill, in the open skill format,
for composing, resolving and debugging addresses and writing address books. Its examples run in the test suite.

## License

Apache-2.0.
