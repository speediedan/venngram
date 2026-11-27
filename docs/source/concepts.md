# Concepts, grouped

venngram has more names than ideas. This page groups the names that belong to one idea, so a reader learns the idea
once. Nothing here changes a document: each name keeps its own field, kind or section.

## Selecting and treating

An **address** selects: it names tensors, activation sites, units or features, and an address book names its sets.
A **treatment** acts: its ops change what an address selects, at a stage of the forward pass (`stages`), in the
weights (`weights`) or in the gradients (`gradients`). venngram declares both and executes neither: a consumer
applies a treatment, and says what it honours in its support record's `treatments` section.

Two address fields modify a selection rather than narrow it, and say so here because they look like filters:

- `of` says what a consumer acts on, the value (the default) or the gradient (`of: grad`). It never filters: `of:
  grad` selects the same tensors as the address without it, and reads their gradients.
- `axis` says which dimension of a tensor's gradient its `units` index, for a gradient mask.

## Intervention, the role, and treatment, the act

`intervention` is a role kind: a tensor that is a parameter of an inserted intervention module (a steering module,
a ReFT intervention) has that role, as an adapter's tensor has the `adapter` role. What a treatment does to a model is
called applying the treatment. Where these docs mean the act, they say "treatment" ("a treatment applied as an
intervention" reads fine, "the intervention's stages" does not): the stages, ops and values belong to the treatment.

## One coordinate system: `features`, `subspace` and `space`

Three address fields name the coordinate system a target is read in, rather than which tensors or sites it is:

| field | reads the target in | provided by |
| --- | --- | --- |
| `features` | a latent model's features (an SAE, a transcoder) | the latent model, loaded beside the model |
| `subspace` | a named basis at a site | an operand the book or treatment carries |
| `space` | a registered linear map the site is read through | the map's registration |

They are one idea, a change of basis at a site, and they stay three fields because their providers differ, which is
why each validates differently. A fourth such field would be the time to merge them into one.

## Sidecars: one tier, four kinds

A **sidecar** is a binary file (safetensors) that carries a document's bulk data, is cited by the document as
`{file, sha256}`, and is never opened on its own:

| sidecar kind | carries | for the document |
| --- | --- | --- |
| `units` | large unit sets | an address book |
| `weight_delta` | a weight patch's tensors | a `weight_patch` |
| `graph_table` | a graph subset's node values | a `graph_subset` |
| `unit_table_rows` | a unit table's rows | a `unit_table` |

So a reader meets nine document kinds (address books, component maps, rules, verification records, support records,
op declarations, weight patches, graph subsets and unit tables, plus a store's manifest and the registry index), and
one tier of sidecars beneath them. Each sidecar keeps its own kind in its header so a misplaced file is refused by
name.

## Versions

Every document kind carries a `schema_version` that is a feature level: each version adds one capability, and a
writer stamps the lowest version its content needs, so an older reader still reads every document that uses none of
the newer features. `AddressBook.needed_schema` gives a book's. See "Versions and evolution" on the
[home page](index.md#versions-and-evolution) for each kind's window and what a version adds.
