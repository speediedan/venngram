# Inkling 2 Mini execution map

The [candidate map](inkling2_mini.yaml) describes the text forward of the
[Inkling 2 Mini checkpoint at `b85c3cfa`](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/b85c3cfa615fc413cc9af3379f491b963f3b586a).
Its module paths and visits follow the
[pinned model implementation](https://github.com/huggingface/transformers/blob/e3a65e2c21abb9c25a03704f2f25852f9ad03a18/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py).
The vision tower is mapped as one parameter component. The map's execution section covers text sites.

## Weight layers and execution visits

The component rows describe 24 weight layers. The execution timeline expands layers 6 through 17 to three passes,
for 48 visits in one forward. Ordinary parameter addresses such as `component:attn.q & layers:12` select the same
weight for all three visits. The measured query, key, value and output rows place the actual projection weights
inside their wrapper modules, so unit axes and whole-head selections still work there.

```python
from venngram.maps import load_component_map

cmap = load_component_map("registry/candidates/inkling2_mini.yaml")
active, = cmap.execution.resolve("point:state.active & layers:12 & passes:3")
raw, = cmap.execution.resolve("point:message.raw.q & layers:12 & passes:3")
gate, = cmap.execution.resolve("point:message.gated.q & layers:12 & passes:3")
assert (active.call_index, raw.call_index, gate.call_index) == (2, 18, 1)
```

Each resolved site gives a module path relative to the mapped model, a zero-based call index, an input or output hook,
and a path into its value. The call index counts that module's own calls within one forward. Generation and gradient
checkpoint recomputation add calls beyond this timeline. The map binds a module at most once at each visit.
The layer runs on every pass, the message gate starts on pass two, and the shared message module runs at each layer
of passes two and three. An explicit message address for a layer on pass one is refused. With `passes` omitted,
the resolver returns every visit that exposes the point. Treatments require sets without `passes`.

Call `site.read(args, token=-1)` on a positional input hook or `site.read(output, token=-1)` on an output hook to
read the last token. Leave out `token` to keep the full sequence. The consumer owns hooks and any writes.

| Point | Value |
| --- | --- |
| `state.enter`, `state.active` | The input to the weight layer and the active block sum it returns |
| `state.completed`, `state.completed.<index>` | All completed blocks at layer entry or one block by zero-based index. The set is empty at layer 0. Index 0 holds the embedding output once the first block boundary has been crossed. |
| `attn.mix`, `attn.mix.k`, `attn.mix.v` | The query-side block mix, plus separate key and value mixes on full-attention layers. Sliding attention shares the query-side mix. The first layer has no separate attention mix. |
| `mlp.mix` | The MoE block mix before adding a message |
| `message.raw.q`, `message.raw.k`, `message.raw.v`, `message.raw.mlp` | The four retrieved messages before their gates |
| `message.gated.q`, `message.gated.k`, `message.gated.v`, `message.gated.mlp` | The mixed inputs returned by the four message gates |
| `message.bank.0`, `message.bank.1` | The two previous-pass core blocks supplied to message retrieval |
| `state.final_mix`, `logit_softcap` | The final block mix before normalization and the logits after softcapping |

## Reproduce the checks

Run the generator with the pinned implementation and a local copy of the checkpoint configuration.

```bash
python scripts/verify_inkling2_mini.py /path/to/checkpoint-config --out /tmp/inkling2_mini.yaml
cmp /tmp/inkling2_mini.yaml registry/candidates/inkling2_mini.yaml
```

The generator builds the full module tree on the meta device and checks parameter ownership, widths, layouts, head
sizes and every hook path. It also runs the same implementation on the CPU at reduced widths, preserving all 48
visits and attention variants. Hooks check that all 609 mapped calls occur and that their paths return finite tensors
at the chosen token. The forward checks the layer order and confirms each pass uses the same query parameter object
for a given layer.
