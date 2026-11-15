# Registry test evidence

The canonical seed documents live in the [venngram registry on the Hub](https://huggingface.co/datasets/speediedan/venngram-registry).
`snapshot/` is an offline export of the preferred component map per subject from one commit-pinned generation.
`snapshot.yaml` records the registry commit, generation, exact document references and SHA-256 digests. These are
generated copies whose exact bytes are preserved by `.gitattributes`, not the source to edit when adding an architecture or overlay. They are not package data.

The ordinary suite checks every exported digest before rebuilding a local registry. It uses no Hub requests.
Synthetic documents and module trees exercise the reusable base-map and overlay contracts in
`tests/map_contracts.py`. The independent expectations detect missing or duplicated ownership, swapped component
labels, extra adapter tensors and collapsed adapter keys. A small real-library sample exercises fused projections,
grouped-query attention, hybrid stacks, keyed scaling vectors, magnitude vectors and adapter buffers. Adding a new
registry subject does not require a fixture or expand the sample. Specialized regression tests remain separate.

To refresh the export, choose the full registry commit explicitly and write to a **new** directory:

```bash
python scripts/snapshot_registry.py --registry hub-dataset:speediedan/venngram-registry/index@<40-character-commit> --out /tmp/venngram-snapshot
```

Review `snapshot.yaml`, replace `registry/snapshot/` with the generated export, and run `python -m pytest -q`.
Keep the sample bounded; change it when a new structural behavior warrants coverage. The scheduled published-examples
workflow fetches the same pinned generation, compares the export byte for byte, and runs the same sampled contracts
against the fresh export. Network failures fail that scheduled check without affecting pull request CI.

This initial contract covers base and overlay maps. Address books, recipes and other documents keep their existing
schema, support and published-document conformance tests; they can gain model-level contracts when needed.

## Inspecting and rebuilding the seed index

The seed index lives at `hub-dataset:speediedan/venngram-registry/index`; its header is `index/index.yaml`, read by
the same code a lookup uses. To see what it holds (header, subjects and document schemas per kind, mined rules) and
the documents the registry's own manifest lists:

```bash
python -m venngram.registry inspect --manifest hub-dataset:speediedan/venngram-registry
```

A `null` in a kind's `schema_versions` means the index was built before rows recorded their document's schema.

To rebuild, crawl the registry and every Hub repository tagged `venngram` into a **new** local directory, keeping the
seed index's name, and read the printed inspection before publishing anything:

```bash
python -m venngram.registry rebuild --source hub-dataset:speediedan/venngram-registry --discover-hub \
    --name venngram-seed --out /tmp/venngram-index
python -m venngram.registry inspect local:/tmp/venngram-index
```

Publishing writes a new generation and then the header to the Hub, so every venngram reading the default registry
serves it at once. **It needs the maintainer's go**, and a venngram old enough to fail on a document schema the new
index lists would break for its users, so check what consumers pin first. With that go, either rerun `rebuild` with
`--publish hub-dataset:speediedan/venngram-registry/index` or push the reviewed directory:

```bash
python -m venngram.registry publish /tmp/venngram-index hub-dataset:speediedan/venngram-registry/index
```

Credentials come from the environment (`HF_TOKEN`, with write access to the registry for a publish); no command takes
a token or a credential path. After a publish, refresh `snapshot/` as described above.
