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
