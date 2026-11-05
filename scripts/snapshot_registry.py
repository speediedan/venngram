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
"""Export the preferred map per subject from a commit-pinned registry, with its source references and digests.

Run from the repo root with venngram installed. The output must be a new directory. No model weights are fetched.
"""
import argparse
import re
from pathlib import Path

import yaml

from venngram import registry
from venngram.store import cache_dir, fetch, get_store, Reference


def snapshot(location: str, out: Path) -> None:
    store, sep, ref = location.partition(":")
    ref, at, revision = ref.rpartition("@")
    if not sep or not at or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("The registry must pin a full 40-character commit: <store>:<ref>@<commit>.")
    # Read the pinned header as well as the rows: the generation id is part of the evidence.
    root = get_store(store).fetch(ref, ["index.yaml"], revision, cache_dir() / "snapshot-header" / revision)
    header = yaml.safe_load((root / "index.yaml").read_text(encoding="utf-8"))
    subjects = registry.subjects("component_map", locations=[location])
    if not subjects:
        raise ValueError("The pinned registry contains no component maps.")
    docs, contents = [], {}
    for subject in subjects:
        candidates = registry.rows("component_map", subject, locations=[location])
        row = next((r for r in candidates if r.status not in ("failed", "invalidated")), None)
        if row is None:
            raise ValueError(f"No usable map for {subject!r} in {location}.")
        reference = Reference.parse(row.reference)
        if not reference.revision or not reference.sha256:
            raise ValueError(f"The indexed document is not pinned by revision and digest: {row.reference}.")
        path, _ = fetch(reference)  # checks the manifest and the indexed digest before returning bytes
        group = "overlays" if subject.startswith("overlay:") else "maps"
        relative = f"{group}/{path.name}"
        if relative in contents:
            raise ValueError(f"Snapshot filename collision: {relative}.")
        contents[relative] = path.read_bytes()
        docs.append(dict(subject=subject, path=relative, reference=row.reference, sha256=row.sha256))
    out.mkdir(parents=True, exist_ok=False)
    for relative, data in contents.items():
        path = out / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
    (out / "snapshot.yaml").write_text(yaml.safe_dump(dict(
        registry=location, generation=header["current"], documents=docs), sort_keys=False),
                                     encoding="utf-8", newline="\n")
    print(f"{location}: generation {header['current']}, {len(docs)} documents -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    snapshot(args.registry, args.out)
