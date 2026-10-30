import shutil
from pathlib import Path

import pytest
import yaml

from venngram.book import AddressBook
from venngram.errors import AddressError, StoreError
from venngram.manifest import build_manifest, Manifest, MANIFEST_NAME, satisfies, sha256_of, UnknownKindWarning
from venngram.recipe import Recipe
from venngram.store import fetch, fetch_manifest, load, pin, publish, Reference
from venngram.units import units_value

MODEL = {"id": "Qwen/Qwen3.5-4B", "revision": "abc123"}
DENSE = list(range(0, 2000, 3))


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VENNGRAM_CACHE", str(tmp_path / "cache"))


def write_docs(root: Path) -> None:
    """A book whose set's units are a sidecar, a book it imports, and a recipe over the first book."""
    (root / "books").mkdir(parents=True)
    value = units_value({15: DENSE, 16: [7, 8]}, 9216, root / "books" / "chess.safetensors", relative_to=root / "books")
    (root / "books" / "common.yaml").write_text(yaml.safe_dump(
        {"schema_version": 1, "name": "common", "sets": {"mlp": {"component": "mlp.out"}}}))
    (root / "books" / "chess.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "name": "chess", "portability": "checkpoint", "applies_to": {"models": [MODEL]},
        "imports": {"base": "common.yaml"}, "sets": {"chess": {"point": "mlp.out.hook_in", "units": value}}}))
    (root / "recipes").mkdir()
    (root / "recipes" / "chess.yaml").write_text(yaml.safe_dump({
        "schema_version": 2, "kind": "recipe", "name": "chess", "model": {**MODEL, "revision_source": "pinned"},
        "book": "../books/chess.yaml",
        "stages": [{"site": "mlp.out.hook_in", "ops": [{"op": "add", "target": {"set": "chess"},
                                                        "value": {"dose": 0.8}}]}]}))


DOCS = [("recipe", "recipes/chess.yaml")]


def test_a_manifest_lists_each_document_with_what_it_needs(tmp_path):
    write_docs(tmp_path)
    manifest = build_manifest(tmp_path, DOCS, requires={"venngram": ">=0.1.0.dev0"})
    by_path = {e.path: e for e in manifest.documents}
    assert set(by_path) == {"recipes/chess.yaml", "books/chess.yaml", "books/common.yaml", "books/chess.safetensors"}
    assert by_path["books/chess.yaml"].kind == "address_book" and by_path["books/chess.safetensors"].kind == "units"
    assert by_path["books/chess.yaml"].needs == ("books/chess.safetensors", "books/common.yaml")
    assert by_path["recipes/chess.yaml"].schema_version == 2
    assert {e.path for e in manifest.closure("recipes/chess.yaml")} == set(by_path)
    assert all(e.sha256 == sha256_of(tmp_path / e.path) for e in manifest.documents)
    manifest.write(tmp_path)
    assert Manifest.from_file(tmp_path / MANIFEST_NAME).to_mapping() == manifest.to_mapping()


def test_unknown_kinds_are_skipped_and_kept():
    doc = {"venngram_schema_version": 1, "documents": [{"kind": "address_book", "path": "a.yaml"},
                                                       {"kind": "hologram", "path": "h.bin"}]}
    with pytest.warns(UnknownKindWarning, match="hologram"):
        manifest = Manifest.from_mapping(doc)
    assert [e.path for e in manifest.documents] == ["a.yaml"] and [e.path for e in manifest.skipped] == ["h.bin"]
    assert manifest.to_mapping()["documents"][1] == {"kind": "hologram", "path": "h.bin"}
    with pytest.raises(StoreError, match="kind this venngram does not know"):
        manifest.entry("h.bin")


@pytest.mark.parametrize("change, match", [
    (lambda d: d.update(venngram_schema_version=2), "must be in 1..1"),
    (lambda d: d.update(requires={"venngram": ">=99"}), "requires venngram >=99"),
    (lambda d: d["documents"].append({"kind": "recipe", "path": "a.yaml"}), "listed twice"),
    (lambda d: d["documents"][0].update(path="../a.yaml"), "relative POSIX path"),
    (lambda d: d["documents"][0].update(needs=["missing.yaml"]), "does not list"),
    (lambda d: d["documents"][0].update(sha256="abc"), "64 lowercase hex"),
    (lambda d: d["documents"][0].update(extra=1), "unknown entry keys"),
])
def test_malformed_manifests_are_refused(change, match):
    doc = {"venngram_schema_version": 1, "documents": [{"kind": "address_book", "path": "a.yaml"}]}
    change(doc)
    with pytest.raises(StoreError, match=match):
        Manifest.from_mapping(doc)


def test_version_specifiers():
    assert satisfies("0.2.0", ">=0.1, <1") and not satisfies("1.0", "<1") and satisfies("0.1.0", "==0.1")
    assert not satisfies("0.1.0.dev0", "==0.1") and satisfies("0.1.0.dev0", ">=0.1.0.dev0")
    with pytest.raises(StoreError, match="not a version clause"):
        satisfies("0.1", "~=0.1")


def test_references_parse_and_print():
    ref = Reference.parse(f"hub-model:speediedan/maps@3f1e:maps/Qwen3ForCausalLM.yaml#{'a' * 64}")
    assert (ref.store, ref.ref, ref.revision, ref.path, ref.sha256) == (
        "hub-model", "speediedan/maps", "3f1e", "maps/Qwen3ForCausalLM.yaml", "a" * 64)
    assert str(ref) == f"hub-model:speediedan/maps@3f1e:maps/Qwen3ForCausalLM.yaml#{'a' * 64}"
    local = Reference.parse("local:/data/v@2:books/a.yaml")  # an unversioned store's ref keeps its `@`
    assert (local.ref, local.revision) == ("/data/v@2", None)
    for bad, match in (("nowhere:x:y", "starts with a store name"), ("local:books.yaml", "is `<store>:<ref>"),
                       ("local:/d:a.yaml#beef", "sha256")):
        with pytest.raises(StoreError, match=match):
            Reference.parse(bad)


def published(tmp_path) -> tuple[Path, Manifest]:
    src, dest = tmp_path / "src", tmp_path / "pub"
    write_docs(src)
    manifest, revision = publish("local", str(dest), src, documents=DOCS)
    assert revision is None and (dest / MANIFEST_NAME).is_file()
    return dest, manifest


def test_a_published_book_and_recipe_load_by_reference(tmp_path):
    dest, manifest = published(tmp_path)
    sha = {e.path: e.sha256 for e in manifest.documents}
    book = load(f"local:{dest}:books/chess.yaml#{sha['books/chess.yaml']}")
    assert isinstance(book, AddressBook) and book.sets()["chess"].units.for_layer(16, 9216) == (7, 8)
    assert "base.mlp" in book.sets()
    recipe = load(f"local:{dest}:recipes/chess.yaml#{sha['recipes/chess.yaml']}")
    assert isinstance(recipe, Recipe) and recipe.stages[0].ops[0].unit_doses(recipe.book)[(16, 7)] == 0.8


def test_fetching_refuses_what_it_cannot_vouch_for(tmp_path):
    dest, manifest = published(tmp_path)
    sha = {e.path: e.sha256 for e in manifest.documents}
    with pytest.raises(StoreError, match="is not listed"):
        fetch(f"local:{dest}:books/other.yaml#{'0' * 64}")
    with pytest.raises(StoreError, match="cites another content"):
        fetch(f"local:{dest}:books/chess.yaml#{'0' * 64}")
    with pytest.raises(StoreError, match="no loader reads kind `units`"):
        load(f"local:{dest}:books/chess.safetensors#{sha['books/chess.safetensors']}")
    (dest / "books" / "common.yaml").write_text("schema_version: 1\nname: changed\nsets: {}\n")
    with pytest.raises(StoreError, match="changed after it was cited"):  # a needed file changed under its manifest
        fetch(f"local:{dest}:books/chess.yaml#{sha['books/chess.yaml']}")


def test_a_reference_without_a_hash_follows_the_latest_snapshot_and_pins(tmp_path):
    dest, manifest = published(tmp_path)
    sha = {e.path: e.sha256 for e in manifest.documents}["books/chess.yaml"]
    path, entry = fetch(f"local:{dest}:books/chess.yaml")  # latest: checked against the current manifest
    assert entry.sha256 == sha and path == dest / "books" / "chess.yaml"
    assert pin(f"local:{dest}:books/chess.yaml") == Reference("local", str(dest), "books/chess.yaml", None, sha)
    (dest / "books" / "common.yaml").write_text("schema_version: 1\nname: changed\nsets: {}\n")
    with pytest.raises(StoreError, match="changed after it was cited"):  # still checked against the manifest
        fetch(f"local:{dest}:books/chess.yaml")


def test_a_book_imports_another_by_reference(tmp_path):
    dest, manifest = published(tmp_path)
    sha = {e.path: e.sha256 for e in manifest.documents}["books/chess.yaml"]
    mine = tmp_path / "mine.yaml"
    mine.write_text(yaml.safe_dump({"schema_version": 1, "name": "mine",
                                    "imports": {"chess": f"local:{dest}:books/chess.yaml#{sha}"},
                                    "sets": {"late": {"layers": "16:", "exclude": "@chess.base.mlp"}}}))
    sets = AddressBook.from_file(mine).sets()
    assert sets["chess.chess"].units.for_layer(15, 9216) == tuple(DENSE)
    mine.write_text(yaml.safe_dump({"schema_version": 1, "name": "mine",
                                    "imports": {"chess": f"local:{dest}:books/chess.yaml#{'0' * 64}"},
                                    "sets": {"all": {"layers": ":"}}}))
    with pytest.raises(AddressError, match="did not fetch"):
        AddressBook.from_file(mine)


class FakeHub:
    """Serves ``repos[(repo_id, commit)]`` directories the way the Hub client does, and records commits."""

    def __init__(self, tmp_path, monkeypatch):
        import huggingface_hub

        self.root, self.repos, self.branches, self.commits = tmp_path / "hub", {}, {}, []
        hub = self

        class Info:
            def __init__(self, sha):
                self.sha, self.oid = sha, sha

        class Api:
            def repo_info(self, repo_id, repo_type=None, revision=None):
                return Info(hub.branches.get((repo_id, revision or "main"), revision))

            def create_commit(self, repo_id, operations, commit_message, repo_type=None, revision=None):
                hub.commits.append((repo_id, repo_type, [op.path_in_repo for op in operations], commit_message))
                return Info("c" * 40)

        def download(repo_id, filename, repo_type=None, revision=None):
            path = hub.root / repo_type / repo_id / "snapshots" / revision / filename
            if not path.is_file():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(hub.repos[(repo_id, revision)] / filename, path)
            return str(path)

        def download_bucket_files(bucket_id, files, raise_on_missing_files=False):
            for remote, local in files:
                shutil.copyfile(hub.repos[(bucket_id, None)] / remote, local)

        monkeypatch.setattr(huggingface_hub, "HfApi", Api)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
        monkeypatch.setattr(huggingface_hub, "download_bucket_files", download_bucket_files)


def test_a_hub_repository_is_read_manifest_first_at_one_commit(tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    hub = FakeHub(tmp_path, monkeypatch)
    dest, manifest = published(tmp_path)
    hub.repos[("me/sets", "1" * 40)] = dest.parent  # the documents sit in the repository's `pub` directory
    hub.branches[("me/sets", "main")] = "1" * 40
    book = load("hub-model:me/sets/pub:books/chess.yaml")  # no revision: main resolves to its commit first
    assert book.sets()["chess"].units.for_layer(16, 9216) == (7, 8)
    _, root, revision = fetch_manifest("hub-model", "me/sets/pub")
    assert revision == "1" * 40 and root.name == "pub" and root.parent.name == "1" * 40
    sha = {e.path: e.sha256 for e in manifest.documents}["books/chess.yaml"]
    pinned = f"hub-model:me/sets/pub@{'1' * 40}:books/chess.yaml#{sha}"
    assert str(pin("hub-model:me/sets/pub:books/chess.yaml")) == pinned
    _, written = publish("hub-dataset", "me/sets/pub", dest, message="seed")
    assert written == "c" * 40 and hub.commits[-1][0] == "me/sets" and hub.commits[-1][1] == "dataset"
    assert set(hub.commits[-1][2]) == {f"pub/{e.path}" for e in manifest.documents} | {f"pub/{MANIFEST_NAME}"}


def test_a_hub_bucket_is_cached_by_its_manifest(tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    hub = FakeHub(tmp_path, monkeypatch)
    dest, manifest = published(tmp_path)
    hub.repos[("me/bucket", None)] = dest.parent
    sha = {e.path: e.sha256 for e in manifest.documents}["books/chess.yaml"]
    path, _ = fetch(f"hub-bucket:me/bucket/pub:books/chess.yaml#{sha}")
    assert fetch("hub-bucket:me/bucket/pub:books/chess.yaml")[0] == path  # latest resolves to the same snapshot
    assert path.parent.parent.name == sha256_of(dest / MANIFEST_NAME)
    assert AddressBook.from_file(path).sets()["chess"].units.for_layer(16, 9216) == (7, 8)
