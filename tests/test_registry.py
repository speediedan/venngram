"""The registry: an index over published documents, federated across locations, with lookup, ranking and offline use."""
import json
import shutil
from pathlib import Path

import pytest
import yaml

pytest.importorskip("pyarrow")

from conftest import SEED_MAPS  # noqa: E402

from venngram import registry  # noqa: E402
from venngram.maps import component_map_for, known_architectures, register, unregister  # noqa: E402
from venngram.maps.schema import load_component_map  # noqa: E402
from venngram.store import LocalStore, publish, register_store  # noqa: E402

TINY = {"schema_version": 1, "architecture": "TinyDecoder", "provenance": {"source": "module_tree", "version": "0.1"},
        "components": {"blocks.{i}": {"module": "layers.{i}", "kind": "block"},
                       "blocks.{i}.attn": {"module": "layers.{i}.attn", "kind": "attn"}}}


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VENNGRAM_CACHE", str(tmp_path / "cache"))
    registry.clear_cache()


def directory(root: Path, maps: dict[str, dict], records: dict[str, dict] = (), books: dict[str, dict] = ()) -> str:
    """A published manifest directory of maps, verification records and books; returns its location."""
    root.mkdir(parents=True)
    docs = []
    for name, doc in maps.items():
        (root / name).write_text(yaml.safe_dump(doc))
        docs.append(("component_map", name))
    for name, doc in dict(records).items():
        (root / name).write_text(yaml.safe_dump(doc))
        docs.append(("verification_record", name))
    for name, doc in dict(books).items():
        (root / name).write_text(yaml.safe_dump(doc))
        docs.append(("address_book", name))
    publish("local", str(root), root, documents=docs)
    return f"local:{root}"


def seed_doc(name: str) -> dict:
    return yaml.safe_load((SEED_MAPS / name).read_text())


def build(tmp_path, locations, name="idx") -> str:
    registry.write_index(registry.crawl(locations), tmp_path / name, name=name, sources=locations)
    return f"local:{tmp_path / name}"


def test_maps_from_any_source_are_indexed_and_found(tmp_path):
    seeds = directory(tmp_path / "seeds", {"llama.yaml": seed_doc("llama.yaml")})
    contrib = directory(tmp_path / "contrib", {"tiny.yaml": TINY})  # a contributor's drafted map, not a seed
    registry.set_registries([build(tmp_path, [seeds, contrib])])
    assert component_map_for("TinyDecoder").architecture == "TinyDecoder"
    (row,) = registry.rows("component_map", "TinyDecoder")
    assert row.source == "module_tree 0.1" and row.location == contrib and row.reference.endswith(f"#{row.sha256}")
    assert known_architectures() == ["LlamaForCausalLM", "TinyDecoder"]


def test_a_verified_map_outranks_an_unverified_one_and_failed_maps_are_never_served(tmp_path):
    llama = seed_doc("llama.yaml")
    other = {**llama, "components": {**llama["components"], "embed": {"module": "model.embed", "kind": "embed"}}}
    seeds = directory(tmp_path / "seeds", {"llama.yaml": llama})
    verified = directory(tmp_path / "verified", {"llama.yaml": other},
                         {"llama.record.yaml": {"architecture": "LlamaForCausalLM", "status": "verified",
                                                "config": {"source": "hf:some/llama"}}})
    failed = directory(tmp_path / "failed", {"tiny.yaml": TINY},
                       {"tiny.record.yaml": {"architecture": "TinyDecoder", "status": "failed"}})
    registry.set_registries([build(tmp_path, [seeds, verified, failed])])
    candidates = registry.rows("component_map", "LlamaForCausalLM")
    assert [(r.status, r.location) for r in candidates] == [("verified", verified), ("unverified", seeds)]
    assert component_map_for("LlamaForCausalLM").components["embed"].module == "model.embed"
    (record,) = registry.rows("verification_record", "LlamaForCausalLM")
    assert record.model_id == "some/llama"
    with pytest.raises(KeyError, match="TinyDecoder"):
        component_map_for("TinyDecoder")  # its only map failed verification: listed, not served
    assert registry.rows("component_map", "TinyDecoder")[0].status == "failed"


def test_strict_lookup_serves_only_verified_maps(tmp_path):
    registry.set_registries([build(tmp_path, [directory(tmp_path / "s", {"tiny.yaml": TINY})])])
    assert component_map_for("TinyDecoder") is not None
    with pytest.raises(KeyError):
        component_map_for("TinyDecoder", strict=True)


def test_registries_are_consulted_in_order_after_in_process_registrations(tmp_path):
    first = build(tmp_path, [directory(tmp_path / "a", {"tiny.yaml": TINY})], "first")
    second = build(tmp_path, [directory(tmp_path / "b", {"llama.yaml": seed_doc("llama.yaml")})], "second")
    registry.set_registries([first, second])
    assert component_map_for("LlamaForCausalLM").architecture == "LlamaForCausalLM"  # found in the second
    mine = load_component_map(SEED_MAPS / "opt.yaml")
    register(mine)
    try:
        assert component_map_for("OPTForCausalLM") is mine
    finally:
        unregister("OPTForCausalLM")


def test_a_rebuild_is_a_new_generation_and_leaves_no_stale_row(tmp_path):
    src = directory(tmp_path / "s", {"tiny.yaml": TINY, "llama.yaml": seed_doc("llama.yaml")})
    location = build(tmp_path, [src])
    first = yaml.safe_load((tmp_path / "idx" / "index.yaml").read_text())["current"]
    (tmp_path / "s" / "tiny.yaml").unlink()
    publish("local", str(tmp_path / "s"), tmp_path / "s", documents=[("component_map", "llama.yaml")])
    build(tmp_path, [src])
    header = yaml.safe_load((tmp_path / "idx" / "index.yaml").read_text())
    assert header["current"] != first and header["counts"] == {"component_map": 1}
    registry.set_registries([location])
    assert registry.rows("component_map", "TinyDecoder") == [] and registry.subjects("component_map") == [
        "LlamaForCausalLM"]


def test_checkpoint_documents_are_partitioned_by_model(tmp_path):
    book = {"schema_version": 1, "name": "b", "portability": "checkpoint",
            "applies_to": {"models": [{"id": "Qwen/Qwen3.5-4B", "revision": "abc"}]},
            "sets": {"s": {"component": "mlp.out", "layers": [3]}}}
    location = build(tmp_path, [directory(tmp_path / "s", {}, books={"b.yaml": book})])
    (row,) = registry.rows("address_book", "Qwen/Qwen3.5-4B", locations=[location])
    assert (row.model_id, row.revision) == ("Qwen/Qwen3.5-4B", "abc")
    assert list((tmp_path / "idx").glob("generations/*/kind=address_book/subject=Qwen%2FQwen3.5-4B/rows.parquet"))


class Flaky(LocalStore):
    """A versioned local store whose revision lookups can be made to fail, standing in for an unreachable Hub."""

    name, versioned, down = "flaky", True, False

    def resolve_revision(self, ref, revision):
        if Flaky.down:
            raise ConnectionError("unreachable")
        return "0" * 40

    def fetch(self, ref, paths, revision, cache):
        return Path(ref)


def test_an_unreachable_registry_is_served_from_the_cache_with_a_warning(tmp_path):
    register_store("flaky", Flaky(), versioned=True, replace=True)
    location = build(tmp_path, [directory(tmp_path / "s", {"tiny.yaml": TINY})]).replace("local:", "flaky:")
    registry.set_registries([location])
    assert component_map_for("TinyDecoder")  # fetched once while reachable
    registry.clear_cache()
    Flaky.down = True
    try:
        with pytest.warns(registry.RegistryOfflineWarning, match="0000000000"):
            assert component_map_for("TinyDecoder")
        with pytest.raises(registry.RegistryError, match="none was fetched before"):
            registry.rows("component_map", "TinyDecoder", locations=["flaky:" + str(tmp_path / "never")])
    finally:
        Flaky.down = False


def test_configuration_and_the_command(tmp_path, monkeypatch, capsys):
    registry.set_registries(None)
    monkeypatch.delenv("VENNGRAM_REGISTRIES", raising=False)
    assert registry.registries() == list(registry.DEFAULT_REGISTRIES)
    monkeypatch.setenv("VENNGRAM_REGISTRIES", "")
    assert registry.registries() == []
    monkeypatch.setenv("VENNGRAM_REGISTRIES", "local:/a, local:/b local:/c")
    assert registry.registries() == ["local:/a", "local:/b", "local:/c"]
    src = directory(tmp_path / "s", {"tiny.yaml": TINY})
    assert registry.main(["build", "--source", src, "--out", str(tmp_path / "cli"), "--name", "cli"]) == 0
    assert registry.main(["publish", str(tmp_path / "cli"), f"local:{tmp_path / 'pub'}"]) == 0
    monkeypatch.setenv("VENNGRAM_REGISTRIES", f"local:{tmp_path / 'pub'}")
    registry.clear_cache()
    assert registry.main(["lookup", "TinyDecoder"]) == 0
    assert "unverified" in capsys.readouterr().out
    shutil.rmtree(tmp_path / "pub")


def test_inspect_and_rebuild_read_the_index_through_its_header(tmp_path, capsys):
    src = directory(tmp_path / "s", {"tiny.yaml": TINY})
    pub = f"local:{tmp_path / 'pub'}"
    assert registry.main(["rebuild", "--source", src, "--out", str(tmp_path / "built"), "--name", "seed"]) == 0
    assert not (tmp_path / "pub").exists()  # no --publish, no write
    built = capsys.readouterr().out
    assert '"name": "seed"' in built and '"TinyDecoder"' in built
    assert registry.main(["rebuild", "--source", src, "--out", str(tmp_path / "again"), "--name", "seed",
                          "--publish", pub]) == 0
    capsys.readouterr()
    registry.clear_cache()
    assert registry.main(["inspect", pub, "--manifest", src]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["name"] == "seed" and report["current"].startswith("generations/")
    assert report["kinds"]["component_map"]["subjects"] == ["TinyDecoder"]
    assert report["kinds"]["component_map"]["schema_versions"] and report["mined_rules"]["kind"] == "rules"
    assert report["manifest"]["documents"]["component_map"] == ["tiny.yaml"]
    assert registry.main(["inspect", f"local:{tmp_path / 'never'}"]) == 1


# ----- cache-only lookup ---------------------------------------------------------------------------------------------

COMMIT = "c0ffee" + "0" * 34
HUB_INDEX = "hub-dataset:owner/vg/index"


@pytest.fixture
def requests(monkeypatch) -> list:
    """Sockets that record and refuse every connection, in a process otherwise online (``HF_HUB_OFFLINE`` unset)."""
    import socket

    from huggingface_hub import constants

    attempts: list = []

    def refuse(*args, **kwargs):
        attempts.append(args)
        raise ConnectionRefusedError("a request was attempted")

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    for name in ("create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    return attempts


@pytest.fixture
def hub(tmp_path, monkeypatch, requests) -> Path:
    """A Hub dataset ``owner/vg`` in the Hub's cache at ``COMMIT``: a maps directory and the index crawled from it, as a
    lookup with network access would have left them. Returns the snapshot directory."""
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hf"))
    repo = tmp_path / "hf" / "datasets--owner--vg"
    snapshot = repo / "snapshots" / COMMIT
    maps = directory(tmp_path / "maps", {"tiny.yaml": TINY, "llama.yaml": seed_doc("llama.yaml")})
    shutil.copytree(maps.removeprefix("local:"), snapshot / "maps")
    index = registry.crawl([f"hub-dataset:owner/vg/maps@{COMMIT}"])  # a pinned crawl reads the cache, no request
    registry.write_index(index, tmp_path / "idx", name="hub", sources=["hub-dataset:owner/vg/maps"])
    shutil.copytree(tmp_path / "idx", snapshot / "index")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(COMMIT)
    remember(HUB_INDEX)
    registry.set_registries([HUB_INDEX])
    assert not requests
    return snapshot


def remember(location: str) -> None:
    """What a lookup with network access remembers of a registry: its resolved revision, in venngram's cache."""
    from venngram.store import _safe, cache_dir

    memo = cache_dir() / "registry" / _safe(location) / "last_revision"
    memo.parent.mkdir(parents=True, exist_ok=True)
    memo.write_text(COMMIT)


def partition(snapshot: Path, subject: str | None = None) -> Path:
    current = yaml.safe_load((snapshot / "index" / "index.yaml").read_text())["current"]
    leaf = "subjects.parquet" if subject is None else f"subject={subject}/rows.parquet"
    return snapshot / "index" / current / "kind=component_map" / leaf


def test_a_cache_only_lookup_serves_the_online_answer_with_no_request(hub, requests):
    from venngram.errors import CacheMissError

    cold = component_map_for("TinyDecoder", cache_only=True)  # a warm disk and a cold process
    assert cold.architecture == "TinyDecoder" and component_map_for("LlamaForCausalLM", cache_only=True)
    assert not requests and HUB_INDEX not in registry._STATE  # nothing remembered for an online lookup to reuse
    with pytest.raises(KeyError, match="NoSuchForCausalLM"):  # the cached subject list proves it absent
        component_map_for("NoSuchForCausalLM", cache_only=True)
    assert registry.lookup_map("NoSuchForCausalLM", cache_only=True) is None and not requests
    registry.clear_cache()
    with pytest.warns(registry.RegistryOfflineWarning):  # an online lookup does make a request
        assert component_map_for("TinyDecoder").source == cold.source
    assert requests
    requests.clear()
    assert component_map_for("TinyDecoder", cache_only=True).source == cold.source  # a warm process
    assert not requests
    with pytest.raises(CacheMissError, match="pinned commit"):  # a branch cannot be resolved without a request
        registry.rows("component_map", "TinyDecoder", locations=[f"{HUB_INDEX}@main"], cache_only=True)
    registry.set_registries([f"{HUB_INDEX}@{COMMIT}"])  # a pinned registry needs nothing remembered
    assert component_map_for("TinyDecoder", cache_only=True).architecture == "TinyDecoder" and not requests


def test_a_cache_only_lookup_refuses_what_the_cache_cannot_decide(hub, tmp_path, requests):
    from venngram.errors import CacheMissError, StoreError

    (hub / "maps" / "tiny.yaml").unlink()
    with pytest.raises(CacheMissError, match="not in the Hub's cache"):  # never another row in its place
        component_map_for("TinyDecoder", cache_only=True)
    partition(hub, "TinyDecoder").unlink()
    with pytest.raises(CacheMissError, match="cannot decide"):  # listed, but its rows were never fetched
        component_map_for("TinyDecoder", cache_only=True)
    partition(hub).unlink()
    with pytest.raises(CacheMissError, match="subject list"):  # an unavailable partition proves nothing
        component_map_for("NoSuchForCausalLM", cache_only=True)
    no_exist = hub.parents[1] / ".no_exist" / COMMIT / partition(hub, "NoSuchForCausalLM").relative_to(hub)
    no_exist.parent.mkdir(parents=True)
    no_exist.touch()  # the Hub's cache records that this commit has no such file: a proven absence
    with pytest.raises(KeyError):
        component_map_for("NoSuchForCausalLM", cache_only=True)
    (hub / "maps" / "llama.yaml").write_text("tampered: true\n")
    with pytest.raises(StoreError, match="sha256") as err:  # the cached copy is checked like a fetched one
        component_map_for("LlamaForCausalLM", cache_only=True)
    assert not isinstance(err.value, CacheMissError)
    registry.set_registries(["hub-dataset:owner/never/index", HUB_INDEX])
    with pytest.raises(CacheMissError, match="never fetched"):  # an undecidable registry stops the walk
        component_map_for("LlamaForCausalLM", cache_only=True)
    register(load_component_map(SEED_MAPS / "llama.yaml"))
    try:
        assert component_map_for("LlamaForCausalLM", cache_only=True)  # a registration needs no registry
    finally:
        unregister("LlamaForCausalLM")
    assert not requests


def test_cache_only_scopes_the_calling_context_and_refuses_stores_that_cannot(tmp_path):
    import threading

    from venngram.errors import CacheMissError
    from venngram.store import cache_only, cache_only_active

    class Remote:
        name, versioned = "remote", True

        def resolve_revision(self, ref, revision):
            return COMMIT

        def fetch(self, ref, paths, revision, cache):
            return Path(ref)

        def push(self, ref, files, message, revision):
            return None

    register_store("remote", Remote(), versioned=True, replace=True)
    registry.set_registries([build(tmp_path, [directory(tmp_path / "s", {"tiny.yaml": TINY})])])
    assert component_map_for("TinyDecoder", cache_only=True)  # a local registry reads in place
    seen = []
    with cache_only():
        thread = threading.Thread(target=lambda: seen.append(cache_only_active()))
        thread.start()
        thread.join()
        assert cache_only_active() and seen == [False]  # another thread stays online
        with pytest.raises(CacheMissError, match="cannot read cache-only"):
            registry.rows("component_map", "TinyDecoder", locations=["remote:x"])
    assert not cache_only_active()
