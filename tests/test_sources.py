import hashlib
import json

import pytest
import yaml

from venngram.errors import Refusal
from venngram.manifest import build_manifest
from venngram.maps.schema import ComponentMap
from venngram.maps.transformer_lens import TransformerLensSource
from venngram.recipe import Recipe
from venngram.sources import (AdapterListing, derive_map, format_for, formats, map_sources, Provenance, ReadContext,
                              register_format, register_map_source, SourceError, SourceLoadWarning, Translation,
                              unregister, write_translation)
from venngram.store import load, publish


class ToyFormat:
    """A made-up foreign format: ``{"toy": 1, "model": id, "neurons": [[layer, unit, dose], ...]}``, no revision."""

    name, version, kinds, unexpressed = "toy", "0.1", ("address_book", "recipe"), ("experts",)

    def detect(self, doc):
        return isinstance(doc, dict) and doc.get("toy") == 1

    def read(self, doc, ctx):
        neurons = [n for n in doc["neurons"] if len(n) == 3]
        skipped = [Refusal(f"neuron {n} names an expert") for n in doc["neurons"] if len(n) != 3]
        if skipped and not ctx.lenient:
            raise SourceError(f"toy: {skipped[0]}, which venngram does not address yet")
        revision = ctx.resolve_revision(doc["model"])
        units = {}
        for layer, unit, _ in neurons:
            units.setdefault(layer, []).append(unit)
        book = {"schema_version": 1, "name": "toy", "portability": "checkpoint",
                "applies_to": {"models": [{"id": doc["model"], "revision": revision}]},
                "sets": {"toy": {"point": "mlp.out.hook_in", "layers": sorted(units),
                                 "units": {k: {"file": "toy.safetensors", "key": f"L{k}", "sha256": None}
                                           for k in sorted(units)}}}}
        doses: dict = {}
        for layer, u, d in neurons:
            doses.setdefault(layer, {})[u] = d
        recipe = {"schema_version": 2, "kind": "recipe", "name": "toy",
                  "model": {"id": doc["model"], "revision": revision, "revision_source": "resolved-at-import"},
                  "book": "toy.yaml",
                  "stages": [{"site": "mlp.out.hook_in",
                              "ops": [{"op": "add", "target": {"set": "toy"}, "value": {"doses": doses}}]}],
                  "metadata": {"unexpressed": [str(r) for r in skipped]} if skipped else {}}
        sha = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
        return Translation({"toy.yaml": ("address_book", book), "toy_recipe.yaml": ("recipe", recipe)},
                           Provenance("format", self.name, self.version, sha, "resolved-at-import"),
                           {"toy.safetensors": {f"L{k}": (v, 64) for k, v in units.items()}}, tuple(skipped))


class FakeMapSource:
    name, version, unexpressed = "fake", "1.0", ()

    def architectures(self):
        return ("FakeForCausalLM",)

    def derive(self, architecture, *, config=None, model=None):
        return ComponentMap(architecture=architecture, components={}, source=self.name)


@pytest.fixture(autouse=True)
def clean(tmp_path, monkeypatch):
    monkeypatch.setenv("VENNGRAM_CACHE", str(tmp_path / "cache"))
    yield
    for name in ("toy", "fake", "other", "good"):
        unregister(name)


DOC = {"toy": 1, "model": "Qwen/Qwen3.5-4B", "neurons": [[13, 7, -1.0], [14, 9, 0.5]]}
CTX = ReadContext(resolve_revision=lambda model_id: "abc123")


def test_a_translation_is_written_listed_published_and_loaded(tmp_path):
    register_format(ToyFormat())
    fmt = format_for(DOC)
    translation = fmt.read(DOC, CTX)
    documents = write_translation(translation, tmp_path / "src")
    book = yaml.safe_load((tmp_path / "src" / "toy.yaml").read_text())
    assert all(v["sha256"] for v in book["sets"]["toy"]["units"].values())  # filled in from the written sidecar
    manifest = build_manifest(tmp_path / "src", documents)
    assert {e.path for e in manifest.documents} == {"toy.yaml", "toy_recipe.yaml", "toy.safetensors"}
    publish("local", str(tmp_path / "pub"), tmp_path / "src", documents=documents)
    recipe = load(f"local:{tmp_path / 'pub'}:toy_recipe.yaml")
    assert isinstance(recipe, Recipe)
    assert recipe.stages[0].ops[0].unit_doses(recipe.book) == {(13, 7): -1.0, (14, 9): 0.5}
    assert translation.provenance.to_mapping()["revision_source"] == "resolved-at-import"


def test_what_a_format_cannot_carry_is_refused_unless_lenient():
    register_format(ToyFormat())
    doc = {**DOC, "neurons": [*DOC["neurons"], [12, 1, 3, 1.0]]}
    with pytest.raises(SourceError, match="names an expert"):
        format_for(doc).read(doc, CTX)
    lenient = format_for(doc).read(doc, ReadContext(resolve_revision=lambda m: "abc123", lenient=True))
    assert len(lenient.unexpressed) == 1 and lenient.documents["toy_recipe.yaml"][1]["metadata"]["unexpressed"]


def test_format_selection_refuses_none_and_several():
    register_format(ToyFormat())
    with pytest.raises(SourceError, match="No format"):
        format_for({"other": 1})

    class Other(ToyFormat):
        name = "other"

    register_format(Other())
    with pytest.raises(SourceError, match="Several formats"):
        format_for(DOC)
    assert format_for(DOC, name="other").name == "other"
    with pytest.raises(SourceError, match="already registered"):
        register_format(Other())
    with pytest.raises(SourceError, match="does not implement Format"):
        register_format(object())  # type: ignore[arg-type]


def test_a_map_source_derives_with_provenance():
    register_map_source(FakeMapSource())
    cmap, provenance = derive_map("fake", "FakeForCausalLM")
    assert cmap.architecture == "FakeForCausalLM" and provenance == Provenance("map_source", "fake", "1.0")
    with pytest.raises(SourceError, match="No source `missing`"):
        derive_map("missing", "X")


def test_entry_points_load_on_first_use_and_a_broken_one_is_skipped(monkeypatch):
    import venngram.sources as sources

    class EP:
        def __init__(self, name, obj):
            self.name, self.value, self._obj = name, f"pkg:{name}", obj

        def load(self):
            if isinstance(self._obj, Exception):
                raise self._obj
            return self._obj

    class Good(FakeMapSource):
        name = "good"

    groups = {sources.MAP_SOURCES_GROUP: [EP("good", Good), EP("broken", ImportError("no module"))],
              sources.FORMATS_GROUP: []}
    monkeypatch.setattr(sources, "entry_points", lambda group: groups[group])
    monkeypatch.setattr(sources._MAP_SOURCES, "_loaded", False)
    with pytest.warns(SourceLoadWarning, match="`broken`"):
        found = map_sources()
    assert "good" in found and "broken" not in found
    assert "toy" not in formats()


def test_the_transformer_lens_source_reads_an_adapter_duck_typed():
    def bridge(cls_name, name=None, **subs):
        return type(cls_name, (), {"name": name, "submodules": subs})()

    adapter = type("Adapter", (), {"component_mapping": {
        "embed": bridge("EmbeddingBridge", "model.embed_tokens"),
        "blocks": bridge("BlockBridge", "model.layers", attn=bridge("AttentionBridge", "self_attn"),
                         mlp=bridge("MLPBridge", "mlp")),
        "rotary_emb": bridge("RotaryEmbeddingBridge", "model.rotary_emb"),
    }})()
    source = TransformerLensSource(lambda arch, config: adapter, ["ToyForCausalLM"], version="3.0")
    register_map_source(source)
    cmap, provenance = derive_map("transformer_lens", "ToyForCausalLM")
    assert cmap.components["blocks.{i}.attn"].module == "model.layers.{i}.self_attn"
    assert provenance.version == "3.0" and "RotaryEmbeddingBridge" in source.unexpressed
    unregister("transformer_lens")
    with pytest.raises(SourceError, match="no adapter for Missing"):
        TransformerLensSource(lambda arch, config: (_ for _ in ()).throw(KeyError(arch))).derive("Missing")


def test_an_adapter_listing_is_data(tmp_path):
    doc = {"schema_version": 1, "kind": "format_adapter", "name": "toy", "version": "0.1", "formats": ["toy"],
           "install": {"git": "https://github.com/someone/venngram-toy", "commit": "f" * 40}}
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "toy_adapter.yaml").write_text(yaml.safe_dump(doc))
    publish("local", str(tmp_path / "pub"), tmp_path / "src", documents=[("format_adapter", "toy_adapter.yaml")])
    listing = load(f"local:{tmp_path / 'pub'}:toy_adapter.yaml")
    assert listing == AdapterListing("toy", "0.1", ("toy",), doc["install"])
    with pytest.raises(SourceError, match="pins a `commit`"):
        AdapterListing.from_mapping({**doc, "install": {"git": "https://example.com/x"}})
