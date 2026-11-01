"""A published directory checked from the outside: every document loads, recipes against support records."""
import pytest
import yaml

from venngram.conformance import check, main, parse_location, read_support
from venngram.errors import VenngramError
from venngram.ops import Support
from venngram.store import publish
from venngram.units import units_value

MODEL = {"id": "Qwen/Qwen3.5-4B", "revision": "abc123"}
STEERER = {"schema_version": 1, "kind": "support_record", "name": "steerer", "ops": ["add", "scale"],
           "targets": ["units"], "tokens": ["all"], "sites": ["mlp.*.hook_in"]}


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VENNGRAM_CACHE", str(tmp_path / "cache"))


def published(tmp_path, tokens="all"):
    """A book with a sidecar set, a recipe over it, and a support record, published to a local store."""
    src = tmp_path / "src"
    (src / "books").mkdir(parents=True)
    value = units_value({15: list(range(0, 2000, 3))}, 9216, src / "books" / "chess.safetensors",
                        relative_to=src / "books")
    (src / "books" / "chess.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "name": "chess", "portability": "checkpoint", "applies_to": {"models": [MODEL]},
        "sets": {"chess": {"point": "mlp.out.hook_in", "units": value}}}))
    (src / "recipes").mkdir()
    (src / "recipes" / "chess.yaml").write_text(yaml.safe_dump({
        "schema_version": 2, "kind": "recipe", "name": "chess", "model": {**MODEL, "revision_source": "pinned"},
        "book": "../books/chess.yaml", "stages": [{"site": "mlp.out.hook_in", "tokens": tokens, "ops": [
            {"op": "add", "target": {"set": "chess"}, "value": {"dose": 0.8}}]}]}))
    (src / "support").mkdir()
    (src / "support" / "steerer.yaml").write_text(yaml.safe_dump(STEERER))
    pub = tmp_path / "pub"
    publish("local", str(pub), src, documents=[("recipe", "recipes/chess.yaml"),
                                               ("support_record", "support/steerer.yaml")])
    return f"local:{pub}", pub


def test_every_document_loads_and_sidecars_are_listed_not_loaded(tmp_path):
    location, _ = published(tmp_path)
    report = check(location)
    assert report.ok and {(d.kind, d.path) for d in report.documents} == {
        ("recipe", "recipes/chess.yaml"), ("address_book", "books/chess.yaml"),
        ("support_record", "support/steerer.yaml")}
    assert report.unloaded == (("units", "books/chess.safetensors"),)
    assert "0 failing" in report.render()


def test_recipes_are_checked_against_each_support_record(tmp_path):
    location, _ = published(tmp_path, tokens="prompt")
    steerer = read_support(f"{location}:support/steerer.yaml")  # a support record read by reference
    assert steerer.name == "steerer" and steerer.tuning is None  # schema 1: a steering section only
    assert steerer.steering.site_ok("mlp.out.hook_in") and not steerer.steering.site_ok("attn.hook_z")
    report = check(location, supports=[steerer, Support(ops=frozenset({"add"}), tokens=frozenset({"prompt"}))])
    (recipe,) = [d for d in report.documents if d.kind == "recipe"]
    assert recipe.refusals == {"steerer": ("stage 0 (mlp.out.hook_in) op 0 (`add`): token scope `prompt` is not "
                                           "supported",), "support 1": ()}
    assert not report.ok and "refused by steerer" in report.render()


def test_a_document_that_fails_to_load_is_reported_not_raised(tmp_path):
    location, pub = published(tmp_path)
    (pub / "books" / "chess.yaml").write_text("tampered: true\n")
    report = check(location)
    failing = {d.path: d.error for d in report.documents if not d.ok}
    assert set(failing) == {"books/chess.yaml", "recipes/chess.yaml"}  # the recipe needs the book
    assert all("sha256" in e for e in failing.values())


def test_the_command_exits_by_outcome(tmp_path, capsys):
    location, pub = published(tmp_path, tokens="generated")
    assert main([location]) == 0
    assert main([location, "--support", str(pub / "support" / "steerer.yaml")]) == 1
    assert "token scope `generated`" in capsys.readouterr().out
    assert main([f"local:{tmp_path / 'nowhere'}"]) == 1
    assert main([location, "--support", str(pub / "books" / "chess.yaml")]) == 2


def test_locations_and_support_records_are_refused_by_name():
    assert parse_location("hub-dataset:o/r/dir@main") == ("hub-dataset", "o/r/dir", "main")
    with pytest.raises(VenngramError, match="not a location"):
        parse_location("nostore")
    for doc, match in [({**STEERER, "kind": "book"}, "kind: support_record"), ({**STEERER, "speed": 1}, "unknown"),
                       ({**STEERER, "tokens": ["first"]}, "does not define"), ({**STEERER, "ops": "add"}, "list of")]:
        with pytest.raises(ValueError, match=match):
            Support.from_mapping(doc)
