"""Bug 3: one manifest.json (written once, read by the engine) replaces two conflicting category_map.json files."""
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_category_map_json_is_no_longer_written_by_two_scripts():
    writers = [f for f in ("extract_embeddings.py", "trainer.py")
               if "category_map.json" in (ROOT / f).read_text()]
    assert writers == [], f"{writers} still reference category_map.json (id->name vs name->id conflict)"


def test_manifest_roundtrip_and_index_is_class_id(tmp_path):
    from manifest import Manifest, read_manifest, write_manifest
    m = Manifest(model_name="google/siglip-base-patch16-224", embed_dim=768,
                 classes=["Mountains", "Rivers", "Aircraft maintenance"], index_type="flat", n_vectors=10)
    write_manifest(tmp_path, m)
    got = read_manifest(tmp_path)
    assert got.classes == ["Mountains", "Rivers", "Aircraft maintenance"]
    assert got.class_to_id["Rivers"] == 1 and got.id_to_class[2] == "Aircraft maintenance"
    assert got.embed_dim == 768 and got.schema_version == 1
    assert (tmp_path / "manifest.json").is_file()


def test_manifest_rejects_duplicates_wrong_dim_and_future_schema(tmp_path):
    from manifest import Manifest, ManifestError, read_manifest, write_manifest
    with pytest.raises(ManifestError):
        write_manifest(tmp_path, Manifest(model_name="m", embed_dim=768, classes=["A", "A"]))
    with pytest.raises(ManifestError):
        write_manifest(tmp_path, Manifest(model_name="m", embed_dim=512, classes=["A"]), expected_dim=768)
    write_manifest(tmp_path, Manifest(model_name="m", embed_dim=768, classes=["A"]))
    raw = json.loads((tmp_path / "manifest.json").read_text())
    raw["schema_version"] = 99
    (tmp_path / "manifest.json").write_text(json.dumps(raw))
    with pytest.raises(ManifestError):
        read_manifest(tmp_path)


def test_manifest_missing_file_gives_clear_error(tmp_path):
    from manifest import ManifestError, read_manifest
    with pytest.raises(ManifestError, match="manifest.json"):
        read_manifest(tmp_path)


def test_write_is_atomic_no_partial_file(tmp_path):
    from manifest import Manifest, write_manifest
    write_manifest(tmp_path, Manifest(model_name="m", embed_dim=768, classes=["A"]))
    assert [p.name for p in tmp_path.iterdir()] == ["manifest.json"]      # no leftover .tmp
