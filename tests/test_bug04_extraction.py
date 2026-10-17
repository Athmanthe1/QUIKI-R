"""Bug 4: label alignment after failed images, streaming, multi-label from metadata, train-only index, dedup after extraction."""
import json
import math

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import extract_embeddings as ex
from encoders import FakeEncoder
from indexing import load_index
from manifest import read_manifest

COLORS = {"red": (230, 20, 20), "green": (20, 200, 20), "blue": (20, 20, 230), "yellow": (230, 230, 20)}


def make_images(tmp_path, specs):
    """specs: list of (filename, colour name | None=corrupt). Returns {filename: path}."""
    paths = {}
    for name, colour in specs:
        p = tmp_path / name
        if colour is None:
            p.write_bytes(b"not an image")
        else:
            rng = np.random.default_rng(abs(hash(name)) % 2**32)
            arr = np.clip(np.array(COLORS[colour])[None, None, :] + rng.integers(-25, 25, size=(32, 32, 3)), 0, 255)
            Image.fromarray(arr.astype("uint8")).save(p)
        paths[name] = str(p)
    return paths


def frame(paths, cats, start="2020-01-01", uploader_prefix="u"):
    n = len(paths)
    return pd.DataFrame({"image_path": paths, "categories": cats,
                         "timestamp": pd.date_range(start, periods=n, freq="D", tz="UTC").astype(str),
                         "uploader": [f"{uploader_prefix}{i}" for i in range(n)],
                         "title": [""] * n, "description": [""] * n})


def test_old_slicing_misaligned_labels_after_a_failed_image():
    batch_paths, batch_cats = ["a", "BROKEN", "c", "d"], ["A", "B", "C", "D"]
    valid = ["a", "c", "d"]                                       # 'BROKEN' failed to load
    assert batch_cats[:len(valid)] == ["A", "B", "C"]             # old code: 'c' labelled B, 'd' labelled C


def test_labels_stay_aligned_when_an_image_fails_mid_batch(tmp_path):
    spec = [("r1.png", "red"), ("bad.png", None), ("g1.png", "green"), ("b1.png", "blue")]
    paths = make_images(tmp_path, spec)
    cats = [["Red"], ["Broken"], ["Green"], ["Blue"]]
    recs = ex.records_from_frames({"train": frame([paths[n] for n, _ in spec], cats)})
    store = ex.MemoryStore()
    stats = ex.stream_encode(recs, FakeEncoder(), store, batch_size=4)
    assert stats.ok == 3 and [p for p, _ in stats.failed] == [paths["bad.png"]]
    by_path = {r["path"]: r for r in store.rows}
    assert by_path[paths["g1.png"]]["categories"] == ["Green"]    # old code would have said ['Broken']
    assert by_path[paths["b1.png"]]["categories"] == ["Blue"]
    # and the embedding really belongs to that image
    enc = FakeEncoder()
    want = enc.encode_images([Image.open(paths["g1.png"]).convert("RGB")])[0]
    assert np.allclose(by_path[paths["g1.png"]]["embedding"], want, atol=1e-5)


def test_streaming_writes_one_batch_at_a_time(tmp_path):
    spec = [(f"i{i}.png", "red") for i in range(10)]
    paths = make_images(tmp_path, spec)
    recs = ex.records_from_frames({"train": frame([paths[n] for n, _ in spec], [["Red"]] * 10)})

    class CountingStore(ex.MemoryStore):
        appends = []
        def append_batch(self, rows):
            self.appends.append(len(rows)); super().append_batch(rows)

    ex.stream_encode(recs, FakeEncoder(), CountingStore(), batch_size=4)
    assert CountingStore.appends == [4, 4, 2]


def test_label_space_is_train_only_multilabel_and_min_freq():
    tr = frame(["a", "b", "c", "d"], [["X", "Y"], ["X"], ["X", "Z"], ["Y"]])
    va = frame(["e"], [["Only in val"]])
    recs = ex.records_from_frames({"train": tr, "val": va})
    classes, counts = ex.build_label_space(recs, min_freq=2)
    assert classes == ["X", "Y"] and counts == [3, 2]             # multi-label counted; 'Z' (1) and val-only dropped


def test_categories_are_canonicalised_and_deduped():
    recs = ex.records_from_frames({"train": frame(["a"], [["Category:Mount_Everest", "Mount Everest", "category:Rivers"]])})
    assert recs[0].categories == ["Mount Everest", "Rivers"]


def build(tmp_path, spec, cats_by_split, min_freq=1, **kw):
    paths = make_images(tmp_path, spec)
    frames = {}
    for split, (names, cats, start) in cats_by_split.items():
        frames[split] = frame([paths[n] for n in names], cats, start=start, uploader_prefix=split)
    out = tmp_path / "artifacts"
    kw.setdefault("dedup_threshold", 1.01)             # disabled here: same-colour fake embeddings are >0.95 similar
    summary = ex.build_artifacts(frames, out, FakeEncoder(), min_freq=min_freq, batch_size=3,
                                 store_factory=ex.MemoryStore, backend="numpy", **kw)
    return paths, out, summary


def test_index_is_built_from_train_split_only_and_aligned(tmp_path):
    spec = [("t1.png", "red"), ("t2.png", "green"), ("t3.png", "blue"), ("v1.png", "yellow"), ("s1.png", "red")]
    paths, out, summary = build(tmp_path, spec, {
        "train": (["t1.png", "t2.png", "t3.png"], [["Red"], ["Green", "Nature"], ["Blue"]], "2020-01-01"),
        "val": (["v1.png"], [["Yellow"]], "2020-02-01"),
        "test": (["s1.png"], [["Red"]], "2020-03-01")})
    man = read_manifest(out)
    idx = load_index(out / man.files["index"])
    labels = json.loads((out / man.files["index_labels"]).read_text())
    assert idx.ntotal == 3 == len(labels) == man.n_vectors               # val/test are NOT in the index
    enc = FakeEncoder()
    val_emb = enc.encode_images([Image.open(paths["v1.png"]).convert("RGB")])
    D, I = idx.search(val_emb, 1)
    assert D[0, 0] < 0.999                                               # the val image itself is absent
    t2 = enc.encode_images([Image.open(paths["t2.png"]).convert("RGB")])
    D, I = idx.search(t2, 1)
    assert D[0, 0] > 0.999
    assert sorted(man.classes[i] for i in labels[I[0, 0]]) == ["Green", "Nature"]   # multi-label survived into the index


def test_manifest_and_artifacts_are_complete(tmp_path):
    spec = [(f"t{i}.png", "red") for i in range(3)] + [("v.png", "green")]
    paths, out, summary = build(tmp_path, spec, {
        "train": ([f"t{i}.png" for i in range(3)], [["Red", "Warm"], ["Red"], ["Red"]], "2020-01-01"),
        "val": (["v.png"], [["Green"]], "2020-02-01")}, min_freq=2)
    man = read_manifest(out)
    assert man.classes == ["Red"] and man.class_counts == [3]            # min_freq=2 removed Warm(1)
    assert man.embed_dim == 64 and man.model_name == "fake-encoder"
    assert man.logit_scale == pytest.approx(math.log(10.0)) and man.logit_bias == -5.0
    assert man.index_type == "flat"
    for key in ("index", "index_labels", "thumbs", "text_prototypes"):
        assert (out / man.files[key]).is_file(), key
    thumbs = np.load(out / man.files["thumbs"])
    assert thumbs.shape == (3, 64, 64, 3) and thumbs.dtype == np.uint8
    protos = np.load(out / man.files["text_prototypes"])
    assert protos.shape == (1, 64)


def test_thumbnails_are_row_aligned_with_the_index(tmp_path):
    spec = [("a.png", "red"), ("b.png", "green"), ("c.png", "blue")]
    paths, out, _ = build(tmp_path, spec, {"train": (["a.png", "b.png", "c.png"], [["R"], ["G"], ["B"]], "2020-01-01")})
    man = read_manifest(out)
    idx = load_index(out / man.files["index"])
    thumbs = np.load(out / man.files["thumbs"])
    enc = FakeEncoder()
    for name, dom in (("a.png", 0), ("b.png", 1), ("c.png", 2)):
        _, I = idx.search(enc.encode_images([Image.open(paths[name]).convert("RGB")]), 1)
        assert thumbs[I[0, 0]].reshape(-1, 3).mean(axis=0).argmax() == dom      # row i's thumbnail is image i


def test_embedding_dedup_runs_after_extraction_and_keeps_the_earliest(tmp_path):
    spec = [("orig.png", "red"), ("other.png", "green")]
    paths = make_images(tmp_path, spec)
    import shutil
    shutil.copy(paths["orig.png"], tmp_path / "copy.png")                       # identical pixels -> identical embedding
    frames = {"train": frame([paths["orig.png"], paths["other.png"]], [["R"], ["G"]], start="2020-01-01"),
              "test": frame([str(tmp_path / "copy.png")], [["R"]], start="2020-06-01", uploader_prefix="s")}
    out = tmp_path / "artifacts"
    summary = ex.build_artifacts(frames, out, FakeEncoder(), min_freq=1, batch_size=2,
                                 store_factory=ex.MemoryStore, backend="numpy")
    assert summary["duplicates_removed"] == 1 and summary["rows_final"] == 2
    assert summary["rows_by_split"] == {"train": 2}                              # the later copy (test) was dropped


def test_folder_mode_still_works(tmp_path):
    for cat, col in (("Category:Reds", "red"), ("Category:Greens", "green")):
        d = tmp_path / "data" / cat
        d.mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (16, 16), COLORS[col]).save(d / f"{i}.png")
    recs = ex.records_from_folders(str(tmp_path / "data"))
    assert len(recs) == 4 and {tuple(r.categories) for r in recs} == {("Reds",), ("Greens",)}
    assert all(r.split == "train" for r in recs)


def test_parquet_store_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    store = ex.ParquetStore(tmp_path / "x.parquet")
    row = {"path": "p", "split": "train", "categories": ["A", "B"], "uploader": "u", "timestamp": "2020-01-01",
           "width": 3, "height": 4, "thumb": np.zeros((64, 64, 3), np.uint8),
           "embedding": np.arange(8, dtype=np.float32), "meta_embedding": None}
    store.append_batch([row]); store.close()
    got = list(ex.ParquetStore(tmp_path / "x.parquet").iter_batches(10))[0][0]
    assert got["categories"] == ["A", "B"] and np.allclose(got["embedding"], row["embedding"])
