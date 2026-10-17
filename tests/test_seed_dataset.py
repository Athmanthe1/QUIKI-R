import io
import json

from PIL import Image

import build_seed_dataset as sd
from cleaning_and_split import run_pipeline


def test_license_filter():
    assert sd.license_ok("CC BY-SA 4.0") and sd.license_ok("CC0") and sd.license_ok("Public domain") and sd.license_ok("CC BY 2.0")
    assert not sd.license_ok("CC BY-NC 4.0") and not sd.license_ok("CC BY-ND 2.0") and not sd.license_ok("") and not sd.license_ok("All rights reserved")


def fake_api(params):
    if params.get("list") == "categorymembers":
        return {"query": {"categorymembers": [{"title": "File:A.jpg"}, {"title": "File:B.jpg"}, {"title": "File:C.jpg"}]}}
    titles = params["titles"].split("|")
    pages = []
    for t in titles:
        lic = "CC BY-NC 4.0" if t.endswith("C.jpg") else "CC BY-SA 4.0"
        pages.append({"title": t, "imageinfo": [{"thumburl": "x", "width": 800, "height": 600, "timestamp": "2021-05-01T00:00:00Z",
                                                 "user": "bob", "mime": "image/jpeg",
                                                 "extmetadata": {"LicenseShortName": {"value": lic}, "ImageDescription": {"value": "<b>hi</b> there"}}}],
                      "categories": [{"title": "Category:Cats"}, {"title": "Category:Uploaded with X", "hidden": True}]})
    return {"query": {"pages": pages}}


def png():
    b = io.BytesIO()
    Image.new("RGB", (32, 32), (200, 10, 10)).save(b, "PNG")
    return b.getvalue()


def test_build_from_api_mocked(tmp_path):
    rep = sd.build_from_api(tmp_path, ["Cats"], per_category=10, get_json=fake_api, get_bytes=lambda u: png())
    assert rep["n_images"] == 2 and rep["rejected_by_licence"] == 1 and not rep["synthetic"]
    rec = json.loads((tmp_path / "raw_records.jsonl").read_text().splitlines()[0])
    assert rec["categories"] == ["Cats", "Uploaded with X"] and rec["description"] == "hi there"
    assert (tmp_path / "hidden_categories.txt").read_text().strip() == "Uploaded with X"


def test_synthetic_seed_runs_through_cleaning(tmp_path):
    rep = sd.build_synthetic(tmp_path, n_categories=6, per_category=20)
    assert rep["synthetic"] and rep["n_images"] == 120
    hidden = (tmp_path / "hidden_categories.txt").read_text().split("\n")
    tr, va, te, _ = run_pipeline(str(tmp_path / "raw_records.jsonl"), hidden_categories=[h for h in hidden if h])
    assert len(tr) and len(te)
    assert all("Uploaded with synthetic tool" not in c for c in tr["categories"])
    assert set(tr["uploader"]).isdisjoint(set(te["uploader"]))
