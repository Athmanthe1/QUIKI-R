"""build_seed_dataset.py - small open-licence seed dataset from the MediaWiki API (3-5k images, ~100 categories).

    python build_seed_dataset.py --out_dir seed --per_category 40            # needs network access to commons.wikimedia.org
    python build_seed_dataset.py --out_dir seed_synth --synthetic             # offline, SYNTHETIC images (CI / plumbing only)

Output (``--out_dir``):  images/*.jpg, raw_records.jsonl (input of cleaning_and_split.py), hidden_categories.txt,
seed_report.json. Records keep ALL categories of a file (multi-label) and the uploader / timestamp / licence.
Only files whose licence is CC0, public domain, CC BY or CC BY-SA are kept; licence + attribution are stored per record.
Synthetic data is NOT Commons data: never report metrics from it as Commons results.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

logger = logging.getLogger("quiki.seed")
API = "https://commons.wikimedia.org/w/api.php"
UA = "QUIKI-R-hackathon/1.4 (https://github.com/Athmanthe1/QUIKI-R; research prototype)"
THUMB_WIDTH = 320

# Content categories that exist on Commons (object / scene / event / abstract mix). The script reports the ones it
# could not fill; it does not pretend they were present.
DEFAULT_CATEGORIES = [
    "Eiffel Tower", "Brooklyn Bridge", "Taj Mahal", "Colosseum", "Big Ben", "Golden Gate Bridge", "Sydney Opera House",
    "Machu Picchu", "Great Wall of China", "Statue of Liberty", "Notre-Dame de Paris", "Sagrada Família",
    "Mount Fuji", "Matterhorn", "Grand Canyon", "Niagara Falls", "Lake Como", "Iguazu Falls", "Yosemite National Park",
    "Sunsets", "Rainbows", "Snow", "Beaches", "Waterfalls", "Forests", "Deserts", "Glaciers", "Volcanoes", "Rivers", "Lakes",
    "Cats", "Dogs", "Horses", "Cattle", "Sheep", "Elephants", "Lions", "Tigers", "Giraffes", "Penguins", "Owls", "Eagles",
    "Butterflies", "Bees", "Spiders", "Frogs", "Dolphins", "Sharks", "Roses", "Sunflowers", "Tulips", "Orchids", "Oak trees",
    "Apples", "Bananas", "Pizza", "Bread", "Coffee", "Cheese", "Sushi", "Wine", "Chocolate",
    "Bicycles", "Trains", "Locomotives", "Airplanes", "Helicopters", "Ships", "Sailboats", "Buses", "Trams", "Motorcycles",
    "Cars", "Tractors", "Lighthouses", "Windmills", "Castles", "Churches", "Mosques", "Temples", "Bridges", "Towers",
    "Skyscrapers", "Street art", "Graffiti", "Statues", "Fountains", "Clocks", "Maps", "Flags", "Coins", "Stamps",
    "Fireworks", "Carnivals", "Marathons", "Football matches", "Concerts", "Protests", "Parades", "Markets",
    "Paintings", "Sculptures", "Manuscripts", "Musical instruments", "Guitars", "Pianos", "Chess", "Telescopes",
]

ALLOWED_LICENSE = re.compile(r"^(cc0|public domain|pd\b|pd-|cc[- ]by(-sa)?\b)", re.I)
BLOCKED_LICENSE = re.compile(r"(\bnc\b|-nc|\bnd\b|-nd|non-?commercial|no derivatives)", re.I)


def license_ok(short_name: str) -> bool:
    s = (short_name or "").strip()
    return bool(s) and bool(ALLOWED_LICENSE.search(s)) and not BLOCKED_LICENSE.search(s)


def _strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s or "")).strip()


def http_get_json(params: Dict[str, Any], retries: int = 3, pause: float = 0.2) -> Dict[str, Any]:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
                data = json.loads(r.read().decode("utf-8"))
            time.sleep(pause)
            return data
        except Exception as exc:                                  # network errors are retried, then raised
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"MediaWiki API unreachable: {last!r}")


def http_get_bytes(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
        return r.read()


def category_files(category: str, limit: int, get_json: Callable = http_get_json) -> List[str]:
    titles: List[str] = []
    cont: Dict[str, Any] = {}
    while len(titles) < limit:
        data = get_json({"action": "query", "list": "categorymembers", "cmtitle": f"Category:{category}",
                         "cmtype": "file", "cmlimit": min(500, limit - len(titles) + 20), **cont})
        titles += [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
        if "continue" not in data:
            break
        cont = data["continue"]
    return titles[:limit]


def file_records(titles: List[str], get_json: Callable = http_get_json) -> List[Dict[str, Any]]:
    """imageinfo + categories (with the hidden flag) for up to 50 titles."""
    pages: Dict[str, Dict[str, Any]] = {}
    cont: Dict[str, Any] = {}
    while True:
        data = get_json({"action": "query", "titles": "|".join(titles), "prop": "imageinfo|categories",
                         "iiprop": "url|size|timestamp|user|extmetadata|mime", "iiurlwidth": THUMB_WIDTH,
                         "iiextmetadatafilter": "LicenseShortName|ImageDescription|Artist",
                         "cllimit": "max", "clprop": "hidden", **cont})
        for p in data.get("query", {}).get("pages", []):
            cur = pages.setdefault(p["title"], {"title": p["title"], "imageinfo": p.get("imageinfo"), "categories": []})
            if p.get("imageinfo") and not cur["imageinfo"]:
                cur["imageinfo"] = p["imageinfo"]
            cur["categories"] += p.get("categories", [])
        if "continue" not in data:
            break
        cont = data["continue"]
    out = []
    for p in pages.values():
        ii = (p.get("imageinfo") or [None])[0]
        if not ii or not str(ii.get("mime", "")).startswith("image/") or ii.get("mime") in ("image/svg+xml", "image/gif"):
            continue
        meta = ii.get("extmetadata", {})
        lic = (meta.get("LicenseShortName") or {}).get("value", "")
        out.append({"title": p["title"], "thumburl": ii.get("thumburl") or ii.get("url"), "width": ii.get("width"),
                    "height": ii.get("height"), "timestamp": ii.get("timestamp"), "uploader": ii.get("user"),
                    "license": lic, "license_ok": license_ok(lic),
                    "description": _strip_html((meta.get("ImageDescription") or {}).get("value", ""))[:500],
                    "attribution": _strip_html((meta.get("Artist") or {}).get("value", ""))[:200],
                    "source_url": ii.get("descriptionurl") or ii.get("url"),
                    "categories": [c["title"].split(":", 1)[-1] for c in p["categories"]],
                    "hidden": sorted({c["title"].split(":", 1)[-1] for c in p["categories"] if c.get("hidden")})})
    return out


def build_from_api(out_dir: Path, categories: Iterable[str], per_category: int = 40, max_images: int = 5000,
                   get_json: Callable = http_get_json, get_bytes: Callable = http_get_bytes) -> Dict[str, Any]:
    from io import BytesIO
    from PIL import Image
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    seen: Dict[str, Dict[str, Any]] = {}
    hidden: set = set()
    empty_cats, rejected_license, failed_dl = [], 0, 0
    for cat in categories:
        if len(seen) >= max_images:
            break
        try:
            titles = [t for t in category_files(cat, per_category, get_json) if t not in seen]
        except Exception as exc:
            logger.warning("category %r failed: %s", cat, exc)
            empty_cats.append(cat)
            continue
        if not titles:
            empty_cats.append(cat)
            continue
        for s in range(0, len(titles), 40):
            for rec in file_records(titles[s:s + 40], get_json):
                if rec["title"] in seen:
                    continue
                if not rec["license_ok"]:
                    rejected_license += 1
                    continue
                try:
                    img = Image.open(BytesIO(get_bytes(rec["thumburl"]))).convert("RGB")
                    path = out_dir / "images" / f"{len(seen):06d}.jpg"
                    img.save(path, quality=88)
                except Exception as exc:
                    failed_dl += 1
                    logger.warning("download failed for %s: %s", rec["title"], exc)
                    continue
                hidden.update(rec["hidden"])
                seen[rec["title"]] = {**rec, "image_path": str(path), "seed_category": cat}
        logger.info("%s: %d images so far", cat, len(seen))
    return _write(out_dir, list(seen.values()), hidden, {
        "source": "MediaWiki API (commons.wikimedia.org)", "synthetic": False, "categories_requested": len(list(categories)) if not isinstance(categories, list) else len(categories),
        "categories_empty_or_failed": empty_cats, "rejected_by_licence": rejected_license, "download_failures": failed_dl})


# ---- synthetic fallback (plumbing only) ---------------------------------------------------------------------------
def build_synthetic(out_dir: Path, n_categories: int = 12, per_category: int = 40, seed: int = 0) -> Dict[str, Any]:
    """Colour/shape images with multi-label categories, uploaders and timestamps, plus maintenance noise. NOT Commons."""
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    hues = [tuple(int(v) for v in np.array([np.cos(i), np.cos(i + 2.1), np.cos(i + 4.2)]) * 100 + 128) for i in np.linspace(0, 6.28, n_categories, endpoint=False)]
    shapes = ["ellipse", "rectangle"]
    recs = []
    t0 = np.datetime64("2019-01-01T00:00:00")
    for ci in range(n_categories):
        for j in range(per_category):
            n = len(recs)
            img = Image.new("RGB", (128, 128), tuple(int(np.clip(c + rng.integers(-25, 25), 0, 255)) for c in hues[ci]))
            d = ImageDraw.Draw(img)
            box = (int(rng.integers(10, 40)), int(rng.integers(10, 40)), int(rng.integers(80, 118)), int(rng.integers(80, 118)))
            fill = tuple(int(255 - c) for c in hues[(ci + 1) % n_categories])
            (d.ellipse if ci % 2 == 0 else d.rectangle)(box, fill=fill)
            path = out_dir / "images" / f"{n:06d}.jpg"
            img.save(path, quality=90)
            cats = [f"Synthetic colour {ci}", f"Synthetic {shapes[ci % 2]} family"]
            if rng.random() < 0.3:
                cats.append("Uploaded with synthetic tool")           # maintenance noise to exercise cleaning
            recs.append({"title": f"File:synthetic_{n}.jpg", "image_path": str(path), "width": 128, "height": 128,
                         "timestamp": str(t0 + np.timedelta64(int(n * 6), "h")) + "Z", "uploader": f"synth_user_{n // 8}",
                         "license": "CC0", "license_ok": True, "description": f"synthetic image {n}", "attribution": "generated",
                         "source_url": "", "categories": cats, "hidden": ["Uploaded with synthetic tool"], "seed_category": cats[0]})
    return _write(out_dir, recs, {"Uploaded with synthetic tool"}, {"source": "SYNTHETIC (not Wikimedia Commons)", "synthetic": True})


def _write(out_dir: Path, records: List[Dict[str, Any]], hidden: set, extra: Dict[str, Any]) -> Dict[str, Any]:
    with open(out_dir / "raw_records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({"image_path": r["image_path"], "categories": r["categories"], "timestamp": r["timestamp"],
                                "uploader": r["uploader"], "title": r["title"], "description": r["description"],
                                "width": r["width"], "height": r["height"], "license": r["license"],
                                "attribution": r["attribution"], "source_url": r["source_url"]}) + "\n")
    (out_dir / "hidden_categories.txt").write_text("\n".join(sorted(hidden)) + "\n", encoding="utf-8")
    report = {**extra, "n_images": len(records), "n_distinct_categories": len({c for r in records for c in r["categories"]}),
              "n_uploaders": len({r["uploader"] for r in records}), "n_hidden_categories_seen": len(hidden)}
    (out_dir / "seed_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("seed report: %s", report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="seed")
    ap.add_argument("--per_category", type=int, default=40)
    ap.add_argument("--max_images", type=int, default=5000)
    ap.add_argument("--categories_file", help="one Commons category name per line (default: built-in ~100)")
    ap.add_argument("--synthetic", action="store_true", help="offline synthetic images (plumbing only)")
    ap.add_argument("--fallback_synthetic", action="store_true", help="if the API is unreachable, write synthetic data instead (clearly labelled)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    if a.synthetic:
        build_synthetic(Path(a.out_dir), per_category=a.per_category)
        return
    cats = DEFAULT_CATEGORIES
    if a.categories_file:
        cats = [ln.strip() for ln in Path(a.categories_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    try:
        rep = build_from_api(Path(a.out_dir), cats, a.per_category, a.max_images)
        if rep["n_images"] == 0:
            raise RuntimeError("API returned no usable images")
    except RuntimeError as exc:
        if not a.fallback_synthetic:
            raise SystemExit(f"{exc}\nRe-run with network access, or use --synthetic / --fallback_synthetic (synthetic data is not Commons).")
        logger.error("%s -> writing SYNTHETIC data (not Commons)", exc)
        build_synthetic(Path(a.out_dir), per_category=a.per_category)


if __name__ == "__main__":
    main()
