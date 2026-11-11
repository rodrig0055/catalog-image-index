"""Turn a catalog CSV into the index's input: data/<name>/images.json + images/<id>.webp (600 px).

  uv run prepare.py examples/demo.csv              # name defaults to the CSV's stem: data/demo/
  uv run prepare.py my_catalog.csv --name shop

One row per product image: product_id, name, variant, slot, image, description[, url]. `image` is a
URL or a local path; `slot` orders a variant's images (1 = the main image, the one feeds use).
`description` may be HTML: the composition block (Blooms, Components, Inclusions, Ingredients...)
is extracted, because that is what the photo should show.
"""

import argparse
import csv
import html
import io
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent / "data"
SIZE = 600


def plain(text: str | None) -> str:
    text = html.unescape(html.unescape(text or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Section headers that open the composition, and the ones that end it. A header must be followed by a
# capitalised word, digit or list dash, so "Noble Blooms is designed..." (a product name) is not a header.
START = r"(?:Blooms(?: Includes)?|Components|Inclusions|The Assortment|Product Highlights|Ingredients|Material(?: & Packaging)?)"
STOP = (r"(?:Perfect For|Additional Notes|Dimensions|Care Instructions|Variants|Availability|Size(?: & Weight)?|"
        r"Approximate(?:ly)? (?:height|size)|Packaging Measurement|Personali[sz]ation Instructions|Notes)\b")
COMP = re.compile(rf"(?:^|(?<=[\s.!?…:]))({START})\s*:?\s+(?=[A-Za-z0-9\-•])(.{{10,600}}?)(?=\s{STOP}|$)")
SENTENCE = re.compile(r"[^.!?…]*\b(?:consists of|features|includes|composed of|made up of|comes with|filled with)\b"
                      r"[^.!?…]{5,300}[.!?…]?", re.I)


def composition(desc: str, name: str = "") -> str:
    """What is in the product: the composition block (never truncated), else the first 'consists of /
    features / includes' sentence, else the opening of the copy."""
    for m in COMP.finditer(desc):
        if name and desc[m.start(1):].startswith(name):
            continue
        return m.group(2).strip().rstrip(".")
    m = SENTENCE.search(desc)
    return m.group(0).strip() if m else desc[:300].strip()


def load_image(src: str) -> Image.Image:
    if src.startswith(("http://", "https://")):
        for attempt in range(4):
            try:
                req = urllib.request.Request(src, headers={"User-Agent": "catalog-image-index"})
                return Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=30).read()))
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    return Image.open(src)


def main(csv_path: str, name: str | None) -> None:
    name = name or Path(csv_path).stem
    out = ROOT / name
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    images = []
    for k, r in enumerate(rows, 1):
        desc = plain(r.get("description"))
        images.append({
            "id": k,
            "catalog": name,
            "product_id": r["product_id"],
            "variant_id": f"{r['product_id']}:{r.get('variant') or ''}",
            "slot": str(r.get("slot") or 1),
            "name": r["name"].strip(),
            "variant_name": (r.get("variant") or "").strip(),
            "url": r.get("url") or "",
            "image_url": r["image"],
            "image_file": f"{name}/images/{k}.webp",
            "composition": composition(desc, r["name"].strip()),
            "copy": desc[:6000],
        })
    (out / "images.json").write_text(json.dumps(images, ensure_ascii=False, indent=1))

    def get(i: dict) -> str | None:
        dest = ROOT / i["image_file"]
        if dest.exists():
            return None
        try:
            im = load_image(i["image_url"]).convert("RGB")
            im.thumbnail((SIZE, SIZE))
            im.save(dest, "WEBP", quality=85)
        except Exception as e:  # noqa: BLE001 - reported, and embedded as unreadable
            return f"{i['image_url']}: {e}"
        return None

    with ThreadPoolExecutor(6) as pool:
        failed = [f for f in pool.map(get, images) if f]
    for f in failed:
        print("failed:", f)
    print(f"{name}: {len(images)} images over {len({i['product_id'] for i in images})} products -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--name")
    a = ap.parse_args()
    main(a.csv, a.name)
