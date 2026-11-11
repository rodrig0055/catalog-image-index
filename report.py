"""Mismatch, image-type and duplicate report from the EmbeddingGemma 2 vectors.

  uv run report.py demo                                  # one catalog
  uv run report.py demo -q "kraft-wrapped sunflowers"    # plus text-search examples in the HTML
  uv run report.py shop_a shop_b                         # several catalogs, plus duplicates across them

Writes data/report/<name>_images.csv (one row per image, with type, own-text rank and flags),
data/report/<name>_duplicates.csv, data/report/<name>.html (the review page),
data/report/cross_catalog_duplicates.csv and data/report/summary.json.

Flags:
- type: zero-shot against the label prompts below. "graphic" = instructions, notices, size charts: not a photo
  of the product. Fine as a late slot; a problem as the first slot (feeds use it as the main image).
- mismatch: a photo whose own product text ranks poorly among all products AND whose best-matching other
  product beats its own clearly. Secondary slots of bundles legitimately show one component, so bundle names
  are surfaced for review, not flagged as errors.
- duplicates: byte-identical files (md5) and near-identical photos (cosine >= DUP) on DIFFERENT products.

On a real catalog (1,629 images of 410 products), checked by eye: product text must carry the real
composition (with marketing copy only, 15 correct main images were flagged; with composition, 1), and
duplicates were the strongest signal (291 images on 135 products shared a photo with another product).
Treat mismatch flags as a review queue, not as errors.
"""

import argparse
import base64
import csv
import hashlib
import html
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent / "data"
OUT = ROOT / "report"
DUP = 0.99
MISMATCH_RANK = 50      # own product outside the top 50 products (10% of a small catalog, at least 2)
MISMATCH_MARGIN = 0.03  # best other product's text beats own text by at least this cosine

LABELS = {
    "product": "a studio product photo of a flower bouquet, plant, cake or gift box on a plain background",
    "lifestyle": "a photo of a person holding or presenting a bouquet or gift",
    "printed": "a product with printed words on it, such as a mug, greeting card, poster or t-shirt",
    "graphic": "a graphic made of text and icons: ordering instructions, a notice with an email address, steps, or a size chart",
    "placeholder": "a blank, grey placeholder or coming soon image",
}


def first_slot(slots: list[str]) -> str:
    return sorted(slots, key=lambda s: (int("".join(c for c in s if c.isdigit()) or 99), s))[0]


def load(name: str):
    d = ROOT / name
    im = json.loads((d / "images.json").read_text())
    return im, np.load(d / "img_vecs.npy"), np.load(d / "txt_vecs.npy")


def catalog_report(name: str, L: np.ndarray) -> dict:
    im, I, T = load(name)
    n = len(im)
    pids = np.array([i["product_id"] for i in im])
    up, first = np.unique(pids, return_index=True)
    own_col = np.searchsorted(up, pids)

    rank_cut = min(MISMATCH_RANK, max(2, len(up) // 10))
    S = I @ T[first].T                                  # image x product-text similarity
    own = S[np.arange(n), own_col]
    rank = (S > own[:, None]).sum(1)
    S_other = S.copy()
    S_other[np.arange(n), own_col] = -1
    best_other = S_other.argmax(1)
    margin = S_other[np.arange(n), best_other] - own

    P = I @ L.T
    keys = list(LABELS)
    typ = [keys[j] for j in P.argmax(1)]
    blank = ~I.any(1)                                   # unreadable image files were zeroed in embed.py

    slots_by_variant = defaultdict(list)
    for i in im:
        slots_by_variant[i["variant_id"]].append(i["slot"])
    is_first = np.array([i["slot"] == first_slot(slots_by_variant[i["variant_id"]]) for i in im])

    md5 = [hashlib.md5((ROOT / i["image_file"]).read_bytes()).hexdigest() if not b else "" for i, b in zip(im, blank)]

    # duplicates on different products: union-find over exact md5 matches and near-identical vectors
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_hash = defaultdict(list)
    for k, h in enumerate(md5):
        if h:
            by_hash[h].append(k)
    for ks in by_hash.values():
        for k in ks[1:]:
            parent[find(k)] = find(ks[0])
    for s in range(0, n, 2048):
        G = I[s:s + 2048] @ I.T
        a, b = np.nonzero(G >= DUP)
        for x, y in zip(a + s, b):
            if x < y:
                parent[find(y)] = find(x)
    groups = defaultdict(list)
    for k in range(n):
        groups[find(k)].append(k)
    dup_groups = [g for g in groups.values() if len({pids[k] for k in g}) > 1]
    dup_of = {k: gi for gi, g in enumerate(dup_groups) for k in g}

    rows = []
    for k, i in enumerate(im):
        photo = typ[k] in ("product", "lifestyle", "printed")
        mismatch = bool(photo and not blank[k] and rank[k] >= rank_cut and margin[k] >= MISMATCH_MARGIN)
        bundle = any(w in i["name"].lower() for w in ("bundle", "set", "hamper", "box", "combo", "duo", "trio", "basket"))
        flags = []
        if blank[k]:
            flags.append("unreadable")
        if typ[k] == "placeholder":
            flags.append("placeholder")
        if typ[k] == "graphic" and is_first[k]:
            flags.append("graphic_as_main_image")
        if mismatch and k in dup_of:
            flags.append("shared_component")  # the same photo sits on another product: a bundle part, not an error
        elif mismatch:
            flags.append("mismatch_main" if is_first[k] else ("mismatch_bundle_component" if bundle else "mismatch_secondary"))
        if k in dup_of:
            flags.append("duplicate_across_products")
        rows.append({
            "catalog": name, "image_id": i["id"], "product_id": i["product_id"], "variant_id": i["variant_id"],
            "slot": i["slot"], "first_slot": int(is_first[k]), "name": i["name"], "variant": i["variant_name"],
            "type": typ[k], "own_text_cos": round(float(own[k]), 4), "own_rank": int(rank[k]),
            "best_other_product": im[first[best_other[k]]]["name"], "margin": round(float(margin[k]), 4),
            "dup_group": dup_of.get(k, ""), "flags": ";".join(flags), "url": i["url"], "image_url": i["image_url"],
        })

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}_images.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["flags"] == "", -r["own_rank"])))
    # A group whose photo is the MAIN image of 2+ different products: those products look identical in the
    # feed, on listing pages and to Google. The rest is expected reuse (bundle parts, size or instruction cards).
    main_shared = {gi for gi, g in enumerate(dup_groups) if len({pids[k] for k in g if is_first[k]}) >= 2}
    with open(OUT / f"{name}_duplicates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "main_photo_shared", "products", "image_id", "product_id", "name", "variant", "slot",
                    "first_slot", "exact_file_match", "url", "image_url"])
        for gi in sorted(range(len(dup_groups)), key=lambda x: (x not in main_shared, -len(dup_groups[x]))):
            g = dup_groups[gi]
            hs = Counter(md5[k] for k in g)
            for k in g:
                w.writerow([gi, int(gi in main_shared), len({pids[x] for x in g}), im[k]["id"], im[k]["product_id"],
                            im[k]["name"], im[k]["variant_name"], im[k]["slot"], int(is_first[k]),
                            int(hs[md5[k]] > 1), im[k]["url"], im[k]["image_url"]])

    flag_counts = Counter(fl for r in rows for fl in r["flags"].split(";") if fl)
    return {
        "_rows": rows, "_dups": [[int(k) for k in g] for g in dup_groups], "_main_shared": sorted(main_shared),
        "images": n, "products": int(len(up)),
        "types": dict(Counter(typ)),
        "own_product_top1_pct": round(100 * float((rank == 0).mean()), 1),
        "own_product_top5_pct": round(100 * float((rank < 5).mean()), 1),
        "flags": dict(flag_counts),
        "duplicate_groups": len(dup_groups),
        "images_in_duplicate_groups": sum(len(g) for g in dup_groups),
        "products_in_duplicate_groups": len({pids[k] for g in dup_groups for k in g}),
        "main_photo_shared_groups": len(main_shared),
        "products_with_shared_main_photo": len({pids[k] for gi in main_shared for k in dup_groups[gi] if is_first[k]}),
    }


def cross_catalog(names: list[str]) -> dict:
    data = {m: load(m) for m in names}
    out = []
    for a_i, a in enumerate(names):
        for b in names[a_i + 1:]:
            ia, Ia, _ = data[a]
            ib, Ib, _ = data[b]
            best, arg = np.full(len(ia), -1.0), np.zeros(len(ia), int)
            for s in range(0, len(ib), 4096):
                G = Ia @ Ib[s:s + 4096].T
                j = G.argmax(1)
                v = G[np.arange(len(ia)), j]
                better = v > best
                best[better], arg[better] = v[better], j[better] + s
            for k in np.nonzero(best >= DUP)[0]:
                out.append({"catalog_a": a, "image_a": ia[k]["id"], "product_a": ia[k]["name"], "url_a": ia[k]["url"],
                            "catalog_b": b, "image_b": ib[arg[k]]["id"], "product_b": ib[arg[k]]["name"], "url_b": ib[arg[k]]["url"],
                            "cos": round(float(best[k]), 4)})
    if out:
        with open(OUT / "cross_catalog_duplicates.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
    return dict(Counter(r["catalog_a"] + "-" + r["catalog_b"] for r in out))




# --- Review page -------------------------------------------------------------------------------

def thumb(image_file: str, px: int = 220) -> str:
    from PIL import Image
    try:
        im = Image.open(ROOT / image_file).convert("RGB")
    except Exception:  # noqa: BLE001
        return ""
    im.thumbnail((px, px))
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=72)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def html_page(name: str, rep: dict, images: list[dict], searches: list[tuple[str, list[tuple[int, float]]]]) -> str:
    rows = rep["_rows"]
    by_id = {i["id"]: i for i in images}
    idx = {i["id"]: k for k, i in enumerate(images)}
    best_img = {}
    for i in images:   # first-slot image per product name, to show "looks like" next to a mismatch
        best_img.setdefault(i["name"], i)
    e = html.escape

    def card(i, note="", tag=""):
        t = thumb(i["image_file"])
        img = f'<img src="{t}" alt="">' if t else '<div class="noimg">unreadable</div>'
        return (f'<figure>{img}<figcaption>{f"<span class=tag>{e(tag)}</span>" if tag else ""}'
                f'<b>{e(i["name"])}</b><small>slot {e(i["slot"])}{" · " + e(i["variant_name"]) if i["variant_name"] else ""}</small>'
                f'{f"<small>{note}</small>" if note else ""}</figcaption></figure>')

    sections = []
    not_photo = [r for r in rows if any(f in r["flags"] for f in ("graphic_as_main_image", "placeholder", "unreadable"))]
    if not_photo:
        sections.append(("Main image is not a product photo",
                         "Instruction graphics, placeholders and broken files in the slot that feeds and listing pages show.",
                         "".join(card(by_id[r["image_id"]], tag=r["flags"].split(";")[0].replace("_", " ")) for r in not_photo)))
    mism = [r for r in rows if "mismatch" in r["flags"] or "shared_component" in r["flags"]]
    if mism:
        pairs = "".join(
            f'<div class="pair">{card(by_id[r["image_id"]], note="own text ranks #" + str(r["own_rank"] + 1), tag="this photo" + (", shared" if "shared_component" in r["flags"] else ""))}'
            f'<div class="arrow">looks more like</div>{card(best_img[r["best_other_product"]], tag="product")}</div>'
            for r in mism)
        sections.append(("Photo does not match its product",
                         "The photo is closer to another product's description than to its own: a stale, swapped or wrong image.",
                         pairs))
    if rep["_dups"]:
        groups = '<div class="row">' + "".join('<div class="group">' + "".join(card(images[k], tag="main" if images[k]["slot"] == "1" else "") for k in g) + "</div>"
                         for g in rep["_dups"]) + "</div>"
        sections.append(("Same photo on different products",
                         "Products that look identical in the feed, on listing pages and to Google Images.", groups))
    for q, hits in searches:
        sections.append((f"Search: “{q}”", "Text search over every product photo, no tags or alt text involved.",
                         '<div class="grid">' + "".join(card(images[k], note=f"{s:.2f}") for k, s in hits) + "</div>"))

    flags = rep["flags"]
    stats = [(rep["images"], f"images of {rep['products']} products"),
             (f"{rep['own_product_top1_pct']}%", "of photos match their own product's text best"),
             (flags.get("graphic_as_main_image", 0) + flags.get("placeholder", 0) + flags.get("unreadable", 0), "main images that are not product photos"),
             (sum(v for k, v in flags.items() if k.startswith("mismatch") or k == "shared_component"), "photos that look like another product"),
             (rep["products_in_duplicate_groups"], "products sharing a photo with another product")]
    tiles = "".join(f'<div class="stat"><b>{v}</b><span>{e(str(t))}</span></div>' for v, t in stats)
    body = "".join(f'<section><h2>{e(h)}</h2><p class="lede">{e(d)}</p>{c}</section>' for h, d, c in sections)
    return TEMPLATE.replace("{{NAME}}", e(name)).replace("{{TILES}}", tiles).replace("{{BODY}}", body)


TEMPLATE = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Catalog Image Review</title>
<style>
:root{--bg:#f5f4f0;--card:#fff;--ink:#1f2421;--muted:#66706a;--line:#e0ded6;--accent:#2f5d50;--tag:#b4472c}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,"Helvetica Neue",Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:40px 24px 64px}
.eyebrow{font:600 12px ui-monospace,Menlo,monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--tag)}
h1{font-size:38px;line-height:1.1;margin:8px 0 6px}h2{font-size:22px;margin:0 0 4px}
.lede{color:var(--muted);margin:0 0 16px}
.stats{display:grid;grid-template-columns:repeat(5,1fr);border:1px solid var(--line);border-radius:12px;background:var(--card);margin:28px 0 36px}
.stat{padding:18px;border-left:1px solid var(--line)}.stat:first-child{border-left:0}
.stat b{display:block;font-size:30px}.stat span{color:var(--muted);font-size:13px}
section{margin:0 0 40px}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;width:200px;flex:none}
figure img,.noimg{display:block;width:200px;height:200px;object-fit:cover;background:#ddd}
figcaption{padding:8px 10px;display:flex;flex-direction:column;gap:2px}
figcaption small{color:var(--muted);font-size:12px}
.tag{font:600 10px ui-monospace,Menlo,monospace;text-transform:uppercase;color:var(--tag);letter-spacing:.06em}
.grid,section>figure{display:inline-block;margin-right:12px}
.grid{display:flex;flex-wrap:wrap;gap:12px}
section{display:block}
section>figure{vertical-align:top;margin:0 12px 12px 0}
.row{display:flex;flex-wrap:wrap;gap:14px}.row .group{margin:0}
.pair,.group{display:flex;align-items:center;gap:14px;margin:0 0 14px;padding:12px;border:1px dashed var(--line);border-radius:12px;width:fit-content}
.arrow{color:var(--accent);font:600 12px ui-monospace,Menlo,monospace;text-transform:uppercase;width:90px;text-align:center}
@media (max-width:700px){.stats{grid-template-columns:1fr 1fr}.pair,.group{flex-wrap:wrap}}
</style>
<main><div class="eyebrow">{{NAME}} · catalog image review</div><h1>Every product photo, checked against its product</h1>
<p class="lede">EmbeddingGemma 2 puts product photos and product descriptions in one space, locally, with no API calls. Photos far from their own product, photos that are not photos, and photos shared between products show up on their own.</p>
<div class="stats">{{TILES}}</div>{{BODY}}</main></html>"""


def main(names: list[str], queries: list[str], k: int) -> None:
    from embed import load_model
    model = load_model()
    L = model.encode(list(LABELS.values()), normalize_embeddings=True)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"catalogs": {}}
    for n in names:
        rep = catalog_report(n, L)
        images, I, _ = load(n)
        searches = []
        for q in queries:
            v = model.encode([q], normalize_embeddings=True)[0]
            s = I @ v
            searches.append((q, [(int(j), float(s[j])) for j in np.argsort(-s)[:k]]))
        (OUT / f"{n}.html").write_text(html_page(n, rep, images, searches))
        summary["catalogs"][n] = {kk: v for kk, v in rep.items() if not kk.startswith("_")}
    if len(names) > 1:
        summary["cross_catalog_duplicate_images"] = cross_catalog(names)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("-q", "--query", action="append", default=[], help="text-search example for the HTML (repeatable)")
    ap.add_argument("-k", type=int, default=5, help="results per search")
    a = ap.parse_args()
    main(a.names, a.query, a.k)
