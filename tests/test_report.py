"""Offline tests: composition extraction and the flags, on hand-made vectors (no model needed)."""

import json

import numpy as np

import report
from prepare import composition, plain


def test_composition_block():
    d = plain("<p>Our best seller.</p><strong>Blooms:</strong> 12 red roses, eucalyptus <strong>Dimensions:</strong> 40 cm")
    assert composition(d) == "12 red roses, eucalyptus"


def test_composition_sentence_fallback():
    assert "includes three cupcakes" in composition("A cheerful gift. It includes three cupcakes and a card. Order today.")


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "OUT", tmp_path / "report")
    d = tmp_path / "t"
    (d / "images").mkdir(parents=True)
    names = ["Rose", "Tulip", "Cake", "Plant", "Cookies", "Honey", "Vase", "Card", "Balloon", "Box", "Wine", "Tea"]
    images = []
    for k, n in enumerate(names):
        images.append({"id": k, "product_id": str(k), "variant_id": f"{k}:", "slot": "1", "name": n,
                       "variant_name": "", "url": "", "image_url": "", "image_file": f"t/images/{k}.webp"})
        (d / "images" / f"{k}.webp").write_bytes(f"img{k}".encode())
    (d / "images" / "11.webp").write_bytes(b"img10")          # Tea reuses Wine's file: exact duplicate
    (d / "images.json").write_text(json.dumps(images))
    rng = np.random.default_rng(0)
    T = np.stack([_unit(rng.normal(size=16) + 2 * np.eye(16)[0]) for _ in names])  # all share a 'product' axis
    I = T.copy()
    I[5] = _unit(T[2] - 0.6 * T[5] + 0.1 * rng.normal(size=16))  # Honey's photo shows a cake, not honey
    I[11] = I[10]
    np.save(d / "img_vecs.npy", I)
    np.save(d / "txt_vecs.npy", T)
    L = np.stack([np.eye(16)[0]] + [-np.eye(16)[0]] * 4)     # the "product" label wins for every photo
    rep = report.catalog_report("t", L)
    flags = {r["name"]: r["flags"] for r in rep["_rows"]}
    assert "duplicate_across_products" in flags["Tea"] and "duplicate_across_products" in flags["Wine"]
    assert "mismatch_main" in flags["Honey"]
    assert flags["Rose"] == ""
