"""Embed a catalog's product images and product texts with EmbeddingGemma 2 (open weights, runs locally).

  uv run embed.py demo
  uv run embed.py demo --text-only     # product texts changed, images did not

Resumable: image vectors are written in chunks to data/<name>/vec/img_<start>.npy; a chunk that exists is
skipped. Merged output: data/<name>/img_vecs.npy (float32, 768-d, L2-normalised, row order = images.json),
data/<name>/txt_vecs.npy (same order; text = name, variant, composition, no task prefix: a prefix lowered
agreement with gemini-embedding-2 in a baseline), data/<name>/bad_images.json.

Runs on Apple GPUs (MPS), CUDA or CPU in float32: the model card forbids float16.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

# A Hugging Face id (accept the Gemma licence on the model page first) or a local folder.
MODEL = os.environ.get("EG2_MODEL", "google/embeddinggemma-2")
ROOT = Path(__file__).parent / "data"
CHUNK = 256
BATCH = 8


def text_of(i: dict) -> str:
    v = f" ({i['variant_name']})" if i["variant_name"] and i["variant_name"].lower() not in ("standard", "default") else ""
    return f"{i['name']}{v}. {i['composition']}"


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL, device=device(), model_kwargs={"torch_dtype": torch.float32},
                               config_kwargs={"audio_config": None})


def main(name: str, text_only: bool = False) -> None:
    out = ROOT / name
    images = json.loads((out / "images.json").read_text())
    (out / "vec").mkdir(exist_ok=True)

    model = load_model()
    print(f"{name}: {len(images)} images on {device()}", flush=True)

    if text_only:  # product texts changed (re-fetched images.json); image vectors stay valid
        img = np.load(out / "img_vecs.npy")
        assert img.shape[0] == len(images), "images.json no longer matches img_vecs.npy: rerun the full embed"
        txt = model.encode([text_of(i) for i in images], batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
        np.save(out / "txt_vecs.npy", txt.astype(np.float32))
        print(f"{name}: text vectors refreshed", flush=True)
        return

    bad_path = out / "bad_images.json"
    bad = json.loads(bad_path.read_text()) if bad_path.exists() else {}
    t0 = time.time()
    for start in range(0, len(images), CHUNK):
        f = out / "vec" / f"img_{start:06d}.npy"
        if f.exists():
            continue
        batch = images[start:start + CHUNK]
        pil, ok = [], []
        for i in batch:
            try:
                pil.append(Image.open(ROOT / i["image_file"]).convert("RGB"))
                ok.append(True)
            except Exception as e:
                bad[str(i["id"])] = str(e)[:120]
                pil.append(Image.new("RGB", (64, 64)))
                ok.append(False)
        vecs = model.encode([{"image": p} for p in pil], batch_size=BATCH, normalize_embeddings=True, convert_to_numpy=True)
        vecs = vecs.astype(np.float32)
        vecs[~np.array(ok)] = 0.0
        np.save(f, vecs)
        bad_path.write_text(json.dumps(bad, indent=1))
        done = start + len(batch)
        rate = (time.time() - t0) / max(1, done)
        print(f"{name}: {done}/{len(images)} images, {rate:.2f}s each, bad {len(bad)}", flush=True)

    img = np.concatenate([np.load(out / "vec" / f"img_{s:06d}.npy") for s in range(0, len(images), CHUNK)])
    assert img.shape[0] == len(images)
    np.save(out / "img_vecs.npy", img)

    texts = [text_of(i) for i in images]
    txt = model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    np.save(out / "txt_vecs.npy", txt)
    (out / "embedded.ok").write_text(time.strftime("%Y-%m-%d %H:%M"))
    print(f"{name}: done in {(time.time() - t0) / 60:.0f} min; bad images {len(bad)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], text_only="--text-only" in sys.argv)
