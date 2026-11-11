"""Text search over a catalog's product photos.

  uv run search.py demo "kraft-wrapped pink roses" -k 5
"""

import argparse

import numpy as np

from report import load


def main(name: str, query: str, k: int) -> None:
    from embed import load_model
    images, I, _ = load(name)
    v = load_model().encode([query], normalize_embeddings=True)[0]
    s = I @ v
    for j in np.argsort(-s)[:k]:
        i = images[j]
        print(f"{s[j]:.3f}  {i['name']} (slot {i['slot']})  {i['url'] or i['image_url']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args()
    main(a.name, a.query, a.k)
