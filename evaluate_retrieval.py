"""
Evaluate retrieval quality of the Vectorize index using standard IR metrics.

Ground truth:
- If your local images are organized as data/<product_id>/... then all images that share the
  same product_id are considered relevant to each other.

This script:
- Iterates local images under DATA_FOLDER
- Re-embeds each query image using the same DINOv3 pipeline as ingestion
- Queries Cloudflare Vectorize (v2 REST API) for topK nearest neighbors
- Computes metrics: hit_rate@k, recall@k, precision@k, MRR, MAP@k, nDCG@k
"""

from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests
from dotenv import load_dotenv
from numpy.linalg import norm
from PIL import Image
from transformers import pipeline


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
DINO_MODEL = os.getenv("DINO_MODEL", "facebook/dinov3-vitb16-pretrain-lvd1689m")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "8"))


def is_image_file(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)


def is_thumbnail(key: str, filename: str) -> bool:
    key_lower = key.lower()
    filename_lower = filename.lower()

    thumbnail_indicators = [
        "thumbnail",
        "thumbnails",
        "thumb",
        "_thumb",
        "thumb_",
        "tn_",
        "_tn",
    ]
    return any(ind in key_lower or ind in filename_lower for ind in thumbnail_indicators) or (
        "/thumbnail" in key_lower or "/thumbnails" in key_lower
    )


def _product_id_from_relative_path(relative_posix: str) -> Optional[str]:
    parts = [p for p in relative_posix.split("/") if p]
    # data/<product_id>/<image>
    if len(parts) >= 2:
        return parts[0]
    return None


def _iter_local_images(data_folder: Path) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    """
    Yield (path, info) where info contains:
    - relative_path
    - product_id (optional)
    """
    for file_path in sorted(data_folder.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.name == "manifest.jsonl":
            continue

        filename = file_path.name
        if not is_image_file(filename):
            continue

        rel_posix = file_path.relative_to(data_folder).as_posix()
        if is_thumbnail(rel_posix, filename):
            continue

        product_id = _product_id_from_relative_path(rel_posix)
        yield file_path, {"relative_path": rel_posix, "filename": filename, "product_id": product_id}


def _extract_embeddings(pipe, images: List[Image.Image]) -> List[List[float]]:
    vecs = pipe(images, batch_size=EMBED_BATCH_SIZE)
    out: List[List[float]] = []
    for v in vecs:
        arr = np.array(v)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        pooled = np.mean(arr, axis=0)
        pooled = pooled / (norm(pooled) + 1e-10)
        out.append(pooled.astype("float32").tolist())
    return out


def _vectorize_query_url(account_id: str, index_name: str) -> str:
    return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/vectorize/v2/indexes/{index_name}/query"


def _vectorize_headers(api_token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}


def vectorize_query(
    *,
    account_id: str,
    index_name: str,
    api_token: str,
    vector: Sequence[float],
    top_k: int,
    return_metadata: str = "all",
    filter_expr: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    url = _vectorize_query_url(account_id, index_name)
    payload: Dict[str, Any] = {"vector": list(vector), "topK": int(top_k), "returnMetadata": return_metadata}
    if filter_expr is not None:
        payload["filter"] = filter_expr

    resp = requests.post(url, headers=_vectorize_headers(api_token), json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Vectorize query failed: HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    if not data.get("success", False):
        raise RuntimeError(f"Vectorize query error: {data.get('errors')}")
    result = data.get("result") or {}
    return list(result.get("matches") or [])


@dataclass
class QueryExample:
    path: Path
    relative_path: str
    product_id: str


def _dcg(relevances: List[float]) -> float:
    # Standard DCG with log2 discount, positions starting at 1
    total = 0.0
    for i, rel in enumerate(relevances, start=1):
        total += (2.0**rel - 1.0) / math.log2(i + 1)
    return total


def evaluate_one(
    *,
    query: QueryExample,
    retrieved: List[Dict[str, Any]],
    relevant_ids: Optional[set] = None,
    ks: Sequence[int],
) -> Dict[str, float]:
    """
    Computes per-query metrics. Relevance is based on metadata.product_id match.

    If `relevant_ids` is provided, recall is computed as (#relevant retrieved)/len(relevant_ids).
    Otherwise recall@k is computed as hit_rate@k (binary), since total relevant is unknown.
    """
    out: Dict[str, float] = {}

    retrieved_pids: List[Optional[str]] = []
    retrieved_ids: List[Optional[str]] = []
    for m in retrieved:
        retrieved_ids.append(m.get("id"))
        md = m.get("metadata") or {}
        retrieved_pids.append(md.get("product_id"))

    # Rank-wise relevance signals (1/0) by product_id match.
    rel01 = [1.0 if pid == query.product_id else 0.0 for pid in retrieved_pids]

    # MRR (first relevant rank)
    rr = 0.0
    for rank, r in enumerate(rel01, start=1):
        if r > 0:
            rr = 1.0 / float(rank)
            break
    out["mrr"] = rr

    for k in ks:
        k = int(k)
        rel_at_k = rel01[:k]
        hit = 1.0 if any(r > 0 for r in rel_at_k) else 0.0
        out[f"hit_rate@{k}"] = hit
        out[f"precision@{k}"] = float(sum(rel_at_k)) / float(k) if k > 0 else 0.0

        # recall@k
        if relevant_ids is not None and len(relevant_ids) > 0:
            # Use ids if available (prevents counting duplicates twice)
            got_relevant = 0
            for rid, r in zip(retrieved_ids[:k], rel01[:k]):
                if r > 0 and rid is not None and rid in relevant_ids:
                    got_relevant += 1
            out[f"recall@{k}"] = float(got_relevant) / float(len(relevant_ids))
        else:
            out[f"recall@{k}"] = hit

        # AP@k (average precision at k)
        num_rel = 0.0
        sum_prec = 0.0
        for i, r in enumerate(rel_at_k, start=1):
            if r > 0:
                num_rel += 1.0
                sum_prec += (float(sum(rel_at_k[:i])) / float(i))
        denom = num_rel
        out[f"map@{k}"] = (sum_prec / denom) if denom > 0 else 0.0

        # nDCG@k (binary relevance)
        dcg = _dcg(rel_at_k)
        ideal = sorted(rel_at_k, reverse=True)
        idcg = _dcg(ideal)
        out[f"ndcg@{k}"] = (dcg / idcg) if idcg > 0 else 0.0

    return out


def mean_dict(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for r in rows for k in r.keys()})
    out: Dict[str, float] = {}
    for k in keys:
        vals = [r.get(k, 0.0) for r in rows]
        out[k] = float(sum(vals)) / float(len(vals))
    return out


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", default=os.getenv("DATA_FOLDER", "data"))
    parser.add_argument("--max-queries", type=int, default=int(os.getenv("MAX_QUERIES", "200")))
    parser.add_argument("--topks", default=os.getenv("TOPKS", "1,5,10,20"))
    parser.add_argument("--return-metadata", default=os.getenv("RETURN_METADATA", "all"), choices=["none", "indexed", "all"])
    args = parser.parse_args()

    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    index_name = os.getenv("CLOUDFLARE_VECTORIZE_INDEX")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")
    if not all([account_id, index_name, api_token]):
        raise SystemExit("Missing CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_VECTORIZE_INDEX / CLOUDFLARE_API_TOKEN in environment.")

    ks = [int(x.strip()) for x in str(args.topks).split(",") if x.strip()]
    if not ks:
        raise SystemExit("--topks must be a comma-separated list like 1,5,10,20")
    max_topk = max(ks)

    data_folder = Path(args.data_folder)
    if not data_folder.exists():
        raise SystemExit(f"Data folder does not exist: {data_folder}")

    # Build query set from local images with product_id labels
    by_product: Dict[str, List[QueryExample]] = defaultdict(list)
    for p, info in _iter_local_images(data_folder):
        pid = info.get("product_id")
        if not pid:
            continue
        by_product[str(pid)].append(QueryExample(path=p, relative_path=info["relative_path"], product_id=str(pid)))

    # Keep only products with at least 2 images (otherwise no positives)
    candidates: List[QueryExample] = []
    for pid, exs in by_product.items():
        if len(exs) < 2:
            continue
        candidates.extend(exs)

    if not candidates:
        raise SystemExit(
            "No evaluatable examples found.\n"
            "- Ensure images are organized as data/<product_id>/image.jpg so product_id can be inferred\n"
            "- Ensure each product_id has at least 2 images (so there are relevant positives)"
        )

    candidates = candidates[: max(1, int(args.max_queries))]

    # Precompute relevant sets per product_id (by relative_path; ids are stored in Vectorize metadata as product_id only)
    # We can still estimate total relevant per query based on local counts per product.
    total_relevant_per_pid: Dict[str, int] = {pid: max(0, len(exs) - 1) for pid, exs in by_product.items()}

    print(f"Loading embedding model: {DINO_MODEL}")
    pipe = pipeline("image-feature-extraction", model=DINO_MODEL)

    metrics_rows: List[Dict[str, float]] = []

    # Embed/query one-by-one to keep implementation simple and predictable.
    # If you want this faster, we can batch embeddings and parallelize queries.
    for i, q in enumerate(candidates, start=1):
        try:
            img = Image.open(q.path).convert("RGB")
        except Exception as e:
            print(f"[skip] failed to load {q.path}: {e}")
            continue

        emb = _extract_embeddings(pipe, [img])[0]
        matches = vectorize_query(
            account_id=account_id,
            index_name=index_name,
            api_token=api_token,
            vector=emb,
            top_k=max_topk,
            return_metadata=args.return_metadata,
        )

        # Some indexes will return the query image itself among results if it was indexed.
        # We don't know its id from local-only info; so we do NOT attempt to exclude by id here.
        # (If needed, we can store ids in metadata during ingestion and exclude exact self-match.)

        relevant_count = total_relevant_per_pid.get(q.product_id, 0)
        relevant_ids = None if relevant_count <= 0 else set()  # unknown exact ids; we only know count

        row = evaluate_one(query=q, retrieved=matches, relevant_ids=relevant_ids, ks=ks)
        metrics_rows.append(row)

        if i % 25 == 0:
            agg = mean_dict(metrics_rows)
            snapshot = " | ".join([f"R@{k}={agg.get(f'recall@{k}', 0.0):.3f}" for k in ks])
            print(f"[{i}/{len(candidates)}] {snapshot} | MRR={agg.get('mrr', 0.0):.3f}")

    agg = mean_dict(metrics_rows)
    print("\n" + "=" * 60)
    print(f"Evaluated queries: {len(metrics_rows)}")
    print("Aggregate metrics (mean over queries):")
    for k in ks:
        print(f"  recall@{k}:     {agg.get(f'recall@{k}', 0.0):.4f}")
        print(f"  hit_rate@{k}:   {agg.get(f'hit_rate@{k}', 0.0):.4f}")
        print(f"  precision@{k}:  {agg.get(f'precision@{k}', 0.0):.4f}")
        print(f"  map@{k}:        {agg.get(f'map@{k}', 0.0):.4f}")
        print(f"  ndcg@{k}:       {agg.get(f'ndcg@{k}', 0.0):.4f}")
    print(f"  mrr:            {agg.get('mrr', 0.0):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

