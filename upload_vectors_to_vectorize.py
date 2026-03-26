"""
Script to convert images to vectors using DINOv3 model and upload to Cloudflare Vectorize.

Data source: put images under the `data/` folder (flat or nested subfolders). That folder is the only source.

Prerequisites:
1. Create a Cloudflare Vectorize index with 768 dimensions and cosine metric:
   npx wrangler vectorize create <index-name> --dimensions=768 --metric=cosine
2. Set the following in your .env file:
   - CLOUDFLARE_ACCOUNT_ID
   - CLOUDFLARE_VECTORIZE_INDEX
   - CLOUDFLARE_API_TOKEN
"""
import os
import json
import hashlib
import math
from pathlib import Path
from PIL import Image
import numpy as np
from numpy.linalg import norm
from transformers import pipeline
from dotenv import load_dotenv
import requests
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Load environment variables from .env file
load_dotenv()

# Cloudflare Vectorize Configuration from .env
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_VECTORIZE_INDEX = os.getenv("CLOUDFLARE_VECTORIZE_INDEX")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")

# Image processing configuration
DATA_FOLDER = os.getenv("DATA_FOLDER", "data")
DINO_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
VECTOR_DIM = 768  # DINOv3 ViT-B/16 produces 768-dimensional embeddings
UPLOAD_BATCH_SIZE = int(os.getenv("UPLOAD_BATCH_SIZE", "5"))  # vectors per API call (keep small to avoid 413)
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "8"))  # model inference batch size
MAX_IMAGES = os.getenv("MAX_IMAGES")
START_AT = int(os.getenv("START_AT", "0"))

# Image extensions to process
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}


def is_image_file(filename):
    """Check if file is an image based on extension."""
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)


def is_thumbnail(key, filename):
    """Check if file is a thumbnail based on path or filename."""
    key_lower = key.lower()
    filename_lower = filename.lower()
    
    thumbnail_indicators = [
        'thumbnail', 'thumbnails', 'thumb',
        '_thumb', 'thumb_', 'tn_', '_tn',
    ]
    
    for indicator in thumbnail_indicators:
        if indicator in key_lower or indicator in filename_lower:
            return True
    
    if '/thumbnail' in key_lower or '/thumbnails' in key_lower:
        return True
    
    return False


def _stable_id_from_string(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]


def _product_id_from_relative_path(relative_posix: str) -> Optional[str]:
    """If images live under data/<segment>/..., use first segment as product_id."""
    parts = [p for p in relative_posix.split("/") if p]
    if len(parts) >= 2:
        return parts[0]
    return None


def _iter_local_images(data_folder: Path) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    """
    Walk `data/` recursively. Vector id is stable per relative path.
    """
    for file_path in sorted(data_folder.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.name == "manifest.jsonl":
            continue

        filename = file_path.name
        if not is_image_file(filename):
            continue
        rel = file_path.relative_to(data_folder)
        rel_posix = rel.as_posix()
        if is_thumbnail(rel_posix, filename):
            continue

        vector_id = _stable_id_from_string(rel_posix)
        product_id = _product_id_from_relative_path(rel_posix)
        metadata: Dict[str, Any] = {
            "filename": filename,
            "local_filename": filename,
            "relative_path": rel_posix,
            "source": "catalog",
        }
        if product_id is not None:
            metadata["product_id"] = product_id

        yield file_path, {"id": vector_id, "metadata": metadata}


def _extract_embeddings(pipe, images: List[Image.Image]) -> List[List[float]]:
    """
    Extract normalized embeddings for a batch of images.
    Returns list of python lists (JSON serializable).
    """
    vecs = pipe(images, batch_size=EMBED_BATCH_SIZE)
    out: List[List[float]] = []
    for v in vecs:
        arr = np.array(v)
        # Some pipelines return shape (1, seq_len, dim) per image; normalize to (seq_len, dim)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        pooled = np.mean(arr, axis=0)
        pooled = pooled / (norm(pooled) + 1e-10)
        out.append(pooled.astype("float32").tolist())
    return out


def _vectorize_upsert_url() -> str:
    # Validate environment variables
    if not all([CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_VECTORIZE_INDEX, CLOUDFLARE_API_TOKEN]):
        raise ValueError("Missing required Cloudflare Vectorize environment variables. Please check your .env file.")
    return f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/vectorize/v2/indexes/{CLOUDFLARE_VECTORIZE_INDEX}/upsert"


def _vectorize_headers() -> Dict[str, str]:
    if not CLOUDFLARE_API_TOKEN:
        raise ValueError("Missing CLOUDFLARE_API_TOKEN")
    return {"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}", "Content-Type": "application/json"}


def embed_and_upload_from_folder(data_folder: str) -> None:
    """
    Stream embeddings from local `data/` and upload in batches.
    This avoids holding all embeddings in memory (important for large datasets).
    """
    folder_path = Path(data_folder)
    if not folder_path.exists():
        raise ValueError(f"Data folder '{data_folder}' does not exist. Add images under {data_folder}/ first.")

    max_images: Optional[int] = None
    if MAX_IMAGES:
        try:
            max_images = int(MAX_IMAGES)
        except ValueError:
            raise ValueError("MAX_IMAGES must be an integer if set")

    url = _vectorize_upsert_url()
    headers = _vectorize_headers()

    print("Loading DINOv3 model...")
    pipe = pipeline("image-feature-extraction", model=DINO_MODEL)

    uploaded_count = 0
    failed_count = 0
    seen = 0

    upload_vectors: List[Dict[str, Any]] = []
    embed_imgs: List[Image.Image] = []
    embed_infos: List[Dict[str, Any]] = []

    def flush_embed_to_upload():
        nonlocal embed_imgs, embed_infos, upload_vectors
        if not embed_imgs:
            return
        embeddings = _extract_embeddings(pipe, embed_imgs)
        for info, emb in zip(embed_infos, embeddings):
            if not emb or any((not isinstance(x, (int, float)) or not math.isfinite(float(x))) for x in emb):
                # Skip vectors with NaN/Inf (Cloudflare rejects non-JSON numbers)
                # Count as failed so totals still reflect reality.
                nonlocal failed_count
                failed_count += 1
                continue
            upload_vectors.append({"id": info["id"], "values": emb, "metadata": info["metadata"]})
        embed_imgs = []
        embed_infos = []

    def flush_upload():
        nonlocal upload_vectors, uploaded_count, failed_count
        if not upload_vectors:
            return
        payload = {"vectors": upload_vectors}
        try:
            body = json.dumps(payload, allow_nan=False)
            response = requests.post(url, headers=headers, data=body)
            if response.status_code == 200:
                result = response.json()
                if result.get("success", False):
                    uploaded_count += len(upload_vectors)
                    print(f"Uploaded {uploaded_count} vectors...")
                else:
                    failed_count += len(upload_vectors)
                    print(f"Vectorize error: {result.get('errors')}")
            else:
                failed_count += len(upload_vectors)
                print(f"HTTP {response.status_code} error uploading batch: {response.text[:500]}")
                if response.status_code == 413:
                    print(
                        "Tip: reduce UPLOAD_BATCH_SIZE (e.g. 1-3). "
                        "Each vector is 768 floats; JSON payloads can exceed Cloudflare limits quickly."
                    )
                if response.status_code == 400:
                    # Print a tiny sample to help debug schema issues without dumping massive payloads
                    sample = upload_vectors[0]
                    print(f"Sample vector id={sample.get('id')} values_len={len(sample.get('values', []))} metadata_keys={list((sample.get('metadata') or {}).keys())}")
        except ValueError as e:
            # allow_nan=False will throw if any NaN/Inf sneaks in
            failed_count += len(upload_vectors)
            print(f"JSON serialization error (likely NaN/Inf in values): {e}")
        except Exception as e:
            failed_count += len(upload_vectors)
            print(f"Exception uploading batch: {e}")
        finally:
            upload_vectors = []
            time.sleep(0.1)

    for idx, (path, info) in enumerate(_iter_local_images(folder_path)):
        if idx < START_AT:
            continue
        if max_images is not None and seen >= max_images:
            break

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

        embed_imgs.append(img)
        embed_infos.append(info)
        seen += 1

        if seen % 200 == 0:
            print(f"Prepared {seen} images...")

        if len(embed_imgs) >= EMBED_BATCH_SIZE:
            flush_embed_to_upload()

        if len(upload_vectors) >= UPLOAD_BATCH_SIZE:
            flush_upload()

    flush_embed_to_upload()
    flush_upload()

    print("-" * 50)
    print("Upload complete!")
    print(f"Successfully uploaded: {uploaded_count} vectors")
    print(f"Failed: {failed_count} vectors")
    print(f"Index: {CLOUDFLARE_VECTORIZE_INDEX}")


def verify_vectors_in_index():
    """Verify that vectors exist in the Vectorize index."""
    try:
        # List vectors endpoint
        url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/vectorize/v2/indexes/{CLOUDFLARE_VECTORIZE_INDEX}/list"
        
        headers = {
            "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        params = {
            "limit": 10  # Just check first 10
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success", False):
                data = result.get("result", {})
                total_count = data.get("totalCount", 0)
                vectors = data.get("vectors", [])
                print(f"✓ Index contains {total_count} vectors")
                if vectors:
                    print(f"  Sample vector IDs: {[v.get('id') for v in vectors[:5]]}")
                else:
                    print("  ⚠ Warning: No vectors found in index!")
                    print(f"  Full response: {json.dumps(result, indent=2)[:1000]}")
            else:
                errors = result.get("errors", [])
                print(f"✗ Error verifying index: {errors}")
        else:
            print(f"✗ HTTP {response.status_code} error verifying index: {response.text}")
            
    except Exception as e:
        print(f"✗ Exception verifying index: {e}")


def main():
    """Main function to process images and upload to Vectorize."""
    try:
        print(f"Reading images from {DATA_FOLDER}/ ...")
        print(f"Embedding batch size: {EMBED_BATCH_SIZE} | Upload batch size: {UPLOAD_BATCH_SIZE}")
        if MAX_IMAGES:
            print(f"MAX_IMAGES={MAX_IMAGES}")
        if START_AT:
            print(f"START_AT={START_AT}")

        embed_and_upload_from_folder(DATA_FOLDER)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
