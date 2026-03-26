# Image Embeddings — Vector-Based Image Search (Cloudflare Vectorize)

This project implements **visual similarity search** over a retail product catalog by:

- Converting images into **DINOv3** embeddings (768-dim, L2-normalized)
- Storing vectors in **Cloudflare Vectorize**
- Querying with a new image to return the **top-\(K\)** most similar catalog images

It includes:

- A **Cloudflare Worker** API for `add`, `search`, and `delete` over Vectorize
- A **Python embedding service** (FastAPI) used by the Worker to generate embeddings
- A **Python bulk uploader** to ingest a local dataset into Vectorize
- A **React (Vite) UI** to run searches and manage vectors

## Dataset

This implementation is built and tested with **Myntra’s fashion catalog dataset (~44k product images)**.

- **Local layout**: images live under `data/` (flat or nested folders)
- **Optional product grouping**: if you organize as `data/<product_id>/image.jpg`, the uploader stores `product_id` in Vectorize metadata

## Architecture (high level)

- **Bulk ingestion**: `data/` → DINOv3 embeddings (Python) → Vectorize upsert (Cloudflare API)
- **Query**: query image → embedding (Python FastAPI) → Vectorize query (Worker) → top matches (+ metadata)

## Repository structure

- `worker.ts`: Cloudflare Worker API (Vectorize-backed)
- `wrangler.toml`: Worker config and Vectorize binding
- `embedding_api.py`: FastAPI image→embedding service (DINOv3)
- `upload_vectors_to_vectorize.py`: bulk embed + upload from `data/` to Vectorize
- `verify_vectorize_index.py`: sanity check / index inspection
- `web/`: React + TypeScript + MUI frontend (visual search UI)

## Prerequisites

- **Cloudflare account** with Vectorize enabled
- **Node.js** (for Wrangler + web UI)
- **Python 3.10+** recommended (for embedding API + bulk uploader)

## Configuration

### 1) Local environment (`.env`)

Copy `.env.example` → `.env` and fill in:

- `DATA_FOLDER`: defaults to `data`
- `CLOUDFLARE_ACCOUNT_ID`
- `CLOUDFLARE_VECTORIZE_INDEX`
- `CLOUDFLARE_API_TOKEN`
- `EMBED_BATCH_SIZE` / `UPLOAD_BATCH_SIZE`: tune for throughput vs request size limits

### 2) Worker secrets (Wrangler)

Set secrets for the Worker (recommended instead of hardcoding):

```bash
wrangler secret put ACCOUNT_ID
wrangler secret put API_TOKEN
wrangler secret put VECTORIZE_INDEX
wrangler secret put EMBEDDING_API_URL
```

Notes:

- `EMBEDDING_API_URL` **must be publicly reachable** (the Worker cannot call `localhost`).
- `wrangler.toml` defines the Vectorize binding name `VECTORIZE` and an `index_name`. Keep it consistent with the `VECTORIZE_INDEX` secret you use in the Worker.

## Create a Vectorize index

Vectorize must match the embedding dimension and metric used by DINOv3:

- **Dimensions**: `768`
- **Metric**: `cosine`

Create an index with Wrangler:

```bash
npx wrangler vectorize create <index-name> --dimensions=768 --metric=cosine
```

## Bulk ingestion (Myntra ~44k images)

1) Place your dataset under `data/` (you can nest by product id):

```text
data/
  12345/
    img1.jpg
    img2.jpg
  67890/
    main.png
```

2) Install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3) Upload vectors to Vectorize:

```bash
python upload_vectors_to_vectorize.py
```

Optional tuning via env:

- `MAX_IMAGES`: ingest only first N images
- `START_AT`: resume ingestion from an offset
- `UPLOAD_BATCH_SIZE`: lower if you hit `413 Payload Too Large`

4) Verify index has vectors:

```bash
python verify_vectorize_index.py
```

## Embedding API (Python / FastAPI)

The Worker relies on this service to convert images into vectors at query time (and for `/add-vector` when uploading via the Worker).

Run locally:

```bash
python embedding_api.py
```

Endpoints:

- `POST /embed` (multipart): accepts `file`
- `POST /embed-binary` (octet-stream): accepts raw image bytes
- `GET /health`

Deploy it (Render/Railway/Fly/etc.) and set the deployed URL as `EMBEDDING_API_URL` in Worker secrets.

## Worker API (Vectorize)

Run locally:

```bash
npm install
npm run dev
```

Deploy:

```bash
npm run deploy
```

Available endpoints (see `worker.ts`):

- `GET /health`
- `POST /create-index`: create a Vectorize index via Cloudflare API
- `POST /add-vector`: upload an image, embed, and upsert into Vectorize
- `DELETE /delete-vector/:id`: delete a vector by id
- `POST /search`: embed a query image and return top matches (default top 20)

Example search request (JSON, base64 image):

```bash
curl -sS -X POST "$WORKER_URL/search" \
  -H "Content-Type: application/json" \
  -d '{"image":"data:image/jpeg;base64,<...>","topK":20}'
```

## Web UI (React / Vite)

1) Configure:

- Copy `web/.env.example` → `web/.env`
- Set `VITE_WORKER_URL` to your deployed (or local) Worker URL
- Optionally set `VITE_IMAGE_BASE_URL` if your catalog images are publicly accessible (the UI uses `relative_path` metadata to render thumbnails)

2) Run:

```bash
cd web
npm install
npm run dev
```

## Metadata stored with vectors

The bulk uploader stores useful metadata alongside each vector, including:

- `filename`
- `relative_path`
- `product_id` (when images are nested under `data/<product_id>/...`)
- `source` (e.g. `catalog`)

The Worker also stores metadata for vectors added via `/add-vector` (e.g. `uploaded_at`, `source`).

## Security & operational notes

- **Do not commit secrets**: keep `.env` local; use `wrangler secret put` for Worker secrets.
- **Request sizing**: Vectorize upserts are JSON; large batches can trigger `413` — tune `UPLOAD_BATCH_SIZE`.
- **Deterministic IDs**: bulk ingestion uses a stable id derived from each image’s `relative_path`, so re-running ingestion is idempotent.

## License

MIT (see `package.json`).
