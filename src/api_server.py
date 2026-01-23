from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
from numpy.linalg import norm
import faiss
from transformers import pipeline
import io

app = FastAPI()

import os

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory="data"), name="images")


# Load pipeline and index with error handling
pipe = pipeline("image-feature-extraction", model="facebook/dinov3-vitb16-pretrain-lvd1689m")
try:
    index = faiss.read_index("image_index.faiss")
    image_paths = np.load("image_paths.npy")
    try:
        image_embeddings = np.load("image_embeddings.npy")
    except Exception:
        image_embeddings = np.empty((0, 768), dtype='float32')
except Exception as e:
    print(f"Warning: Could not load index or image paths: {e}")
    # Create IVF/PQ index setup and empty paths/embeddings
    d = 768
    nlist = 256
    m = 64
    nbits = 8
    quantizer = faiss.IndexFlatL2(d)
    ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
    ivfpq.nprobe = 8
    # Fallback flat index for searches before IVF/PQ is trained
    flat_index = faiss.IndexFlatL2(d)
    index = ivfpq
    image_paths = np.array([])
    image_embeddings = np.empty((0, d), dtype='float32')


@app.post("/search")
def search_image(file: UploadFile = File(...), top_k: int = 5):
    # Read image from upload
    image_bytes = file.file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Extract and normalize embedding
    vec = pipe(img)
    pooled_vec = np.mean(np.array(vec[0]), axis=0)
    pooled_vec = pooled_vec / (norm(pooled_vec) + 1e-10)
    test_vec = pooled_vec.astype('float32').reshape(1, -1)
    # Use IVF/PQ index when trained, otherwise fallback to flat index
    if hasattr(index, 'is_trained') and index.is_trained:
        D, I = index.search(test_vec, top_k)
    else:
        # Use flat_index if available, otherwise search on the (possibly untrained) index isn't allowed
        if 'flat_index' in globals():
            D, I = flat_index.search(test_vec, top_k)
        else:
            D, I = index.search(test_vec, top_k)
    # Prepare response
    results = []
    for idx, dist in zip(I[0], D[0]):
        img_name = str(image_paths[idx]).replace("\\", "/").split("/")[-1]
        img_url = f"/images/{img_name}"
        results.append({
            "image_url": img_url,
            "distance": float(dist)
        })
    return JSONResponse(content={"results": results})


# New route to add images to the vector store
@app.post("/add-image")
def add_image(file: UploadFile = File(...)):
    try:
        # Save uploaded image to data directory
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        img_bytes = file.file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        filename = file.filename
        save_path = os.path.join(data_dir, filename)
        # Avoid overwriting existing files
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(save_path):
            filename = f"{base}_{counter}{ext}"
            save_path = os.path.join(data_dir, filename)
            counter += 1
        img.save(save_path)

        # Extract embedding
        vec = pipe(img)
        pooled_vec = np.mean(np.array(vec[0]), axis=0)
        pooled_vec = pooled_vec / (norm(pooled_vec) + 1e-10)
        pooled_vec = pooled_vec.astype('float32').reshape(1, -1)

        # Persist embedding and path
        global image_paths, image_embeddings, index
        image_paths = np.append(image_paths, save_path)
        image_embeddings = np.append(image_embeddings, pooled_vec, axis=0)
        np.save("image_paths.npy", image_paths)
        np.save("image_embeddings.npy", image_embeddings)

        # If index is trained, add new vector; otherwise add to flat_index fallback
        if hasattr(index, 'is_trained') and index.is_trained:
            index.add(pooled_vec)
        else:
            # ensure a flat_index exists for immediate search
            if 'flat_index' not in globals():
                flat_index = faiss.IndexFlatL2(pooled_vec.shape[1])
            flat_index.add(pooled_vec)

        # If not trained yet, attempt to train once we have enough vectors
        try:
            if hasattr(index, 'is_trained') and not index.is_trained:
                # require at least nlist vectors to train; use nlist from the index
                nlist = index.nlist if hasattr(index, 'nlist') else 256
                if image_embeddings.shape[0] >= max(256, nlist):
                    index.train(image_embeddings)
                    index.add(image_embeddings)
                    # remove flat_index now that ivfpq is trained
                    if 'flat_index' in globals():
                        del flat_index
        except Exception as e:
            # training may fail with too few vectors or other issues; continue using flat_index
            print(f"Info: IVF/PQ training deferred or failed: {e}")

        # Save updated index (convert from GPU if needed)
        try:
            idx_to_save = faiss.index_gpu_to_cpu(index) if hasattr(faiss, 'index_gpu_to_cpu') and faiss.get_num_gpus() > 0 else index
            faiss.write_index(idx_to_save, "image_index.faiss")
        except Exception as e:
            print(f"Warning: failed to save index: {e}")

        return {"message": "✅ Image added successfully!", "image_path": save_path}
    except Exception as e:
        return {"message": f"❌ Failed to add image: {str(e)}"}
