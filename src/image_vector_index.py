import os
from PIL import Image
import numpy as np
from numpy.linalg import norm
from transformers import pipeline
import faiss

# 1. Load the image feature extraction pipeline
pipe = pipeline("image-feature-extraction", model="facebook/dinov3-vitb16-pretrain-lvd1689m")

def load_images_from_folder(folder):
    images = []
    paths = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder, filename)
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                paths.append(path)
            except Exception as e:
                print(f"Error loading {path}: {e}")
    return images, paths


# Check if index and paths exist
index_path = "image_index.faiss"
paths_path = "image_paths.npy"
if os.path.exists(index_path) and os.path.exists(paths_path):
    print("Loading existing vector store...")
    index = faiss.read_index(index_path)
    image_paths = np.load(paths_path)
    print(f"Loaded index with {len(image_paths)} images.")
else:
    # 2. Extract features for all images in the data folder
    data_folder = "data"
    images, image_paths = load_images_from_folder(data_folder)
    features = []
    for img in images:
        vec = pipe(img)
        pooled_vec = np.mean(np.array(vec[0]), axis=0)
        # Normalize the embedding to unit length
        pooled_vec = pooled_vec / (norm(pooled_vec) + 1e-10)
        features.append(pooled_vec)
    features = np.stack(features).astype('float32')

    # 3. Build and save a Faiss IVF+PQ index (fall back to IndexFlatL2 if too few vectors)
    dim = features.shape[1]
    nlist = 256
    m = 64
    nbits = 8
    quantizer = faiss.IndexFlatL2(dim)
    ivfpq = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
    ivfpq.nprobe = 8

    # Save raw embeddings and paths for later training / reindexing
    np.save('image_embeddings.npy', features)
    np.save(paths_path, np.array(image_paths))

    # If we have enough vectors, train the IVF/PQ index and add features
    try:
        if features.shape[0] >= max(256, nlist):
            ivfpq.train(features)
            ivfpq.add(features)
            index = ivfpq
            # Move to GPU if available
            if faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                print("Trained IVFPQ and using GPU for Faiss index.")
            else:
                print("Trained IVFPQ and using CPU for Faiss index.")
            faiss.write_index(faiss.index_gpu_to_cpu(index) if faiss.get_num_gpus() > 0 else index, index_path)
            print(f"Indexed {len(image_paths)} images with IVF+PQ.")
        else:
            # Not enough vectors to train IVFPQ - use a flat index for now
            flat_index = faiss.IndexFlatL2(dim)
            flat_index.add(features)
            index = flat_index
            print(f"Not enough vectors to train IVF+PQ ({features.shape[0]} < {nlist}), using IndexFlatL2.")
            faiss.write_index(index, index_path)
    except Exception as e:
        print(f"Failed to build IVF/PQ index, falling back to flat index: {e}")
        flat_index = faiss.IndexFlatL2(dim)
        flat_index.add(features)
        index = flat_index
        faiss.write_index(index, index_path)

# 4. Example: Query with a test image
import time
def query_image(test_image_path, top_k=5):
    start_time = time.time()
    test_img = Image.open(test_image_path).convert("RGB")
    vec = pipe(test_img)
    test_vec = np.mean(np.array(vec[0]), axis=0)
    # Normalize the test embedding
    test_vec = test_vec / (norm(test_vec) + 1e-10)
    test_vec = test_vec.astype('float32').reshape(1, -1)
    # Use trained IVF/PQ when available, otherwise fallback to flat index
    if hasattr(index, 'is_trained') and getattr(index, 'is_trained'):
        D, I = index.search(test_vec, top_k)
    else:
        if 'flat_index' in globals():
            D, I = flat_index.search(test_vec, top_k)
        else:
            # Fall back to brute-force over saved embeddings if available
            if os.path.exists('image_embeddings.npy'):
                emb = np.load('image_embeddings.npy')
                # compute L2 distances
                dists = np.linalg.norm(emb - test_vec, axis=1)
                idxs = np.argsort(dists)[:top_k]
                I = idxs.reshape(1, -1)
                D = dists[idxs].reshape(1, -1)
            else:
                D, I = index.search(test_vec, top_k)
    elapsed = time.time() - start_time
    print("Top matches:")
    for idx, dist in zip(I[0], D[0]):
        print(f"{image_paths[idx]} (distance: {dist:.4f})")
    print(f"Query time: {elapsed:.3f} seconds")

# Example usage:

# Utility: Export all vectors from FAISS index to .npy for visualization
def export_faiss_to_npy(index, output_path):
    n = index.ntotal
    dim = index.d
    vectors = np.zeros((n, dim), dtype='float32')
    for i in range(n):
        vectors[i] = index.reconstruct(i)
    np.save(output_path, vectors)
    print(f"Exported {n} vectors to {output_path}")

if __name__ == "__main__":
    # Example usage:
    query_image('car1.jpg')
    
