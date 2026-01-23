"""
Script to convert images to vectors using DINOv3 model and upload to Cloudflare Vectorize.

Prerequisites:
1. Ensure images are downloaded to the 'data' folder using download_images_from_r2.py
2. Create a Cloudflare Vectorize index with 768 dimensions and cosine metric:
   npx wrangler vectorize create <index-name> --dimensions=768 --metric=cosine
3. Set the following in your .env file:
   - CLOUDFLARE_ACCOUNT_ID
   - CLOUDFLARE_VECTORIZE_INDEX
   - CLOUDFLARE_API_TOKEN
"""
import os
import json
import hashlib
from pathlib import Path
from PIL import Image
import numpy as np
from numpy.linalg import norm
from transformers import pipeline
from dotenv import load_dotenv
import requests
import time

# Load environment variables from .env file
load_dotenv()

# Cloudflare Vectorize Configuration from .env
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_VECTORIZE_INDEX = os.getenv("CLOUDFLARE_VECTORIZE_INDEX")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")

# Image processing configuration
DATA_FOLDER = "data"
DINO_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
VECTOR_DIM = 768  # DINOv3 ViT-B/16 produces 768-dimensional embeddings
BATCH_SIZE = 50  # Number of vectors to upload per API call

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


def generate_vector_id(image_path):
    """Generate a unique ID for the vector based on image path."""
    # Use hash of the full path to ensure uniqueness
    path_str = str(image_path.resolve())
    return hashlib.sha256(path_str.encode()).hexdigest()[:32]


def load_images_from_folder(folder):
    """Load all images from the specified folder."""
    images = []
    paths = []
    folder_path = Path(folder)
    
    if not folder_path.exists():
        raise ValueError(f"Data folder '{folder}' does not exist. Please download images first.")
    
    for file_path in folder_path.iterdir():
        if not file_path.is_file():
            continue
        
        filename = file_path.name
        key = str(file_path)
        
        # Skip if not an image file
        if not is_image_file(filename):
            continue
        
        # Skip thumbnails
        if is_thumbnail(key, filename):
            continue
        
        try:
            img = Image.open(file_path).convert("RGB")
            images.append(img)
            paths.append(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    return images, paths


def extract_embeddings(images, paths):
    """Extract embeddings from images using DINOv3 model."""
    print("Loading DINOv3 model...")
    pipe = pipeline("image-feature-extraction", model=DINO_MODEL)
    
    print(f"Extracting embeddings from {len(images)} images...")
    embeddings = []
    vector_ids = []
    metadata_list = []
    
    for i, (img, path) in enumerate(zip(images, paths)):
        try:
            # Extract features using the pipeline
            vec = pipe(img)
            # Pool the features (mean pooling)
            pooled_vec = np.mean(np.array(vec[0]), axis=0)
            # Normalize the embedding to unit length
            pooled_vec = pooled_vec / (norm(pooled_vec) + 1e-10)
            
            # Convert to list for JSON serialization
            embedding = pooled_vec.tolist()
            
            # Generate unique ID
            vector_id = generate_vector_id(path)
            
            # Prepare metadata
            metadata = {
                "filename": path.name,
                "path": str(path),
                "relative_path": str(path.relative_to(Path(DATA_FOLDER)))
            }
            
            embeddings.append(embedding)
            vector_ids.append(vector_id)
            metadata_list.append(metadata)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(images)} images...")
                
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    
    print(f"Successfully extracted {len(embeddings)} embeddings")
    return vector_ids, embeddings, metadata_list


def upload_vectors_to_vectorize(vector_ids, embeddings, metadata_list):
    """Upload vectors to Cloudflare Vectorize."""
    # Validate environment variables
    if not all([CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_VECTORIZE_INDEX, CLOUDFLARE_API_TOKEN]):
        raise ValueError("Missing required Cloudflare Vectorize environment variables. Please check your .env file.")
    
    # Cloudflare Vectorize API endpoint (v2 API)
    # Using upsert instead of insert to handle existing vectors
    url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/vectorize/v2/indexes/{CLOUDFLARE_VECTORIZE_INDEX}/upsert"
    
    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    total_vectors = len(vector_ids)
    uploaded_count = 0
    failed_count = 0
    
    print(f"Uploading {total_vectors} vectors to Cloudflare Vectorize...")
    print(f"Index: {CLOUDFLARE_VECTORIZE_INDEX}")
    print("-" * 50)
    
    # Upload in batches
    for i in range(0, total_vectors, BATCH_SIZE):
        batch_ids = vector_ids[i:i + BATCH_SIZE]
        batch_embeddings = embeddings[i:i + BATCH_SIZE]
        batch_metadata = metadata_list[i:i + BATCH_SIZE]
        
        # Prepare vectors for this batch
        vectors = []
        for vec_id, embedding, metadata in zip(batch_ids, batch_embeddings, batch_metadata):
            vector = {
                "id": vec_id,
                "values": embedding,
                "metadata": metadata
            }
            vectors.append(vector)
        
        # Prepare request body
        payload = {
            "vectors": vectors
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            
            # Print response for debugging (first batch only)
            if i == 0:
                print(f"API Response Status: {response.status_code}")
                print(f"API Response: {response.text[:500]}...")  # First 500 chars
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success", False):
                    uploaded_count += len(vectors)
                    # Print result details for first batch
                    if i == 0:
                        print(f"Success response: {json.dumps(result, indent=2)[:500]}")
                    print(f"[{uploaded_count}/{total_vectors}] Uploaded batch of {len(vectors)} vectors")
                else:
                    errors = result.get("errors", [])
                    print(f"Error uploading batch: {errors}")
                    print(f"Full response: {json.dumps(result, indent=2)}")
                    failed_count += len(vectors)
            else:
                print(f"HTTP {response.status_code} error uploading batch: {response.text}")
                failed_count += len(vectors)
            
            # Rate limiting: add a small delay between batches
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Exception uploading batch: {e}")
            import traceback
            traceback.print_exc()
            failed_count += len(vectors)
            continue
    
    print("-" * 50)
    print(f"Upload complete!")
    print(f"Successfully uploaded: {uploaded_count} vectors")
    print(f"Failed: {failed_count} vectors")
    print(f"Index: {CLOUDFLARE_VECTORIZE_INDEX}")
    
    # Verify vectors were inserted
    if uploaded_count > 0:
        print("\nVerifying vectors in index...")
        verify_vectors_in_index()


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
        # Load images
        print("Loading images from data folder...")
        images, paths = load_images_from_folder(DATA_FOLDER)
        
        if not images:
            print("No images found in data folder. Please download images first.")
            return
        
        print(f"Found {len(images)} images to process")
        print("-" * 50)
        
        # Extract embeddings
        vector_ids, embeddings, metadata_list = extract_embeddings(images, paths)
        
        if not embeddings:
            print("No embeddings extracted. Exiting.")
            return
        
        # Verify embedding dimensions
        if len(embeddings[0]) != VECTOR_DIM:
            print(f"Warning: Embedding dimension is {len(embeddings[0])}, expected {VECTOR_DIM}")
            print("Please ensure your Vectorize index is created with the correct dimensions.")
        
        # Upload to Cloudflare Vectorize
        upload_vectors_to_vectorize(vector_ids, embeddings, metadata_list)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
