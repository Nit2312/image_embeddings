"""
Quick start script to upload images to vector store via the worker.
Update WORKER_URL before running.
"""
import os
import requests
import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Update this with your worker URL
WORKER_URL = os.getenv("WORKER_URL", "https://your-worker.your-subdomain.workers.dev")
DATA_FOLDER = "data"
MAX_IMAGES = 100  # Limit for testing, set to None for all images


def upload_image(image_path, worker_url):
    """Upload a single image to vector store via worker."""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Determine MIME type
        ext = image_path.suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.gif': 'image/gif',
        }
        mime_type = mime_types.get(ext, 'image/jpeg')
        
        # Use filename (without extension) as ID
        vector_id = image_path.stem
        
        payload = {
            "image": f"data:{mime_type};base64,{base64_image}",
            "id": vector_id,
            "metadata": {
                "filename": image_path.name,
                "path": str(image_path.relative_to(Path(DATA_FOLDER))),
                "source": "bulk_upload"
            }
        }
        
        response = requests.post(f"{worker_url}/add-vector", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                return True, result
            else:
                return False, result.get("error", "Unknown error")
        else:
            return False, f"HTTP {response.status_code}: {response.text}"
            
    except Exception as e:
        return False, str(e)


def main():
    """Upload images to vector store."""
    if not WORKER_URL or "your-worker" in WORKER_URL:
        print("⚠️  Please set WORKER_URL in .env file or update this script")
        print(f"   Current URL: {WORKER_URL}")
        return
    
    print("=" * 60)
    print("Quick Start - Upload Images to Vector Store")
    print("=" * 60)
    print(f"Worker URL: {WORKER_URL}")
    print(f"Data folder: {DATA_FOLDER}")
    print("-" * 60)
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
        image_files.extend(Path(DATA_FOLDER).glob(f"*{ext}"))
    
    if not image_files:
        print(f"❌ No images found in {DATA_FOLDER} folder")
        return
    
    # Limit if specified
    if MAX_IMAGES:
        image_files = image_files[:MAX_IMAGES]
    
    print(f"Found {len(image_files)} images to upload")
    print("-" * 60)
    
    # Test connection first
    try:
        health_response = requests.get(f"{WORKER_URL}/health", timeout=10)
        if health_response.status_code == 200:
            print("✓ Worker is accessible")
        else:
            print(f"⚠️  Worker health check returned: {health_response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to worker: {e}")
        print("   Please check your WORKER_URL and ensure the worker is deployed")
        return
    
    print("-" * 60)
    
    # Upload images
    uploaded = 0
    failed = 0
    
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Uploading {image_path.name}...", end=" ")
        
        success, result = upload_image(image_path, WORKER_URL)
        
        if success:
            uploaded += 1
            print("✓")
        else:
            failed += 1
            print(f"✗ Error: {result}")
    
    print("-" * 60)
    print(f"Upload complete!")
    print(f"  ✓ Uploaded: {uploaded}")
    print(f"  ✗ Failed: {failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
