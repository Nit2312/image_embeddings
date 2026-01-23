"""Quick test script for the worker endpoints"""
import requests
import base64
import json
from pathlib import Path

# UPDATE THIS with your worker URL
WORKER_URL = "https://image-vector-store-worker.patelnit2341.workers.dev/"  # CHANGE THIS!

def test_health():
    print("1. Testing Health Check...")
    r = requests.get(f"{WORKER_URL}/health")
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.json()}")
    print()

def test_create_index():
    print("2. Testing Create Index...")
    payload = {"dimensions": 768, "metric": "cosine"}
    r = requests.post(f"{WORKER_URL}/create-index", json=payload)
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.json()}")
    print()

def test_add_vector():
    print("3. Testing Add Vector...")
    # Use first image from data folder
    image_path = Path("data/0.webp")
    if not image_path.exists():
        print("   ❌ No image found in data/0.webp")
        return
    
    with open(image_path, 'rb') as f:
        img_data = f.read()
        base64_img = base64.b64encode(img_data).decode('utf-8')
    
    payload = {
        "image": f"data:image/webp;base64,{base64_img}",
        "id": "test-image-1",
        "metadata": {"filename": "0.webp", "test": True}
    }
    
    r = requests.post(f"{WORKER_URL}/add-vector", json=payload)
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.json()}")
    print()

def test_search():
    print("4. Testing Search...")
    # Use same image for search
    image_path = Path("data/0.webp")
    if not image_path.exists():
        print("   ❌ No image found")
        return
    
    with open(image_path, 'rb') as f:
        img_data = f.read()
        base64_img = base64.b64encode(img_data).decode('utf-8')
    
    payload = {
        "image": f"data:image/webp;base64,{base64_img}",
        "topK": 5
    }
    
    r = requests.post(f"{WORKER_URL}/search", json=payload)
    print(f"   Status: {r.status_code}")
    result = r.json()
    print(f"   Found {result.get('count', 0)} matches")
    if result.get('matches'):
        print(f"   Top match: {result['matches'][0]}")
    print()

def test_delete():
    print("5. Testing Delete Vector...")
    r = requests.delete(f"{WORKER_URL}/delete-vector/test-image-1")
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.json()}")
    print()

if __name__ == "__main__":
    if "your-worker-url" in WORKER_URL:
        print("⚠️  Please update WORKER_URL in this script first!")
        print(f"   Current: {WORKER_URL}")
    else:
        print("=" * 60)
        print(f"Testing Worker: {WORKER_URL}")
        print("=" * 60)
        print()
        
        test_health()
        test_create_index()
        test_add_vector()
        test_search()
        test_delete()
        
        print("=" * 60)
        print("Testing complete!")
