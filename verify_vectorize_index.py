"""
Script to verify Cloudflare Vectorize index status and list vectors.
"""
import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_VECTORIZE_INDEX = os.getenv("CLOUDFLARE_VECTORIZE_INDEX")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")

def get_index_info():
    """Get information about the Vectorize index."""
    url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/vectorize/v2/indexes/{CLOUDFLARE_VECTORIZE_INDEX}"
    
    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        print(f"Index Info Request Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result.get('success', False)}")
            if result.get("success"):
                index_data = result.get("result", {})
                print(f"\nIndex Information:")
                print(f"  Name: {index_data.get('name')}")
                print(f"  Dimensions: {index_data.get('dimensions')}")
                print(f"  Metric: {index_data.get('metric')}")
                print(f"  Description: {index_data.get('description', 'N/A')}")
                print(f"  Created: {index_data.get('created_on', 'N/A')}")
                print(f"  Modified: {index_data.get('modified_on', 'N/A')}")
            else:
                print(f"Errors: {result.get('errors', [])}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()


def list_vectors(limit=10):
    """List vectors in the index."""
    url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/vectorize/v2/indexes/{CLOUDFLARE_VECTORIZE_INDEX}/list"
    
    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    params = {
        "limit": limit
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        print(f"\nList Vectors Request Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result.get('success', False)}")
            if result.get("success"):
                data = result.get("result", {})
                total_count = data.get("totalCount", 0)
                vectors = data.get("vectors", [])
                print(f"\nVectors in Index:")
                print(f"  Total Count: {total_count}")
                print(f"  Returned: {len(vectors)}")
                
                if vectors:
                    print(f"\n  Sample vectors:")
                    for i, vec in enumerate(vectors[:5], 1):
                        vec_id = vec.get("id", "N/A")
                        metadata = vec.get("metadata", {})
                        filename = metadata.get("filename", "N/A")
                        print(f"    {i}. ID: {vec_id[:20]}... | Filename: {filename}")
                else:
                    print("  ⚠ No vectors found in index!")
            else:
                print(f"Errors: {result.get('errors', [])}")
                print(f"Full response: {json.dumps(result, indent=2)}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if not all([CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_VECTORIZE_INDEX, CLOUDFLARE_API_TOKEN]):
        print("Missing required environment variables. Please check your .env file.")
    else:
        print("=" * 60)
        print("Cloudflare Vectorize Index Verification")
        print("=" * 60)
        get_index_info()
        list_vectors(limit=10)
        print("=" * 60)
