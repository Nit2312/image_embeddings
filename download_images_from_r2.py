"""
Script to download 1000 images from Cloudflare R2 storage.
Images are downloaded to the data folder.
"""
import os
import boto3
from botocore.config import Config
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# R2 Configuration from .env
R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET = os.getenv("R2_BUCKET")
R2_PREFIX = os.getenv("R2_PREFIX", "")
 
# Create data folder if it doesn't exist
DATA_FOLDER = "data"
Path(DATA_FOLDER).mkdir(exist_ok=True)

# Number of images to download
MAX_IMAGES = 1000

# Image extensions to filter
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}


def is_image_file(filename):
    """Check if file is an image based on extension."""
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)


def is_thumbnail(key, filename):
    """Check if file is a thumbnail based on path or filename."""
    key_lower = key.lower()
    filename_lower = filename.lower()
    
    # Check for thumbnail in path or filename
    thumbnail_indicators = [
        'thumbnail',
        'thumbnails',
        'thumb',
        '_thumb',
        'thumb_',
        'tn_',
        '_tn',
    ]
    
    # Check if any thumbnail indicator is in the key path or filename
    for indicator in thumbnail_indicators:
        if indicator in key_lower or indicator in filename_lower:
            return True
    
    # Check if file is in a thumbnail directory
    if '/thumbnail' in key_lower or '/thumbnails' in key_lower:
        return True
    
    return False


def download_images_from_r2():
    """Download images from R2 storage."""
    # Validate environment variables
    if not all([R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET]):
        raise ValueError("Missing required R2 environment variables. Please check your .env file.")
    
    # Configure boto3 for R2 (S3-compatible)
    # R2 requires region_name='auto' instead of AWS region names
    s3_client = boto3.client(
        's3',
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name='auto',  # R2 uses 'auto' or specific R2 regions (wnam, enam, weur, eeur, apac, oc)
        config=Config(signature_version='s3v4')
    )
    
    print(f"Connecting to R2 bucket: {R2_BUCKET}")
    print(f"Prefix: {R2_PREFIX}")
    print(f"Downloading up to {MAX_IMAGES} images...")
    print("-" * 50)
    
    # List objects in the bucket with the specified prefix
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=R2_BUCKET, Prefix=R2_PREFIX)
    
    downloaded_count = 0
    skipped_count = 0
    duplicate_count = 0
    thumbnail_count = 0
    seen_r2_keys = set()  # Track R2 keys to avoid downloading duplicates
    filename_counter = {}  # Track filename usage to handle collisions
    
    def get_unique_filename(base_filename):
        """Generate a unique filename by appending a counter if needed."""
        if base_filename not in filename_counter:
            filename_counter[base_filename] = 0
            return base_filename
        
        # File with this name already exists, create unique name
        filename_counter[base_filename] += 1
        name, ext = os.path.splitext(base_filename)
        return f"{name}_{filename_counter[base_filename]}{ext}"
    
    for page in page_iterator:
        if 'Contents' not in page:
            continue
        
        for obj in page['Contents']:
            if downloaded_count >= MAX_IMAGES:
                break
            
            key = obj['Key']
            filename = os.path.basename(key)
            
            # Skip if not an image file
            if not is_image_file(filename):
                skipped_count += 1
                continue
            
            # Skip directories
            if key.endswith('/'):
                continue
            
            # Skip thumbnails
            if is_thumbnail(key, filename):
                thumbnail_count += 1
                skipped_count += 1
                continue
            
            # Check for duplicate R2 keys (same file in R2)
            if key in seen_r2_keys:
                print(f"[{downloaded_count + 1}/{MAX_IMAGES}] Skipping duplicate R2 key: {key}")
                duplicate_count += 1
                skipped_count += 1
                continue
            
            seen_r2_keys.add(key)
            
            # Get unique local filename (handles filename collisions)
            unique_filename = get_unique_filename(filename)
            local_path = os.path.join(DATA_FOLDER, unique_filename)
            
            # If the unique filename already exists locally, skip it
            if os.path.exists(local_path):
                print(f"[{downloaded_count + 1}/{MAX_IMAGES}] Skipping (exists locally): {unique_filename}")
                skipped_count += 1
                continue
            
            try:
                # Download the file
                s3_client.download_file(R2_BUCKET, key, local_path)
                downloaded_count += 1
                if unique_filename != filename:
                    print(f"[{downloaded_count}/{MAX_IMAGES}] Downloaded: {filename} -> {unique_filename} (renamed to avoid collision)")
                else:
                    print(f"[{downloaded_count}/{MAX_IMAGES}] Downloaded: {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                skipped_count += 1
                # Remove from seen set if download failed
                seen_r2_keys.discard(key)
                continue
        
        if downloaded_count >= MAX_IMAGES:
            break
    
    print("-" * 50)
    print(f"Download complete!")
    print(f"Downloaded: {downloaded_count} images")
    print(f"Skipped: {skipped_count} files")
    print(f"Thumbnails skipped: {thumbnail_count}")
    print(f"Duplicate R2 keys skipped: {duplicate_count}")
    print(f"Filename collisions handled: {sum(1 for count in filename_counter.values() if count > 0)}")
    print(f"Images saved to: {os.path.abspath(DATA_FOLDER)}")


if __name__ == "__main__":
    try:
        download_images_from_r2()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
