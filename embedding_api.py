"""
Simple FastAPI service to convert images to vectors using DINOv3.
This service can be deployed separately and called by the Cloudflare Worker.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
from numpy.linalg import norm
from transformers import pipeline
import io

app = FastAPI(title="Image Embedding API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load DINOv3 model
print("Loading DINOv3 model...")
pipe = pipeline("image-feature-extraction", model="facebook/dinov3-vitb16-pretrain-lvd1689m")
print("Model loaded successfully!")


@app.post("/embed")
async def embed_image(file: UploadFile = File(...)):
    """
    Convert an image to a vector embedding using DINOv3.
    
    Accepts: multipart/form-data with image file
    Returns: JSON with embedding vector
    """
    try:
        # Read image file
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Extract features using DINOv3
        vec = pipe(img)
        
        # Pool the features (mean pooling)
        pooled_vec = np.mean(np.array(vec[0]), axis=0)
        
        # Normalize the embedding to unit length
        pooled_vec = pooled_vec / (norm(pooled_vec) + 1e-10)
        
        # Convert to list for JSON serialization
        embedding = pooled_vec.tolist()
        
        return JSONResponse(content={
            "embedding": embedding,
            "vector": embedding,  # Alias for compatibility
            "values": embedding,  # Alias for compatibility
            "dimension": len(embedding),
            "filename": file.filename
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/embed-binary")
async def embed_image_binary(request: bytes):
    """
    Convert an image to a vector embedding from raw binary data.
    
    Accepts: application/octet-stream (raw image bytes)
    Returns: JSON with embedding vector
    """
    try:
        # Read image from bytes
        img = Image.open(io.BytesIO(request)).convert("RGB")
        
        # Extract features using DINOv3
        vec = pipe(img)
        
        # Pool the features (mean pooling)
        pooled_vec = np.mean(np.array(vec[0]), axis=0)
        
        # Normalize the embedding to unit length
        pooled_vec = pooled_vec / (norm(pooled_vec) + 1e-10)
        
        # Convert to list for JSON serialization
        embedding = pooled_vec.tolist()
        
        return JSONResponse(content={
            "embedding": embedding,
            "vector": embedding,
            "values": embedding,
            "dimension": len(embedding)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "embedding-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
