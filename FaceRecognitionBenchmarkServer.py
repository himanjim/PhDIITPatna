# face_server.py

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import faiss
import uvicorn

# Configuration for embedding dimensions and count
DIMENSIONS = 512
NUM_EMBEDDINGS = 100

# Initialize FAISS index with dummy embeddings
faiss_index = faiss.IndexFlatL2(DIMENSIONS)
dummy_embeddings = np.random.rand(NUM_EMBEDDINGS, DIMENSIONS).astype('float32')
faiss_index.add(dummy_embeddings)

# Initialize FastAPI server
app = FastAPI()

# Define the input model for face embeddings
class FaceEmbedding(BaseModel):
    face_embedding: list

# API endpoint to verify face embedding against FAISS index
@app.post("/verify")
async def verify_face(data: FaceEmbedding):
    try:
        embedding = np.array(data.face_embedding, dtype='float32').reshape(1, -1)
        distances, indices = faiss_index.search(embedding, 1)
        return {"match_index": int(indices[0][0]), "distance": float(distances[0][0])}
    except Exception as e:
        return {"error": str(e)}

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
