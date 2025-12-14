from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import faiss
import uvicorn
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from deepface import DeepFace
import cv2
import io

# Initialize FAISS index
DIMENSIONS = 512
NUM_EMBEDDINGS = 1000
faiss_index = faiss.IndexFlatL2(DIMENSIONS)
dummy_embeddings = np.random.rand(NUM_EMBEDDINGS, DIMENSIONS).astype('float32')
faiss_index.add(dummy_embeddings)

# Initialize FastAPI app
app = FastAPI()

@app.post("/verify")
async def verify_face(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Generate embedding using DeepFace
        rep = DeepFace.represent(
            img_path=img,
            model_name="Facenet512",
            enforce_detection=False
        )
        embedding_vector = np.array(rep[0]["embedding"], dtype='float32').reshape(1, -1)

        # Search in FAISS index
        distances, indices = faiss_index.search(embedding_vector, 1)
        return {
            "match_index": int(indices[0][0]),
            "distance": float(distances[0][0])
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
