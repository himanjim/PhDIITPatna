# This module implements a minimal face-verification microservice that
# exposes a single HTTP endpoint for receiving an uploaded image,
# generating a facial embedding using DeepFace, and performing a nearest-
# neighbour lookup in a FAISS index. The service is intended primarily for
# functional validation and latency benchmarking of an embedding-plus-
# search pipeline rather than for production-scale identity verification.
# A synthetic FAISS index is initialised at startup to ensure that search
# behaviour can be exercised even when no real enrolment database is
# available. The returned response contains only the closest match index
# and its distance score so that external benchmarking clients can measure
# end-to-end behaviour without exposing unnecessary internal state.
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

# Handle a single face-verification request by decoding the uploaded image,
# extracting a fixed-length embedding vector, and querying the FAISS index
# for the nearest stored representation. The function performs minimal
# preprocessing and does not enforce strict face detection constraints so
# that benchmarking runs remain robust to imperfect inputs. The response
# provides the index position of the closest candidate together with the
# associated distance value, enabling downstream systems or evaluation
# scripts to interpret similarity outcomes.
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
