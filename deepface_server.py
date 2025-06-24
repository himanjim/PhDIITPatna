# deepface_server.py
# FastAPI server that accepts images, runs DeepFace embedding, and sends the embedding to a FAISS microservice

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import requests
from deepface import DeepFace
import os
import time

# ----------- Configuration -----------
FAISS_API_URL = "http://localhost:9000/search"  # FAISS microservice endpoint

# Enable TensorFlow GPU memory growth before any TF/DL imports
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import tensorflow as tf
print("[INFO] Available GPUs:", tf.config.list_physical_devices('GPU'))

# Initialize FastAPI
app = FastAPI()

# ✅ Preload and cache the model at server start
print("[INFO] Loading DeepFace model (Facenet512) once...")
DeepFace.build_model("Facenet512")
print("[INFO] DeepFace model loaded and cached.")

# Define the /verify endpoint to handle image-based face verification
@app.post("/verify")
async def verify_face(file: UploadFile = File(...)):
    try:
        # Read and decode uploaded image file into OpenCV format
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Time the embedding process
        embed_start = time.perf_counter()
        rep = DeepFace.represent(
            img_path=img,
            model_name="Facenet512",
            detector_backend ="retinaface",
            enforce_detection=False
        )
        embed_end = time.perf_counter()
        embedding_time_ms = (embed_end - embed_start) * 1000

        # Prepare vector to send to FAISS service
        embedding_vector = rep[0]["embedding"]

        # Time the round-trip FAISS call
        faiss_start = time.perf_counter()
        response = requests.post(FAISS_API_URL, json={"vector": embedding_vector})
        faiss_end = time.perf_counter()
        faiss_time_ms = (faiss_end - faiss_start) * 1000

        # Return combined result with detailed timing
        if response.status_code == 200:
            faiss_result = response.json()
            return {
                **faiss_result,
                "embedding_time_ms": round(embedding_time_ms, 3),
                "faiss_call_time_ms": round(faiss_time_ms, 3)
            }
        else:
            return JSONResponse(status_code=500, content={"error": response.text})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ----------- Entry Point for Standalone Run -----------
# import uvicorn
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

#gunicorn deepface_server:app -w 1 --threads 8 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
