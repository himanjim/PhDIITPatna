# faiss_microservice.py
# A standalone FAISS REST service to handle vector search centrally,
# with live index updating upon each query and separate timing for search and update steps.

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import faiss
import os
import time
import threading


# ----------- Configuration -----------
INDEX_FILE = "voter_index_flatl2.faiss"  # Path to FAISS index file
EMBEDDING_DIM = 512                     # Dimension of face embeddings
IO_FLAGS = faiss.IO_FLAG_MMAP           # Memory-mapped I/O to avoid loading full index into RAM

# ----------- Load FAISS Index -----------
print("[INFO] Loading FAISS index with mmap...")
if not os.path.exists(INDEX_FILE):
    print(f"[WARNING] Index file {INDEX_FILE} not found. Creating dummy index...")
    embeddings = np.random.rand(1000, EMBEDDING_DIM).astype('float32')
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)

# Load the index from disk (shared via copy-on-write when using Gunicorn --preload)
faiss_index = faiss.read_index(INDEX_FILE, IO_FLAGS)
print(f"[INFO] Loaded index with {faiss_index.ntotal} entries.")


# Try to move index to GPU if available
try:
    if hasattr(faiss, "StandardGpuResources"):
        print("✅ FAISS GPU support detected. Loading index to GPU...")
        res = faiss.StandardGpuResources()
        faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
    else:
        print("⚠️ FAISS GPU not available. Using CPU index.")
except Exception as e:
    print(f"❌ Failed to move FAISS index to GPU: {e}")
    print("⚠️ Falling back to CPU index.")

# ----------- FastAPI App Initialization -----------
app = FastAPI()

# Thread lock to synchronize access to the FAISS index
index_lock = threading.Lock()

# Define input data model for vector search
class QueryVector(BaseModel):
    vector: list  # Must contain EMBEDDING_DIM floats representing the face embedding

# Define the /search endpoint to accept POST requests with embedding vectors
@app.post("/search")
def search(query: QueryVector):
    try:
        # Convert input list to NumPy array and reshape to (1, EMBEDDING_DIM)
        vec = np.array(query.vector, dtype='float32').reshape(1, -1)

        # Lock the index during both search and update operations to ensure thread safety
        with index_lock:
            # Time the FAISS search (nearest neighbor lookup)
            t0 = time.perf_counter()
            distances, indices = faiss_index.search(vec, k=1)
            t1 = time.perf_counter()

            # Time the FAISS update (add new embedding to index)
            faiss_index.add(vec)
            t2 = time.perf_counter()

        # Return the results with detailed timing and updated index size
        return {
            "match_index": int(indices[0][0]),                      # Closest existing index
            "distance": float(distances[0][0]),                     # Distance to the matched vector
            "faiss_search_time_ms": round((t1 - t0) * 1000, 3),           # Time to perform search
            "faiss_index_update_time_ms": round((t2 - t1) * 1000, 3),          # Time to add new vector
            "new_index_size": faiss_index.ntotal                   # Total number of vectors after insertion
        }

    except Exception as e:
        # Return any errors that occur during the request processing
        return {"error": str(e)}

# ----------- Entry Point for Standalone Run -----------
import uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)

# Note: Run under Gunicorn with --preload and a single process to avoid index duplication.
# Example:
# gunicorn faiss_microservice:app -w 1 --threads 8 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:9000
