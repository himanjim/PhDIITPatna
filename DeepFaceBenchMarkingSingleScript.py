# ----------- Imports -----------
import os
# Enable TensorFlow GPU memory growth before any TF/DL imports
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import time
import psutil
import faiss
import numpy as np
import pandas as pd
from deepface import DeepFace
import platform
import random
import threading

import tensorflow as tf
print("[INFO] Available GPUs:", tf.config.list_physical_devices('GPU'))

# ----------- Configuration -----------
INDEX_FILE = "voter_index_flatl2.faiss"   # FAISS index file
EMBEDDING_DIM = 512                        # Facenet512 dimension
IO_FLAGS = faiss.IO_FLAG_MMAP              # Memory-map for FAISS
if platform.system() != "Windows":
    IMAGE_FOLDER = "/content/drive/MyDrive/voter_images"  # Image source
    CONCURRENCY_LEVELS = [1, 10, 100, 1000]  # Batch sizes to test
    FAISS_INDEX_SIZE = 1000000
else:
    IMAGE_FOLDER = "C:/Users/himan/Downloads/voter_images"  # Image source
    CONCURRENCY_LEVELS = [1, 10]  # Batch sizes to test
    FAISS_INDEX_SIZE = 1000

TOP_K = 5                                  # FAISS KNN

# Thread lock to synchronize access to the FAISS index
index_lock = threading.Lock()
# ----------- Load FAISS Index -----------
print("[INFO] Loading FAISS index...")
if not os.path.exists(INDEX_FILE):
    print("[WARNING] Index file not found. Creating dummy index...")
    dummy_embeddings = np.random.rand(FAISS_INDEX_SIZE, EMBEDDING_DIM).astype('float32')
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(dummy_embeddings)
    faiss.write_index(index, INDEX_FILE)

faiss_index = faiss.read_index(INDEX_FILE, IO_FLAGS)
print(f"[INFO] Loaded FAISS index with {faiss_index.ntotal} entries.")

# ----------- Try moving to GPU -----------
try:
    if hasattr(faiss, "StandardGpuResources"):
        print("✅ FAISS GPU support detected. Loading index to GPU...")
        res = faiss.StandardGpuResources()
        faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
except Exception as e:
    print(f"⚠️ Failed to move FAISS index to GPU: {e}")


# ----------- Helper Functions -----------
def measure_resources():
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().used / (1024 * 1024)
    return cpu, mem


def process_batch(image_paths):
    stats = {
        "embed_time": 0,
        "faiss_call_time": 0,
        "faiss_search_time": 0,
        "faiss_update_time": 0,
        "failed": 0
    }

    try:
        # Embedding all images in one batch
        t0 = time.time()

        # Simulate image upload latency (e.g., 50ms to 300ms)
        time.sleep(random.uniform(0.05, 0.3))

        reps = DeepFace.represent(
            img_path=image_paths,
            model_name="Facenet512",
            detector_backend="retinaface",
            enforce_detection=False
        )
        stats["embed_time"] = (time.time() - t0) * 1000

        if not isinstance(reps, list):
            reps = [reps]  # Ensure list of dicts

        # Build numpy array from embeddings
        valid_embeddings = []

        # Normalize reps: always make it a list of dicts
        if isinstance(reps[0], list):  # batch mode: list of lists
            flattened = [item[0] for item in reps if isinstance(item, list) and item]
        else:  # single image mode
            flattened = reps

        # Convert each embedding
        for r in flattened:
            try:
                emb = np.array(r["embedding"], dtype='float32').reshape(1, -1)
                valid_embeddings.append(emb)
            except Exception as e:
                stats["failed"] += 1
                print(f"❌ Exception occurred in converting embeddings:", str(e))

        if not valid_embeddings:
            stats["failed"] = len(image_paths)
            return stats

        all_embeddings = np.vstack(valid_embeddings)

        # FAISS search
        t1 = time.time()
        # Simulate network delay between DeepFace and FAISS microservice (e.g., 20ms to 100ms)
        time.sleep(random.uniform(0.02, 0.1))

        t2 = time.time()
        D, I = faiss_index.search(all_embeddings, TOP_K)
        t3 = time.time()

        # Lock the index during both search and update operations to ensure thread safety
        with index_lock:
            # Time the FAISS update (add new embedding to index)
            faiss_index.add(all_embeddings)
        t4 = time.perf_counter()

        stats["faiss_call_time"] = (t2 - t1) * 1000
        stats["faiss_search_time"] = (t3 - t2) * 1000
        stats["faiss_update_time"] = (t4 - t3) * 1000

    except Exception as e:
        stats["failed"] = len(image_paths)
        print(f"❌ Exception occurred:", str(e))

    return stats

# ----------- Benchmark Function -----------
def benchmark(batch_size):
    print(f"\n🚀 Benchmarking {batch_size} concurrent voters...")
    # Get all image files from the folder
    image_files = [os.path.join(IMAGE_FOLDER, f)
                   for f in os.listdir(IMAGE_FOLDER)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Randomly select N images with replacement
    selected_images = random.choices(image_files, k=batch_size)

    start_time = time.time()
    stats = process_batch(selected_images)
    end_time = time.time()

    cpu, mem = measure_resources()

    return {
        "Concurrent Voters": batch_size,
        "Avg Total Time (ms)": stats["embed_time"] + stats["faiss_call_time"],
        "Avg Embed Time (ms)": stats["embed_time"] / batch_size,
        "Avg FAISS Call Time (ms)": stats["faiss_call_time"] / batch_size,
        "Avg FAISS Search Time (ms)": stats["faiss_search_time"] / batch_size,
        "Avg FAISS Update Time (ms)": stats["faiss_update_time"] / batch_size,
        "Total Batch Time (ms)": (end_time - start_time) * 1000,
        "Failed Requests": stats["failed"],
        "CPU Usage (%)": cpu,
        "Memory Usage (MB)": mem
    }

# ----------- Run Benchmarks -----------
results = []
for batch_size in CONCURRENCY_LEVELS:
    results.append(benchmark(batch_size))

# ----------- Output CSV Table -----------
df = pd.DataFrame(results)
print("\n📊 Benchmark Results:\n")
print(df.to_string(index=False))
