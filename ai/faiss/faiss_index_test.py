# This script is a standalone experimental harness for building and timing
# an exact `IndexFlatL2` FAISS index backed by memory-mapped storage. It
# generates a large synthetic embedding set, writes the index to disk,
# reloads it through FAISS memory mapping, and then measures cold and warm
# search behaviour. The script is intended for one-off systems
# experimentation rather than for production service deployment.

import numpy as np
import faiss
import os
import time

# ---------- Configuration ----------
EMBEDDING_DIM = 512
NUM_VOTERS = 10_000_000  # Simulate 10 million voters for demo
INDEX_FILE = "voter_index_flatl2.faiss"
NUM_BENCHMARK_QUERIES = 10

# ---------- Step 1: Generate synthetic embeddings ----------
# Generate the synthetic embedding matrix used both to populate the index
# and to supply query vectors for the timing checks.
print("[INFO] Generating synthetic voter embeddings...")
embeddings = np.random.rand(NUM_VOTERS, EMBEDDING_DIM).astype('float32')

# ---------- Step 2: Create and populate IndexFlatL2 ----------
# Build and populate an exact FlatL2 index in memory before writing it to
# disk.
print("[INFO] Creating IndexFlatL2 and adding vectors...")
index = faiss.IndexFlatL2(EMBEDDING_DIM)
index.add(embeddings)

# ---------- Step 3: Save index to disk ----------
print(f"[INFO] Saving index to disk as {INDEX_FILE}...")
faiss.write_index(index, INDEX_FILE)

# ---------- Step 4: Load index via mmap ----------
# Reload the saved index through FAISS memory-mapped I/O so that the test
# exercises the disk-backed access path rather than only the original
# in-memory object.
print(f"[INFO] Loading index from disk with memory-mapped I/O...")
mmap_index = faiss.read_index(INDEX_FILE, faiss.IO_FLAG_MMAP)

# ---------- Step 4.5: Warm up index by reconstructing entries ----------
# Touch representative entries to warm the mapped index before the later
# warm-search measurements.
print("[INFO] Warming up index to preload into memory...")
for i in range(0, mmap_index.ntotal, 100_000):
    _ = mmap_index.reconstruct(i)

# ---------- Step 5: Perform initial search and time it ----------
print("[INFO] Performing cold-start search...")
test_query = embeddings[123].reshape(1, -1)
start_time = time.perf_counter()
distances, indices = mmap_index.search(test_query, k=1)
end_time = time.perf_counter()
search_time_ms = (end_time - start_time) * 1000
print("✅ Cold-start search completed:")
print("Matched Index:", indices[0][0])
print("Distance:", distances[0][0])
print(f"Cold-start Search Time: {search_time_ms:.4f} ms")

# ---------- Step 6: Benchmark warm search performance ----------
print(f"\n[INFO] Benchmarking {NUM_BENCHMARK_QUERIES} warm searches...")
timings = []
for i in range(NUM_BENCHMARK_QUERIES):
    query = embeddings[np.random.randint(0, NUM_VOTERS)].reshape(1, -1)
    start = time.perf_counter()
    mmap_index.search(query, k=1)
    end = time.perf_counter()
    timings.append((end - start) * 1000)

avg_time = np.mean(timings)
min_time = np.min(timings)
max_time = np.max(timings)

print("✅ Warm search benchmark completed:")
print(f"Average Search Time: {avg_time:.4f} ms")
print(f"Min Search Time: {min_time:.4f} ms")
print(f"Max Search Time: {max_time:.4f} ms")

# ---------- Optional: Delete index file ----------
# os.remove(INDEX_FILE)
