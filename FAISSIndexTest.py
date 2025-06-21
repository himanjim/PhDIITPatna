# FAISS IndexFlatL2 with Memory-Mapped Storage for Exact Matching + Timing

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
print("[INFO] Generating synthetic voter embeddings...")
embeddings = np.random.rand(NUM_VOTERS, EMBEDDING_DIM).astype('float32')

# ---------- Step 2: Create and populate IndexFlatL2 ----------
print("[INFO] Creating IndexFlatL2 and adding vectors...")
index = faiss.IndexFlatL2(EMBEDDING_DIM)
index.add(embeddings)

# ---------- Step 3: Save index to disk ----------
print(f"[INFO] Saving index to disk as {INDEX_FILE}...")
faiss.write_index(index, INDEX_FILE)

# ---------- Step 4: Load index via mmap ----------
print(f"[INFO] Loading index from disk with memory-mapped I/O...")
mmap_index = faiss.read_index(INDEX_FILE, faiss.IO_FLAG_MMAP)

# ---------- Step 4.5: Warm up index by reconstructing entries ----------
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


# Cold-start Search Time: 1086.0795 ms
#
# [INFO] Benchmarking 10 warm searches...
# ✅ Warm search benchmark completed:
# Average Search Time: 1087.3747 ms
# Min Search Time: 1083.6219 ms
# Max Search Time: 1089.9090 ms
