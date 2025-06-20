# Optimized FaceRecognitionBenchmarkClient.py

import os
import time
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import requests
import random
import psutil
from statistics import mean
from deepface import DeepFace

# ----------- Configuration -----------
NUM_VOTERS_LIST = [1]  # Adjust upper bound as needed
SERVER_URL = "http://localhost:8000/verify"
SIMULATED_NETWORK_LATENCY = (10, 50)  # in milliseconds
TIMEOUT = 5  # in seconds

# ----------- Paths -----------
downloads_path = str(Path.home() / "Downloads")
IMAGE_FOLDER = downloads_path + "/voter_images"

# ----------- Load Image Paths -----------
def load_face_embeddings(folder_path):
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            image_paths.append(img_path)
    return image_paths

image_paths = load_face_embeddings(IMAGE_FOLDER)
if not image_paths:
    raise Exception("No images found in folder: " + IMAGE_FOLDER)

# ----------- Voter Simulation Function -----------
def simulate_request(results, index):
    start_time = time.perf_counter()

    img_path = random.choice(image_paths)
    try:
        rep = DeepFace.represent(
            img_path=img_path,
            model_name="Facenet512",
            enforce_detection=False
        )
        embedding_vector = rep[0]["embedding"]
        data = {"face_embedding": embedding_vector}
    except Exception as e:
        print(f"[{index}] ❌ Embedding failed: {e}")
        results[index] = (-1, 500)
        return

    time.sleep(random.uniform(*SIMULATED_NETWORK_LATENCY) / 1000.0)

    try:
        response = requests.post(SERVER_URL, json=data, timeout=TIMEOUT)
        end_time = time.perf_counter()
        status = response.status_code
        duration = (end_time - start_time) * 1000
        print(f"[{index}] ✅ Response {status}")
    except Exception as e:
        print(f"[{index}] ❌ Request failed: {e}")
        duration = -1
        status = 500

    results[index] = (duration, status)


# ----------- System Resource Monitoring -----------
def get_cpu_memory_usage():
    process = psutil.Process(os.getpid())
    cpu_percent = psutil.cpu_percent(interval=1)
    mem_usage = process.memory_info().rss / (1024 * 1024)
    return round(cpu_percent, 2), round(mem_usage, 2)

# ----------- Run Benchmark -----------
def run_benchmark():
    results_summary = []

    for num_voters in NUM_VOTERS_LIST:
        results = [None] * num_voters

        cpu_before, mem_before = get_cpu_memory_usage()
        start_all = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_voters) as executor:
            futures = [executor.submit(simulate_request, results, i) for i in range(num_voters)]
            for f in futures:
                f.result()

        end_all = time.perf_counter()
        cpu_after, mem_after = get_cpu_memory_usage()

        durations = [r[0] for r in results if r and r[1] == 200]
        failed = sum(1 for r in results if r[1] != 200)

        results_summary.append({
            "Concurrent Voters": num_voters,
            "Avg Time per Voter (ms)": round(mean(durations), 2) if durations else 'N/A',
            "Total Time (ms)": round((end_all - start_all) * 1000, 2),
            "Max Time Observed (ms)": round(max(durations), 2) if durations else 'N/A',
            "Failed Requests": failed,
            "CPU Usage (%)": f"{cpu_before} → {cpu_after}",
            "Memory Usage (MB)": f"{mem_before} → {mem_after}"
        })

    df = pd.DataFrame(results_summary)
    print(df)
    df.to_csv(downloads_path + "/benchmark_results.csv", index=False)

# ----------- Main Entry -----------
if __name__ == "__main__":
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    run_benchmark()
