# FaceRecognitionBenchmarkClient.py
# Simulates concurrent voters submitting images to the DeepFace server
# and collects metrics from both DeepFace and FAISS microservices.
# Enhanced with detailed comments and exception logging.

import time
from pathlib import Path
import os
import pandas as pd
from threading import Thread
import requests
import random
import psutil
from statistics import mean
import cv2
import traceback

# Configuration parameters
NUM_VOTERS_LIST = [10]  # Adjust for load testing
SERVER_URL = "http://localhost:8000/verify"
SIMULATED_NETWORK_LATENCY = (10, 50)  # Simulated network delay in milliseconds
TIMEOUT = 100  # HTTP request timeout in seconds

# Get path to voter_images folder
# Assumes images are located in Downloads/voter_images
downloads_path = os.path.join(os.environ["USERPROFILE"], "Downloads")
posix_path = Path(downloads_path).as_posix()
IMAGE_FOLDER = posix_path + "/voter_images"

# Load all image file paths from voter_images directory
def load_face_images(folder_path):
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(folder_path, filename))
    return image_paths

image_paths = load_face_images(IMAGE_FOLDER)

# Simulate a single voter request
# Measures total round-trip time and extracts metrics from server responses
def simulate_request(results, index):
    start_time = time.perf_counter()
    img_path = random.choice(image_paths)

    try:
        with open(img_path, "rb") as f:
            files = {'file': (os.path.basename(img_path), f, 'image/jpeg')}
            time.sleep(random.uniform(*SIMULATED_NETWORK_LATENCY) / 1000.0)
            response = requests.post(SERVER_URL, files=files, timeout=TIMEOUT)
            end_time = time.perf_counter()

            duration = (end_time - start_time) * 1000  # Total request duration in ms
            status = response.status_code

            if status == 200:
                data = response.json()
                print(data)
                embedding_time = data.get('embedding_time_ms')
                faiss_call_time = data.get('faiss_call_time_ms')
                faiss_search_time = data.get('faiss_search_time_ms')
                faiss_update_time = data.get('faiss_index_update_time_ms')
                distance = data.get('distance')
                match_index = data.get('match_index')

                results[index] = (
                    duration, status,
                    embedding_time,
                    faiss_call_time,
                    faiss_search_time,
                    faiss_update_time,
                    distance, match_index
                )
            else:
                print(f"[{index}] ❌ Request failed with status code: {status}")
                results[index] = (duration, status, None, None, None, None, None, None)

    except Exception as e:
        print(f"[{index}] ❌ Exception occurred:", str(e))
        traceback.print_exc()
        results[index] = (-1, 500, None, None, None, None, None, None)

# Capture CPU and memory usage before and after test batches
def get_cpu_memory_usage():
    process = psutil.Process(os.getpid())
    cpu_percent = psutil.cpu_percent(interval=1)
    mem_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    return round(cpu_percent, 2), round(mem_usage, 2)

# Run benchmark loop for different numbers of concurrent voters
summary = []
for num_voters in NUM_VOTERS_LIST:
    results = [None] * num_voters
    threads = [Thread(target=simulate_request, args=(results, i)) for i in range(num_voters)]

    # Capture system stats before load
    cpu_before, mem_before = get_cpu_memory_usage()
    t0 = time.perf_counter()

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Capture system stats after load
    t1 = time.perf_counter()
    cpu_after, mem_after = get_cpu_memory_usage()

    # Extract metrics from results
    durations = [r[0] for r in results if r and r[1] == 200]
    embeds = [r[2] for r in results if r and isinstance(r[2], (int, float))]
    faiss_call_times = [r[3] for r in results if r and isinstance(r[3], (int, float))]
    faiss_search_times = [r[4] for r in results if r and isinstance(r[4], (int, float))]
    faiss_update_times = [r[5] for r in results if r and isinstance(r[5], (int, float))]
    failed = sum(1 for r in results if r and r[1] != 200)

    # Log summary row
    summary.append({
        "Concurrent Voters": num_voters,
        "Avg Total Time (ms)": round(mean(durations), 2) if durations else 'N/A',
        "Avg Embed Time (ms)": round(mean(embeds), 2) if embeds else 'N/A',
        "Avg FAISS Call Time (ms)": round(mean(faiss_call_times), 2) if faiss_call_times else 'N/A',
        "Avg FAISS Search Time (ms)": round(mean(faiss_search_times), 2) if faiss_search_times else 'N/A',
        "Avg FAISS Update Time (ms)": round(mean(faiss_update_times), 2) if faiss_update_times else 'N/A',
        "Total Batch Time (ms)": round((t1 - t0) * 1000, 2),
        "Failed Requests": failed,
        "CPU Usage (%)": f"{cpu_before} → {cpu_after}",
        "Memory Usage (MB)": f"{mem_before} → {mem_after}"
    })

# Output results to CSV and console
output_csv = posix_path + "/benchmark_results_detailed.csv"
df = pd.DataFrame(summary)
print(df)
df.to_csv(output_csv, index=False)
print(f"\n✅ Metrics saved to {output_csv}")
