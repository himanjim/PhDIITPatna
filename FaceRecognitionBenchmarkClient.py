import time
from pathlib import Path
import os
import pandas as pd
from threading import Thread
import requests
import random
import psutil
import os
from statistics import mean
import cv2
from deepface import DeepFace

# Configuration parameters
NUM_VOTERS_LIST = [1]  # Number of concurrent simulated voters
SERVER_URL = "http://localhost:8000/verify"  # Local server endpoint
SIMULATED_NETWORK_LATENCY = (10, 50)  # Simulated network delay range in ms
TIMEOUT = 5  # Request timeout in seconds

# Get the Downloads path (Windows style)
downloads_path = os.path.join(os.environ["USERPROFILE"], "Downloads")
# Convert to POSIX (Linux-style) path
posix_path = Path(downloads_path).as_posix()
IMAGE_FOLDER = posix_path + "/voter_images"  # Folder containing face images

# Load image paths from the specified folder
def load_face_embeddings(folder_path):
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            image_paths.append(img_path)
    return image_paths

# Load all eligible image file paths into memory
image_paths = load_face_embeddings(IMAGE_FOLDER)

# Simulate a single voter request: from reading the image to posting to server
def simulate_request(results, index):
    start_time = time.perf_counter()

    # Randomly select an image and compute its embedding
    img_path = random.choice(image_paths)
    try:
        img = cv2.imread(img_path)  # Read image from disk
        rep = DeepFace.represent(img_path=img, model_name='Facenet512', enforce_detection=False)
        embedding_vector = rep[0]["embedding"]
        data = {'face_embedding': embedding_vector}  # Prepare request payload
    except Exception:
        results[index] = (-1, 500)  # Mark failure if any error
        return

    # Simulate network delay
    time.sleep(random.uniform(*SIMULATED_NETWORK_LATENCY) / 1000.0)

    # POST request to the local verification server
    try:
        response = requests.post(SERVER_URL, json=data, timeout=TIMEOUT)
        end_time = time.perf_counter()
        status = response.status_code
        duration = (end_time - start_time) * 1000  # Convert to milliseconds
    except Exception:
        duration = -1
        status = 500  # Mark failed request

    # Save result tuple (duration, HTTP status code)
    results[index] = (duration, status)

# Capture CPU and memory usage before and after request batch
def get_cpu_memory_usage():
    process = psutil.Process(os.getpid())
    cpu_percent = psutil.cpu_percent(interval=1)
    mem_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    return round(cpu_percent, 2), round(mem_usage, 2)

# Run the benchmark for different numbers of concurrent voters
def run_benchmark():
    results_summary = []

    for num_voters in NUM_VOTERS_LIST:
        results = [None] * num_voters  # Store individual results
        threads = [Thread(target=simulate_request, args=(results, i)) for i in range(num_voters)]

        # Capture pre-execution resource usage
        cpu_before, mem_before = get_cpu_memory_usage()
        start_all = time.perf_counter()

        # Start all voter threads
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        end_all = time.perf_counter()
        # Capture post-execution resource usage
        cpu_after, mem_after = get_cpu_memory_usage()

        # Filter out valid timings and count failures
        durations = [r[0] for r in results if r and r[1] == 200]
        failed = sum(1 for r in results if r[1] != 200)

        # Log summary metrics
        results_summary.append({
            'Concurrent Voters': num_voters,
            'Avg Time per Voter (ms)': round(mean(durations), 2) if durations else 'N/A',
            'Total Time (ms)': round((end_all - start_all) * 1000, 2),
            'Max Time Observed (ms)': round(max(durations), 2) if durations else 'N/A',
            'Failed Requests': failed,
            'CPU Usage (%)': f"{cpu_before} → {cpu_after}",
            'Memory Usage (MB)': f"{mem_before} → {mem_after}"
        })

    # Convert to DataFrame and export to CSV
    df = pd.DataFrame(results_summary)
    print(df)
    df.to_csv(posix_path + "/benchmark_results.csv", index=False)

# Main execution
if __name__ == "__main__":
    run_benchmark()
