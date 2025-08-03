"""
Locust benchmark for Triton + FAISS
-----------------------------------
* Measures four latencies per request:
  1. Triton embedding time            ->   name='triton'       request_type='model'
  2. FAISS nearest-neighbor search    ->   name='faiss_search' request_type='search'
  3. FAISS index update               ->   name='faiss_update' request_type='update'
  4. End-to-end workflow              ->   name='end_to_end'   request_type='login'
* All metrics are reported through Locust’s built-in stats; no ad-hoc CSV code.
"""

from locust import HttpUser, task, between
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
import numpy as np
import random, os, cv2, json, time, requests, logging
import os, atexit, csv

CSV_PATH = "run_metrics.csv"
write_header = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0

_csv_file = open(CSV_PATH, "a", buffering=1, newline="")
atexit.register(_csv_file.close)           # close file cleanly at exit
CSV = csv.writer(_csv_file)

if write_header:
    CSV.writerow(["triton_ms", "faiss_search_ms",
                  "faiss_update_ms", "total_time_ms"])

# ───────────── Config ───────────── #
TRITON        = InferenceServerClient("localhost:8001")   # Triton gRPC endpoint
FAISS_URL     = "http://localhost:9000/search"            # FAISS FastAPI endpoint
IMAGE_FOLDER  = "face_benchmark_gdrive/voter_images"      # flat list of JPG/PNG files
IMG_SIZE      = (112, 112)                                # InsightFace input size
EMB_LAYER     = "683"                                     # Output tensor name

LOG = logging.getLogger(__name__)

# ───────────── Helper: preload all images ───────────── #
def load_images(folder: str):
    imgs = []
    for fname in sorted(os.listdir(folder)):
        p = os.path.join(folder, fname)
        if not os.path.isfile(p):
            continue
        try:
            img = cv2.imread(p).astype("float32") / 255.0
            img = cv2.resize(img, IMG_SIZE).transpose(2, 0, 1)  # CHW
            imgs.append(img)
        except Exception as e:
            LOG.warning("Skipping %s (%s)", p, e)
    return imgs

IMGS = load_images(IMAGE_FOLDER)
if not IMGS:
    raise RuntimeError(f"No images found in {IMAGE_FOLDER}")

# ───────────── Locust user ───────────── #
class Voter(HttpUser):
    """
    Simulates a single voter sending one face per task.
    Each task fires FOUR separate Locust request_success events so
    Triton/FAISS timings show up as distinct endpoints.
    """
    host      = "http://localhost"       # dummy; not used
    wait_time = between(0.01, 0.05)

    @task
    def login(self):
        # 1) Prepare image & Triton input
        img = random.choice(IMGS)
        inp = np.expand_dims(img, 0)                 # [1,3,112,112]
        ii  = InferInput("input.1", inp.shape, "FP32")
        ii.set_data_from_numpy(inp)
        out = InferRequestedOutput(EMB_LAYER)

        # ── Triton inference ──
        t0 = time.time()
        resp = TRITON.infer("buffalo_l", inputs=[ii], outputs=[out])
        emb  = resp.as_numpy(EMB_LAYER)[0].tolist()
        t1 = time.time()
        triton_ms = (t1 - t0) * 1000
        # record
        self.environment.events.request_success.fire(
            request_type="model",
            name="triton",
            response_time=triton_ms,
            response_length=0,
        )

        # ── FAISS search + (optional) update ──
        payload = {"voter_id": random.randint(1_000_000, 9_999_999), "vector": emb}
        r = requests.post(FAISS_URL, json=payload).json()

        # If micro-service errors out, count as request_failure
        if "faiss_search_time_ms" not in r:
            self.environment.events.request_failure.fire(
                request_type="search",
                name="faiss_error",
                response_time=0,
                exception=RuntimeError(str(r)),
                response_length=0,
            )
            return

        faiss_search_ms  = r["faiss_search_time_ms"]
        faiss_update_ms  = r["faiss_index_update_time_ms"]

        # record them as two separate “endpoints”
        self.environment.events.request_success.fire(
            request_type="search",
            name="faiss_search",
            response_time=faiss_search_ms,
            response_length=0,
        )
        self.environment.events.request_success.fire(
            request_type="update",
            name="faiss_update",
            response_time=faiss_update_ms,
            response_length=0,
        )

        # ── End-to-end ──
        t2 = time.time()
        total_ms = (t2 - t0) * 1000
        self.environment.events.request_success.fire(
            request_type="login",
            name="end_to_end",
            response_time=total_ms,
            response_length=0,
        )