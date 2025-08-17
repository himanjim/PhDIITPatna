"""
Locust benchmark for Triton + FAISS
-----------------------------------
* Measures four latencies per request:
  1. Triton embedding time            ->   name='triton'       request_type='model'
  2. FAISS nearest-neighbor search    ->   name='faiss_search' request_type='search'
  3. FAISS index update               ->   name='faiss_update' request_type='update'
  4. End-to-end workflow              ->   name='end_to_end'   request_type='login'
* Uses Locust's unified events.request API (>=2.15).
* Logs one CSV row per task, even if there are failures/partials.
"""

from locust import HttpUser, task, between
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
import numpy as np
import random, os, cv2, requests, logging, atexit, csv
from time import perf_counter

# ───────────── CSV setup ───────────── #
CSV_PATH = "run_metrics.csv"
_write_header = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
_csv_file = open(CSV_PATH, "a", buffering=1, newline="")
atexit.register(_csv_file.close)           # close file cleanly at exit
CSV = csv.writer(_csv_file)
if _write_header:
    CSV.writerow(["triton_ms", "faiss_search_ms", "faiss_update_ms", "total_time_ms"])

# ───────────── Config ───────────── #
TRITON        = InferenceServerClient("localhost:8001")   # Triton gRPC endpoint
FAISS_URL     = "http://localhost:9000/search"            # FAISS FastAPI endpoint
IMAGE_FOLDER  = "face_benchmark_gdrive/voter_images"      # flat list of JPG/PNG files
IMG_SIZE      = (112, 112)                                # InsightFace input size
EMB_LAYER     = "683"                                     # Output tensor name

LOG = logging.getLogger(__name__)
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ───────────── Helper: preload all images ───────────── #
def load_images(folder: str):
    imgs = []
    for fname in sorted(os.listdir(folder)):
        p = os.path.join(folder, fname)
        if not os.path.isfile(p):
            continue
        try:
            img = cv2.imread(p)
            if img is None:
                LOG.warning("Skipping %s (cv2.imread returned None)", p)
                continue
            img = img.astype("float32") / 255.0
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
    Simulates a voter sending one face per task.
    Fires FOUR Locust request events so timings show as distinct endpoints.
    """
    host      = "http://localhost"       # dummy; not used
    wait_time = between(0.01, 0.05)

    # Unified request recorder (Locust ≥ 2.15)
    def _record(self, request_type, name, response_time_ms, response_length=0, exception=None, response=None):
        self.environment.events.request.fire(
            request_type=request_type,
            name=name,
            response_time=response_time_ms,
            response_length=response_length,
            exception=exception,
            context={"user": self},
            response=response
        )

    @task
    def login(self):
        # Timers & per-iteration metrics (None -> will print as 0.0 in CSV)
        t_iter_start = perf_counter()
        triton_ms = None
        faiss_search_ms = None
        faiss_update_ms = None

        try:
            # 1) Prepare image & Triton input
            img = random.choice(IMGS)
            inp = np.expand_dims(img, 0)                 # [1,3,112,112]
            ii  = InferInput("input.1", inp.shape, "FP32")
            ii.set_data_from_numpy(inp)
            out = InferRequestedOutput(EMB_LAYER)

            # ── Triton inference ──
            t_triton_start = perf_counter()
            try:
                resp = TRITON.infer("buffalo_l", inputs=[ii], outputs=[out])
                emb  = resp.as_numpy(EMB_LAYER)[0].tolist()
                triton_ms = (perf_counter() - t_triton_start) * 1000.0
                self._record("model", "triton", triton_ms, response_length=0, exception=None)
            except Exception as e:
                triton_ms = (perf_counter() - t_triton_start) * 1000.0
                self._record("model", "triton", triton_ms, response_length=0, exception=e)
                # Propagate; E2E will be recorded in outer except/finally
                raise

            # ── FAISS search + (optional) update ──
            payload = {"voter_id": random.randint(1_000_000, 9_999_999), "vector": emb}

            t_faiss_start = perf_counter()
            try:
                r = requests.post(FAISS_URL, json=payload, timeout=30)
                r.raise_for_status()
                rj = r.json()
            except Exception as e:
                rt_ms = (perf_counter() - t_faiss_start) * 1000.0
                # Record a single failure event under "faiss_error"
                self._record("search", "faiss_error", rt_ms, response_length=0, exception=e)
                raise

            # Validate JSON shape
            if "faiss_search_time_ms" not in rj:
                e = RuntimeError(f"Malformed FAISS response: keys={list(rj.keys())}")
                # Record and exit early; update metrics remain None
                self._record("search", "faiss_error", 0.0, response_length=0, exception=e)
                return

            faiss_search_ms  = float(rj.get("faiss_search_time_ms", 0.0))
            faiss_update_ms  = float(rj.get("faiss_index_update_time_ms", 0.0))

            # Record them as two separate “endpoints”
            self._record("search", "faiss_search", faiss_search_ms, response_length=0, exception=None)
            self._record("update", "faiss_update", faiss_update_ms, response_length=0, exception=None)

        except Exception as e:
            # Let Locust count the exception; we still log CSV in finally
            LOG.debug("Iteration error: %s", e, exc_info=True)
            # Re-raise so Locust marks the task as failed
            raise
        finally:
            # ── End-to-end ──
            total_ms = (perf_counter() - t_iter_start) * 1000.0

            # Always emit E2E event: success if no exception pending, else failure
            # Note: if we're in an except branch above, the exception has already been fired for sub-steps.
            # Here we only record success when all earlier steps completed.
            # We infer success if both Triton and FAISS timings are present or FAISS JSON was valid.
            e2e_exception = None
            # Heuristic: if Triton failed, triton_ms exists but an exception was raised; Locust already counted it.
            # We'll emit success only if Triton ran AND we saw FAISS search timing (meaning FAISS JSON was valid).
            if triton_ms is None or faiss_search_ms is None:
                # mark as failure for E2E
                e2e_exception = RuntimeError("end_to_end_incomplete")

            self._record("login", "end_to_end", total_ms, response_length=0, exception=e2e_exception)

            # ── CSV: write one row per iteration (zeros for missing parts) ──
            try:
                CSV.writerow([
                    f"{(triton_ms or 0.0):.3f}",
                    f"{(faiss_search_ms or 0.0):.3f}",
                    f"{(faiss_update_ms or 0.0):.3f}",
                    f"{total_ms:.3f}",
                ])
            except Exception as e:
                LOG.warning("CSV write failed: %s", e)
