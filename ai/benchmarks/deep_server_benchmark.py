# This script performs a small-scale batch embedding test using DeepFace on
# a local folder of face images. Its purpose is to verify that the selected
# recognition model, detector backend, and runtime environment can process
# multiple images in one call and return embeddings with the expected
# structure. The program is therefore best understood as a diagnostic and
# benchmarking utility rather than a production inference pipeline. It
# reports total runtime, average runtime per image, and a brief per-image
# summary so that both implementation behaviour and output format can be
# inspected during development or experimental reporting.
import os
import platform
import time
import numpy as np
import cv2
from deepface import DeepFace
from pathlib import Path

# -------- CONFIG --------
MAX_IMAGES = 10  # 🔁 Change to load more or fewer images
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "retinaface"
ENFORCE_DETECTION = False

# -------- TensorFlow GPU Config --------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# -------- Determine OS and Image Folder --------
# Select the image source directory according to the host operating system.
# On Windows, the script expects a folder named "voter_images" under the
# user's Downloads directory; on Linux and macOS, it expects the same
# folder under the current working directory. This keeps the script easy to
# run across common development environments without changing the core
# embedding logic.

if platform.system() == "Windows":
    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
    image_dir = os.path.join(downloads_path, "voter_images")
else:  # Linux / macOS
    image_dir = os.path.join(os.getcwd(), "voter_images")

print(f"[INFO] Using image directory: {image_dir}")
if not os.path.isdir(image_dir):
    raise FileNotFoundError(f"Directory not found: {image_dir}")


# -------- Load images from folder --------
# Load a bounded set of image files into memory before invoking batch
# embedding. Only images that OpenCV can decode successfully are retained,
# and their filenames are stored separately so that the printed results can
# be matched back to the source files. This explicit loading stage ensures
# that later timing measurements reflect the DeepFace batch call itself
# rather than repeated file discovery during inference.

image_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
image_paths = image_paths[:MAX_IMAGES]

images = []
valid_filenames = []

for path in image_paths:
    img = cv2.imread(str(path))
    if img is not None:
        images.append(img)
        valid_filenames.append(path.name)
    else:
        print(f"[WARNING] Failed to load: {path}")

if not images:
    raise RuntimeError("No valid images loaded.")

print(f"[INFO] Loaded {len(images)} image(s) for batch embedding.")

# -------- Perform Batch Embedding --------
# Measure the execution time of a single batch embedding call over all
# loaded images. The DeepFace API is invoked with in-memory image arrays,
# which allows the script to test whether batch-style processing works
# correctly for the chosen model and detector settings. The reported timing
# is a wall-clock measurement for the entire batch under the current
# runtime conditions and should therefore be interpreted as an empirical
# performance observation rather than a hardware-independent constant.

start = time.perf_counter()
reps = DeepFace.represent(
    img_path=images,
    model_name=MODEL_NAME,
    detector_backend=DETECTOR_BACKEND,
    enforce_detection=ENFORCE_DETECTION
)
end = time.perf_counter()

# -------- Results --------
# Report both aggregate and per-image outcomes from the batch embedding
# call. The summary includes total batch time and average time per image,
# while the per-image output confirms whether an embedding was produced and
# shows its shape. This is useful for checking consistency of the returned
# data structure, especially when detector settings permit images with weak
# or ambiguous facial content.
print("\n✅ Embedding Summary:")
print(f"Total images processed: {len(reps)}")
print(f"Total time: {(end - start)*1000:.2f} ms")
print(f"Avg time per image: {(end - start)*1000/len(reps):.2f} ms")

for i, r in enumerate(reps):
    if not r:
        print(f"{valid_filenames[i]} → ❌ No face detected")
        continue

    first_face = r[0]  # Always get the first face's data
    embedding = first_face["embedding"]
    emb_shape = np.array(embedding).shape
    print(f"{valid_filenames[i]} → shape: {emb_shape}, sample[0]: {embedding}")


