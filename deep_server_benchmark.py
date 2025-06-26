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
if platform.system() == "Windows":
    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
    image_dir = os.path.join(downloads_path, "voter_images")
else:  # Linux / macOS
    image_dir = os.path.join(os.getcwd(), "voter_images")

print(f"[INFO] Using image directory: {image_dir}")
if not os.path.isdir(image_dir):
    raise FileNotFoundError(f"Directory not found: {image_dir}")


# -------- Load images from folder --------
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
start = time.perf_counter()
reps = DeepFace.represent(
    img_path=images,
    model_name=MODEL_NAME,
    detector_backend=DETECTOR_BACKEND,
    enforce_detection=ENFORCE_DETECTION
)
end = time.perf_counter()

# -------- Results --------
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


