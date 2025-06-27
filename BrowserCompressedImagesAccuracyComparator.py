import os
from pathlib import Path
from deepface import DeepFace
from deepface.commons import distance as dst
from tqdm import tqdm

# --- Step 1: Configuration ---
model_name = "Facenet512"
detector_backend = "retinaface"
distance_metric = "cosine"  # 'cosine', 'euclidean', or 'euclidean_l2'

# Get threshold based on model and metric
threshold = dst.findThreshold(model_name, distance_metric)

# Detect Downloads folder path (Windows)
downloads_path = os.path.join(os.environ["USERPROFILE"], "Downloads")
downloads_folder = Path(downloads_path).as_posix()

# Define directories
original_dir = "voter_images"
compressed_dir = os.path.join(downloads_folder, "voter_images_compressed")

# --- Step 2: Helper - Collect images ---
def collect_images(folder, exts=('.jpg', '.jpeg', '.png')):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ])

print("📥 Loading image paths...")
original_paths = collect_images(original_dir)
compressed_paths = collect_images(compressed_dir)

if not original_paths:
    raise FileNotFoundError(f"No images found in original folder: {original_dir}")
if not compressed_paths:
    raise FileNotFoundError(f"No images found in compressed folder: {compressed_dir}")

# --- Step 3: Generate embeddings (bulk) ---
print(f"\n🧠 Generating embeddings using {model_name} + {detector_backend}...")

original_embeddings = DeepFace.represent(
    img_path=original_paths,
    model_name=model_name,
    detector_backend=detector_backend,
    enforce_detection=False,
    prog_bar=True
)

compressed_embeddings = DeepFace.represent(
    img_path=compressed_paths,
    model_name=model_name,
    detector_backend=detector_backend,
    enforce_detection=False,
    prog_bar=True
)

# --- Step 4: Build {name → embedding} maps ---
orig_dict = {
    os.path.splitext(os.path.basename(obj["instance"]))[0]: obj["embedding"]
    for obj in original_embeddings
}

comp_dict = {
    os.path.splitext(os.path.basename(obj["instance"]))[0].replace("_deepface", ""): obj["embedding"]
    for obj in compressed_embeddings
}

# --- Step 5: Compare embeddings and report accuracy ---
print("\n🔍 Comparing embeddings...")

correct = 0
total = len(orig_dict)

for name, orig_embed in tqdm(orig_dict.items(), total=total):
    if name not in comp_dict:
        print(f"⚠️ Skipping: No compressed match found for '{name}'")
        continue

    comp_embed = comp_dict[name]
    distance = dst.findDistance(orig_embed, comp_embed, distance_metric)
    matched = distance <= threshold
    result = "✅" if matched else "❌"

    print(f"{result} {name}: Distance = {distance:.4f} | Threshold = {threshold:.4f}")

    if matched:
        correct += 1

# --- Step 6: Final accuracy summary ---
accuracy = (correct / total) * 100 if total else 0
print(f"\n📊 Final Accuracy: {accuracy:.2f}% ({correct}/{total} matched)")
