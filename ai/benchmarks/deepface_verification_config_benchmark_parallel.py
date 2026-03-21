# This script performs parallel benchmarking of DeepFace-based face verification on an identity-organised image dataset. It evaluates multiple combinations of recognition model, detector backend, and distance metric by first grouping images by person, then generating genuine and impostor pairs, computing embeddings in parallel, and finally verifying image pairs in parallel. The purpose is to compare configuration-level verification performance using standard classification metrics such as accuracy, precision, recall, F1 score, and confusion-matrix counts. The design emphasises experimental throughput on large pair sets, while preserving a transparent evaluation pipeline that can be inspected by both developers and academic reviewers.

import os
import time
import pandas as pd
from deepface import DeepFace
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import gc
from tensorflow import keras

# ----------- CONFIGURATION -----------
base_dir = "C:/Users/himan/Downloads/archive/Test/"
# base_dir = "/mnt/c/Users/himan/Downloads/archive/Test/"
# base_dir = "/content/drive/MyDrive/DeepFace/Test"
models = ["Facenet512", "Facenet", "VGG-Face", "ArcFace"]
metrics = ["cosine", "euclidean_l2"]
detector_backends = ["retinaface", "fastmtcnn", "centerface", "yunet"]
MAX_WORKERS = os.cpu_count() - 8

# ----------- FUNCTION DEFINITIONS -----------
# Load the dataset from the base directory and organise image paths by identity, assuming that each subfolder represents one person. This grouped representation is used later to construct positive pairs from images of the same person and negative pairs from images belonging to different persons. Only supported image formats are included, and non-directory entries are ignored.

def load_grouped_images(base_dir):
    grouped = {}
    for person in sorted(os.listdir(base_dir)):
        person_path = os.path.join(base_dir, person)
        if os.path.isdir(person_path):
            images = [os.path.join(person_path, f)
                      for f in os.listdir(person_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                grouped[person] = images
    return grouped

# Construct the labelled verification set for evaluation. Positive pairs are formed from all unique within-person image combinations, while negative pairs are formed from cross-person image combinations. This exhaustive pairing strategy improves coverage of the available dataset, but it can also increase computational cost substantially as the number of identities and images grows.

def generate_pairs(grouped_images):
    pairs = []
    persons = list(grouped_images.keys())
    for person, imgs in grouped_images.items():
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                pairs.append((imgs[i], imgs[j], 1))
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            for img1 in grouped_images[persons[i]]:
                for img2 in grouped_images[persons[j]]:
                    pairs.append((img1, img2, 0))
    return pairs

# Compute the facial embedding for a single image under a specified model and detector configuration. This function is written as an isolated worker task so that it can be executed safely inside a process pool. On success it returns the image path together with its embedding; on failure it returns the image path and an error marker so that the calling process can exclude the image from later pairwise verification.
def compute_embedding_task(args):
    img_path, model, detector = args
    try:
        rep = DeepFace.represent(img_path=img_path, model_name=model,
                                 detector_backend=detector,
                                 enforce_detection=True)
        return img_path, rep[0]["embedding"], None
    except Exception as e:
        print(f"[⚠️] Embedding failed for {img_path} using {detector}: {e}")
        return img_path, None, str(e)

# Compute embeddings for all images in parallel using multiple worker processes. The function distributes one embedding task per image, collects successful embeddings into a lookup dictionary, and records failures separately. This precomputation step avoids repeated feature extraction during pair comparison and is therefore central to making large-scale verification experiments tractable.
def compute_embeddings_parallel(image_paths, model, detector):
    embeddings = {}
    failed = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = [(img, model, detector) for img in image_paths]
        futures = [executor.submit(compute_embedding_task, t) for t in tasks]
        for future in as_completed(futures):
            img_path, embedding, error = future.result()
            if embedding is not None:
                embeddings[img_path] = embedding
            else:
                failed.append(img_path)
    return embeddings, failed

# Verify a single labelled image pair using precomputed embeddings rather than re-reading and re-processing the original image files. The function returns the ground-truth label together with the predicted verification outcome when both embeddings are available. If either embedding is missing or verification fails, the pair is skipped so that downstream metrics are computed only on valid comparisons.
def verify_pair_task(args):
    img1, img2, label, embeddings, model, detector, metric = args
    if img1 not in embeddings or img2 not in embeddings:
        return None
    try:
        result = DeepFace.verify(
            img1_path=embeddings[img1],
            img2_path=embeddings[img2],
            model_name=model,
            detector_backend=detector,
            distance_metric=metric,
            enforce_detection=False,
            silent=True
        )
        return (label, 1 if result["verified"] else 0)
    except:
        return None

# Evaluate all labelled pairs in parallel for one model-detector-metric configuration. Each worker performs one pairwise verification task, and the results are aggregated into ground-truth and predicted label lists for metric computation. The function also counts skipped pairs, which is important for interpreting final scores when some embeddings are unavailable or some comparisons fail during execution.
def compare_pairs_parallel(pairs, embeddings, model, detector, metric):
    y_true, y_pred = [], []
    skipped = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = [(img1, img2, label, embeddings, model, detector, metric) for img1, img2, label in pairs]
        futures = [executor.submit(verify_pair_task, t) for t in tasks]
        for future in as_completed(futures):
            result = future.result()
            if result:
                y_true.append(result[0])
                y_pred.append(result[1])
            else:
                skipped += 1
    return y_true, y_pred, skipped

# ----------- MAIN EXECUTION (SAFE FOR WINDOWS) -----------

# Run the full experimental pipeline: load the dataset, generate labelled pairs, iterate over all detector, model, and metric combinations, compute embeddings, verify pairs, and store the resulting evaluation statistics to CSV. This function acts as the main driver for the benchmark and is intentionally structured as a single, reproducible workflow for configuration comparison. TensorFlow session clearing and garbage collection are invoked between detector-level runs to reduce memory carry-over across long experiments.
def main():
    start_time = time.time()
    grouped_images = load_grouped_images(base_dir)
    pairs = generate_pairs(grouped_images)
    all_image_paths = [img for imgs in grouped_images.values() for img in imgs]
    results = []

    for detector_backend in detector_backends:
        print(f"\n🚀 Detector: {detector_backend}")
        for model in models:
            print(f"🔵 Embedding with model: {model}")
            embeddings, failed_images = compute_embeddings_parallel(all_image_paths, model, detector_backend)
            print(f"✅ Success: {len(embeddings)} | ❌ Failed: {len(failed_images)}")

            for metric in metrics:
                print(f"🧪 Verifying: {model} + {detector_backend} + {metric}")
                y_true, y_pred, skipped = compare_pairs_parallel(pairs, embeddings, model, detector_backend, metric)

                if y_true:
                    acc = accuracy_score(y_true, y_pred)
                    prec = precision_score(y_true, y_pred, zero_division=0)
                    rec = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

                    results.append({
                        "Model": model,
                        "Metric": metric,
                        "Detector": detector_backend,
                        "Accuracy": acc,
                        "Precision": prec,
                        "Recall": rec,
                        "F1 Score": f1,
                        "TP": tp,
                        "TN": tn,
                        "FP": fp,
                        "FN": fn,
                        "Failed Embeddings": len(failed_images),
                        "Skipped Pairs": skipped
                    })
        del embeddings, failed_images
        keras.backend.clear_session()
        gc.collect()

    df = pd.DataFrame(results)
    output_file = os.path.join(base_dir, "deepface_detector_comparison_parallel.csv")
    df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")
    print(f"⏱️ Total execution time: {time.time() - start_time:.2f} seconds")

# ----------- LAUNCH GUARD FOR WINDOWS -----------

# Protect the multiprocessing entry point so that worker processes can be spawned safely on Windows. Without this guard, the process-pool sections of the script may recursively re-execute the module during child-process startup.
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
