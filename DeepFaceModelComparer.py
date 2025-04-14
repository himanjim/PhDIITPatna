import os
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from deepface import DeepFace

# ------------- CONFIGURATION SECTION -------------

# Base directory containing folders for each person with images inside
# Example structure:
# /path/to/indian_faces/
#   └── person1/
#         ├── img1.jpg
#         └── img2.jpg
#   └── person2/
#         ├── img3.jpg
#         └── img4.jpg
base_dir = "C:/Users/himan/Downloads/archive/Test"  # <<< 🔁 CHANGE THIS to the actual path on your machine

# List of all DeepFace-supported models, detectors, and similarity metrics you want to test
models = ["Facenet512", "Facenet", "VGG-Face", "ArcFace", "Dlib", "GhostFaceNet",
          "SFace", "OpenFace", "DeepFace", "DeepID"]

detectors = ["retinaface", "mtcnn", "fastmtcnn", "dlib", "yolov8", "yunet",
             "centerface", "mediapipe", "ssd", "opencv", "skip"]

metrics = ["euclidean", "euclidean_l2", "cosine"]

# Batch size determines how many image pairs are processed in each batch to avoid GPU overuse
BATCH_SIZE = 100  # You can adjust based on available GPU RAM


# ------------- STEP 1: Load Images into Grouped Dictionary -------------

def load_grouped_images(base_dir):
    """
    Reads the image dataset from folders.
    Returns a dictionary where:
    - key: person name (i.e., folder name)
    - value: list of full image paths belonging to that person
    """
    grouped = {}
    for person in sorted(os.listdir(base_dir)):
        person_path = os.path.join(base_dir, person)
        if os.path.isdir(person_path):
            images = [os.path.join(person_path, f) for f in os.listdir(person_path)
                      if f.lower().endswith(('jpg', 'jpeg', 'png'))]
            if images:
                grouped[person] = images
    return grouped


grouped_images = load_grouped_images(base_dir)


# ------------- STEP 2: Generate All Positive and Negative Pairs -------------

def generate_pairs(grouped_images):
    """
    Generates all image pairs:
    - Positive pairs: same person → label = 1
    - Negative pairs: different persons → label = 0
    Returns a list of tuples: (image1, image2, label)
    """
    pairs = []
    persons = list(grouped_images.keys())

    # Positive pairs: same folder (same person)
    for person, imgs in grouped_images.items():
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                pairs.append((imgs[i], imgs[j], 1))

    # Negative pairs: between different persons
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            for img1 in grouped_images[persons[i]]:
                for img2 in grouped_images[persons[j]]:
                    pairs.append((img1, img2, 0))

    return pairs


# This contains thousands of (img1, img2, label) tuples
pairs = generate_pairs(grouped_images)


# ------------- STEP 3: Evaluate One Combination in Batches -------------

def evaluate_in_batches(model, detector, metric, pairs, batch_size=BATCH_SIZE):
    """
    Evaluates DeepFace.verify() over all image pairs in batches.
    This is serial (non-parallel) but memory-safe for GPU usage.

    Returns:
    - y_true: ground truth labels (0 for negative, 1 for positive)
    - y_pred: predicted labels from DeepFace
    """
    y_true, y_pred = [], []
    total = len(pairs)

    for i in range(0, total, batch_size):
        batch = pairs[i:i + batch_size]
        print(f"   → Processing batch {i + 1} to {min(i + batch_size, total)} / {total}")

        for img1, img2, label in batch:
            try:
                result = DeepFace.verify(
                    img1_path=img1,
                    img2_path=img2,
                    model_name=model,
                    detector_backend=detector,
                    distance_metric=metric,
                    enforce_detection=False,  # skip if face not detected
                    silent=True  # suppress logs
                )
                y_true.append(label)
                y_pred.append(1 if result["verified"] else 0)
            except:
                # Skip this pair if any error (e.g., face not detected, broken file, etc.)
                continue

    return y_true, y_pred


# ------------- STEP 4: Evaluate All Combinations -------------

results = []

# Total combinations = 10 models × 11 detectors × 3 metrics = 330
for model, detector, metric in itertools.product(models, detectors, metrics):
    print(f"\n🧪 Evaluating combination: {model} + {detector} + {metric}...")

    y_true, y_pred = evaluate_in_batches(model, detector, metric, pairs)

    if y_true:  # Only add results if some valid predictions were made
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results.append({
            "Model": model,
            "Detector": detector,
            "Metric": metric,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        })

# ------------- STEP 5: Save Results to CSV -------------

df = pd.DataFrame(results)

# Save results for further analysis in Excel or Jupyter
df.to_csv(base_dir + "/deepface_model_comparison.csv", index=False)

print("\n✅ All evaluations completed. Results saved to 'deepface_model_comparison.csv'.")
