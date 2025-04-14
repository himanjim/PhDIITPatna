import os
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from deepface import DeepFace

# --- CONFIGURATION ---

# Base directory where your dataset is located
# Each subfolder should contain images of a single person, e.g.:
# /path/to/indian_faces/person1/*.jpg
# /path/to/indian_faces/person2/*.jpg
base_dir = "C:/Users/himan/Downloads/archive/Test"  # <<< CHANGE THIS TO YOUR ACTUAL PATH

# Define all the DeepFace models, face detectors, and distance metrics you want to test
models = ["Facenet512", "Facenet", "VGG-Face", "ArcFace", "Dlib", "GhostFaceNet",
          "SFace", "OpenFace", "DeepFace", "DeepID"]
detectors = ["retinaface", "mtcnn", "fastmtcnn", "dlib", "yolov8", "yunet",
             "centerface", "mediapipe", "ssd", "opencv", "skip"]
metrics = ["euclidean", "euclidean_l2", "cosine"]


# --- STEP 1: Load all images grouped by person ---

def load_grouped_images(base_dir):
    """
    Walk through base_dir and organize image paths in a dictionary where:
    key = person_name (folder name)
    value = list of image paths belonging to that person
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

# Dictionary: {person_name: [img1, img2, ...]}
grouped_images = load_grouped_images(base_dir)


# --- STEP 2: Generate all image pairs for evaluation ---

def generate_pairs(grouped_images):
    """
    Generate labeled image pairs:
    - Positive pairs (same person) → label 1
    - Negative pairs (different persons) → label 0
    Returns a list of tuples: (img1, img2, label)
    """
    pairs = []
    persons = list(grouped_images.keys())

    # Generate positive pairs
    for person, imgs in grouped_images.items():
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                pairs.append((imgs[i], imgs[j], 1))  # same person → positive label

    # Generate negative pairs
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            for img1 in grouped_images[persons[i]]:
                for img2 in grouped_images[persons[j]]:
                    pairs.append((img1, img2, 0))  # different persons → negative label

    return pairs

# Prepare all (img1, img2, label) combinations
pairs = generate_pairs(grouped_images)


# --- STEP 3: Evaluate each model-detector-metric combination ---

def evaluate(model, detector, metric, pairs):
    """
    Use DeepFace.verify to compare image pairs and collect predictions.
    Returns:
    - y_true: list of actual labels (0 or 1)
    - y_pred: list of predicted labels (0 or 1)
    """
    y_true, y_pred = [], []

    for img1, img2, label in pairs:
        try:
            # Perform face verification using DeepFace
            result = DeepFace.verify(
                img1, img2,
                model_name=model,
                detector_backend=detector,
                distance_metric=metric,
                enforce_detection=False,  # skip detection errors
                silent=True
            )
            # Append actual label
            y_true.append(label)

            # Append prediction: 1 if verified match, else 0
            y_pred.append(1 if result["verified"] else 0)

        except:
            # In case of error (e.g., bad face detection), skip the pair
            continue

    return y_true, y_pred


# --- STEP 4: Loop through all combinations and record performance ---

results = []

# This will take time: 10 models × 11 detectors × 3 metrics = 330 iterations
for model, detector, metric in itertools.product(models, detectors, metrics):
    print(f"Evaluating {model} + {detector} + {metric}...")

    y_true, y_pred = evaluate(model, detector, metric, pairs)

    if y_true:
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Store result
        results.append({
            "Model": model,
            "Detector": detector,
            "Metric": metric,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })


# --- STEP 5: Save results to CSV file ---

# Convert to pandas DataFrame
df = pd.DataFrame(results)

# Save all results for analysis
df.to_csv("deepface_model_comparison.csv", index=False)

print("\n✅ Evaluation complete. Results saved to 'deepface_model_comparison.csv'.")
