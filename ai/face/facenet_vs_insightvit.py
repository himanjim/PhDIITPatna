# This script compares two face-recognition embedding pipelines,
# FaceNet512 through DeepFace and InsightFace ViT through the buffalo_l
# application wrapper, on an identity-organised image dataset. It groups
# images by person, constructs positive and negative verification pairs,
# computes embeddings once per image, evaluates pairwise similarity using
# fixed thresholds, and reports classification metrics together with
# confusion matrices. The purpose is comparative model evaluation rather
# than deployment benchmarking.

!pip install deepface insightface opencv-python-headless scikit-learn seaborn matplotlib onnxruntime

from google.colab import drive
drive.mount('/content/drive')

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from deepface import DeepFace
import insightface
import time

# Set your dataset path here
base_dir = "/content/drive/MyDrive/PhD/Data/archive/Image_Train"

# Load the dataset from the base directory and organise image paths by
# identity, assuming that each subfolder corresponds to one person.
def load_grouped_images(base_dir):
    grouped = {}
    for person in sorted(os.listdir(base_dir)):
        person_path = os.path.join(base_dir, person)
        if os.path.isdir(person_path):
            images = [os.path.join(person_path, f)
                      for f in os.listdir(person_path)
                      if f.lower().endswith(('jpg', 'jpeg', 'png'))]
            if images:
                grouped[person] = images
    return grouped

grouped_images = load_grouped_images(base_dir)
print(f"✅ Loaded {len(grouped_images)} identities.")

# Construct the labelled verification set by creating positive pairs from
# images of the same identity and negative pairs from images belonging to
# different identities.
def generate_pairs(grouped_images):
    pairs = []
    persons = list(grouped_images.keys())

    # Positive pairs
    for person, imgs in grouped_images.items():
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                pairs.append((imgs[i], imgs[j], 1))

    # Negative pairs: all combinations between different identities
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            for img1 in grouped_images[persons[i]]:
                for img2 in grouped_images[persons[j]]:
                    pairs.append((img1, img2, 0))

    return pairs

pairs = generate_pairs(grouped_images)
print(f"✅ Total image pairs: {len(pairs)}")

# Compute FaceNet512 embeddings once per image so that pairwise evaluation
# can reuse cached features rather than repeating feature extraction for
# every comparison.
def compute_facenet_embeddings(image_paths, detector_backend='retinaface'):
    embeddings = {}
    for img_path in tqdm(image_paths, desc="🧠 FaceNet512 embeddings"):
        try:
            rep = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet512",
                detector_backend=detector_backend,
                enforce_detection=True
            )
            embeddings[img_path] = rep[0]["embedding"]
        except Exception as e:
            print(f"[⚠️ FaceNet] Failed {img_path}: {e}")
    return embeddings

# Initialize InsightFace ViT
insight_model = insightface.app.FaceAnalysis(name='buffalo_l')
insight_model.prepare(ctx_id=0)

# Compute InsightFace embeddings once per image using the prepared
# InsightFace application model, retaining the first detected face per
# image when available.
def compute_insight_embeddings(image_paths):
    embeddings = {}
    for img_path in tqdm(image_paths, desc="🧠 InsightFace embeddings"):
        try:
            img = cv2.imread(img_path)
            faces = insight_model.get(img)
            if faces:
                embeddings[img_path] = faces[0].embedding
        except Exception as e:
            print(f"[⚠️ InsightFace] Failed {img_path}: {e}")
    return embeddings

# Get all unique image paths
image_paths = list({p for pair in pairs for p in pair[:2]})

# Compute embeddings
start = time.time()
facenet_embeddings = compute_facenet_embeddings(image_paths)
insight_embeddings = compute_insight_embeddings(image_paths)
print(f"⏱️ Total embedding time: {time.time() - start:.2f} seconds")

# Evaluate a verification model by comparing cosine similarity between
# paired embeddings against a fixed decision threshold and collecting the
# resulting true and predicted labels.
def evaluate_model(pairs, embedding_dict, threshold):
    y_true, y_pred = [], []
    for img1, img2, label in pairs:
        if img1 in embedding_dict and img2 in embedding_dict:
            emb1, emb2 = embedding_dict[img1], embedding_dict[img2]
            sim = cosine_similarity([emb1], [emb2])[0][0]
            pred = 1 if sim > threshold else 0
            y_true.append(label)
            y_pred.append(pred)
    return y_true, y_pred

# Evaluate both models
y_true_facenet, y_pred_facenet = evaluate_model(pairs, facenet_embeddings, threshold=0.3)
y_true_insight, y_pred_insight = evaluate_model(pairs, insight_embeddings, threshold=0.35)

# Compute and print the main verification metrics for one model together
# with its confusion matrix, and return the results in dictionary form for
# later comparison or plotting.
def print_metrics(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n📌 {name} Results")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    return {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "CM": cm}

results = []
results.append(print_metrics("FaceNet512", y_true_facenet, y_pred_facenet))
results.append(print_metrics("InsightFace_ViT", y_true_insight, y_pred_insight))

# Visualise a confusion matrix as a heatmap so that the distribution of
# correct and incorrect verification decisions can be inspected more
# easily.
def plot_confusion_matrix(cm, title):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Diff', 'Same'], yticklabels=['Diff', 'Same'])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

plot_confusion_matrix(results[0]['CM'], "FaceNet512 Confusion Matrix")
plot_confusion_matrix(results[1]['CM'], "InsightFace ViT Confusion Matrix")

def evaluate_model(pairs, embedding_dict, threshold):
    y_true, y_pred = [], []
    for img1, img2, label in pairs:
        if img1 in embedding_dict and img2 in embedding_dict:
            emb1, emb2 = embedding_dict[img1], embedding_dict[img2]
            sim = cosine_similarity([emb1], [emb2])[0][0]
            pred = 1 if sim > threshold else 0
            y_true.append(label)
            y_pred.append(pred)
    return y_true, y_pred

# Evaluate both models
y_true_facenet, y_pred_facenet = evaluate_model(pairs, facenet_embeddings, threshold=0.5)
y_true_insight, y_pred_insight = evaluate_model(pairs, insight_embeddings, threshold=0.35)

def print_metrics(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n📌 {name} Results")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    return {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "CM": cm}

results = []
results.append(print_metrics("FaceNet512", y_true_facenet, y_pred_facenet))
results.append(print_metrics("InsightFace_ViT", y_true_insight, y_pred_insight))

def plot_confusion_matrix(cm, title):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Diff', 'Same'], yticklabels=['Diff', 'Same'])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

plot_confusion_matrix(results[0]['CM'], "FaceNet512 Confusion Matrix")
plot_confusion_matrix(results[1]['CM'], "InsightFace ViT Confusion Matrix")
