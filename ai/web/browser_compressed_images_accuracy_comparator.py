# This script evaluates whether browser-compressed face images remain
# identifiable relative to their originals when embeddings are generated
# with DeepFace FaceNet512 and compared using cosine distance. It loads
# image pairs from two folders, computes embeddings once for all original
# and compressed images, performs all pairwise original-versus-compressed
# comparisons, and reports standard verification metrics together with
# detailed false-positive and false-negative examples. The script is
# intended for empirical comparison of compression impact rather than for
# production inference.

import sys
import re
# --- Step 1: Local DeepFace path ---
deepface_path = "C:/Users/himan/PycharmProjects/PhDIITPatna/deepface_local"
if deepface_path not in sys.path:
    sys.path.insert(0, deepface_path)


import os
import sys
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from deepface import DeepFace
from scipy.spatial.distance import cosine

# Local DeepFace import (if not pip installed)
deepface_path = "C:/Users/himan/PycharmProjects/PhDIITPatna/deepface_local"
if deepface_path not in sys.path:
    sys.path.insert(0, deepface_path)

# --- Step 1: Configuration ---
model_name = "Facenet512"
detector_backend = "retinaface"
distance_threshold = 0.3

# --- Step 2: Define folders ---
downloads_path = os.path.join(os.environ["USERPROFILE"], "Downloads")
original_dir = os.path.join(downloads_path, "voter_images_faces")
compressed_dir = os.path.join(downloads_path, "voter_images_faces_compressed")

# --- Step 3: Collect image paths ---
# Collect all supported image files from the specified folder in sorted
# order so that evaluation remains reproducible.
def collect_images(folder, exts=('.jpg', '.jpeg', '.png')):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ])

# Derive a simple identity label from the leading alphabetic portion of
# the filename so that comparisons can be labelled as same-person or
# different-person cases.
def extract_identity(filename):
    basename = os.path.basename(filename)
    match = re.match(r'^([A-Za-z]+)', basename)
    return match.group(1)


original_paths = collect_images(original_dir)
compressed_paths = collect_images(compressed_dir)

# --- Step 4: Generate embeddings ---
# Compute one embedding per image for both the original-image set and the
# compressed-image set before any pairwise comparisons are made.
print("🔍 Generating embeddings...")
original_embeddings = DeepFace.represent(
    img_path=original_paths,
    model_name=model_name,
    detector_backend=detector_backend,
    enforce_detection=False,
)

compressed_embeddings = DeepFace.represent(
    img_path=compressed_paths,
    model_name=model_name,
    detector_backend=detector_backend,
    enforce_detection=False,
)

# --- Step 5: Compare embeddings ---
print("\n📊 Comparing for confusion matrix...")

y_true, y_pred = [], []

# --- For tracking detailed results ---
false_positives = []
false_negatives = []

# Compare every original image embedding against every compressed-image
# embedding and classify each pair according to cosine distance and the
# configured decision threshold.
for orig_path, orig_embed_list in zip(original_paths, original_embeddings):
    if not orig_embed_list:
        continue
    orig_identity = extract_identity(orig_path)
    orig_embedding = np.array(orig_embed_list[0]["embedding"])

    for comp_path, comp_embed_list in zip(compressed_paths, compressed_embeddings):
        if not comp_embed_list:
            continue
        comp_identity = extract_identity(comp_path)
        comp_embedding = np.array(comp_embed_list[0]["embedding"])

        distance = cosine(orig_embedding, comp_embedding)
        is_match = distance <= distance_threshold
        same_person = orig_identity == comp_identity

        y_true.append(1 if same_person else 0)
        y_pred.append(1 if is_match else 0)

        # Track mismatches
        if is_match and not same_person:
            false_positives.append((orig_path, comp_path, distance))
        elif not is_match and same_person:
            false_negatives.append((orig_path, comp_path, distance))

# --- Step 6: Metrics ---
print("\n✅ Evaluation Metrics:")
print(f"Accuracy:  {accuracy_score(y_true, y_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y_true, y_pred):.2f}")
print(f"Recall:    {recall_score(y_true, y_pred):.2f}")
print(f"F1 Score:  {f1_score(y_true, y_pred):.2f}")

print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, labels=[0, 1], target_names=["Different", "Same"]))


# --- Step 7: Confusion Matrix ---
# --- Compute Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # 0 = Different, 1 = Same

# --- Create Confusion Matrix DataFrame ---
cm_df = pd.DataFrame(
    cm,
    index=["True: Different", "True: Same"],
    columns=["Pred: Different", "Pred: Same"]
)

# --- Print Confusion Matrix in Tabular Form ---
print("\n📊 Confusion Matrix:")
print(cm_df.to_string())


# --- Print Results ---
print("\n❌ False Positives (Matched but different identities):")
for orig, comp, dist in false_positives:
    print(f"FP: {os.path.basename(orig)} ↔ {os.path.basename(comp)} | Distance: {dist:.4f}")

print("\n❌ False Negatives (Same identity but not matched):")
for orig, comp, dist in false_negatives:
    print(f"FN: {os.path.basename(orig)} ↔ {os.path.basename(comp)} | Distance: {dist:.4f}")


# 📊 Comparing for confusion matrix...
#
# ✅ Evaluation Metrics:
# Accuracy:  99.30%
# Precision: 1.00
# Recall:    0.89
# F1 Score:  0.94
#
# 📋 Classification Report:
#               precision    recall  f1-score   support
#
#    Different       0.99      1.00      1.00     45784
#         Same       1.00      0.89      0.94      3057
#
#     accuracy                           0.99     48841
#    macro avg       0.99      0.95      0.97     48841
# weighted avg       0.99      0.99      0.99     48841
#
#
# 📊 Confusion Matrix:
#                  Pred: Different  Pred: Same
# True: Different            45774          10
# True: Same                   332        2725
