import os
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from deepface import DeepFace
from sklearn.metrics import confusion_matrix

# ----------- CONFIGURATION -----------

base_dir = "C:/Users/himan/Downloads/archive/Test"  # <<< CHANGE THIS to your dataset folder

models = ["Facenet512", "Facenet", "VGG-Face", "ArcFace"]
metrics = ["cosine", "euclidean_l2"]
detector_backends = ["retinaface", "fastmtcnn", "centerface", "yunet"]  # <<< All detectors to test


# ----------- STEP 1: Load Image Paths Grouped by Person -----------

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


# ----------- STEP 2: Generate Positive and Negative Pairs -----------

def generate_pairs(grouped_images):
    pairs = []
    persons = list(grouped_images.keys())

    # Positive pairs: within each folder
    for person, imgs in grouped_images.items():
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                pairs.append((imgs[i], imgs[j], 1))

    # Negative pairs: between folders
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            for img1 in grouped_images[persons[i]]:
                for img2 in grouped_images[persons[j]]:
                    pairs.append((img1, img2, 0))
    return pairs

pairs = generate_pairs(grouped_images)


# ----------- STEP 3: Precompute Embeddings -----------

def compute_embeddings(image_paths, model_name, detector_backend):
    embeddings = {}
    for img_path in image_paths:
        try:
            rep = DeepFace.represent(
                img_path=img_path,
                model_name=model_name,
                detector_backend=detector_backend,
               )
            embeddings[img_path] = rep[0]["embedding"]
        except Exception as e:
            print(f"[⚠️] Embedding failed for {img_path} using {detector_backend}: {e}")
    return embeddings


# ----------- STEP 4: Evaluate All Combinations -----------

results = []

# Flatten list of all image paths once
all_image_paths = [img for imgs in grouped_images.values() for img in imgs]

# Loop over all detector backends
for detector_backend in detector_backends:
    print(f"\n🚀 Starting evaluation with detector: {detector_backend}")

    for model in models:
        print(f"🔵 Computing embeddings for model: {model} using detector: {detector_backend}")
        embeddings = compute_embeddings(all_image_paths, model, detector_backend)
        print(f"✅ Total embeddings computed: {len(embeddings)}")

        for metric in metrics:
            print(f"🧪 Evaluating {model} + {detector_backend} + {metric}...")
            y_true, y_pred = [], []

            for img1, img2, label in pairs:
                if img1 in embeddings and img2 in embeddings:
                    try:
                        # Pass precomputed embeddings directly as img1_path / img2_path
                        result = DeepFace.verify(
                            img1_path=embeddings[img1],
                            img2_path=embeddings[img2],
                            model_name=model,
                            detector_backend=detector_backend,
                            distance_metric=metric,
                        )
                        pred = 1 if result["verified"] else 0
                        y_true.append(label)
                        y_pred.append(pred)
                    except Exception as e:
                        print(f"[❌] Error comparing {img1} & {img2}: {e}")
                        continue

            if y_true:
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                # Confusion Matrix: order is [[TN, FP], [FN, TP]]
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()

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
                    "FN": fn
                })

# ----------- STEP 5: Save Results -----------

df = pd.DataFrame(results)
output_file = os.path.join(base_dir, "deepface_detector_comparison.csv")
df.to_csv(output_file, index=False)

print(f"\n✅ All evaluations completed. Results saved to: {output_file}")
