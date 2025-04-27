import os  # For handling file and directory operations
import pandas as pd  # For storing and exporting results as DataFrame/CSV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # Evaluation metrics
from deepface import DeepFace  # DeepFace library for face recognition
from sklearn.metrics import confusion_matrix  # To compute TP, FP, TN, FN
import time
import gc
from tensorflow import keras

# ----------- CONFIGURATION -----------

base_dir = "C:/Users/himan/Downloads/archive/Test/"  # <<< CHANGE THIS to your dataset folder path
# base_dir = "/media/himanshu/OS/Users/himan/Downloads/archive/Test/"  # <<< CHANGE THIS to your dataset folder path

# base_dir = "/content/drive/MyDrive/DeepFace/Test"

models = ["Facenet512", "Facenet", "VGG-Face", "ArcFace"]  # List of DeepFace models to evaluate
metrics = ["cosine", "euclidean_l2"]  # Distance metrics to test for face comparison
detector_backends = ["retinaface", "fastmtcnn", "centerface", "yunet"]  # Face detectors to evaluate

# ----------- START TIMER -----------
start_time = time.time()

# ----------- STEP 1: Load Image Paths Grouped by Person -----------


def load_grouped_images(base_dir):
    grouped = {}  # Dictionary to store image paths grouped by person
    for person in sorted(os.listdir(base_dir)):  # Iterate over folders in base_dir
        person_path = os.path.join(base_dir, person)
        if os.path.isdir(person_path):  # Ensure it's a directory (not a file)
            images = [os.path.join(person_path, f)
                      for f in os.listdir(person_path)
                      if f.lower().endswith(('jpg', 'jpeg', 'png'))]  # Collect image paths with valid extensions
            if images:
                grouped[person] = images  # Add to dictionary if images exist
    return grouped  # Return grouped dictionary


grouped_images = load_grouped_images(base_dir)  # Load image paths grouped by person

# ----------- STEP 2: Generate Positive and Negative Pairs -----------


def generate_pairs(grouped_images):
    pairs = []  # List to store image pairs with labels
    persons = list(grouped_images.keys())  # Get list of person IDs (folder names)

    # Positive pairs: all combinations of two images from the same person
    for person, imgs in grouped_images.items():
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                pairs.append((imgs[i], imgs[j], 1))  # Label 1 for positive match

    # Negative pairs: all combinations of one image from different persons
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            for img1 in grouped_images[persons[i]]:
                for img2 in grouped_images[persons[j]]:
                    pairs.append((img1, img2, 0))  # Label 0 for negative match
    return pairs  # Return list of labeled pairs


pairs = generate_pairs(grouped_images)  # Generate all image pairs

# ----------- STEP 3: Precompute Embeddings -----------


def compute_embeddings(image_paths, model_name, detector_backend):
    failed_images = []  # To track failed embeddings

    embeddings = {}  # Dictionary to hold image path to embedding mapping
    for img_path in image_paths:
        try:
            # Extract embedding using DeepFace
            rep = DeepFace.represent(
                img_path=img_path,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=True
            )
            embeddings[img_path] = rep[0]["embedding"]  # Store embedding
        except Exception as e:
            print(f"[⚠️] Embedding failed for {img_path} using {detector_backend}: {e}")
            failed_images.append(img_path)  # Log failed image
    return embeddings, failed_images  # Return successful embeddings and failures


# ----------- STEP 4: Evaluate All Combinations -----------

results = []  # Store evaluation metrics for all combinations

# Flatten all image paths from grouped_images into one list
all_image_paths = [img for imgs in grouped_images.values() for img in imgs]

# Iterate over all detector backends
for detector_backend in detector_backends:
    print(f"\n🚀 Starting evaluation with detector: {detector_backend}")

    for model in models:
        print(f"🔵 Computing embeddings for model: {model} using detector: {detector_backend}")
        embeddings, failed_images = compute_embeddings(all_image_paths, model, detector_backend)
        print(f"✅ Embeddings created: {len(embeddings)} | ❌ Failed: {len(failed_images)}")

        for metric in metrics:
            print(f"🧪 Evaluating {model} + {detector_backend} + {metric}...")
            y_true, y_pred = [], []  # Lists to hold actual and predicted labels
            skipped_pairs = 0  # Track pairs that were skipped due to failure

            for img1, img2, label in pairs:
                if img1 in embeddings and img2 in embeddings:
                    try:
                        # Use precomputed embeddings instead of image paths
                        result = DeepFace.verify(
                            img1_path=embeddings[img1],
                            img2_path=embeddings[img2],
                            model_name=model,
                            detector_backend=detector_backend,
                            distance_metric=metric,
                            enforce_detection=False,
                            silent=True
                        )

                        y_true.append(label)
                        y_pred.append(1 if result["verified"] else 0)  # Convert boolean result to int
                    except Exception as e:
                        print(f"[❌] Error comparing {img1} & {img2}: {e}")
                        skipped_pairs += 1
                else:
                    skipped_pairs += 1  # Skip if embeddings not available

            if y_true:
                # Compute performance metrics
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                # Compute confusion matrix to get TP, TN, FP, FN
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()

                # Append result for this configuration
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
                    "Skipped Pairs": skipped_pairs
                })
            else:
                print(f"[⚠️] Skipping scoring for {model} + {detector_backend} + {metric} — No valid pairs.")

        del embeddings, failed_images
        keras.backend.clear_session()
        gc.collect()

# ----------- STEP 5: Save Results -----------

df = pd.DataFrame(results)  # Create a DataFrame from the results list
output_file = os.path.join(base_dir, "deepface_detector_comparison.csv")  # Set output file path
df.to_csv(output_file, index=False)  # Save results to CSV file

end_time = time.time()
print(f"\n✅ All evaluations completed. Results saved to: {output_file}")
print(f"⏱️ Total execution time: {end_time - start_time:.2f} seconds")
