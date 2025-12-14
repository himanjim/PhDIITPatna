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


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
