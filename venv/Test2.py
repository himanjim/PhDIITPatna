import os
import cv2
import numpy as np
import faiss
from deepface import DeepFace

# Configs
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
DATABASE_PATH = "C:/Users/himan/Downloads/Test Faces/"  # Your root image directory
THRESHOLD = 0.6  # Adjust based on accuracy needs

# Storage
embedding_list = []
metadata_list = []

# Step 1: Scan folder and subfolders
def scan_images(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(root, file))
    return image_paths

# Step 2: Build index from existing images
def build_index(image_paths):
    global embedding_list, metadata_list

    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))  # Resize here

        try:
            embedding = DeepFace.represent(
                img_path=img,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
                align=True,
            )[0]["embedding"]

            embedding_list.append(np.array(embedding, dtype="float32"))
            metadata_list.append(img_path)

        except Exception as e:
            print(f"Failed on {img_path}: {e}")

    if not embedding_list:
        raise Exception("No valid face embeddings found.")

    embeddings_np = np.vstack(embedding_list).astype("float32")

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)

    return index

# Step 3: Match a new face
def match_new_face(img_path, index):
    try:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))  # Resize here

        result = DeepFace.represent(
            img_path=img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True,
        )[0]["embedding"]

        query = np.array(result, dtype="float32").reshape(1, -1)

        distances, indices = index.search(query, k=1)
        best_distance = distances[0][0]
        best_index = indices[0][0]

        if best_distance < THRESHOLD:
            print(f"Match found: {metadata_list[best_index]} (distance: {best_distance:.4f})")
        else:
            print(f"No match found. (distance: {best_distance:.4f}). Adding to database.")
            index.add(query)
            embedding_list.append(query[0])
            metadata_list.append(img_path)

    except Exception as e:
        print(f"Failed to process new face {img_path}: {e}")

# ---------- MAIN EXECUTION ----------
if __name__ == "__main__":
    image_paths = scan_images(DATABASE_PATH + 'Faces/')
    print(f"Found {len(image_paths)} images in database.")

    face_index = build_index(image_paths)

    # Example: match a new face
    new_face_path = DATABASE_PATH+ "my_passport_photo (110 x 140).png"
    match_new_face(new_face_path, face_index)
    match_new_face(new_face_path, face_index)

    # (Optional) Save metadata_list and embeddings for persistence
