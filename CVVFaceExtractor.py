import cv2
import mediapipe as mp
import os
from pathlib import Path

import tensorflow as tf
print("[INFO] Available GPUs:", tf.config.list_physical_devices('GPU'))


# Detect Downloads folder path (Windows)
downloads_path = os.path.join(os.environ["USERPROFILE"], "Downloads")
downloads_folder = Path(downloads_path).as_posix()

input_folder = os.path.join(downloads_folder, "voter_images")
output_folder = os.path.join(downloads_folder, "voter_images_faces")
os.makedirs(output_folder, exist_ok=True)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# MediaPipe face detector
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        path = os.path.join(input_folder, filename)
        image = cv2.imread(path)
        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = face_detection.process(image_rgb)
        if not results.detections:
            print(f"No face found in {filename}")
            continue

        # Find the largest face based on bounding box area
        faces = []
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            xmin = int(bbox.xmin * w)
            ymin = int(bbox.ymin * h)
            box_w = int(bbox.width * w)
            box_h = int(bbox.height * h)
            area = box_w * box_h
            faces.append((area, xmin, ymin, box_w, box_h))

        largest = max(faces, key=lambda x: x[0])
        _, x, y, bw, bh = largest

        # Add padding like webcam
        pad_ratio = 0.5
        pad_x = int(bw * pad_ratio)
        pad_y = int(bh * pad_ratio)
        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_y, 0)
        x2 = min(x + bw + pad_x, w)
        y2 = min(y + bh + pad_y, h)

        face_crop = image[y1:y2, x1:x2]
        face_resized = cv2.resize(face_crop, (224, 224))

        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_face.png")
        cv2.imwrite(output_path, face_resized)  # PNG is lossless by default

        print(f"Saved: {output_path}")

print("✅ Done.")
