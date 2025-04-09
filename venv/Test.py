from deepface import DeepFace
import cv2
import numpy as np

def enhance_image(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def sharpen(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

DATABASE_PATH = "C:/Users/himan/Downloads/Test Faces/"
# Load images
img1 = cv2.imread(DATABASE_PATH + "hs_passport_photo.jpg")
# img2 = cv2.imread(DATABASE_PATH + "my_passport_photo (110 x 140).png")
# img2 = cv2.imread(DATABASE_PATH + "SS passport photo.jpg")
# img2 = cv2.imread(DATABASE_PATH + "Akanksha Jaswal passport photo.png")
img2 = cv2.imread(DATABASE_PATH + "Shama Himanshu passport photo (2).jpg")

# Step 1: Resize both to 224x224 (or at least the low-res one)
img1 = cv2.resize(img1, (224, 224))  # Upscale low-res image
img2 = cv2.resize(img2, (224, 224))  # Upscale low-res image

# # Step 2: Pre-enhance contrast (Histogram Equalization)
# img1 = enhance_image(img1)
# img2 = enhance_image(img2)

# Step 3: Sharpen Image
# img1 = sharpen(img1)
# img2 = sharpen(img2)

# Step 3: Compare using DeepFace
result = DeepFace.verify(
    img1_path=img1,
    img2_path=img2,
    model_name="ArcFace",
    detector_backend="retinaface",
    enforce_detection=True
)

print("Is Match:", result["verified"])
print("Distance:", result["distance"])
print("Threshold used:", result["threshold"])
