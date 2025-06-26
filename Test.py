# import numpy as np
# print(np.__version__)
# print(np.arange(5))
# print(np.integer)
# exit(0)

# import tensorflow as tf
#
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     print("✅ GPU(s) detected by TensorFlow:")
#     for gpu in gpus:
#         print(gpu)
# else:
#     print("❌ No GPU detected by TensorFlow.")

import tensorflow as tf
from deepface.models import Facenet512
import numpy as np
import time

# Initialize model once
model = Facenet512.loadModel()


# GPU-optimized pipeline
def get_embedding(image_array):
    # Preprocess on GPU
    img_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    img_tensor = tf.image.resize(img_tensor, (160, 160))
    img_tensor = (img_tensor - 127.5) / 128.0  # Facenet specific
    img_tensor = tf.expand_dims(img_tensor, axis=0)

    # Run inference
    return model(img_tensor, training=False)[0]


# Benchmark
for i in range(10):
    random_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    start = time.time()
    embedding = get_embedding(random_img)
    print(f"Run {i + 1}: {(time.time() - start) * 1000:.2f}ms")