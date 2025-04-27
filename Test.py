# import numpy as np
# print(np.__version__)
# print(np.arange(5))
# print(np.integer)
# exit(0)

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU(s) detected by TensorFlow:")
    for gpu in gpus:
        print(gpu)
else:
    print("❌ No GPU detected by TensorFlow.")