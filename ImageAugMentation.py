import os
import imageio
import imgaug.augmenters as iaa

# Define the parent directory that contains subfolders of images
parent_dir = "C:/Users/himan/Downloads/archive/Image_Train/"  # <<< CHANGE this to your root folder

# Define image augmentation pipeline
augmenter = iaa.Sequential([
    iaa.Multiply((0.6, 1.4)),                    # Brightness variation
    iaa.GaussianBlur(sigma=(0.0, 1.5)),          # Blur
    iaa.Affine(rotate=(-25, 25)),                # Rotation
    iaa.AddToHueAndSaturation((-20, 20)),        # Hue/saturation shift
    iaa.Crop(percent=(0, 0.1)),                  # Slight crop
    iaa.Fliplr(0.5)                              # Horizontal flip
])

# Traverse each subdirectory (representing a person)
for person_dir in os.listdir(parent_dir):
    full_path = os.path.join(parent_dir, person_dir)

    # Process only if it's a directory
    if os.path.isdir(full_path):
        print(f"📂 Processing folder: {person_dir}")
        image_files = [f for f in os.listdir(full_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_file in image_files:
            img_path = os.path.join(full_path, img_file)

            try:
                image = imageio.imread(img_path)

                # Generate 5 augmented versions of the image
                augmented_images = augmenter(images=[image for _ in range(5)])

                for idx, aug_img in enumerate(augmented_images):
                    out_filename = f"{os.path.splitext(img_file)[0]}_aug{idx + 1}.jpg"
                    out_path = os.path.join(full_path, out_filename)
                    imageio.imwrite(out_path, aug_img)

            except Exception as e:
                print(f"⚠️ Failed to augment {img_path}: {e}")
