import os
import random
import string
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_DIR = "my_dataset"
IMG_SIZE = (32, 32)
TRAIN_SPLIT = 0.8  # 80% train, 20% test

# Customize the amount of images per class here
IMAGES_PER_CLASS = 500

# ONLY DIGITS (0-9) - This creates exactly 10 classes
CLASSES = list(string.digits)


# ==========================================
# DATASET GENERATION
# ==========================================

def create_directory_structure():
    """Creates the train/test folder structure for each digit class."""
    for split in ['train', 'test']:
        for char in CLASSES:
            # Simple clean folder names since they are just digits '0' through '9'
            path = os.path.join(OUTPUT_DIR, split, char)
            os.makedirs(path, exist_ok=True)


def generate_character_image(char):
    """Generates a 32x32 white background image with a random black digit, then blurs it."""
    # 1. Create a white canvas
    img = Image.new("RGB", IMG_SIZE, color="white")
    draw = ImageDraw.Draw(img)

    # 2. Load default font
    font = ImageFont.load_default()

    # 3. Calculate position with slight random jitter to vary the data
    bbox = draw.textbbox((0, 0), char, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Randomly nudge the digit around the center
    max_x_jitter = max(0, IMG_SIZE[0] - text_width)
    max_y_jitter = max(0, IMG_SIZE[1] - text_height)
    x = random.randint(0, max_x_jitter)
    y = random.randint(0, max_y_jitter)

    # 4. Draw the black text
    draw.text((x, y), char, fill="black", font=font)

    # 5. Convert to OpenCV format to apply Gaussian Blur
    open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Adjusted to a stronger blur kernel (5x5) as requested previously!
    # Change to (3,3) for less blur, or (7,7) for even more blur.
    blurred_image = cv2.GaussianBlur(open_cv_image, (5, 5), 0)

    return blurred_image


def main():
    print(f"Generating dataset for {len(CLASSES)} digit classes (0-9)...")
    create_directory_structure()

    num_train = int(IMAGES_PER_CLASS * TRAIN_SPLIT)

    for char in CLASSES:
        for i in range(IMAGES_PER_CLASS):
            # Generate the blurred image
            img = generate_character_image(char)

            # Determine if it goes to train or test split
            split = "train" if i < num_train else "test"

            # Save the image
            file_name = f"img_{i + 1}.jpg"
            save_path = os.path.join(OUTPUT_DIR, split, char, file_name)
            cv2.imwrite(save_path, img)

    print(f"Successfully created digit dataset at './{OUTPUT_DIR}'!")


if __name__ == "__main__":
    main()