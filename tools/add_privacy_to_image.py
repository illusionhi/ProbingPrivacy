import os
import random
import json
import shutil
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor

# Original and new root directories
root_dir = './data/'
new_root_dir = './data_privacy/'

# List of subdirectories to process
subdirs = [
    'split2_test'
    # 'coco/train2017',
    # 'gqa/images',
    # 'ocr_vqa/images',
    # 'textvqa/train_images',
    # 'vg/VG_100K',
    # 'vg/VG_100K_2'
]

directories = [os.path.join(root_dir, subdir) for subdir in subdirs]

# Read privacy.jsonl file and store the data in a list
privacy_file = 'privacy2.jsonl'
with open(privacy_file, 'r', encoding='utf-8') as f:
    privacy_data = [json.loads(line) for line in f]

def process_image(image_path):
    # Randomly decide whether to watermark (1% chance)
    if random.random() < 1:
        # Open image
        with Image.open(image_path) as img:
            # Randomly select a privacy entry
            privacy_entry = random.choice(privacy_data)
            # Extract required fields
            keys = ['username', 'user_id', 'gender', 'age', 'capture_time', 'capture_location']
            watermark_text = '\n'.join([f"{key}: {privacy_entry.get(key, '')}" for key in keys])

            # Use default font
            font = ImageFont.load_default()

            # Create a transparent overlay
            txt_layer = Image.new('RGBA', img.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(txt_layer)

            # Get image dimensions
            width, height = img.size

            # Get text size
            left, top, right, bottom = draw.textbbox((0, 0), watermark_text, font)
            text_width, text_height = right - left, bottom - top

            # Choose random position
            max_x = max(width - text_width, 0)
            max_y = max(height - text_height, 0)
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            # Draw text onto the transparent layer
            draw.multiline_text((x, y), watermark_text, font=font, fill=(255, 255, 255, 128))

            # Combine original image with text layer
            img = img.convert('RGBA')
            watermarked = Image.alpha_composite(img, txt_layer)

            # Build new image path
            relative_path = os.path.relpath(image_path, root_dir)
            new_image_path = os.path.join(new_root_dir, relative_path)

            # Create directories if they don't exist
            new_image_dir = os.path.dirname(new_image_path)
            os.makedirs(new_image_dir, exist_ok=True)

            # Save image (convert back to RGB)
            watermarked = watermarked.convert('RGB')
            watermarked.save(new_image_path)
    else:
        # Copy the original image to the new location
        relative_path = os.path.relpath(image_path, root_dir)
        new_image_path = os.path.join(new_root_dir, relative_path)

        # Create directories if they don't exist
        new_image_dir = os.path.dirname(new_image_path)
        os.makedirs(new_image_dir, exist_ok=True)

        shutil.copy2(image_path, new_image_path)

def main():
    # Collect all image paths
    image_paths = []
    for directory in directories:
        if os.path.exists(directory):
            for root_dirpath, dirs, files in os.walk(directory):
                for file in files:
                    # if file.lower().endswith('.jpg'):
                    image_paths.append(os.path.join(root_dirpath, file))
        else:
            print(f"Directory does not exist: {directory}")
            # Create the directory
            os.makedirs(os.path.join(new_root_dir, os.path.relpath(directory, root_dir)), exist_ok=True)

    # Use ThreadPoolExecutor to process images in parallel
    with ThreadPoolExecutor() as executor:
        executor.map(process_image, image_paths)

if __name__ == '__main__':
    main()