import os
import shutil
import random

SOURCE_DIR = "dataset_original/train"
DEST_DIR = "dataset"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CLASSES = ["signature", "not_signature"]

for split in ["train", "val", "test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)

def split_and_copy_images():
    for cls in CLASSES:
        src_folder = os.path.join(SOURCE_DIR, cls)
        images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        random.shuffle(images)
        total = len(images)

        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split, imgs in splits.items():
            for i, img in enumerate(imgs, start=1):
                new_name = f"{cls}_{i:03d}{os.path.splitext(img)[1]}"
                src_path = os.path.join(src_folder, img)
                dest_path = os.path.join(DEST_DIR, split, cls, new_name)
                shutil.copy(src_path, dest_path)
            print(f"{cls} - {split}: {len(imgs)} images copied.")


split_and_copy_images()
print("Dataset splitting and renaming complete!")
