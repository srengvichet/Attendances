from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import random

# === PIL-compatible augmentation (no ToTensor or Normalize)
augment = transforms.Compose([
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.4, hue=0.08),
    ], p=0.8),

    transforms.RandomApply([
        transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.95, 1.05), shear=5)
    ], p=0.7),

    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.3, 1.0))
    ], p=0.4),

    transforms.RandomApply([
        transforms.RandomAdjustSharpness(sharpness_factor=2)
    ], p=0.3),

    transforms.RandomHorizontalFlip(p=0.5),

    # Optional: add some edge cases, but keep low p
    transforms.RandomApply([
        transforms.Grayscale(num_output_channels=3)
    ], p=0.1),

    transforms.Resize((160, 160))
])


# === Paths ===
src_root = Path("datasets_train")
dst_root = Path("datasets_train_aug")
dst_root.mkdir(exist_ok=True)

NUM_VARIANTS = 5

for class_folder in src_root.iterdir():
    if not class_folder.is_dir():
        continue

    dst_class_folder = dst_root / class_folder.name
    dst_class_folder.mkdir(parents=True, exist_ok=True)

    image_files = list(class_folder.glob("*.jpg"))

    for idx, image_path in enumerate(image_files):
        try:
            img = Image.open(image_path).convert("RGB")

            for i in range(NUM_VARIANTS):
                aug_img = augment(img)
                aug_img.save(dst_class_folder / f"{idx}_{i}.jpg")

        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {e}")

