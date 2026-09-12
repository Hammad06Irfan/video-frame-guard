import os
import random
import cv2
import numpy as np
from tqdm import tqdm

RAW_DIR = "data/raw"
BASE_DATASET_DIR = "data/dataset"
SPLITS = {"train": 0.8, "val": 0.2}
IMG_SIZE = (224, 224)


def create_local_base_images(num_images=200):
    """Generates 200 procedural base images locally to bypass slow server downloads."""
    os.makedirs(RAW_DIR, exist_ok=True)
    existing_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".png")]
    if len(existing_files) >= num_images:
        print(f"[*] Found {len(existing_files)} raw images already in {RAW_DIR}.")
        return

    print(f"[*] Generating {num_images} local base frames (fast offline mode)...")
    np.random.seed(42)
    random.seed(42)

    for i in range(num_images):
        # Create varied gradients and color backgrounds
        img = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
        color1 = np.random.randint(40, 220, size=3)
        color2 = np.random.randint(40, 220, size=3)
        for y in range(IMG_SIZE[0]):
            alpha = y / IMG_SIZE[0]
            img[y, :] = (1 - alpha) * color1 + alpha * color2

        # Draw procedural shapes to mimic natural scenes/textures
        for _ in range(random.randint(3, 8)):
            shape_type = random.choice(["circle", "rect", "line"])
            col = [int(x) for x in np.random.randint(0, 256, size=3)]
            pt1 = (random.randint(0, IMG_SIZE[1]), random.randint(0, IMG_SIZE[0]))
            pt2 = (random.randint(0, IMG_SIZE[1]), random.randint(0, IMG_SIZE[0]))

            if shape_type == "circle":
                radius = random.randint(15, 60)
                cv2.circle(img, pt1, radius, col, -1)
            elif shape_type == "rect":
                cv2.rectangle(img, pt1, pt2, col, -1)
            elif shape_type == "line":
                cv2.line(img, pt1, pt2, col, random.randint(2, 6))

        cv2.imwrite(os.path.join(RAW_DIR, f"frame_{i:04d}.png"), img)

    print(f"[+] Successfully generated {num_images} raw base frames in {RAW_DIR}")


def apply_tearing(img: np.ndarray) -> np.ndarray:
    """Simulates GPU/V-Sync frame tearing by horizontally displacing a slice."""
    corrupted = img.copy()
    h, w, _ = corrupted.shape
    slice_start = random.randint(int(h * 0.2), int(h * 0.7))
    slice_height = random.randint(30, 80)
    slice_end = min(slice_start + slice_height, h)

    shift = random.choice([random.randint(-40, -15), random.randint(15, 40)])
    torn_slice = np.roll(corrupted[slice_start:slice_end, :, :], shift, axis=1)
    corrupted[slice_start:slice_end, :, :] = torn_slice
    return corrupted


def apply_macroblocking(img: np.ndarray) -> np.ndarray:
    """Simulates video compression/decompression macroblock artifacts."""
    h, w, _ = img.shape
    block_h, block_w = (random.randint(16, 28), random.randint(16, 28))
    low_res = cv2.resize(img, (block_w, block_h), interpolation=cv2.INTER_NEAREST)
    blocky = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_NEAREST)
    alpha = random.uniform(0.7, 0.9)
    return cv2.addWeighted(blocky, alpha, img, 1 - alpha, 0)


def apply_bit_noise(img: np.ndarray) -> np.ndarray:
    """Simulates memory buffer bit-flips / PCIe transmission noise."""
    corrupted = img.copy()
    h, w, c = corrupted.shape
    num_salt = int(h * w * random.uniform(0.015, 0.04))

    for _ in range(num_salt):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        corrupted[y, x] = [
            random.choice([0, 255]),
            random.choice([0, 255]),
            random.choice([0, 255]),
        ]
    return corrupted


def generate_dataset():
    create_local_base_images(num_images=200)

    raw_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".png")])
    random.seed(1337)
    random.shuffle(raw_files)

    split_idx = int(len(raw_files) * SPLITS["train"])
    train_files = raw_files[:split_idx]
    val_files = raw_files[split_idx:]

    dataset_splits = [("train", train_files), ("val", val_files)]

    print("[*] Generating corrupted variants across train and val splits...")
    for split_name, files in dataset_splits:
        for fname in tqdm(files, desc=f"Building {split_name} set"):
            raw_path = os.path.join(RAW_DIR, fname)
            img = cv2.imread(raw_path)
            if img is None:
                continue

            base_name = os.path.splitext(fname)[0]

            # 1. Clean
            cv2.imwrite(
                os.path.join(BASE_DATASET_DIR, split_name, "clean", f"{base_name}.png"),
                img,
            )

            # 2. Torn
            torn_img = apply_tearing(img)
            cv2.imwrite(
                os.path.join(BASE_DATASET_DIR, split_name, "torn", f"{base_name}_torn.png"),
                torn_img,
            )

            # 3. Blocky
            blocky_img = apply_macroblocking(img)
            cv2.imwrite(
                os.path.join(BASE_DATASET_DIR, split_name, "blocky", f"{base_name}_blocky.png"),
                blocky_img,
            )

            # 4. Noise
            noise_img = apply_bit_noise(img)
            cv2.imwrite(
                os.path.join(BASE_DATASET_DIR, split_name, "noise", f"{base_name}_noise.png"),
                noise_img,
            )

    print("\n[+] Dataset generation complete! Check data/dataset/ for the 4 classes.")


if __name__ == "__main__":
    generate_dataset()