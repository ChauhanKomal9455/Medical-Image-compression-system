from __future__ import annotations

import os

import cv2
import numpy as np

from compression import ensure_directories


def generate_xray_like(size: int = 512) -> np.ndarray:
    image = np.full((size, size), 25, dtype=np.uint8)
    center = size // 2
    cv2.ellipse(image, (center, center), (150, 210), 0, 0, 360, 120, -1)
    cv2.circle(image, (center - 70, center - 20), 45, 190, -1)
    cv2.circle(image, (center + 70, center - 20), 45, 190, -1)
    cv2.rectangle(image, (center - 25, center + 10), (center + 25, center + 170), 180, -1)
    cv2.line(image, (center - 60, center + 210), (center - 20, center + 340), 160, 12)
    cv2.line(image, (center + 60, center + 210), (center + 20, center + 340), 160, 12)
    return cv2.GaussianBlur(image, (9, 9), 0)


def generate_mri_like(size: int = 512) -> np.ndarray:
    y, x = np.ogrid[:size, :size]
    center = size / 2.0
    radius = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    brain = np.exp(-(radius**2) / (2 * (size / 4.5) ** 2)) * 180
    ventricle = np.exp(-(((x - center) ** 2) / (2 * 40**2) + ((y - center) ** 2) / (2 * 22**2))) * 60
    texture = np.random.default_rng(42).normal(0, 10, (size, size))
    image = brain - ventricle + texture + 35
    return np.clip(image, 0, 255).astype(np.uint8)


def generate_ct_like(size: int = 512) -> np.ndarray:
    image = np.full((size, size), 15, dtype=np.float32)
    center = size // 2
    cv2.circle(image, (center, center), 170, 80, -1)
    cv2.circle(image, (center, center), 140, 110, -1)
    cv2.circle(image, (center - 40, center - 25), 32, 160, -1)
    cv2.circle(image, (center + 55, center - 20), 28, 145, -1)
    cv2.rectangle(image, (center - 18, center + 30), (center + 18, center + 95), 150, -1)
    image += np.random.default_rng(7).normal(0, 8, image.shape)
    return np.clip(cv2.GaussianBlur(image, (7, 7), 0), 0, 255).astype(np.uint8)


def save_samples() -> None:
    ensure_directories()
    output_dir = os.path.join("data", "samples")

    cv2.imwrite(os.path.join(output_dir, "sample_xray.png"), generate_xray_like())
    cv2.imwrite(os.path.join(output_dir, "sample_mri.png"), generate_mri_like())
    cv2.imwrite(os.path.join(output_dir, "sample_ct.png"), generate_ct_like())

    print("Sample images created in data/samples/")


if __name__ == "__main__":
    save_samples()
