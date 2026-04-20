from __future__ import annotations

from typing import Dict

import numpy as np


def mean_squared_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)
    return float(np.mean((original - reconstructed) ** 2))


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    mse = mean_squared_error(original, reconstructed)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    return float(20 * np.log10(max_pixel / np.sqrt(mse)))


def ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_x = original.mean()
    mu_y = reconstructed.mean()
    sigma_x = original.var()
    sigma_y = reconstructed.var()
    sigma_xy = ((original - mu_x) * (reconstructed - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    if denominator == 0:
        return 1.0
    return float(numerator / denominator)


def compression_ratio(original: np.ndarray, compressed_payload: bytes) -> float:
    original_size = int(original.size * original.itemsize)
    compressed_size = len(compressed_payload)
    if compressed_size == 0:
        return float("inf")
    return float(original_size / compressed_size)


def evaluate_metrics(original: np.ndarray, reconstructed: np.ndarray, compressed_payload: bytes) -> Dict[str, float]:
    return {
        "compression_ratio": compression_ratio(original, compressed_payload),
        "mse": mean_squared_error(original, reconstructed),
        "psnr": psnr(original, reconstructed),
        "ssim": ssim(original, reconstructed),
    }
