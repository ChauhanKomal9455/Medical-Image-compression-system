from __future__ import annotations

import io
import os
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pywt
import pydicom


@dataclass
class CompressionResult:
    method: str
    original_image: np.ndarray
    reconstructed_image: np.ndarray
    compressed_payload: bytes
    metadata: Dict[str, Any]


def ensure_directories() -> None:
    for path in ("data", "data/input", "data/output", "data/samples"):
        os.makedirs(path, exist_ok=True)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32)
    min_val = float(image.min())
    max_val = float(image.max())
    if max_val - min_val < 1e-8:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image - min_val) / (max_val - min_val)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def load_medical_image(path: str, resize_to: Optional[Tuple[int, int]] = None) -> np.ndarray:
    extension = os.path.splitext(path)[1].lower()

    if extension == ".dcm":
        dicom = pydicom.dcmread(path)
        image = dicom.pixel_array
        image = normalize_to_uint8(image)
    else:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {path}")
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = normalize_to_uint8(image)

    if resize_to is not None:
        width, height = resize_to
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    return image


def save_image(path: str, image: np.ndarray) -> None:
    ensure_directories()
    cv2.imwrite(path, image)


def _serialize_numpy_arrays(arrays: Dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _deserialize_numpy_arrays(payload: bytes) -> Dict[str, Any]:
    buffer = io.BytesIO(payload)
    with np.load(buffer, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def compress_wavelet(
    image: np.ndarray,
    wavelet: str = "haar",
    level: int = 2,
    threshold_ratio: float = 0.08,
    mode: str = "soft",
) -> CompressionResult:
    image = normalize_to_uint8(image)
    image_float = image.astype(np.float32)

    coeffs = pywt.wavedec2(image_float, wavelet=wavelet, level=level)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

    threshold = float(np.max(np.abs(coeff_arr)) * threshold_ratio)
    thresholded = pywt.threshold(coeff_arr, threshold, mode=mode)

    compressed_payload = _serialize_numpy_arrays(
        {
            "coeff_arr": thresholded.astype(np.float32),
            "shape_meta": np.array(image.shape, dtype=np.int32),
            "coeff_slices": np.array(coeff_slices, dtype=object),
            "wavelet": np.array(wavelet),
        }
    )

    reconstructed = decompress_wavelet(compressed_payload)

    return CompressionResult(
        method="wavelet",
        original_image=image,
        reconstructed_image=reconstructed,
        compressed_payload=compressed_payload,
        metadata={
            "wavelet": wavelet,
            "level": level,
            "threshold_ratio": threshold_ratio,
            "threshold_mode": mode,
            "threshold_value": threshold,
        },
    )


def decompress_wavelet(payload: bytes) -> np.ndarray:
    arrays = _deserialize_numpy_arrays(payload)
    coeff_arr = arrays["coeff_arr"]
    coeff_slices = arrays["coeff_slices"].tolist()
    wavelet = str(arrays["wavelet"])
    original_shape = tuple(int(v) for v in arrays["shape_meta"])

    coeffs = pywt.array_to_coeffs(coeff_arr, coeff_slices, output_format="wavedec2")
    reconstructed = pywt.waverec2(coeffs, wavelet=wavelet)
    reconstructed = reconstructed[: original_shape[0], : original_shape[1]]
    return np.clip(reconstructed, 0, 255).astype(np.uint8)


def _predictor(left: int, top: int, top_left: int) -> int:
    if top_left >= max(left, top):
        return min(left, top)
    if top_left <= min(left, top):
        return max(left, top)
    return left + top - top_left


def compress_jpegls_style(image: np.ndarray, near: int = 0) -> CompressionResult:
    image = normalize_to_uint8(image)
    height, width = image.shape

    quantized_residuals = np.zeros((height, width), dtype=np.int16)
    reconstructed = np.zeros((height, width), dtype=np.uint8)

    for row in range(height):
        for col in range(width):
            left = int(reconstructed[row, col - 1]) if col > 0 else 0
            top = int(reconstructed[row - 1, col]) if row > 0 else 0
            top_left = int(reconstructed[row - 1, col - 1]) if row > 0 and col > 0 else 0

            prediction = _predictor(left, top, top_left)
            residual = int(image[row, col]) - prediction

            if near > 0:
                q_residual = int(np.round(residual / float(near + 1)))
                reconstructed_value = prediction + q_residual * (near + 1)
            else:
                q_residual = residual
                reconstructed_value = prediction + q_residual

            reconstructed[row, col] = np.uint8(np.clip(reconstructed_value, 0, 255))
            quantized_residuals[row, col] = q_residual

    raw_payload = b"".join(
        [
            height.to_bytes(4, "little"),
            width.to_bytes(4, "little"),
            near.to_bytes(2, "little"),
            quantized_residuals.astype(np.int16).tobytes(),
        ]
    )
    compressed_payload = zlib.compress(raw_payload, level=9)

    return CompressionResult(
        method="jpegls_style",
        original_image=image,
        reconstructed_image=reconstructed,
        compressed_payload=compressed_payload,
        metadata={
            "near": near,
            "height": height,
            "width": width,
        },
    )


def decompress_jpegls_style(payload: bytes) -> np.ndarray:
    raw_payload = zlib.decompress(payload)
    height = int.from_bytes(raw_payload[0:4], "little")
    width = int.from_bytes(raw_payload[4:8], "little")
    near = int.from_bytes(raw_payload[8:10], "little")
    residual_bytes = raw_payload[10:]

    residuals = np.frombuffer(residual_bytes, dtype=np.int16).reshape((height, width))
    reconstructed = np.zeros((height, width), dtype=np.uint8)

    for row in range(height):
        for col in range(width):
            left = int(reconstructed[row, col - 1]) if col > 0 else 0
            top = int(reconstructed[row - 1, col]) if row > 0 else 0
            top_left = int(reconstructed[row - 1, col - 1]) if row > 0 and col > 0 else 0

            prediction = _predictor(left, top, top_left)
            if near > 0:
                value = prediction + int(residuals[row, col]) * (near + 1)
            else:
                value = prediction + int(residuals[row, col])
            reconstructed[row, col] = np.uint8(np.clip(value, 0, 255))

    return reconstructed


def run_compression_pipeline(
    image_path: str,
    method: str,
    resize_to: Optional[Tuple[int, int]] = None,
    wavelet: str = "haar",
    level: int = 2,
    threshold_ratio: float = 0.08,
    near: int = 0,
) -> Dict[str, CompressionResult]:
    ensure_directories()
    image = load_medical_image(image_path, resize_to=resize_to)

    results: Dict[str, CompressionResult] = {}

    if method in {"wavelet", "both"}:
        results["wavelet"] = compress_wavelet(
            image=image,
            wavelet=wavelet,
            level=level,
            threshold_ratio=threshold_ratio,
        )

    if method in {"jpegls", "both"}:
        results["jpegls_style"] = compress_jpegls_style(image=image, near=near)

    return results
