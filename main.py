from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

from compression import ensure_directories, run_compression_pipeline, save_image
from metrics import evaluate_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Medical Image Compression System using Wavelet and JPEG-LS-inspired techniques"
    )
    parser.add_argument("--input", required=True, help="Path to input PNG, JPG, or DICOM image")
    parser.add_argument(
        "--method",
        default="both",
        choices=["wavelet", "jpegls", "both"],
        help="Compression method to run",
    )
    parser.add_argument("--width", type=int, default=None, help="Optional resize width")
    parser.add_argument("--height", type=int, default=None, help="Optional resize height")
    parser.add_argument("--wavelet", default="haar", help="Wavelet family name")
    parser.add_argument("--level", type=int, default=2, help="Wavelet decomposition level")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.08,
        help="Wavelet threshold ratio between 0 and 1",
    )
    parser.add_argument(
        "--near",
        type=int,
        default=0,
        help="Near-lossless error bound for JPEG-LS-inspired compression",
    )
    return parser.parse_args()


def get_resize_tuple(width: Optional[int], height: Optional[int]) -> Optional[Tuple[int, int]]:
    if width is None or height is None:
        return None
    return width, height


def print_metrics_block(method_name: str, metrics: dict, output_path: str) -> None:
    print(f"\nMethod: {method_name}")
    print(f"Saved reconstructed image: {output_path}")
    print(f"Compression Ratio: {metrics['compression_ratio']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"PSNR: {metrics['psnr']:.4f}")
    print(f"SSIM: {metrics['ssim']:.4f}")


def main() -> None:
    args = parse_args()
    ensure_directories()

    resize_to = get_resize_tuple(args.width, args.height)
    results = run_compression_pipeline(
        image_path=args.input,
        method=args.method,
        resize_to=resize_to,
        wavelet=args.wavelet,
        level=args.level,
        threshold_ratio=args.threshold,
        near=args.near,
    )

    base_name = os.path.splitext(os.path.basename(args.input))[0]

    for method_name, result in results.items():
        output_path = os.path.join("data", "output", f"{base_name}_{method_name}_reconstructed.png")
        save_image(output_path, result.reconstructed_image)
        metrics = evaluate_metrics(
            original=result.original_image,
            reconstructed=result.reconstructed_image,
            compressed_payload=result.compressed_payload,
        )
        print_metrics_block(method_name, metrics, output_path)

    if len(results) == 2:
        ranked = []
        for method_name, result in results.items():
            metrics = evaluate_metrics(
                original=result.original_image,
                reconstructed=result.reconstructed_image,
                compressed_payload=result.compressed_payload,
            )
            score = metrics["psnr"] + (metrics["ssim"] * 100) + metrics["compression_ratio"]
            ranked.append((score, method_name, metrics))

        ranked.sort(reverse=True)
        best = ranked[0]
        print("\nComparison Summary")
        print(f"Best overall method for this image: {best[1]}")
        print(
            "This score favors high PSNR, high SSIM, and high compression ratio for an educational comparison."
        )


if __name__ == "__main__":
    main()
