# Medical Image Compression System Using Wavelet-Based and JPEG-LS Techniques with Distortion Analysis

## 1. Problem Statement

Medical images such as X-rays, MRI scans, and CT scans are often high in resolution and file size. These large files consume significant storage and are slow to transmit across low-bandwidth networks. In rural or resource-constrained hospitals, this delay can make it difficult to share diagnostic images with specialists in time. A medical image compression system is therefore needed to reduce image size while preserving diagnostic quality.

## 2. Goal of the Project

This project aims to:

- reduce the size of medical images,
- preserve visual and diagnostic quality,
- improve transmission efficiency,
- compare different compression techniques, and
- provide a working prototype that can be used for experimentation and demonstration.

## 3. Project Overview

This project implements two compression approaches:

1. Wavelet-based compression using discrete wavelet transform and coefficient thresholding.
2. JPEG-LS-inspired near-lossless compression using predictive coding and residual quantization.

The system supports PNG, JPG, and DICOM images and includes:

- an input and preprocessing module,
- compression and decompression modules,
- quality analysis metrics,
- a comparison workflow,
- a Streamlit web interface, and
- a command-line entry point.

## 4. Project Architecture

### Workflow

1. Load medical image from PNG, JPG, or DICOM.
2. Convert to grayscale if needed.
3. Normalize and resize if requested.
4. Apply selected compression method.
5. Reconstruct the decompressed image.
6. Compute compression ratio and distortion metrics.
7. Display visual comparison and results.

### Modules

- `main.py`: command-line demo runner
- `compression.py`: wavelet and JPEG-LS-inspired compression logic
- `metrics.py`: PSNR, MSE, SSIM, compression ratio
- `app.py`: Streamlit user interface
- `sample_data.py`: creates synthetic test images if real medical images are not available

## 5. Folder Structure

```text
MedImage/
|-- app.py
|-- compression.py
|-- main.py
|-- metrics.py
|-- sample_data.py
|-- requirements.txt
|-- README.md
|-- data/
|   |-- input/
|   |-- output/
|   |-- samples/
```

## 6. Compression Techniques Used

### A. Wavelet-Based Compression

This method:

- applies discrete wavelet transform (DWT),
- splits the image into approximation and detail coefficients,
- removes or shrinks small coefficients using thresholding,
- stores the reduced coefficients, and
- reconstructs the image using inverse wavelet transform.

This reduces data size while preserving major image features.

### B. JPEG-LS-Inspired Near-Lossless Compression

JPEG-LS is based on predictive coding. In this project, a simplified educational version is used:

- predict each pixel from its neighbors,
- calculate residual error,
- quantize residuals for near-lossless compression,
- compress the residual map using `zlib`,
- reconstruct by reversing the process.

This is not a full standards-compliant JPEG-LS encoder, but it closely demonstrates the idea of prediction-based near-lossless compression.

## 7. Metrics Used

- **Compression Ratio (CR)**: original size divided by compressed size
- **MSE**: mean squared error
- **PSNR**: peak signal-to-noise ratio
- **SSIM**: structural similarity index

Higher PSNR and SSIM indicate better quality. Lower MSE is better. Higher compression ratio means smaller compressed output.

## 8. Installation

Create a virtual environment if you want:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## 9. Generate Sample Images

If you do not have real medical images yet, generate sample images:

```powershell
python sample_data.py
```

This creates synthetic X-ray, MRI, and CT-like grayscale images in `data/samples/`.

## 10. Run the Command-Line Demo

Example with wavelet compression:

```powershell
python main.py --input data\samples\sample_xray.png --method wavelet
```

Example with JPEG-LS-inspired compression:

```powershell
python main.py --input data\samples\sample_mri.png --method jpegls --near 2
```

Optional resize:

```powershell
python main.py --input data\samples\sample_ct.png --method both --width 512 --height 512
```

## 11. Run the Streamlit App

```powershell
streamlit run app.py
```

Features:

- upload image,
- choose compression method,
- set compression parameters,
- compare original and reconstructed images,
- view compression metrics.

## 12. How to Test Each Module

### Input Module

- Test PNG/JPG loading using files in `data/samples/`.
- Test DICOM loading using your own `.dcm` file if available.
- Confirm grayscale output and optional resizing.

### Wavelet Compression Module

- Run `main.py` with `--method wavelet`.
- Check reconstructed image quality and saved output.

### JPEG-LS-Inspired Module

- Run `main.py` with `--method jpegls`.
- Change the `--near` value to observe the quality/compression trade-off.

### Metrics Module

- Confirm that MSE, PSNR, SSIM, and compression ratio are displayed.
- Compare results between methods.

### Streamlit Frontend

- Upload an image.
- Select a method.
- Click the button to compress.
- Verify displayed images and metrics.

## 13. Beginner Notes

- Start with PNG sample files before trying DICOM.
- Use `wavelet` first because it is easier to understand visually.
- Then try `jpegls` with `near=0`, `near=1`, and `near=2`.
- Compare the metrics to understand the trade-off between compression and image quality.

## 14. Expected Learning Outcomes

By completing this project, you will learn:

- medical image handling in Python,
- grayscale preprocessing,
- wavelet transform basics,
- predictive compression ideas,
- distortion analysis using PSNR, MSE, and SSIM,
- Streamlit dashboard creation,
- and how to structure an academic image-processing project end to end.
