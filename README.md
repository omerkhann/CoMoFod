# Copy-Move Forgery Detection (CMFD) Pipeline

A comprehensive Digital Image Processing (DIP) project for detecting copy-move forgeries in digital images. This system implements a complete pipeline using SIFT feature extraction combined with custom-coded manual algorithms for feature matching, robust geometric estimation (RANSAC), and morphological post-processing.

## Overview

Copy-Move Forgery is a specific type of image manipulation where a part of an image is copied and pasted into another part of the same image, typically to hide an object or duplicate a feature. This project provides a robust detection system with a professional dashboard interface built using Gradio.

## Importance & Applications

### The Challenge
Digital images are increasingly used as legal evidence, in news reporting, and for scientific documentation. However, with modern editing tools, **Copy-Move Forgery** has become one of the most common and difficult-to-detect manipulations. Since the cloned region comes from the same source, its statistical properties (noise, color, texture) match perfectly, making it nearly invisible to the human eye and traditional detection methods.

### Rationale: Why it is Needed
*   **Preserving Digital Trust**: In an era of misinformation, verifying the authenticity of visual media is critical for maintaining public trust.
*   **Legal Integrity**: Automated detection ensures that digital evidence presented in legal proceedings is tamper-free.
*   **Scientific Accuracy**: Prevents "image recycling" or fraudulent data duplication in research publications.

### Key Domains
*   **Forensic Investigation**: Verifying courtroom evidence and criminal records.
*   **Journalism & Media**: Fact-checking and verifying the integrity of news photography.
*   **Medical Imaging**: Ensuring biological scans haven't been altered to misrepresent medical results.
*   **Insurance & Finance**: Detecting fraudulent claims involving manipulated property or accident photos.

## Key Features

- **Preprocessing**: Grayscale conversion using the luminance formula and Gaussian smoothing for noise reduction.
- **Feature Extraction**: SIFT (Scale-Invariant Feature Transform) for detecting robust keypoints and descriptors.
- **Custom Matcher**: Custom-coded pairwise Euclidean distance computation with Lowe's Ratio Test and symmetric cross-checking.
- **Sequential RANSAC**: Sequential RANSAC implementation using Direct Linear Transform (DLT) and SVD to detect multiple forgery clusters within a single image.
- **Custom Post-processing**: Implementation of Dilation and Closing operations to refine detection masks.
- **Statistical Analysis**: Calculation of Intersection over Union (IoU) and DICE Coefficient (F1 Score) against ground truth data.

## Dataset

The project is designed to work with the CoMoFoD dataset. A cleaned version (consists of 40 translated, 40 scaled, and 40 rotated images) of the dataset used for this project is available on Kaggle:

[CoMoFoD Cleaned Dataset](https://www.kaggle.com/datasets/muhammadomerkhan03/comofod-cleaned-dataset)

## Technical Implementation

This project prioritizes technical rigor by implementing core image processing algorithms manually using NumPy vectorization rather than relying solely on high-level library functions:

- **Distance Matrix**: Computed using algebraic expansion to maintain performance without using standard library matchers.
- **Spatial Filtering**: Integrated directly into the nearest neighbor search to prevent overlapping keypoints from compromising the ratio test.
- **Homography Estimation**: Manual implementation of the DLT algorithm using Singular Value Decomposition (SVD).
- **Segmentation**: Custom morphological operations using shift-and-accumulate techniques for binary mask refinement.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/omerkhann/CoMoFod.git
   cd CoMoFod
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset and place it in the `data/` directory (optional for dashboard browsing).

## Usage

Run the Gradio dashboard:

```bash
python app.py
```

The interface allows you to:
- Select images from the integrated CoMoFoD browser.
- Upload custom suspect images for analysis.
- Adjust algorithm parameters (Lowe's Ratio, RANSAC threshold) in real-time.
- View detailed metrics and detection masks.

## License

This project was developed for academic purposes as part of a 6th-semester Computer Science curriculum.
