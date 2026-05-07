# Project Context: Copy-Move Forgery Detection (Forensic Imaging)

## 1. Project Overview
- **Goal:** Detect copy-move forgeries (cloned regions) within a single image.
- **Academic Context:** 6th-semester Computer Science Digital Image Processing (DIP) project.
- **Core Strategy:** Use SIFT for feature extraction, but implement the "heavy logic" (Matching, RANSAC, Post-processing) manually using NumPy to demonstrate technical rigor.

## 2. Technical Pipeline (The "Manual" Flex)
The project is divided into the following stages:
1.  **Preprocessing:** Manual Grayscale conversion (Luminance formula).
2.  **Feature Extraction:** Use `cv2.SIFT_create()` to detect keypoints and descriptors.
3.  **Manual Matching:** - Implement Euclidean Distance ($L_2$ norm) using NumPy vectorization.
    - Implement **Lowe's Ratio Test** ($dist_1 / dist_2 < 0.7$) to filter ambiguous features.
4.  **Manual RANSAC:**
    - Iterate through random subsets of matches.
    - Calculate Homography matrix manually/via SVD.
    - Identify inliers to filter out geometric outliers (false matches).
5.  **Post-Processing:**
    - Generate a Binary Mask based on inlier coordinates.
    - **Manual Morphological Ops:** Implement Dilation and Closing from scratch to fill the detection mask.
6.  **Validation:** Calculate **IoU (Intersection over Union)** by comparing predicted masks against Ground Truth (`_B.png` files).

## 3. Dataset Information
- **Source:** CoMoFoD_small_v2 (Cleaned/Sniped).
- **Scope:** 120 base images (40 Translation, 40 Rotation, 40 Scaling).
- **Structure:** - `_F`: Forged Image (Input).
    - `_B`: Binary Mask (Ground Truth).
    - `_O`: Original Image (Reference).
- **Hosting:** Dataset is sniped and hosted as a private Kaggle dataset.

## 4. Tech Stack & Environment
- **Language:** Python 3.x
- **Libraries:** NumPy (Primary), OpenCV (Minimal usage for SIFT/IO), Matplotlib.
- **Environment:** VS Code for local development (logic) and Kaggle Notebooks for execution/demo.
- **Frontend:** Gradio (Targeted for interactive demo).

## 5. Development Roadmap
- [x] Data Sniping & Kaggle Dataset Setup.
- [ ] Manual Euclidean Matcher & Ratio Test.
- [ ] Manual RANSAC implementation.
- [ ] Manual Morphological functions (Dilation/Closing).
- [ ] Gradio UI Integration (Dataset selector + Image Upload).

## 6. Coding Agent Instructions
- Focus on **NumPy vectorization** rather than nested `for` loops for performance.
- Prioritize "clean" math explanations in comments.
- Ensure all logic can run within a Kaggle Notebook environment.
- Use **Gradio** for the UI to allow both dataset selection and custom image uploads.
