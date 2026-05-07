"""
Gradio Application — Copy-Move Forgery Detection Dashboard.

Provides an interactive UI with:
  • Dropdown to select images from the local /data directory
    (organised by forgery type: Translation / Rotation / Scaling).
  • File uploader for custom "wild" images.
  • Sliders for Lowe's Ratio and RANSAC pixel threshold.
  • Outputs: Original image, Match visualisation, Binary mask, IoU score.

Launch:  python app.py
"""

import os
import glob
import numpy as np
import cv2
import gradio as gr

from preprocessing   import preprocess
from matching        import extract_sift_features, match_keypoints, draw_matches
from ransac          import ransac_homography
from postprocessing  import postprocess

# ──────────────────────────────────────────────────────────────────────
# DATA CATALOGUE
# ──────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _build_catalogue():
    """
    Scan the /data folder and return a dict mapping display names to
    file-path info.  Images 001-040 → Translation, 041-080 → Rotation,
    081-120 → Scaling.
    """
    catalogue = {}
    forged_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_F.png")))

    for fpath in forged_files:
        basename = os.path.basename(fpath)           # e.g. "023_F.png"
        img_id   = int(basename.split("_")[0])       # 23

        if   1  <= img_id <= 40:  category = "Translation"
        elif 41 <= img_id <= 80:  category = "Rotation"
        else:                     category = "Scaling"

        label = f"{category}  —  {basename}"
        catalogue[label] = {
            "forged": fpath,
            "gt":     fpath.replace("_F.png", "_B.png"),
            "orig":   fpath.replace("_F.png", "_O.png"),
        }

    return catalogue


CATALOGUE = _build_catalogue()


# ──────────────────────────────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────────────────────────────

def run_pipeline(image_bgr, gt_mask_path, ratio, ransac_thresh):
    """Run the full CMFD pipeline on a single BGR image."""
    # 1. Preprocess
    gray = preprocess(image_bgr, smooth=True, kernel_size=3, sigma=1.0)

    # 2. SIFT features
    keypoints, descriptors = extract_sift_features(gray, n_features=5000)
    if descriptors is None or len(descriptors) < 10:
        h, w = image_bgr.shape[:2]
        return image_bgr, np.zeros((h, w), dtype=np.uint8), "No features detected."

    # 3. Manual matching (ratio test + spatial filter)
    matches = match_keypoints(keypoints, descriptors, ratio=ratio, min_spatial_dist=10.0)
    if len(matches) < 4:
        h, w = image_bgr.shape[:2]
        return draw_matches(image_bgr, keypoints, matches), \
               np.zeros((h, w), dtype=np.uint8), \
               f"Only {len(matches)} matches — need ≥ 4 for RANSAC."

    # 4. Manual RANSAC
    H, inliers = ransac_homography(
        matches, keypoints, threshold=ransac_thresh, max_iterations=2000
    )

    # 5. Visualise matches (inliers only if available, else raw matches)
    vis_matches = draw_matches(image_bgr, keypoints, inliers if inliers else matches)

    # 6. Post-process → mask + IoU
    gt_mask = None
    if gt_mask_path and os.path.isfile(gt_mask_path):
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

    mask, iou = postprocess(
        image_bgr.shape[:2], keypoints, inliers,
        gt_mask=gt_mask,
        circle_radius=8, dilate_size=7, close_size=15,
    )

    # Build info string
    if iou is not None:
        info = f"Matches: {len(matches)}  |  Inliers: {len(inliers)}  |  IoU: {iou:.4f}"
    else:
        info = f"Matches: {len(matches)}  |  Inliers: {len(inliers)}  |  IoU: N/A (no ground truth)"

    return vis_matches, mask, info


# ──────────────────────────────────────────────────────────────────────
# GRADIO CALLBACKS
# ──────────────────────────────────────────────────────────────────────

def on_dataset_image(selection, ratio, ransac_thresh):
    """Callback when user picks an image from the dropdown."""
    if selection is None or selection not in CATALOGUE:
        return None, None, None, "Select an image from the dropdown."

    entry = CATALOGUE[selection]
    image_bgr = cv2.imread(entry["forged"])
    if image_bgr is None:
        return None, None, None, f"Could not load {entry['forged']}"

    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    vis, mask, info = run_pipeline(image_bgr, entry["gt"], ratio, ransac_thresh)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    return original_rgb, vis_rgb, mask, info


def on_upload_image(upload, ratio, ransac_thresh):
    """Callback when user uploads a custom image."""
    if upload is None:
        return None, None, None, "Upload an image to analyse."

    # Gradio passes a numpy RGB array for gr.Image(type="numpy")
    image_rgb = upload
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    vis, mask, info = run_pipeline(image_bgr, None, ratio, ransac_thresh)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    return image_rgb, vis_rgb, mask, info


# ──────────────────────────────────────────────────────────────────────
# GRADIO UI
# ──────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Dark premium theme overrides ──────────────────────────────────── */
.gradio-container {
    max-width: 1280px !important;
    margin: auto;
}
.title-banner {
    text-align: center;
    padding: 16px 0 4px 0;
}
.title-banner h1 {
    font-size: 1.8rem;
    background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    margin-bottom: 2px;
}
.title-banner p {
    opacity: 0.7;
    font-size: 0.95rem;
}
.param-panel {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px;
}
"""

def build_ui():
    dropdown_choices = list(CATALOGUE.keys())

    with gr.Blocks() as demo:

        # ── Header ─────────────────────────────────────────────────
        gr.HTML("""
        <div class="title-banner">
            <h1>🔍 Copy-Move Forgery Detection</h1>
            <p>SIFT · Manual Matcher · Manual RANSAC · Manual Morphology — NumPy-only pipeline</p>
        </div>
        """)

        with gr.Row():
            # ── Left: Controls ─────────────────────────────────────
            with gr.Column(scale=1, elem_classes="param-panel"):
                gr.Markdown("### 📂 Dataset Image")
                dropdown = gr.Dropdown(
                    choices=dropdown_choices,
                    label="Select from CoMoFoD dataset",
                    info="Translation (001-040) · Rotation (041-080) · Scaling (081-120)",
                )
                btn_dataset = gr.Button("▶  Analyse Dataset Image", variant="primary")

                gr.Markdown("---")
                gr.Markdown("### 📤 Upload Custom Image")
                upload = gr.Image(label="Upload a suspect image", type="numpy")
                btn_upload = gr.Button("▶  Analyse Uploaded Image", variant="primary")

                gr.Markdown("---")
                gr.Markdown("### ⚙️ Parameters")
                ratio_slider = gr.Slider(
                    minimum=0.4, maximum=0.9, value=0.7, step=0.05,
                    label="Lowe's Ratio Threshold",
                    info="Lower = stricter matching (fewer but better matches)",
                )
                ransac_slider = gr.Slider(
                    minimum=1.0, maximum=20.0, value=5.0, step=0.5,
                    label="RANSAC Inlier Threshold (px)",
                    info="Max reprojection error to count as inlier",
                )

            # ── Right: Outputs ─────────────────────────────────────
            with gr.Column(scale=2):
                info_box = gr.Textbox(label="📊 Results", lines=1, interactive=False)

                with gr.Row():
                    out_original = gr.Image(label="Original / Forged Image", type="numpy")
                    out_matches  = gr.Image(label="Detected Matches (inliers)", type="numpy")

                with gr.Row():
                    out_mask = gr.Image(label="Binary Detection Mask", type="numpy")

        # ── Wire callbacks ─────────────────────────────────────────
        btn_dataset.click(
            fn=on_dataset_image,
            inputs=[dropdown, ratio_slider, ransac_slider],
            outputs=[out_original, out_matches, out_mask, info_box],
        )
        btn_upload.click(
            fn=on_upload_image,
            inputs=[upload, ratio_slider, ransac_slider],
            outputs=[out_original, out_matches, out_mask, info_box],
        )

    return demo


# ──────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        share=False,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
        ),
    )
