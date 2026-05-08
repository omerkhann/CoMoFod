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
from ransac          import ransac_homography, sequential_ransac
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

    # 2. SIFT features — no cap, let SIFT find everything
    keypoints, descriptors = extract_sift_features(gray, n_features=0)
    if descriptors is None or len(descriptors) < 10:
        h, w = image_bgr.shape[:2]
        return image_bgr, np.zeros((h, w), dtype=np.uint8), "No features detected."

    # 3. Manual matching (ratio test + spatial filter)
    matches = match_keypoints(keypoints, descriptors, ratio=ratio, min_spatial_dist=10.0)
    if len(matches) < 4:
        h, w = image_bgr.shape[:2]
        return draw_matches(image_bgr, keypoints, matches), \
               np.zeros((h, w), dtype=np.uint8), \
               f"Only {len(matches)} matches found — need at least 4 for RANSAC."

    # 4. Sequential RANSAC — finds multiple copy-move clusters
    ransac_results = sequential_ransac(
        matches, keypoints,
        threshold=ransac_thresh,
        max_iterations=2000,
        min_inliers=6,
        max_models=5,
    )

    # Aggregate all inliers from every detected transformation
    all_inliers = []
    for H, inliers in ransac_results:
        all_inliers.extend(inliers)

    if not all_inliers:
        all_inliers = matches  # fall back to raw matches for visualization

    # 5. Visualise matches
    vis_matches = draw_matches(image_bgr, keypoints, all_inliers)

    # 6. Post-process → mask + metrics
    gt_mask = None
    if gt_mask_path and os.path.isfile(gt_mask_path):
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

    mask, iou, dice = postprocess(
        image_bgr.shape[:2], keypoints, all_inliers,
        gt_mask=gt_mask,
        circle_radius=6, dilate_size=11, close_size=21,
    )

    # Build styled HTML metric cards
    n_clusters = len(ransac_results)
    if iou is not None and dice is not None:
        info = f"""
        <div style="display: flex; gap: 15px; justify-content: space-between; background: var(--background-fill-secondary); padding: 20px; border-radius: 8px; border: 1px solid var(--border-color-primary);">
            <div style="text-align: center; flex: 1;">
                <p style="margin: 0; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; opacity: 0.6;">Matches</p>
                <h3 style="margin: 5px 0 0 0; font-size: 2rem; color: var(--body-text-color); font-weight: 700;">{len(matches)}</h3>
            </div>
            <div style="width: 1px; background: var(--border-color-primary);"></div>
            <div style="text-align: center; flex: 1;">
                <p style="margin: 0; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; opacity: 0.6;">Inliers</p>
                <h3 style="margin: 5px 0 0 0; font-size: 2rem; color: var(--body-text-color); font-weight: 700;">{len(all_inliers)}</h3>
            </div>
            <div style="width: 1px; background: var(--border-color-primary);"></div>
            <div style="text-align: center; flex: 1;">
                <p style="margin: 0; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; opacity: 0.6;">Clusters</p>
                <h3 style="margin: 5px 0 0 0; font-size: 2rem; color: var(--body-text-color); font-weight: 700;">{n_clusters}</h3>
            </div>
            <div style="width: 1px; background: var(--border-color-primary);"></div>
            <div style="text-align: center; flex: 1;">
                <p style="margin: 0; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; opacity: 0.6;">IoU</p>
                <h3 style="margin: 5px 0 0 0; font-size: 2rem; color: var(--body-text-color); font-weight: 700;">{iou:.4f}</h3>
            </div>
            <div style="width: 1px; background: var(--border-color-primary);"></div>
            <div style="text-align: center; flex: 1;">
                <p style="margin: 0; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; opacity: 0.6;">DICE</p>
                <h3 style="margin: 5px 0 0 0; font-size: 2rem; color: var(--body-text-color); font-weight: 700;">{dice:.4f}</h3>
            </div>
        </div>
        """
    else:
        info = f"""
        <div style="display: flex; gap: 15px; justify-content: space-between; background: var(--background-fill-secondary); padding: 20px; border-radius: 8px; border: 1px solid var(--border-color-primary);">
            <div style="text-align: center; flex: 1;">
                <p style="margin: 0; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; opacity: 0.6;">Matches</p>
                <h3 style="margin: 5px 0 0 0; font-size: 2rem; color: var(--body-text-color); font-weight: 700;">{len(matches)}</h3>
            </div>
            <div style="width: 1px; background: var(--border-color-primary);"></div>
            <div style="text-align: center; flex: 1;">
                <p style="margin: 0; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; opacity: 0.6;">Inliers</p>
                <h3 style="margin: 5px 0 0 0; font-size: 2rem; color: var(--body-text-color); font-weight: 700;">{len(all_inliers)}</h3>
            </div>
            <div style="width: 1px; background: var(--border-color-primary);"></div>
            <div style="text-align: center; flex: 1;">
                <p style="margin: 0; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; opacity: 0.6;">Clusters</p>
                <h3 style="margin: 5px 0 0 0; font-size: 2rem; color: var(--body-text-color); font-weight: 700;">{n_clusters}</h3>
            </div>
            <div style="width: 1px; background: var(--border-color-primary);"></div>
            <div style="text-align: center; flex: 2;">
                <p style="margin: 0; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; opacity: 0.6;">Ground Truth</p>
                <h3 style="margin: 5px 0 0 0; font-size: 1.2rem; color: var(--body-text-color); font-weight: 400; padding-top: 8px;">N/A</h3>
            </div>
        </div>
        """

    return vis_matches, mask, info


# ──────────────────────────────────────────────────────────────────────
# GRADIO CALLBACKS
# ──────────────────────────────────────────────────────────────────────

def on_dataset_image(selection, ratio, ransac_thresh):
    """Callback when user picks an image from the dropdown."""
    if selection is None or selection not in CATALOGUE:
        return None, None, None, None, "Select an image from the dropdown."

    entry = CATALOGUE[selection]
    image_bgr = cv2.imread(entry["forged"])
    if image_bgr is None:
        return None, None, None, None, f"Could not load {entry['forged']}"

    original_bgr = cv2.imread(entry["orig"])
    unforged_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB) if original_bgr is not None else None

    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    vis, mask, info = run_pipeline(image_bgr, entry["gt"], ratio, ransac_thresh)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    return unforged_rgb, original_rgb, vis_rgb, mask, info


def on_upload_image(upload, ratio, ransac_thresh):
    """Callback when user uploads a custom image."""
    if upload is None:
        return None, None, None, None, "Upload an image to analyse."

    # Gradio passes a numpy RGB array for gr.Image(type="numpy")
    image_rgb = upload
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    vis, mask, info = run_pipeline(image_bgr, None, ratio, ransac_thresh)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    return None, image_rgb, vis_rgb, mask, info


# ──────────────────────────────────────────────────────────────────────
# GRADIO UI
# ──────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Clean professional theme overrides ──────────────────────────────────── */
.gradio-container {
    max-width: 1600px !important;
    margin: auto;
}
.main-row {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: flex-start !important;
}
.title-banner {
    text-align: center;
    padding: 24px 0 12px 0;
}
.title-banner h1 {
    font-size: 2.2rem;
    color: var(--body-text-color);
    font-weight: 700;
    margin-bottom: 4px;
    letter-spacing: -0.02em;
}
.title-banner p {
    opacity: 0.6;
    font-size: 1.05rem;
    font-weight: 400;
}
.param-panel {
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    padding: 20px;
    background-color: var(--background-fill-secondary);
}
.output-image {
    min-height: 400px;
    background-color: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
}
"""

def build_ui():
    dropdown_choices = list(CATALOGUE.keys())

    with gr.Blocks() as demo:

        # ── Header ─────────────────────────────────────────────────
        gr.HTML("""
        <div class="title-banner">
            <h1>Copy-Move Forgery Detection</h1>
            <p>SIFT · Manual Matcher · Manual RANSAC · Manual Morphology</p>
        </div>
        """)

        with gr.Row(equal_height=False, elem_classes="main-row"):
            # ── Left: Controls ─────────────────────────────────────
            with gr.Column(scale=1, min_width=350, elem_classes="param-panel"):
                gr.Markdown("### Dataset Selection")
                dropdown = gr.Dropdown(
                    choices=dropdown_choices,
                    label="Select from CoMoFoD Dataset",
                    info="Translation (001-040) · Rotation (041-080) · Scaling (081-120)",
                )
                btn_dataset = gr.Button("Analyze Dataset Image", variant="primary")

                gr.Markdown("---")
                gr.Markdown("### Custom Upload")
                upload = gr.Image(label="Upload Image", type="numpy")
                btn_upload = gr.Button("Analyze Uploaded Image", variant="primary")

                gr.Markdown("---")
                gr.Markdown("### Algorithm Parameters")
                ratio_slider = gr.Slider(
                    minimum=0.4, maximum=0.9, value=0.7, step=0.05,
                    label="Lowe's Ratio Threshold",
                    info="Lower = stricter matching",
                )
                ransac_slider = gr.Slider(
                    minimum=1.0, maximum=20.0, value=5.0, step=0.5,
                    label="RANSAC Inlier Threshold (px)",
                    info="Max reprojection error to count as inlier",
                )

            # ── Right: Outputs ─────────────────────────────────────
            with gr.Column(scale=2, min_width=500):
                info_box = gr.HTML(
                    """
                    <div style="background: var(--background-fill-secondary); padding: 20px; border-radius: 8px; border: 1px solid var(--border-color-primary); text-align: center; opacity: 0.6;">
                        Select an image or upload one to see results and metrics.
                    </div>
                    """
                )

                with gr.Row():
                    out_unforged = gr.Image(label="Unforged Original (_O)", type="numpy", elem_classes="output-image")
                    out_forged   = gr.Image(label="Forged Image (_F)", type="numpy", elem_classes="output-image")

                with gr.Row():
                    out_matches  = gr.Image(label="Detected Matches (inliers)", type="numpy", elem_classes="output-image")
                    out_mask     = gr.Image(label="Binary Detection Mask", type="numpy", elem_classes="output-image")

        # ── Wire callbacks ─────────────────────────────────────────
        btn_dataset.click(
            fn=on_dataset_image,
            inputs=[dropdown, ratio_slider, ransac_slider],
            outputs=[out_unforged, out_forged, out_matches, out_mask, info_box],
        )
        btn_upload.click(
            fn=on_upload_image,
            inputs=[upload, ratio_slider, ransac_slider],
            outputs=[out_unforged, out_forged, out_matches, out_mask, info_box],
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
        theme=gr.themes.Base(
            primary_hue="zinc",
            secondary_hue="stone",
            neutral_hue="gray",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
        ),
    )
