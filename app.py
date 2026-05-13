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

def run_pipeline(image_bgr, gt_mask_path, ratio, ransac_thresh, circle_radius, dilate_size, close_size):
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
        vis_inliers = matches  # fall back to raw matches ONLY for visualization
    else:
        vis_inliers = all_inliers

    # 5. Visualise matches
    vis_matches = draw_matches(image_bgr, keypoints, vis_inliers)

    # 6. Post-process → mask + metrics
    gt_mask = None
    if gt_mask_path and os.path.isfile(gt_mask_path):
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

    mask, iou, dice = postprocess(
        image_bgr.shape[:2], keypoints, all_inliers,
        gt_mask=gt_mask,
        circle_radius=circle_radius, dilate_size=dilate_size, close_size=close_size,
    )

    # Build a plain metrics strip for the UI.
    n_clusters = len(ransac_results)
    metric_items = [
        ("Matches", len(matches)),
        ("Inliers", len(all_inliers)),
        ("Clusters", n_clusters),
    ]
    if iou is not None and dice is not None:
        metric_items.extend([("IoU", f"{iou:.4f}"), ("DICE", f"{dice:.4f}")])
    else:
        metric_items.append(("Ground Truth", "N/A"))

    cells = "".join(
        f"""
        <div style="flex: 1; min-width: 110px;">
            <div style="font-size: 0.82rem; margin-bottom: 4px;">{label}</div>
            <div style="font-size: 1.25rem; font-weight: 700;">{value}</div>
        </div>
        """
        for label, value in metric_items
    )
    info = f"""
    <div style="display: flex; flex-wrap: wrap; gap: 14px; background: #ffffff; color: #000000; padding: 14px; border: 1px solid #d0d0d0; border-radius: 4px;">
        {cells}
    </div>
    """

    return vis_matches, mask, info


# ──────────────────────────────────────────────────────────────────────
# GRADIO CALLBACKS
# ──────────────────────────────────────────────────────────────────────

def on_dataset_image(selection, ratio, ransac_thresh, circle_radius, dilate_size, close_size):
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
    vis, mask, info = run_pipeline(image_bgr, entry["gt"], ratio, ransac_thresh, circle_radius, dilate_size, close_size)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    return unforged_rgb, original_rgb, vis_rgb, mask, info


def on_upload_image(upload, ratio, ransac_thresh, circle_radius, dilate_size, close_size):
    """Callback when user uploads a custom image."""
    if upload is None:
        return None, None, None, None, "Upload an image to analyse."

    # Gradio passes a numpy RGB array for gr.Image(type="numpy")
    image_rgb = upload
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    vis, mask, info = run_pipeline(image_bgr, None, ratio, ransac_thresh, circle_radius, dilate_size, close_size)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    return None, image_rgb, vis_rgb, mask, info


# ──────────────────────────────────────────────────────────────────────
# GRADIO UI
# ──────────────────────────────────────────────────────────────────────

def _header_html(title: str) -> str:
    return f"""
    <div class="app-header">
        <div class="app-title">
            <h1>{title}</h1>
            <p>DIP - SP '26</p>
        </div>
        <div class="team-list">Muhammad Omer Khan (23I-0650)
Muhammad Hadeed (23I-0764)
Muhammad Qasim (23I-0552)</div>
    </div>
    """


def show_analyze_page():
    return (
        _header_html("Copy-Move Forgery Detection"),
        gr.update(visible=True),
        gr.update(visible=False),
    )


def show_about_page():
    return (
        _header_html("CoMoFoD"),
        gr.update(visible=False),
        gr.update(visible=True),
    )


CUSTOM_CSS = """
html,
body,
gradio-app,
.gradio-container {
    background: #ffffff !important;
    color: #000000 !important;
}

.gradio-container {
    max-width: 1480px !important;
    margin: auto;
    padding: 24px !important;
}

* {
    color: #000000;
}

.app-header {
    display: flex;
    justify-content: space-between;
    gap: 32px;
    align-items: flex-start;
    padding: 4px 0 22px 0;
    border-bottom: 1px solid #d0d0d0;
    margin-bottom: 22px;
}

.app-title h1 {
    margin: 0;
    font-size: 2rem;
    line-height: 1.15;
    font-weight: 700;
}

.app-title p {
    margin: 8px 0 0 0;
    font-size: 1rem;
    font-weight: 400;
}

.team-list {
    text-align: right;
    line-height: 1.55;
    font-size: 0.84rem;
    white-space: pre-line;
}

.top-workspace,
.parameter-panel {
    border: 1px solid #d0d0d0;
    background: #ffffff;
    border-radius: 4px;
    padding: 16px;
}

.about-panel {
    width: 100%;
    margin-top: 36px;
}

.section-label {
    margin: 0 0 12px 0;
    font-weight: 700;
    font-size: 1rem;
}

.code-note {
    max-width: 760px;
    margin: 0 auto 20px auto;
    padding: 18px;
    border: 1px solid #d0d0d0;
    border-radius: 4px;
    background: #ffffff;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 1.02rem;
    line-height: 1.7;
    white-space: pre-line;
    text-align: left;
}

.page-nav {
    display: flex;
    gap: 12px;
    margin: 0 0 28px 0;
    padding-bottom: 22px;
    border-bottom: 1px solid #d0d0d0;
}

.nav-button,
.nav-button button,
.nav-button .gr-button {
    background: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #000000 !important;
    border-radius: 4px !important;
    min-height: 42px !important;
    min-width: 140px !important;
    padding: 8px 24px !important;
    line-height: 1.3 !important;
    box-shadow: none !important;
}

.nav-button:hover,
.nav-button button:hover,
.nav-button .gr-button:hover {
    background: #f2f2f2 !important;
}

.output-image {
    min-height: 300px;
    background: #ffffff !important;
    border: 1px solid #d0d0d0 !important;
    border-radius: 4px !important;
}

.output-image label,
.output-image .block-label,
.output-image .label-wrap,
.top-workspace .block-label,
.top-workspace .label-wrap {
    background: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #d0d0d0 !important;
    border-radius: 4px !important;
}

.output-image svg,
.top-workspace svg {
    color: #000000 !important;
    stroke: #000000 !important;
}

.action-button,
.action-button button,
.action-button .gr-button {
    background: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #000000 !important;
    border-radius: 4px !important;
    box-shadow: none !important;
    font-weight: 500 !important;
    min-height: 42px !important;
    padding: 8px 16px !important;
}

.action-button:hover,
.action-button button:hover,
.action-button .gr-button:hover {
    background: #f2f2f2 !important;
}

input,
textarea,
select,
.wrap,
.block,
.form,
.panel,
.dropdown,
.svelte-1gfkn6j,
.svelte-1gfkn6j > label {
    background: #ffffff !important;
    color: #000000 !important;
    border-color: #d0d0d0 !important;
}

.metric-box {
    display: flex;
    gap: 12px;
    justify-content: space-between;
    background: #ffffff;
    padding: 14px;
    border: 1px solid #d0d0d0;
    border-radius: 4px;
    margin-bottom: 16px;
}

@media (max-width: 760px) {
    .app-header {
        flex-direction: column;
    }

    .team-list {
        text-align: left;
    }
}

input[type="range"] {
    accent-color: #000000 !important;
}

/* Ensure slider track is visible */
.gradio-container input[type="range"] {
    background: #f0f0f0 !important;
}
"""

def build_ui():
    dropdown_choices = list(CATALOGUE.keys())

    with gr.Blocks() as demo:

        # ── Header + simple page switcher ──────────────────────────
        header = gr.HTML(_header_html("Copy-Move Forgery Detection"))

        with gr.Row(elem_classes="page-nav"):
            nav_analyze = gr.Button("Analyze", elem_classes="nav-button")
            nav_about = gr.Button("About", elem_classes="nav-button")

        with gr.Column(visible=True) as analyze_panel:
            with gr.Column(elem_classes="top-workspace"):
                gr.HTML('<p class="section-label">Images</p>')
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=320):
                        dropdown = gr.Dropdown(
                            choices=dropdown_choices,
                            label="Dataset Image",
                        )
                        btn_dataset = gr.Button("Analyze Dataset Image", elem_classes="action-button")

                        upload = gr.Image(label="Upload Image", type="numpy", elem_classes="output-image")
                        btn_upload = gr.Button("Analyze Uploaded Image", elem_classes="action-button")

                    with gr.Column(scale=2, min_width=560):
                        with gr.Row():
                            out_unforged = gr.Image(label="Unforged Original (_O)", type="numpy", elem_classes="output-image")
                            out_forged   = gr.Image(label="Forged Image (_F)", type="numpy", elem_classes="output-image")

                        with gr.Row():
                            out_matches  = gr.Image(label="Detected Matches", type="numpy", elem_classes="output-image")
                            out_mask     = gr.Image(label="Binary Detection Mask", type="numpy", elem_classes="output-image")

            with gr.Column(elem_classes="parameter-panel"):
                gr.HTML('<p class="section-label">Parameters</p>')
                with gr.Row():
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
                    circle_slider = gr.Slider(
                        minimum=1, maximum=15, value=6, step=1,
                        label="Keypoint Circle Radius",
                        info="Size of circles drawn for inliers",
                    )
                with gr.Row():
                    dilate_slider = gr.Slider(
                        minimum=1, maximum=31, value=11, step=2,
                        label="Dilation Kernel Size",
                        info="Expands the masked regions",
                    )
                    close_slider = gr.Slider(
                        minimum=1, maximum=51, value=21, step=2,
                        label="Closing Kernel Size",
                        info="Fills gaps between regions",
                    )
                    info_box = gr.HTML(
                        """
                        <div style="background: #ffffff; color: #000000; padding: 14px; border-radius: 4px; border: 1px solid #d0d0d0; text-align: center;">
                            Select an image or upload one to see results and metrics.
                        </div>
                        """
                    )

        with gr.Column(visible=False, elem_classes="about-panel") as about_panel:
            gr.HTML("""
            <p class="code-note">This project is made for Digital Image Processing, Spring ’26.

You can upload an image to check whether it has been forged. The program will display a mask showing the tampered region.

The program looks for a specific type of forgery method, copy-move forgery, where part of an image is copied to another location in the same image.</p>
            """)
            gr.HTML("""
            <p class="code-note"><strong>Image Dataset</strong>

We've used the CoMoFoD database, which consists of 120 forged image sets.

These images are grouped in 3 groups according to applied manipulation: translation, rotation, and scaling.

Translation: 1-40
Rotation: 41-80
Scaling: 81-120</p>
            """)

        # ── Wire callbacks ─────────────────────────────────────────
        btn_dataset.click(
            fn=on_dataset_image,
            inputs=[dropdown, ratio_slider, ransac_slider, circle_slider, dilate_slider, close_slider],
            outputs=[out_unforged, out_forged, out_matches, out_mask, info_box],
        )
        btn_upload.click(
            fn=on_upload_image,
            inputs=[upload, ratio_slider, ransac_slider, circle_slider, dilate_slider, close_slider],
            outputs=[out_unforged, out_forged, out_matches, out_mask, info_box],
        )
        nav_analyze.click(
            fn=show_analyze_page,
            inputs=[],
            outputs=[header, analyze_panel, about_panel],
        )
        nav_about.click(
            fn=show_about_page,
            inputs=[],
            outputs=[header, analyze_panel, about_panel],
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
