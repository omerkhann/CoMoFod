import os
import glob
import time
import csv
import cv2

from preprocessing import preprocess
from matching import extract_sift_features, match_keypoints
from ransac import sequential_ransac
from postprocessing import postprocess

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def main():
    forged_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_F.png")))
    
    if not forged_files:
        print("No forged images found in the data directory.")
        return
        
    output_csv = "evaluation_results.csv"
    
    print(f"Found {len(forged_files)} images. Starting evaluation...")
    total_start = time.time()
    
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Category", "Matches", "Inliers", "Clusters", "IoU", "DICE", "Time (s)"])
        
        for idx, fpath in enumerate(forged_files, 1):
            basename = os.path.basename(fpath)
            gt_path = fpath.replace("_F.png", "_B.png")
            
            # Determine category based on your dataset numbering (1-40: Translation, 41-80: Rotation, 81-120: Scaling)
            img_id = int(basename.split("_")[0])
            if 1 <= img_id <= 40: category = "Translation"
            elif 41 <= img_id <= 80: category = "Rotation"
            else: category = "Scaling"
            
            image_bgr = cv2.imread(fpath)
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(gt_path) else None
            
            if image_bgr is None:
                print(f"Skipping {basename} (could not read file)")
                continue
                
            start_t = time.time()
            
            # 1. Preprocess
            gray = preprocess(image_bgr, smooth=True, kernel_size=3, sigma=1.0)
            
            # 2. SIFT Extraction
            keypoints, descriptors = extract_sift_features(gray, n_features=0)
            
            # 3. Matching
            if descriptors is not None and len(descriptors) >= 10:
                matches = match_keypoints(keypoints, descriptors, ratio=0.7, min_spatial_dist=10.0)
            else:
                matches = []
                
            # 4. RANSAC
            ransac_results = []
            if len(matches) >= 4:
                ransac_results = sequential_ransac(
                    matches, keypoints,
                    threshold=5.0, max_iterations=2000,
                    min_inliers=6, max_models=5
                )
                
            all_inliers = []
            for H, inliers in ransac_results:
                all_inliers.extend(inliers)
                
            # 5. Postprocessing & Metrics
            mask, iou, dice = postprocess(
                image_bgr.shape[:2], keypoints, all_inliers,
                gt_mask=gt_mask,
                circle_radius=6, dilate_size=11, close_size=21
            )
            
            elapsed = time.time() - start_t
            
            iou_val = round(iou, 4) if iou is not None else 0.0
            dice_val = round(dice, 4) if dice is not None else 0.0
            
            writer.writerow([basename, category, len(matches), len(all_inliers), len(ransac_results), iou_val, dice_val, round(elapsed, 2)])
            f.flush() # Force write to disk so we can see progress live
            
            print(f"[{idx:03d}/{len(forged_files)}] {basename} ({category:<11}) | IoU: {iou_val:.4f} | DICE: {dice_val:.4f} | Time: {elapsed:.2f}s")
            
    total_time = time.time() - total_start
    print(f"\nEvaluation complete! Results saved to {output_csv}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/len(forged_files):.2f}s per image)")

if __name__ == "__main__":
    main()
