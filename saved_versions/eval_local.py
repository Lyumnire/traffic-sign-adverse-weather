"""
Local Evaluation Script for submit/main.py (Optimized)
======================================================
Simulates MoModel platform inference loop on the local validation split.
Computes Macro-F1 score and checks if the submission script is correct.
Uses GPU acceleration locally for fast evaluation.

Features:
- Per-model evaluation
- Weight search (optional)
- Center crop TTA testing
- ONNX support
"""

import os
import sys
import time
import itertools
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report

# 1. Patch main.py paths dynamically to run locally
_submit_dir = str(Path(__file__).resolve().parent.parent / "submit")
sys.path.insert(0, _submit_dir)
import main

# Redirect platform paths to local directories
_project_root = Path(__file__).resolve().parent.parent
main.RESULTS_DIR = _project_root / "results"
main.CLASSES_PATH = _project_root / "classes.txt"

# Enable CUDA locally for fast evaluation
main.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(os.cpu_count() or 4)

# Patch default argument of load_classes because Python evaluates defaults at definition time
main.load_classes.__defaults__ = (main.CLASSES_PATH,)

# Patch _infer_pytorch to support CUDA execution (moving inputs/outputs between CPU and GPU)
def patched_infer_pytorch(model, bgr_img, size):
    tensor = torch.from_numpy(main.preprocess(bgr_img, size)).to(main.DEVICE)
    with torch.inference_mode():
        logits = model(tensor)
    return F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

main._infer_pytorch = patched_infer_pytorch

# Reset loading flag to force reload from patched paths
main._loaded = False
main._models = []

print("=== Local Path & GPU Alignment ===")
print(f"DEVICE:       {main.DEVICE}")
print(f"RESULTS_DIR:  {main.RESULTS_DIR} (exists: {main.RESULTS_DIR.is_dir()})")
print(f"CLASSES_PATH: {main.CLASSES_PATH} (exists: {main.CLASSES_PATH.is_file()})")

# 2. Get local validation images and labels
def get_validation_data(dataset_path, classes_file):
    def normalize_label(label):
        return ' '.join(str(label).strip().split())

    # Read classes.txt
    indexed = {}
    with open(classes_file, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if not line:
                continue
            idx_str, name = line.split('\t', 1)
            indexed[int(idx_str)] = normalize_label(name)
    categories = [indexed[i] for i in range(len(indexed))]
    class_to_idx = {name: idx for idx, name in enumerate(categories)}

    physical_dirs = {normalize_label(d): d for d in os.listdir(dataset_path)
                     if os.path.isdir(os.path.join(dataset_path, d))}

    # Collect images
    SUPPORTED = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.ppm'}
    all_imgs = []
    all_labels = []
    for class_name in categories:
        if class_name not in physical_dirs:
            continue
        actual_dir = physical_dirs[class_name]
        cat_dir = os.path.join(dataset_path, actual_dir)
        for f in os.listdir(cat_dir):
            if os.path.splitext(f)[1].lower() in SUPPORTED:
                all_imgs.append(os.path.join(cat_dir, f))
                all_labels.append(class_to_idx[class_name])

    # Split (must use exact same random_state=42 and test_size=0.15 as train.py)
    from sklearn.model_selection import train_test_split
    _, val_imgs, _, val_labels = train_test_split(
        all_imgs, all_labels, test_size=0.15, random_state=42, stratify=all_labels
    )

    return val_imgs, val_labels, categories

# 3. Per-Model Inference (returns raw softmax probs for each model)
def infer_single_model(model_or_sess, input_size, is_onnx, bgr_img):
    """Run single model inference, return softmax probs."""
    if is_onnx:
        return main._infer_onnx(model_or_sess, bgr_img, input_size)
    else:
        return main._infer_pytorch(model_or_sess, bgr_img, input_size)

def infer_with_tta(model_or_sess, input_size, is_onnx, bgr_img, use_center_crop=False):
    """Run model with TTA, return averaged probs."""
    p1 = infer_single_model(model_or_sess, input_size, is_onnx, bgr_img)

    # Secondary scale
    secondary_size = max(128, input_size - 32)
    if secondary_size != input_size:
        p2 = infer_single_model(model_or_sess, secondary_size, is_onnx, bgr_img)
        probs = (p1 + p2) / 2.0
    else:
        probs = p1

    # Optional center crop TTA
    if use_center_crop:
        h, w = bgr_img.shape[:2]
        crop_ratio = main.CENTER_CROP_RATIO
        ch, cw = int(h * crop_ratio), int(w * crop_ratio)
        y1, x1 = (h - ch) // 2, (w - cw) // 2
        center_img = bgr_img[y1:y1+ch, x1:x1+cw]
        p_center = infer_single_model(model_or_sess, input_size, is_onnx, center_img)
        probs = (probs + p_center) / 2.0

    return probs

# 4. Search optimal ensemble weights
def search_weights(val_probs_list, val_labels, num_models, num_classes):
    """
    Search optimal ensemble weights using grid search.
    val_probs_list: list of (num_val_samples, num_classes) arrays
    """
    best_f1 = 0.0
    best_weights = None

    # Grid search: weights from 0.0 to 1.0 in 0.1 steps, normalized
    step = 0.1
    for w0 in np.arange(0, 1.0 + step, step):
        for w1 in np.arange(0, 1.0 - w0 + step, step):
            w2 = 1.0 - w0 - w1
            if w2 < -1e-6:
                continue
            w2 = max(0, w2)

            weights = np.array([w0, w1, w2])

            # Ensemble prediction
            ensemble_probs = np.zeros((len(val_labels), num_classes))
            for i in range(num_models):
                ensemble_probs += weights[i] * val_probs_list[i]

            preds = np.argmax(ensemble_probs, axis=1)
            f1 = f1_score(val_labels, preds, average='macro')

            if f1 > best_f1:
                best_f1 = f1
                best_weights = weights.copy()

    return best_weights, best_f1

# 5. Main Evaluation Loop
def evaluate(search_weights_mode=False, use_center_crop=False, test_temperature=False):
    dataset_path = str(_project_root / "datasets" / "train")
    classes_file = str(_project_root / "classes.txt")

    if not os.path.exists(dataset_path):
        print(f"[Error] Dataset path not found: {dataset_path}")
        return

    val_imgs, val_labels, categories = get_validation_data(dataset_path, classes_file)
    print(f"\nCollected {len(val_imgs)} validation images.")

    print("\n=== Initializing models via main.py ===")
    main._load_all()
    print(f"Loaded {len(main._models)} models.")
    if not main._models:
        print("[Error] No models loaded! Please ensure trained pth weights exist in ./results/")
        return

    num_models = len(main._models)
    num_classes = len(categories)

    # Per-model evaluation
    print("\n=== Per-Model Evaluation ===")
    model_names = [m[0] for m in main._models]
    per_model_probs = []  # list of (num_val_samples, num_classes) arrays
    per_model_f1s = []

    for model_idx, (name, model_or_sess, input_size, is_onnx) in enumerate(main._models):
        print(f"\n  Evaluating model {model_idx+1}/{num_models}: {name}")
        model_probs = []

        start_time = time.time()
        for idx, (img_path, label_idx) in enumerate(zip(val_imgs, val_labels)):
            X = cv2.imread(img_path)
            if X is None:
                continue

            # TTA: secondary scale only (no center crop for per-model eval)
            probs = infer_with_tta(model_or_sess, input_size, is_onnx, X, use_center_crop=False)
            model_probs.append(probs)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(val_imgs):
                elapsed = time.time() - start_time
                speed = (idx + 1) / elapsed
                print(f"    Processed {idx + 1}/{len(val_imgs)} images ({speed:.2f} img/sec)")

        model_probs = np.array(model_probs)
        per_model_probs.append(model_probs)

        preds = np.argmax(model_probs, axis=1)
        f1 = f1_score(val_labels, preds, average='macro')
        per_model_f1s.append(f1)
        print(f"  {name} Val Macro-F1: {f1*100:.4f}")

    # Print per-model summary
    print("\n=== Per-Model Summary ===")
    for name, f1 in zip(model_names, per_model_f1s):
        print(f"  {name}: {f1*100:.4f}")

    # Equal weight baseline
    print("\n=== Equal Weight Ensemble ===")
    equal_weights = np.ones(num_models) / num_models
    ensemble_probs_equal = np.zeros((len(val_labels), num_classes))
    for i in range(num_models):
        probs = per_model_probs[i]
        # Apply temperature scaling
        if main.TEMPERATURE != 1.0:
            logits = np.log(probs + 1e-8)
            scaled_logits = logits / main.TEMPERATURE
            exp_l = np.exp(scaled_logits - np.max(scaled_logits))
            probs = exp_l / np.sum(exp_l)
        ensemble_probs_equal += equal_weights[i] * probs
    preds_equal = np.argmax(ensemble_probs_equal, axis=1)
    f1_equal = f1_score(val_labels, preds_equal, average='macro')
    print(f"  Equal Weights F1: {f1_equal*100:.4f}")
    print(f"  Weights: {equal_weights}")
    print(f"  Temperature: {main.TEMPERATURE}")

    # Weight search
    if search_weights_mode:
        print("\n=== Searching Optimal Weights ===")
        best_weights, best_f1 = search_weights(per_model_probs, val_labels, num_models, num_classes)
        print(f"  Best Weights: {best_weights}")
        print(f"  Best F1: {best_f1*100:.4f}")
        print(f"  Improvement over equal: {(best_f1 - f1_equal)*100:+.4f}")

        # Use best weights for final evaluation
        final_weights = best_weights
        final_f1 = best_f1
    else:
        # Use configured weights from main.py
        final_weights = main.ENSEMBLE_WEIGHTS[:num_models]
        final_weights = final_weights / final_weights.sum()  # normalize
        ensemble_probs_final = np.zeros((len(val_labels), num_classes))
        for i in range(num_models):
            probs = per_model_probs[i]
            # Apply temperature scaling
            if main.TEMPERATURE != 1.0:
                logits = np.log(probs + 1e-8)
                scaled_logits = logits / main.TEMPERATURE
                exp_l = np.exp(scaled_logits - np.max(scaled_logits))
                probs = exp_l / np.sum(exp_l)
            ensemble_probs_final += final_weights[i] * probs
        preds_final = np.argmax(ensemble_probs_final, axis=1)
        final_f1 = f1_score(val_labels, preds_final, average='macro')
        print(f"\n=== Configured Weights Ensemble ===")
        print(f"  Weights: {final_weights}")
        print(f"  Temperature: {main.TEMPERATURE}")
        print(f"  F1: {final_f1*100:.4f}")

    # Test center crop TTA
    if use_center_crop:
        print("\n=== Testing Center Crop TTA ===")
        ensemble_probs_center = np.zeros((len(val_labels), num_classes))
        for model_idx, (name, model_or_sess, input_size, is_onnx) in enumerate(main._models):
            model_probs_center = []
            for img_path, label_idx in zip(val_imgs, val_labels):
                X = cv2.imread(img_path)
                if X is None:
                    continue
                probs = infer_with_tta(model_or_sess, input_size, is_onnx, X, use_center_crop=True)
                model_probs_center.append(probs)
            model_probs_center = np.array(model_probs_center)
            weight = final_weights[model_idx]
            ensemble_probs_center += weight * model_probs_center
        preds_center = np.argmax(ensemble_probs_center, axis=1)
        f1_center = f1_score(val_labels, preds_center, average='macro')
        print(f"  Center Crop TTA F1: {f1_center*100:.4f}")
        print(f"  Improvement: {(f1_center - final_f1)*100:+.4f}")

    # Test temperature scaling
    if test_temperature:
        print("\n=== Testing Temperature Scaling ===")
        best_temp = 1.0
        best_f1_temp = final_f1
        for temp in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
            ensemble_probs_temp = np.zeros((len(val_labels), num_classes))
            for i in range(num_models):
                probs = per_model_probs[i]
                # Apply temperature
                logits = np.log(probs + 1e-8)
                scaled_logits = logits / temp
                exp_l = np.exp(scaled_logits - np.max(scaled_logits))
                probs_temp = exp_l / np.sum(exp_l)
                ensemble_probs_temp += final_weights[i] * probs_temp
            preds_temp = np.argmax(ensemble_probs_temp, axis=1)
            f1_temp = f1_score(val_labels, preds_temp, average='macro')
            if f1_temp > best_f1_temp:
                best_f1_temp = f1_temp
                best_temp = temp
            print(f"  Temp={temp:.1f}: F1={f1_temp*100:.4f}")
        print(f"  Best Temperature: {best_temp}")
        print(f"  Best F1: {best_f1_temp*100:.4f}")

    # Final ensemble evaluation with final weights
    print("\n=== Final Ensemble Evaluation ===")
    ensemble_probs_final = np.zeros((len(val_labels), num_classes))
    for i in range(num_models):
        probs = per_model_probs[i]
        # Apply temperature scaling
        if main.TEMPERATURE != 1.0:
            logits = np.log(probs + 1e-8)
            scaled_logits = logits / main.TEMPERATURE
            exp_l = np.exp(scaled_logits - np.max(scaled_logits))
            probs = exp_l / np.sum(exp_l)
        ensemble_probs_final += final_weights[i] * probs
    preds_final = np.argmax(ensemble_probs_final, axis=1)

    print(f"  Total validation samples: {len(val_labels)}")
    print(f"  Ensemble Weights: {final_weights}")
    print(f"  Validation Macro-F1: {final_f1*100:.4f}")

    print("\nDetailed classification report:")
    print(classification_report(val_labels, preds_final, target_names=categories, digits=4, zero_division=0))

    return final_weights, final_f1

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--search-weights', action='store_true', help='Search optimal ensemble weights')
    parser.add_argument('--center-crop', action='store_true', help='Test center crop TTA')
    parser.add_argument('--test-temperature', action='store_true', help='Test temperature scaling')
    args = parser.parse_args()

    evaluate(search_weights_mode=args.search_weights, use_center_crop=args.center_crop,
             test_temperature=args.test_temperature)
