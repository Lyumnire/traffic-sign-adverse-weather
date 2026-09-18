"""
Traffic Sign Classification under Adverse Weather - Single Model Inference Script
==================================================================================
Platform: MoModel (Linux, /home/jovyan/work/)
Interface: predict(X) -> str
  X: np.ndarray BGR image from cv2.imread
  Returns: one of 25 official class name strings
"""

import os
import sys
import warnings
import traceback
from pathlib import Path
from threading import Lock

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ============================================================
# 0b. Auto-install timm if not available (ConvNeXt V2 requires it)
# ============================================================
try:
    import timm
except ImportError:
    print("[INFO] timm not found. Installing timm for ConvNeXt V2 support...", file=sys.stderr)
    os.system(f"{sys.executable} -m pip install timm -q")
    try:
        import timm
        print("[OK] timm installed successfully.", file=sys.stderr)
    except ImportError:
        print("[WARN] Failed to install timm. ConvNeXt V2 will fall back to V1 (different architecture!).", file=sys.stderr)

# ============================================================
# 0. CHOOSE YOUR SINGLE MODEL HERE
# ============================================================
# Choose one of: "efficientnet_v2_m", "convnextv2_base", "swin_v2_b"
CHOSEN_MODEL = "efficientnet_v2_m"  

# ============================================================
# 1. Configuration
# ============================================================
NUM_CLASSES = 25
RESULTS_DIR = Path('/home/jovyan/work/results')
CLASSES_PATH = Path('/home/jovyan/work/classes.txt')
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# CPU threading — match the 2-core inference machine
torch.set_num_threads(2)
torch.set_grad_enabled(False)
DEVICE = torch.device("cpu")

# Try to import timm for ConvNeXt V2
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

# ============================================================
# 2. Label Utilities (must match official platform exactly)
# ============================================================
def normalize_label(label):
    """Clean extra spaces in directory names to match classes.txt"""
    return ' '.join(str(label).strip().split())

def load_classes(path=CLASSES_PATH):
    """Read classes.txt with tab-separated index and class name."""
    indexed = {}
    with path.open('r', encoding='utf-8-sig') as f:
        for line_number, raw in enumerate(f, 1):
            line = raw.rstrip('\r\n')
            if not line:
                continue
            try:
                index_text, class_name = line.split('\t', 1)
                index = int(index_text)
            except ValueError:
                raise ValueError(f'classes.txt line {line_number}: bad format')
            class_name = normalize_label(class_name)
            if index in indexed or not class_name:
                raise ValueError(f'classes.txt line {line_number}: duplicate or empty')
            indexed[index] = class_name
    if sorted(indexed) != list(range(NUM_CLASSES)):
        raise ValueError('classes.txt must contain indices 0 to 24')
    classes = [indexed[i] for i in range(NUM_CLASSES)]
    return classes

# ============================================================
# 3. Model Builders
# ============================================================
def _build_convnextv2_base(num_classes):
    """ConvNeXt V2 Base via timm, fallback to torchvision V1."""
    if HAS_TIMM:
        try:
            model = timm.create_model("convnextv2_base", pretrained=False,
                                      num_classes=num_classes)
            return model
        except Exception:
            pass
    from torchvision.models import convnext_base
    model = convnext_base(weights=None)
    in_f = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_f, num_classes)
    return model

def _build_swin_v2_b(num_classes):
    from torchvision.models import swin_v2_b
    model = swin_v2_b(weights=None)
    in_f = model.head.in_features
    model.head = nn.Linear(in_f, num_classes)
    return model

def _build_efficientnet_v2_m(num_classes):
    from torchvision.models import efficientnet_v2_m
    model = efficientnet_v2_m(weights=None)
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, num_classes)
    return model

# Model registry: (name, builder, weight_file, default_input_size)
MODEL_SPECS = {
    "convnextv2_base":   (_build_convnextv2_base,   "convnextv2_base_best.pth",   384),
    "swin_v2_b":         (_build_swin_v2_b,         "swin_v2_b_best.pth",         256),
    "efficientnet_v2_m": (_build_efficientnet_v2_m, "efficientnet_v2_m_best.pth", 384),
}

# ============================================================
# 4. Preprocessing (pure numpy, no PIL overhead)
# ============================================================
def preprocess(bgr_img, size):
    """BGR -> RGB, resize, normalize, HWC->CHW. Returns float32 (1,3,H,W)."""
    img = cv2.resize(bgr_img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))     # HWC -> CHW
    return np.expand_dims(img, axis=0)      # add batch dim

# ============================================================
# 5. Global State & Lazy Loading
# ============================================================
_model_entry = None  # (model_or_session, input_size, is_onnx)
_classes = None
_load_lock = Lock()
_loaded = False

def _load_all():
    """Load classes and the chosen single model. Called once on first predict()."""
    global _model_entry, _classes, _loaded
    if _loaded:
        return
    with _load_lock:
        if _loaded:
            return

        # Load classes
        _classes = load_classes()

        if CHOSEN_MODEL not in MODEL_SPECS:
            raise ValueError(f"CHOSEN_MODEL must be one of {list(MODEL_SPECS.keys())}")

        builder, weight_file, default_size = MODEL_SPECS[CHOSEN_MODEL]
        pth_path = RESULTS_DIR / weight_file
        onnx_path = RESULTS_DIR / weight_file.replace('.pth', '.onnx')

        # Try ONNX first
        try:
            import onnxruntime as ort
            if onnx_path.is_file():
                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 2
                opts.intra_op_num_threads = 2
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess = ort.InferenceSession(str(onnx_path), sess_options=opts,
                                            providers=["CPUExecutionProvider"])
                _model_entry = (sess, default_size, True)
                print(f"[OK] ONNX Loaded: {CHOSEN_MODEL}", file=sys.stderr)
                _loaded = True
                return
        except Exception as e:
            print(f"[WARN] ONNX load failed: {e}", file=sys.stderr)

        # Fall back to PyTorch
        if not pth_path.is_file():
            raise FileNotFoundError(f"Model weight file not found: {pth_path}")

        try:
            checkpoint = torch.load(str(pth_path), map_location=DEVICE, weights_only=False)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                input_size = checkpoint.get('input_size', default_size)
                if isinstance(input_size, (list, tuple)):
                    input_size = input_size[0]
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                input_size = default_size
            else:
                state_dict = checkpoint
                input_size = default_size

            # Strip "module." prefix from DataParallel
            cleaned = {}
            for k, v in state_dict.items():
                cleaned[k.replace("module.", "")] = v

            model = builder(NUM_CLASSES)

            # Architecture validation: check if checkpoint keys match model
            model_keys = set(model.state_dict().keys())
            ckpt_keys = set(cleaned.keys())
            missing_keys = model_keys - ckpt_keys
            unexpected_keys = ckpt_keys - model_keys

            if missing_keys or unexpected_keys:
                print(f"[ERROR] Architecture mismatch for {CHOSEN_MODEL}!", file=sys.stderr)
                print(f"  Model keys: {len(model_keys)}, Checkpoint keys: {len(ckpt_keys)}", file=sys.stderr)
                print(f"  Missing keys (model expects): {len(missing_keys)}", file=sys.stderr)
                if missing_keys:
                    for k in sorted(missing_keys)[:5]:
                        print(f"    {k}", file=sys.stderr)
                print(f"  Unexpected keys (checkpoint has): {len(unexpected_keys)}", file=sys.stderr)
                if unexpected_keys:
                    for k in sorted(unexpected_keys)[:5]:
                        print(f"    {k}", file=sys.stderr)
                print(f"  CAUSE: Model built with different architecture than training.", file=sys.stderr)
                print(f"  FIX: Ensure timm is installed for ConvNeXt V2.", file=sys.stderr)
                raise RuntimeError(f"Architecture mismatch for {CHOSEN_MODEL}")

            model.load_state_dict(cleaned, strict=True)
            model.to(DEVICE).eval()
            _model_entry = (model, int(input_size), False)
            print(f"[OK] PyTorch Loaded: {CHOSEN_MODEL} (input_size={input_size}, keys={len(model_keys)})", file=sys.stderr)
        except Exception as e:
            print(f"[ERROR] Failed to load PyTorch weight: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise e

        _loaded = True

# ============================================================
# 6. Single-Model Inference
# ============================================================
def _infer_pytorch(model, bgr_img, size):
    """Run PyTorch model, return softmax probs (NUM_CLASSES,)."""
    tensor = torch.from_numpy(preprocess(bgr_img, size))
    with torch.inference_mode():
        logits = model(tensor)
    return F.softmax(logits, dim=1).squeeze(0).numpy()

def _infer_onnx(session, bgr_img, size):
    """Run ONNX session, return softmax probs (NUM_CLASSES,)."""
    np_input = preprocess(bgr_img, size)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: np_input})
    logits = outputs[0]
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
    return probs.squeeze(0)

def _infer_model_tta(model_or_sess, input_size, is_onnx, bgr_img):
    """Multi-scale TTA for single model. NO horizontal flip."""
    infer_fn = _infer_onnx if is_onnx else _infer_pytorch
    
    # Primary scale
    p1 = infer_fn(model_or_sess, bgr_img, input_size)
    
    # Secondary scale (input_size - 32, clamped to minimum 128)
    secondary_size = max(128, input_size - 32)
    if secondary_size != input_size:
        p2 = infer_fn(model_or_sess, bgr_img, secondary_size)
        return (p1 + p2) / 2.0
    return p1

# ============================================================
# 7. predict(X) — Public API
# ============================================================
def predict(X):
    """
    Classify a traffic sign image under adverse weather.

    Parameters
    ----------
    X : np.ndarray
        BGR image from cv2.imread, shape (H, W, 3), dtype uint8.

    Returns
    -------
    str
        One of the 25 official traffic sign class names.
    """
    _load_all()

    # Edge-case guards
    if X is None or not isinstance(X, np.ndarray) or X.size == 0:
        return normalize_label(_classes[0])

    # Ensure 3-channel uint8
    if X.ndim == 2:
        X = cv2.cvtColor(X, cv2.COLOR_GRAY2BGR)
    elif X.ndim == 3 and X.shape[2] == 1:
        X = cv2.cvtColor(X[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif X.ndim == 3 and X.shape[2] == 4:
        X = cv2.cvtColor(X, cv2.COLOR_BGRA2BGR)

    if X.dtype != np.uint8:
        if np.issubdtype(X.dtype, np.integer):
            X = np.clip(X, 0, 255).astype(np.uint8)
        elif np.issubdtype(X.dtype, np.floating):
            if X.max() > 1.0:
                X = np.clip(X, 0, 255).astype(np.uint8)
            else:
                X = np.clip(X * 255, 0, 255).astype(np.uint8)

    if _model_entry is None:
        return normalize_label(_classes[0])

    model_or_sess, input_size, is_onnx = _model_entry
    
    try:
        probs = _infer_model_tta(model_or_sess, input_size, is_onnx, X)
        best_idx = int(np.argmax(probs))
        return normalize_label(_classes[best_idx])
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}", file=sys.stderr)
        return normalize_label(_classes[0])


# ============================================================
# 8. Self-Test
# ============================================================
if __name__ == '__main__':
    _load_all()
    print(f"Classes: {_classes}")
    print(f"Model loaded: {CHOSEN_MODEL} ({_model_entry[1]}px, {'onnx' if _model_entry[2] else 'pt'})")

    # Smoke test
    dummy = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    result = predict(dummy)
    print(f"Dummy prediction: {result}")
