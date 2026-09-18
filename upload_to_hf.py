import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, login, get_token, whoami

def main():
    parser = argparse.ArgumentParser(description="Upload MOPRO models to Hugging Face Hub")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face write token (hf_...)")
    parser.add_argument("--repo_name", type=str, default="traffic-sign-adverse-weather", help="Hugging Face repo name")
    parser.add_argument("--private", action="store_true", help="Make Hugging Face repo private")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or get_token()
    if not token:
        print("[ERROR] No Hugging Face token found!")
        print("Please provide --token hf_... or set HF_TOKEN environment variable.")
        sys.exit(1)

    print("[INFO] Logging in to Hugging Face...")
    login(token=token, add_to_git_credential=True)

    api = HfApi(token=token)
    user_info = whoami(token=token)
    username = user_info["name"]
    repo_id = f"{username}/{args.repo_name}"
    print(f"[OK] Authenticated as: {username}")
    print(f"[INFO] Target repository: {repo_id}")

    # 1. Create repo if not exists
    print(f"[INFO] Creating/Verifying repository {repo_id}...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True
    )
    print(f"[OK] Repository ready: https://huggingface.co/{repo_id}")

    # 2. Prepare Model Card (README.md for HF)
    model_card_content = f"""---
language:
- zh
- en
license: mit
tags:
- image-classification
- traffic-signs
- pytorch
- timm
- adverse-weather
- ensemble
metrics:
- f1
pipeline_tag: image-classification
---

# Traffic Sign Recognition under Adverse Weather (National 2nd Prize)

Official model checkpoints for the **National Second Prize** solution in the Traffic Sign Recognition under Adverse Weather Competition.

- **GitHub Repository**: [https://github.com/Lyumnire/traffic-sign-adverse-weather](https://github.com/Lyumnire/traffic-sign-adverse-weather)
- **Evaluation Metric**: Macro-F1 = **0.945** (Evaluated on 2-Core CPU without GPU)
- **Ensemble Architecture**: ConvNeXt V2-Base (0.60) + Swin V2-Base (0.20) + EfficientNetV2-M (0.20)

## Model Checkpoints

| Model | Checkpoint File | Parameters | Input Resolution | Val Macro-F1 |
| :--- | :--- | :--- | :--- | :--- |
| **ConvNeXt V2-Base** | `convnextv2_base_best.pth` | 89M | 384x384 | 0.938 |
| **Swin Transformer V2-Base** | `swin_v2_b_best.pth` | 88M | 256x256 | 0.916 |
| **EfficientNetV2-M** | `efficientnet_v2_m_best.pth` | 54M | 384x384 | 0.917 |

## Classes
See `classes.txt` for the 25 traffic sign classes.

## Quick Download via Python

```python
from huggingface_hub import hf_hub_download

# Download ConvNeXt V2-Base checkpoint
checkpoint_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="convnextv2_base_best.pth",
    local_dir="./results"
)
```

For inference scripts, training code, and in-depth engineering retrospective, visit the [GitHub Repository](https://github.com/Lyumnire/traffic-sign-adverse-weather).
"""

    model_card_path = Path("results/README_HF.md")
    model_card_path.write_text(model_card_content, encoding="utf-8")

    # 3. Upload model card
    print("[INFO] Uploading Model Card...")
    api.upload_file(
        path_or_fileobj=str(model_card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model"
    )
    if model_card_path.exists():
        model_card_path.unlink()

    # 4. Upload classes.txt
    if Path("classes.txt").exists():
        print("[INFO] Uploading classes.txt...")
        api.upload_file(
            path_or_fileobj="classes.txt",
            path_in_repo="classes.txt",
            repo_id=repo_id,
            repo_type="model"
        )

    # 5. Upload model weights
    models = [
        "convnextv2_base_best.pth",
        "efficientnet_v2_m_best.pth",
        "swin_v2_b_best.pth"
    ]

    for model_file in models:
        file_path = Path("results") / model_file
        if not file_path.exists():
            print(f"[WARN] File not found: {file_path}, skipping.")
            continue
        
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"[INFO] Uploading {model_file} ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=model_file,
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"[OK] Uploaded {model_file}")

    print(f"\n[SUCCESS] All weights uploaded successfully to: https://huggingface.co/{repo_id}")
    return repo_id

if __name__ == "__main__":
    main()
