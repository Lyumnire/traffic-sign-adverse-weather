import os
import json
import zipfile
import io
from PIL import Image

def crop_and_clean_tt100k_in_memory(zip_path, output_dir, padding_ratio=0.1):
    """
    流式/内存中解析 TT100K data.zip，使用 PIL 在内存中解码并裁剪，无需 OpenCV 和 tqdm，
    零额外依赖，保证在默认虚拟环境下即可直接运行。
    """
    if not os.path.exists(zip_path):
        print(f"Error: data.zip not found at {zip_path}")
        return

    print("Opening data.zip...")
    with zipfile.ZipFile(zip_path, "r") as z:
        # 1. 在内存中读取并解析 annotations.json
        ann_name = "data/annotations.json"
        if ann_name not in z.namelist():
            print(f"Error: {ann_name} not found in zip file.")
            return
            
        print("Reading annotations.json from zip...")
        with z.open(ann_name) as f:
            ann = json.load(f)
            
        os.makedirs(output_dir, exist_ok=True)
        print("Processing and cropping traffic signs directly from zip archive to memory...")
        
        imgs = ann.get("imgs", {})
        total_imgs = len(imgs)
        print(f"Total images to scan: {total_imgs}")
        
        # 统计成功导出的样本数
        count = 0
        idx = 0
        
        for img_id, img_info in imgs.items():
            idx += 1
            if idx % 500 == 0 or idx == total_imgs:
                print(f"Progress: {idx}/{total_imgs} images scanned ({idx/total_imgs*100:.1f}%) | Cropped: {count} signs")
                
            rel_path = img_info.get("path", "")
            # zip 内部的完整路径是 "data/" + "train/xxxx.jpg"
            zip_img_path = f"data/{rel_path}"
            
            if zip_img_path not in z.namelist():
                continue
                
            try:
                # 2. 从 zip 直接读取文件字节到内存
                img_bytes = z.read(zip_img_path)
                
                # 3. 将字节解码为 PIL Image (RGB)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                width, height = img.size
                
                objects = img_info.get("objects", [])
                
                for obj_idx, obj in enumerate(objects):
                    category = obj.get("category", "unknown")
                    bbox = obj.get("bbox", {})
                    
                    xmin = int(bbox.get("xmin", 0))
                    ymin = int(bbox.get("ymin", 0))
                    xmax = int(bbox.get("xmax", 0))
                    ymax = int(bbox.get("ymax", 0))
                    
                    # 计算标志尺寸与 padding
                    w_obj = xmax - xmin
                    h_obj = ymax - ymin
                    pad_w = int(w_obj * padding_ratio)
                    pad_h = int(h_obj * padding_ratio)
                    
                    # 添加边缘背景（模拟赛题的“包含一定道路、树木背景”的要求）
                    x1 = max(0, xmin - pad_w)
                    y1 = max(0, ymin - pad_h)
                    x2 = min(width, xmax + pad_w)
                    y2 = min(height, ymax + pad_h)
                    
                    # 裁剪并保存
                    cropped = img.crop((x1, y1, x2, y2))
                    
                    cat_dir = os.path.join(output_dir, category)
                    os.makedirs(cat_dir, exist_ok=True)
                    
                    save_name = f"{img_id}_{obj_idx}.jpg"
                    save_path = os.path.join(cat_dir, save_name)
                    cropped.save(save_path, "JPEG")
                    count += 1
            except Exception as e:
                print(f"\nError processing {zip_img_path}: {e}")
                continue
                
        print(f"Preprocessing completed! Successfully cropped {count} signs to {output_dir}")

def main():
    zip_path = "./datasets/tt100k/data.zip"
    cleaned_dir = "./datasets/tt100k_cropped"
    crop_and_clean_tt100k_in_memory(zip_path, cleaned_dir)

if __name__ == "__main__":
    main()
