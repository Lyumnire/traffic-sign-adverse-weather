import os
import argparse
import torch
import numpy as np
from train import build_model

def export_to_onnx(model_name, checkpoint_path, num_classes, img_size, output_path):
    print(f"Loading PyTorch model: {model_name} from {checkpoint_path}")
    
    # 1. 构建 PyTorch 模型并加载权重
    model = build_model(model_name, num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
        
    # Strip "module." prefix if present
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "")] = v
        
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    
    # 2. 创建虚拟输入 (Batch size = 1)
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    # 3. 导出为 ONNX
    print(f"Exporting to ONNX: {output_path} (img_size={img_size})...")
    # Swin V2 和 ConvNeXt V2 包含复杂的算子，推荐使用 opset_version=16 或 17 以获得最佳兼容性
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # 支持动态 Batch size 推理
    )
    print("ONNX export completed successfully!")
    
    # 4. 验证 PyTorch 与 ONNX 的输出一致性
    try:
        import onnxruntime as ort
        print("Verifying ONNX model output accuracy...")
        
        # PyTorch 推理
        with torch.no_grad():
            torch_out = model(dummy_input).numpy()
            
        # ONNX Runtime 推理
        ort_session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_out = ort_session.run(None, ort_inputs)[0]
        
        # 对比差异
        diff = np.max(np.abs(torch_out - ort_out))
        print(f"Max absolute difference between PyTorch and ONNX: {diff:.6e}")
        if diff < 1e-4:
            print("Verification PASSED! The outputs match perfectly.")
        else:
            print("Warning: Output difference is larger than expected.")
    except ImportError:
        print("onnxruntime is not installed. Skipping numerical verification.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch traffic sign model to ONNX")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["convnextv2_base", "convnext_base", "swin_v2_b", "efficientnet_v2_m"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PyTorch .pth checkpoint")
    parser.add_argument("--num_classes", type=int, default=43, help="Number of classes (GTSRB=43, TT100K=25)")
    parser.add_argument("--img_size", type=int, default=384, choices=[224, 256, 288, 384, 448], help="Input image size")
    parser.add_argument("--output", type=str, default="", help="Output .onnx file path")
    
    args = parser.parse_args()
    
    output_file = args.output if args.output else args.checkpoint.replace(".pth", ".onnx")
    export_to_onnx(args.model, args.checkpoint, args.num_classes, args.img_size, output_file)
