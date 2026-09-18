import torch
import time
import urllib.request
import os

try:
    import timm
except ImportError:
    print("timm not installed. Please install timm to run this benchmark.")
    exit(1)

def benchmark_model(model_name, resolution):
    print(f"Benchmarking {model_name} at {resolution}x{resolution} on CPU...")
    try:
        # Create model (no pretrained weights to save download time)
        model = timm.create_model(model_name, pretrained=False)
        model.eval()
        
        # CPU config
        torch.set_num_threads(2)
        
        # Dummy input
        x = torch.randn(1, 3, resolution, resolution)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(x)
                
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(10):
                start = time.time()
                _ = model(x)
                times.append(time.time() - start)
                
        avg_time = sum(times) / len(times)
        print(f"[{model_name}] Avg CPU Inference Time: {avg_time*1000:.2f} ms")
        return avg_time
    except Exception as e:
        print(f"Failed to benchmark {model_name}: {e}")
        return -1

if __name__ == "__main__":
    models_to_test = [
        ('convnextv2_base', 384),
        ('swinv2_base_window12to16_192to256_22kft1k', 256), # Using generic names
        ('eva02_base_patch14_448.mim_in22k_ft_in22k_in1k', 448),
        ('tf_efficientnetv2_m', 384) # EfficientNetV2-M
    ]
    
    for m, res in models_to_test:
        benchmark_model(m, res)
