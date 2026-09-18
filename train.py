import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
import torchvision.models as models
from sklearn.metrics import f1_score, classification_report

# 尝试导入 timm 库以获取 ConvNeXt V2
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

# 尝试有条件地导入 tqdm，如果未安装则使用简单计数器
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable, desc=None):
            self.iterable = iterable
            self.desc = desc
        def __iter__(self):
            total = len(self.iterable) if hasattr(self.iterable, '__len__') else None
            print(f"Starting {self.desc or ''} (total batches: {total})")
            for idx, item in enumerate(self.iterable):
                yield item
                if idx > 0 and idx % 100 == 0:
                    print(f"  {self.desc or ''}: batch {idx}/{total}")

from dataset import TrafficSignDataset, get_train_transforms, get_val_transforms

# ========== 0. normalize_label (match official platform) ==========
def normalize_label(label):
    """Clean extra spaces in directory names to match classes.txt"""
    return ' '.join(str(label).strip().split())

# ========== 1. 损失函数 (Focal Loss) ==========
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ========== 1b. Mixup / CutMix 数据增强 ==========
def mixup_data(x, y, alpha=0.2):
    """Mixup: 线性插值两张图片及其标签"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss: 加权平均两组标签的损失"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def compute_class_weights(labels, num_classes, device):
    """Compute inverse-frequency class weights to balance Focal Loss"""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalize to mean=1
    return torch.tensor(weights, dtype=torch.float32, device=device)

# ========== 2. 训练与验证核心逻辑 ==========
def train_one_epoch(model, dataloader, optimizer, criterion, device, grad_accum=1, scaler=None, mixup_alpha=0.0):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    optimizer.zero_grad()
    for idx, (images, targets) in enumerate(tqdm(dataloader, desc="Training")):
        images = images.to(device)
        targets = targets.to(device)
        
        # Mixup 数据增强
        use_mixup = mixup_alpha > 0 and np.random.random() < 0.5  # 50% 概率启用 Mixup
        if use_mixup:
            images, targets_a, targets_b, lam = mixup_data(images, targets, mixup_alpha)
        
        # 使用自动混合精度 (AMP) 减少显存占用并加速训练
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                if use_mixup:
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = criterion(outputs, targets)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            
            if (idx + 1) % grad_accum == 0 or (idx + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)
            loss = loss / grad_accum
            loss.backward()
            
            if (idx + 1) % grad_accum == 0 or (idx + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
                
        running_loss += loss.item() * grad_accum
        
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        # 对于 Mixup，用原始标签计算 F1（近似）
        if use_mixup:
            all_targets.extend(targets_a.cpu().numpy())
        else:
            all_targets.extend(targets.cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro')
    return epoch_loss, epoch_f1

def validate(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            targets = targets.to(device)
            
            # 使用 autocast 降低评估时的显存占用
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            running_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
    val_loss = running_loss / len(dataloader)
    val_f1 = f1_score(all_targets, all_preds, average='macro')
    
    # 打印详细分类报告，找出低 F1 的困难类别
    report = classification_report(all_targets, all_preds, zero_division=0)
    print("\n--- Detailed Validation Classification Report ---")
    print(report)
    
    return val_loss, val_f1, all_preds, all_targets

# ========== 3. 创建与修改模型分类头 ==========
def build_model(model_name, num_classes, pretrained=True):
    if model_name == "convnextv2_base" or model_name == "convnext_base":
        if model_name == "convnextv2_base" and HAS_TIMM:
            print("Building ConvNeXt V2 Base from timm...")
            model = timm.create_model("convnextv2_base", pretrained=pretrained, num_classes=num_classes)
        else:
            if model_name == "convnextv2_base":
                print("timm not installed. Falling back to ConvNeXt V1 Base from torchvision.models...")
            else:
                print("Building ConvNeXt V1 Base from torchvision.models...")
            weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            model = models.convnext_base(weights=weights)
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, num_classes)
            
    elif model_name == "swin_v2_b":
        print(f"Building {model_name} from torchvision.models...")
        weights = models.Swin_V2_B_Weights.DEFAULT if pretrained else None
        model = models.swin_v2_b(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        
    elif model_name == "efficientnet_v2_m":
        print(f"Building {model_name} from torchvision.models...")
        weights = models.EfficientNet_V2_M_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_v2_m(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    return model

# ========== 4. 主程序 ==========
def main():
    parser = argparse.ArgumentParser(description="Traffic Sign Adverse Weather Training Pipeline")
    parser.add_argument("--model", type=str, default="convnext_base", 
                        choices=["convnextv2_base", "convnext_base", "swin_v2_b", "efficientnet_v2_m"],
                        help="Model architecture name from torchvision/timm")
    parser.add_argument("--dataset", type=str, default="gtsrb", choices=["gtsrb", "official"], help="Dataset type")
    parser.add_argument("--dataset_path", type=str, default="./datasets/train", help="Path to local dataset directory")
    parser.add_argument("--classes_file", type=str, default="./classes.txt", help="classes.txt with tab-separated index and class name")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--img_size", type=int, default=384, choices=[224, 256, 288, 384, 448], help="Input image size")
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"], help="Loss function")
    parser.add_argument("--pretrained_path", type=str, default="", help="Path to load weights for stage2 fine-tuning")
    parser.add_argument("--swa_start", type=int, default=40, help="Epoch to start SWA")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--no_pretrained", action="store_true", help="Do not load torchvision pre-trained weights")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without F1 improvement)")
    parser.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision (AMP) training")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers (0 to prevent RAM overhead on Windows)")
    parser.add_argument("--mixup_alpha", type=float, default=0.2, help="Mixup interpolation alpha (0=disabled)")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Linear warmup epochs before cosine decay")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 1. 加载数据集 ---
    if args.dataset == "gtsrb":
        import torchvision
        from torchvision.datasets import GTSRB
        num_classes = 43
        data_dir = "./datasets/gtsrb"
        print(f"Loading GTSRB dataset from {data_dir}...")
        
        raw_train = GTSRB(root=data_dir, split='train', download=True)
        raw_val = GTSRB(root=data_dir, split='test', download=True)
        
        train_dataset = TrafficSignDataset(raw_train, transform=get_train_transforms(args.img_size), is_torchvision=True)
        val_dataset = TrafficSignDataset(raw_val, transform=get_val_transforms(args.img_size), is_torchvision=True)
    else:
        dataset_path = args.dataset_path
        print(f"Loading official dataset from folder: {dataset_path}")
        
        # Read classes.txt to get the OFFICIAL class ordering (index -> name)
        # This ensures our label indices match the platform's expected order
        categories = []  # ordered by official index 0..24
        if args.classes_file and os.path.exists(args.classes_file):
            indexed = {}
            with open(args.classes_file, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    line = line.rstrip('\r\n')
                    if not line:
                        continue
                    idx_str, name = line.split('\t', 1)
                    indexed[int(idx_str)] = normalize_label(name)
            categories = [indexed[i] for i in range(len(indexed))]
            print(f"Loaded {len(categories)} classes from {args.classes_file}")
        else:
            # Fallback: use directory names sorted alphabetically
            categories = sorted([normalize_label(d) for d in os.listdir(dataset_path) 
                                if os.path.isdir(os.path.join(dataset_path, d))])
            
        print(f"Classes ({len(categories)}): {categories}")
        num_classes = len(categories)
        
        # Build class_to_idx mapping (matches platform ordering)
        class_to_idx = {name: idx for idx, name in enumerate(categories)}
        idx_to_class = {idx: name for idx, name in enumerate(categories)}
        
        # Map physical directory names to class indices
        physical_dirs = {normalize_label(d): d for d in os.listdir(dataset_path) 
                         if os.path.isdir(os.path.join(dataset_path, d))}
        
        # Collect all images with correct labels
        SUPPORTED = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.ppm'}
        all_imgs = []
        all_labels = []
        for class_name in categories:
            if class_name not in physical_dirs:
                print(f"WARNING: class '{class_name}' not found in dataset directory!")
                continue
            actual_dir = physical_dirs[class_name]
            cat_dir = os.path.join(dataset_path, actual_dir)
            for f in os.listdir(cat_dir):
                if os.path.splitext(f)[1].lower() in SUPPORTED:
                    all_imgs.append(os.path.join(cat_dir, f))
                    all_labels.append(class_to_idx[class_name])
                    
        print(f"Total samples collected: {len(all_imgs)}")
        
        # Show class distribution
        from collections import Counter
        dist = Counter(all_labels)
        for idx in range(num_classes):
            print(f"  class {idx:02d}: {dist.get(idx, 0):4d} images - {categories[idx]}")
        
        # Split train/val (85/15 ratio, stratified)
        from sklearn.model_selection import train_test_split
        train_imgs, val_imgs, train_labels, val_labels = train_test_split(
            all_imgs, all_labels, test_size=0.15, random_state=42, stratify=all_labels
        )
        
        print(f"Split sizes: Train={len(train_imgs)}, Val={len(val_imgs)}")
        
        train_dataset = TrafficSignDataset(train_imgs, train_labels, transform=get_train_transforms(args.img_size))
        val_dataset = TrafficSignDataset(val_imgs, val_labels, transform=get_val_transforms(args.img_size))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # --- 2. 初始化模型 ---
    pretrained = not args.no_pretrained
    try:
        model = build_model(args.model, num_classes, pretrained=pretrained)
    except Exception as e:
        print(f"Failed to build model with pre-trained weights: {e}")
        print("Falling back to random initialization...")
        model = build_model(args.model, num_classes, pretrained=False)
    
    # 加载第一阶段预训练权重（用于第二阶段微调）
    if args.pretrained_path:
        print(f"Loading stage-1 checkpoint from: {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=False)
        
    model.to(device)
    
    # --- 3. 配置训练组件 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    
    # 计算类别权重（仅 Focal Loss）
    class_weights = None
    if args.loss == "focal" and args.dataset == "official":
        try:
            all_train_labels = [train_dataset[i][1].item() if hasattr(train_dataset[i][1], 'item') 
                                else train_dataset[i][1] for i in range(min(1000, len(train_dataset)))]
            class_weights = compute_class_weights(all_train_labels, num_classes, device)
            print(f"Auto-computed class weights for {num_classes} classes")
        except Exception:
            pass
    
    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        
    # 学习率调度：Warmup + Cosine Decay
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs  # 线性预热
        # 余弦衰减
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # 混合精度 GradScaler 声明
    scaler = torch.cuda.amp.GradScaler() if (device.type == 'cuda' and not args.no_amp) else None
    
    # SWA 准备
    swa_model = None
    if args.epochs > args.swa_start:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=args.lr * 0.1)
        
    # --- 4. 训练循环 ---
    best_val_f1 = 0.0
    patience_counter = 0
    os.makedirs("./results", exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n================ Epoch {epoch}/{args.epochs} (lr={optimizer.param_groups[0]['lr']:.2e}) ================")
        
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device, 
            grad_accum=args.grad_accum, scaler=scaler, mixup_alpha=args.mixup_alpha
        )
        
        val_loss, val_f1, _, _ = validate(model, val_loader, criterion, device, num_classes)
        
        print(f"Epoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Macro-F1: {train_f1:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Macro-F1:   {val_f1:.4f}")
        
        # 学习率更新
        if swa_model and epoch >= args.swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
            
        # 保存最佳模型 与 早停逻辑 (使用官方 checkpoint 格式)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_path = f"./results/{args.model}_best.pth"
            # Official platform checkpoint format
            save_dict = {
                'model_state_dict': model.state_dict(),
                'num_classes': num_classes,
                'input_size': args.img_size,
                'model_name': args.model,
                'best_macro_f1': best_val_f1,
                'epoch': epoch,
            }
            # Save class mappings for official dataset
            if args.dataset == 'official' and 'class_to_idx' in dir():
                save_dict['class_to_idx'] = class_to_idx
                save_dict['idx_to_class'] = idx_to_class
            torch.save(save_dict, best_model_path)
            print(f"[Best] New best Val F1: {best_val_f1:.4f}. Model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"[Info] Val F1 did not improve. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"[Early Stopping] Training stopped early after {epoch} epochs due to no improvement in Val F1.")
                break
            
    # SWA 模型评估与保存
    if swa_model:
        print("\n================ SWA Evaluation ================")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        swa_val_loss, swa_val_f1, _, _ = validate(swa_model, val_loader, criterion, device, num_classes)
        print(f"SWA Model - Val Loss: {swa_val_loss:.4f} | Val Macro-F1: {swa_val_f1:.4f}")
        
        if swa_val_f1 > best_val_f1:
            best_val_f1 = swa_val_f1
            swa_model_path = f"./results/{args.model}_swa.pth"
            save_dict = {
                'model_state_dict': swa_model.module.state_dict(),
                'num_classes': num_classes,
                'input_size': args.img_size,
                'model_name': args.model,
                'best_macro_f1': best_val_f1,
            }
            if args.dataset == 'official' and 'class_to_idx' in dir():
                save_dict['class_to_idx'] = class_to_idx
                save_dict['idx_to_class'] = idx_to_class
            torch.save(save_dict, swa_model_path)
            print(f"[SWA] SWA Model outperformed standard best model! Saved to {swa_model_path}")

    print(f"\nTraining completed! Best Validation Macro-F1: {best_val_f1:.4f}")

if __name__ == "__main__":
    main()
