import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

# ========== torchvision.transforms 数据增强 ==========
def get_train_transforms(img_size):
    return T.Compose([
        T.Resize((img_size, img_size)),
        # 几何变换（温和，不进行水平/垂直翻转以防标志混淆）
        T.RandomAffine(
            degrees=(-8, 8),
            translate=(0.08, 0.08),
            scale=(0.9, 1.1),
            interpolation=T.InterpolationMode.BILINEAR
        ),
        # 亮度/对比度/饱和度/色调扰动 (模拟低照度、眩光、退化)
        T.ColorJitter(
            brightness=(0.7, 1.1),
            contrast=(0.7, 1.1),
            saturation=(0.8, 1.2),
            hue=(-0.05, 0.05)
        ),
        # 模糊变换（模拟运动模糊/气象能见度低）
        T.RandomApply([
            T.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0))
        ], p=0.3),
        # 转换为 Tensor
        T.ToTensor(),
        # 标准化
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        # 随机擦除 (模拟遮挡/污损)
        T.RandomErasing(
            p=0.3,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3),
            value=0
        )
    ])

def get_val_transforms(img_size):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

class TrafficSignDataset(Dataset):
    """
    通用交通标志数据集包装器，使用 PIL 和 torchvision.transforms，零额外依赖。
    """
    def __init__(self, data, labels=None, transform=None, is_torchvision=False):
        """
        Args:
            data: 如果 is_torchvision=True, 为 torchvision dataset;
                  否则为图片路径的 list。
            labels: 如果 is_torchvision=False, 为对应的标签 list。
            transform: torchvision Compose 转换。
            is_torchvision: 是否是 torchvision 原生数据集。
        """
        self.data = data
        self.labels = labels
        self.transform = transform
        self.is_torchvision = is_torchvision

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_torchvision:
            # torchvision.datasets.GTSRB 返回 (PIL.Image, label)
            image, label = self.data[idx]
        else:
            img_path = self.data[idx]
            label = self.labels[idx]
            # 使用 PIL 读取
            image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            # 默认简单归一化
            default_trans = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = default_trans(image)

        return image, torch.tensor(label, dtype=torch.long)
