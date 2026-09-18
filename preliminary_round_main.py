import torch
import torch.nn as nn
import numpy as np
import cv2

# ========== 配置 ==========
label = ['cloudy', 'rainy', 'snowy', 'sunny']
im_size = 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(2)  # 限制线程数，避免CPU上下文切换开销


# ========== 公共模块 ==========

class WeatherDecoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_channels * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        avg_pool = self.gap(x).flatten(1)
        max_pool = self.gmp(x).flatten(1)
        combined = torch.cat([avg_pool, max_pool], dim=1)
        return self.classifier(combined)


class RoadDecoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        return self.conv(x).flatten(1)


# ========== ConvNeXt模型 ==========

class ConvNeXtWeather(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        from torchvision import models
        self.backbone = models.convnext_tiny(weights=None)
        self.backbone.classifier = nn.Identity()
        self.weather_decoder = WeatherDecoder(768, num_classes)
        self.road_decoder = RoadDecoder(768)

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), 768, 1, 1)
        weather_out = self.weather_decoder(features)
        road_feat = self.road_decoder(features)
        return weather_out, road_feat


# ========== MobileNetV3模型 ==========

class CoordAtt(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(CoordAtt, self).__init__()
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = torch.mean(x, dim=3, keepdim=True)
        x_w = torch.mean(x, dim=2, keepdim=True).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        out = x * a_h * a_w
        return out


class MobileNetV3Weather(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        from torchvision import models
        backbone = models.mobilenet_v3_large(weights=None)
        self.backbone = backbone
        self.coord_att = CoordAtt(in_channels=960)
        self.weather_decoder = WeatherDecoder(960, num_classes)
        self.road_decoder = RoadDecoder(960)

    def forward(self, x):
        features = self.backbone.features(x)
        features = self.coord_att(features)
        weather_out = self.weather_decoder(features)
        road_feat = self.road_decoder(features)
        return weather_out, road_feat


# ========== EfficientNet-B2模型 ==========

class EfficientNetWeather(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        from torchvision import models
        backbone = models.efficientnet_b2(weights=None)
        self.backbone = backbone.features
        self.in_channels = 1408
        self.weather_decoder = WeatherDecoder(self.in_channels, num_classes)
        self.road_decoder = RoadDecoder(self.in_channels)

    def forward(self, x):
        features = self.backbone(x)
        weather_out = self.weather_decoder(features)
        road_feat = self.road_decoder(features)
        return weather_out, road_feat


# ========== 加载模型 ==========

def load_model(model_class, weight_path, name):
    model = model_class(num_classes=4).to(device)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in checkpoint.items() if k in model_keys}
    model.load_state_dict(filtered)
    model.eval()
    return model

convnext_model = load_model(ConvNeXtWeather, "./results/convnext_weather_final.pth", "ConvNeXt")
mobilenet_model = load_model(MobileNetV3Weather, "./results/mobilenetv3_coordatt_weather_final.pth", "MobileNetV3")
efficientnet_model = load_model(EfficientNetWeather, "./results/efficientnetb2_weather_final.pth", "EfficientNet-B2")


# ========== 核心推断逻辑 ==========

# 类别先验（核心改进：补偿类别不平衡）
train_priors = torch.tensor([0.434, 0.091, 0.083, 0.392], device=device)


def preprocess(X):
    """标准预处理（与训练时一致）"""
    X = cv2.resize(X, (im_size, im_size))
    X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
    X = X.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    X = (X - mean) / std
    X = np.transpose(X, (2, 0, 1))
    return X


def predict_single_tensor(X_tensor):
    """三模型算术平均推理"""
    with torch.no_grad():
        convnext_out, _ = convnext_model(X_tensor)
        mobilenet_out, _ = mobilenet_model(X_tensor)
        efficientnet_out, _ = efficientnet_model(X_tensor)

        probs = (torch.softmax(convnext_out, dim=1) +
                torch.softmax(mobilenet_out, dim=1) +
                torch.softmax(efficientnet_out, dim=1)) / 3.0
    return probs


def predict(X):
    """
    模型预测（三模型 + TTA + 先验校准）
    """
    # 预处理
    X_processed = preprocess(X)
    X_tensor = torch.from_numpy(X_processed).float().unsqueeze(0).to(device)

    # 原图推理
    probs_original = predict_single_tensor(X_tensor)

    # TTA：水平翻转（PyTorch张量操作，更快）
    X_tensor_flipped = torch.flip(X_tensor, dims=[3])
    probs_flipped = predict_single_tensor(X_tensor_flipped)

    # TTA平均
    probs_avg = (probs_original + probs_flipped) / 2.0

    # 先验校准（补偿类别不平衡）
    adjusted_probs = probs_avg / (train_priors ** 0.5)
    adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=1, keepdim=True)

    # 取最大概率类别
    best_class = torch.argmax(adjusted_probs, dim=1).item()

    return label[best_class]
