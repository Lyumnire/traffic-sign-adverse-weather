## 🔴 核心发现：CPU 推理彻底改变选型逻辑

你提供的信息里最关键的一条是：**推理设备是 2核8GB 的 CPU，不是 GPU**。这一点直接否决了 Gemini 推荐中的两个模型。

---

## 一、Gemini 方案 vs 我的方案 — 逐模型评估

### Gemini 推荐的三个模型在 CPU 上的表现：

| 模型 | 架构类型 | 参数量 | GFLOPs (224) | CPU 推理速度 | 国赛可用性 |
|------|---------|--------|-------------|-------------|-----------|
| **ConvNeXt V2-Tiny/Small** | 纯 CNN | 28M/50M | 4.5/8.7 | ✅ 快，CPU 友好 | **✅ 推荐采用** |
| **Swin Transformer V2** | Transformer（窗口注意力） | 28M(T)/50M(S) | 4.5/8.7 | ❌ 慢，注意力机制在 CPU 上开销大 | **⚠️ 有风险** |
| **EVA-02** | ViT（全局注意力） | 87M(Base) | 17.6 | ❌❌ 非常慢，全局 self-attention 在 CPU 上极慢 | **❌ 不建议** |

**关键原因：** CPU 对卷积操作有高度优化的底层库（MKL/BLAS），但 Transformer 的 self-attention 涉及大量矩阵乘、softmax、reshape 操作，这些在 CPU 上没有同等级别的加速支持。实测中，同参数量的 ViT 在 CPU 上比 CNN 慢 **2-5 倍**。

### 我的结论：取两家之长

| 位置 | 最终推荐 | 来源 | 理由 |
|------|---------|------|------|
| 模型1 | **ConvNeXt V2-Small** | ✅ Gemini推荐 + 我认同 | 预训练质量最高的 CNN（FCMAE 自监督），CPU 友好，细粒度能力强 |
| 模型2 | **EfficientNetV2-S** | ✅ 我的推荐 | 专为效率设计的架构，CPU 上速度极快，与 ConvNeXt 特征互补 |
| 模型3 | **ConvNeXt-Small (V1)** | ✅ 我的推荐 | 与 V2 架构相似但预训练方式不同（监督学习 vs 自监督），提供多样性 |
| 备选4 | **EfficientNet-B4** | 备选 | 如果时间富余可加入，进一步提升多样性 |

---

## 二、为什么不用 Swin V2 和 EVA-02？详细推算

### 时间预算分析

初赛信息：
- 3 个小模型（ConvNeXt-Tiny 768ch + MobileNetV3 960ch + EffNet-B2 1408ch）
- 224×224 输入，含 TTA（水平翻转 = 2× 推理量）
- 总耗时 ~10 分钟 → 每模型每张图约 **30-50ms**（估算测试集约 1000-2000 张）

如果用 Swin-Small（Transformer）：
- 同参数量，CPU 上约慢 **2.5-3 倍** → 每张图 ~100-150ms
- 3 个模型 + TTA → 可能 25-40 分钟

如果用 EVA-02-Base（87M ViT）：
- 参数量大 3 倍 + Transformer 架构 → 每张图 ~300-500ms
- 单个模型就可能吃掉 15-20 分钟

**风险太大。** 国赛测试集图片数量未知，可能比初赛多。如果测试集有 3000-5000 张，Transformer 模型很可能超时。

### 纯 CNN 方案的时间估算

ConvNeXt V2-Small (50M) + EfficientNetV2-S (22M) + ConvNeXt-Small (50M)：
- 比初赛三模型平均大约 **1.5-2 倍**
- 预估总耗时 15-25 分钟（取决于测试集大小）
- 距 70 分钟上限仍有 **充足余量**
- 如果用 ONNX Runtime 加速，还能再快 30-50%

---

## 三、ONNX 加速 — 国赛的隐藏武器

在 CPU 推理场景下，将 PyTorch 模型转为 ONNX 是一个**低成本高回报**的优化手段：

```python
# 导出 ONNX
import torch
model.eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, "model.onnx", opset_version=13,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})

# 用 ONNX Runtime 推理
import onnxruntime as ort
session = ort.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])
session.set_providers(['CPUExecutionProvider'], 
                      [{'intra_op_num_threads': 2}])  # 匹配2核CPU

result = session.run(None, {'input': input_numpy})
```

预期加速：**30-50%**（Conv+BN 融合 + 图优化 + 算子融合）。这意味着你可以用更大的模型而不必担心超时。

---

## 四、外部数据的策略 — 你的担忧是对的

你问的"扩充了会不会反而降低 F1"——**这个担忧非常合理**。关键在于怎么用：

### ❌ 错误用法（会降低 F1）
- 把 GTSRB/TT100K 全部混入训练集直接训练
- 外部数据的类别定义和官方不一致
- 外部数据的退化特征（天气、模糊）和官方差异太大
- 域偏移 (domain shift) 导致模型学到错误的分布

### ✅ 正确用法（安全提升）

**方法一：两阶段微调（最安全，强烈推荐）**
```
第1阶段：在外部数据（GTSRB/TT100K 筛选后）上预热 10-15 epochs
         ↓ 学到交通标志的通用形状/颜色特征
第2阶段：在官方数据集上精细微调 30-50 epochs
         ↓ 适应官方的类别定义和退化分布
```
这样外部数据只提供"先验知识"，最终分布完全由官方数据决定。

**方法二：仅用于增强少数类（按需）**
- 如果官方 25 类中某些类别样本特别少
- 只从外部数据中挑选该类别的高质量匹配图片补充
- 用较小权重（如 loss weight 0.3×）训练

**方法三：不用外部分类数据，只用增强技术**
- 如果你担心域偏移，最安全的做法是完全不用外部数据
- 转而投入更多精力在数据增强上模拟恶劣天气
- 用 Albumentations 的 RandomFog/RandomRain/RandomSnow/MotionBlur 等
- 这样"增加"的数据保证和原始分布一致

**我的建议：** 先跑一个只用官方数据的 baseline，再跑一个加入外部数据两阶段微调的版本，在验证集上比较。哪个 macro F1 高用哪个，**用数据说话**。

---

## 五、从 0.922 到 0.95+ 的提分路线图

初赛 0.922，顶尖 0.95-0.96，差距约 **3-4 分**。国赛任务更难（25 类 vs 4 类），但提升空间也更大。以下按预期收益排序：

### 提分手段排序

| 优先级 | 手段 | 预期提升 | 难度 |
|--------|------|---------|------|
| **P0** | 模型升级（Tiny→Small/Medium） | +2~3% | 低 |
| **P0** | 更强的数据增强（模拟恶劣天气） | +1~2% | 低 |
| **P0** | 去掉不适合的 TTA（禁止水平翻转！） | 避免 -1~2% | 低 |
| **P1** | 输入分辨率提升（224→288 或 320） | +1~2% | 低 |
| **P1** | Label Smoothing + Mixup/CutMix | +0.5~1% | 低 |
| **P1** | 集成权重在验证集上搜索优化 | +0.5~1% | 中 |
| **P2** | K-Fold 交叉验证 + OOF 集成 | +0.5~1% | 中 |
| **P2** | SWA（随机权重平均）/ EMA | +0.3~0.5% | 低 |
| **P2** | ONNX 加速 → 省出时间 → 加更多模型或更大分辨率 | 间接提升 | 中 |
| **P3** | 困难样本可视化 + 针对性增强 | +0.5~1% | 高 |
| **P3** | 逐类阈值优化（macro F1 专项） | +0.3~0.5% | 中 |
| **P3** | 外部数据两阶段预训练 | +0~1%（不确定） | 中 |

### 特别提醒：TTA 策略修改

初赛用水平翻转 TTA 对天气分类没问题，但国赛**绝对不能用水平翻转**：
- 「靠右行驶」↔「靠左行驶」
- 「左转弯」↔「右转弯」  
- 「向右急弯」↔「向左急弯」
- 方向性箭头标志全会翻转错误

**安全的 TTA 替代方案：**
- 多尺度推理：分别用 256、288、320 三个尺度推理，概率平均
- 中心裁剪 + 原图：原图 resize + 中心裁剪的两种预处理取平均
- 轻微色彩扰动（对抗天气颜色失真）

---

## 六、最终推荐的技术栈

```
模型架构：
  ├─ ConvNeXt V2-Small (timm: convnextv2_small.fcmae_ft_in22k_in1k)
  ├─ EfficientNetV2-S   (timm: tf_efficientnetv2_s.in21k_ft_in1k)  
  └─ ConvNeXt-Small V1  (timm: convnext_small.fb_in22k_ft_in1k)

输入分辨率：288×288（平衡精度和速度）或 320×320

损失函数：CrossEntropy + Label Smoothing 0.1

优化器：AdamW, lr=2e-4, weight_decay=0.05

训练增强：
  ├─ RandomFog / RandomRain / RandomSnow（模拟天气）
  ├─ MotionBlur / GaussianBlur（模拟模糊）
  ├─ RandomBrightnessContrast（模拟低照度）
  ├─ CoarseDropout（模拟遮挡）
  ├─ ImageCompression（模拟压缩失真）
  ├─ Mixup (α=0.2) + CutMix (α=1.0)
  └─ ShiftScaleRotate（小范围几何变换）

推理优化：
  ├─ 3模型加权软投票（权重在验证集搜索）
  ├─ 安全 TTA（多尺度，不翻转）
  ├─ ONNX Runtime 加速（可选但推荐）
  └─ Temperature Scaling 后处理

正则化：
  ├─ DropPath 0.2
  ├─ SWA（最后10个epoch）
  └─ EMA（decay=0.9999）
```

---

## 七、总结回答你的三个核心问题

**Q1: 还是应该采用初赛的三模型推理策略吗？**
> ✅ **三模型集成框架保留**，这是正确的方向。但自定义的 WeatherDecoder/RoadDecoder/CoordAtt 全部去掉，改用 timm 标准分类头。TTA 策略必须改——禁止水平翻转。

**Q2: 模型选型应不应该变化？**
> ✅ **必须变化**。三个模型全部升级。推荐 **ConvNeXt V2-Small + EfficientNetV2-S + ConvNeXt-Small(V1)**。Gemini 推荐的 ConvNeXt V2 采纳，Swin V2 和 EVA-02 因 CPU 推理太慢而放弃。

**Q3: 需要下载哪些数据集？**
> 推荐下载 **GTSRB** 和 **TT100K**，但**不要直接混入训练**。用两阶段微调法或仅做预热。如果不确定效果，先跑纯官方数据 baseline 对比。最安全的"增强"方式是用 Albumentations 模拟恶劣天气，而不是引入外部数据。