# 恶劣天气下的交通标识识别 — 国赛技术手册

> **用途**：本文档供 Claude Code、Gemini 等 AI 助手及队伍成员快速了解赛题背景、当前技术储备、策略选型及代码架构。

---

## 一、赛题概述

| 项目 | 内容 |
| --- | --- |
| **任务名称** | 恶劣天气下的交通标识识别 |
| **任务类型** | 图像级单标签分类（25 类交通标志） |
| **评估指标** | **Macro-F1 Score** |
| **推理环境** | 2 核 CPU + 8GB 内存（无 GPU），总推理时长限制 70 分钟 |
| **测试集规模** | 约 750 ~ 800 张图片 |
| **模型大小限制** | 无 |
| **外部数据** | 允许使用 |
| **初赛成绩** | F1 = 0.922（二等奖，前 40%；顶尖队伍 0.95 ~ 0.96） |

### 赛题核心挑战
图像来自真实道路场景（车载摄像头、道路监控、行车记录仪），包含以下退化：
- **天气退化**：雨雪、雾霾、沙尘
- **光照退化**：夜间低照度、车灯眩光
- **成像退化**：运动模糊、压缩失真
- **物理退化**：遮挡、污损

### 25 类交通标识
具体类别名称将在**官方数据集发布后**填入 `main.py` 的 `label` 列表和 `selected_25_classes.txt`。根据消息透露，官方数据集可能采用国外标志（如 GTSRB 风格）。涵盖范围：
- **禁令标志**：限速、禁止通行、禁止左/右转等
- **警告标志**：注意路面不平、湿滑等
- **指示标志**：直行、靠右行驶、停车场等
- **道路通行标志**：其他类型

---

## 二、技术选型与策略

### 2.1 模型三剑客

| 编号 | 模型 | 来源 | 参数量 | 推理分辨率 | CPU 特性 |
| --- | --- | --- | --- | --- | --- |
| M1 | **ConvNeXt V2-Base** | timm (`convnextv2_base`) | 89M | 384×384 | 纯 CNN，CPU MKL 加速友好 |
| M2 | **Swin Transformer V2-Base** | torchvision (`swin_v2_b`) | 88M | 256×256 | 窗口注意力，适中的 CPU 开销 |
| M3 | **EfficientNetV2-M** | torchvision (`efficientnet_v2_m`) | 54M | 384×384 | 极致效率设计，CPU 最快 |

**选型逻辑**：
- ConvNeXt V2 → 通过 FCMAE 自监督预训练具有卓越的细粒度特征提取能力
- Swin V2 → Transformer 架构与 CNN 互补，提供特征多样性，使用 256 分辨率控制 CPU 推理时间
- EfficientNetV2 → 效率/精度帕累托最优，作为集成中的"速度担当"

### 2.2 核心策略

#### A. 训练策略
1. **首选方案：直接用官方数据集训练（ImageNet 预训练权重初始化）**：
   - 加载 timm/torchvision 的 ImageNet 预训练权重，直接在官方 25 类数据上训练 50 Epoch
   - 初赛经验证明：当外部数据集（如 BDD100K、GTSRB）与官方数据的域差异较大时，微调反而不如直接训练
   - **这是我们的主线策略**
2. **备用方案：GTSRB 预训练 + 官方数据微调（仅当官方数据确认为德国标志风格时）**：
   - 先在 GTSRB 上预热，再在官方数据上微调
   - 但需跟直接训练的结果做对比，用数据说话
3. **损失函数**：Focal Loss（gamma=2.0）+ 自动类别权重 — 抑制长尾类别不均衡
4. **标签平滑**：Label Smoothing = 0.1 — 防止过度自信
5. **Mixup 数据增强**：alpha=0.2，50% 概率启用 — 增加泛化性
6. **Warmup + Cosine LR**：前 3 个 Epoch 线性预热，之后余弦衰减
7. **SWA（随机权重平均）**：最后 10 个 Epoch 启用 — 提升泛化
8. **早停**：验证集 Macro-F1 连续 5 轮未提升则自动停止

#### B. 数据增强策略（模拟恶劣天气）
- `RandomAffine`：±8° 旋转、8% 平移、0.9~1.1 缩放
- `ColorJitter`：模拟低照度/眩光/退化
- `GaussianBlur`：模拟运动模糊/气象能见度低
- `RandomErasing`：模拟遮挡/污损
- **禁止水平翻转**：交通标志有方向性（左/右转、靠左/右行驶）

#### C. 推理策略
1. **三模型软投票集成**：对 3 个模型的 softmax 概率取算术平均
2. **多尺度 TTA（安全版）**：每模型用 2 个尺度推理取平均（**不做水平翻转**）
3. **推理格式：首选 PyTorch .pth，ONNX 仅作备用**：
   - 默认使用 PyTorch 原生 .pth 推理 — 最安全，无精度损失，无平台兼容风险
   - 如果运行时逆近 70 分钟上限，再考虑切换 ONNX（需确认平台支持 onnxruntime）
   - .pth 可以随时转 .onnx，转换过程不会降低模型精度
4. **CPU 线程控制**：`torch.set_num_threads(2)` 匹配评测 CPU 核数

### 2.3 为什么不用某些模型？
- **EVA-02**：全局 ViT 注意力，CPU 上推理极慢（同参数量比 CNN 慢 3~5 倍），超时风险大
- **MobileNet/EfficientNet-B2**：初赛用过的小模型，容量不足以达到 0.95+ 精度上限

---

## 三、项目文件结构

```
D:\MOPRO\
├── train.py                    # 核心训练脚本（Mixup、Warmup、AMP、早停、SWA、Focal Loss + 类别权重）
├── train_v2_backup.py          # train.py 修改前的备份
├── dataset.py                  # 数据加载与增强（纯 PIL + torchvision.transforms，零额外依赖）
├── main.py                     # 推理脚本（三模型集成 + 多尺度 TTA + ONNX 自适应）
├── export_onnx.py              # PyTorch → ONNX 导出工具（备用）
├── selected_25_classes.txt     # 选定的 25 类标志列表（待官方数据更新）
│
├── run_train_official.bat      # ★ 首选：直接训练官方数据集（ImageNet 权重初始化）
├── run_finetune_official.bat   # 备用：GTSRB 预训练权重微调官方数据
├── run_train_all_gtsrb.bat     # GTSRB 预热训练（Stage 1）
├── run_train_all.bat           # 旧版一键训练
├── run_test_train.bat          # 快速测试训练流程
│
├── results/                    # 模型权重输出目录
│   ├── convnextv2_base_best.pth    # ConvNeXt V2 GTSRB 预训练权重（351MB）
│   ├── swin_v2_b_best.pth         # Swin V2 GTSRB 预训练权重（349MB）
│   └── efficientnet_v2_m_best.pth # EfficientNetV2 GTSRB 预训练权重（213MB）
│
├── datasets/
│   ├── gtsrb/                 # GTSRB 德国标志数据集（43 类）
│   └── tt100k_cropped/        # TT100K 中国标志裁剪数据集（221 类）
│
├── .venv/                     # Python 虚拟环境（含 torch, torchvision, timm, tqdm, sklearn）
├── implementation_plan.md     # 实施计划详细文档
└── 加速建议.txt               # CPU 推理加速策略分析
```

---

## 四、环境与依赖

| 组件 | 版本/说明 |
| --- | --- |
| **Python** | 3.10 |
| **PyTorch** | ~2.1.x（平台兼容 2.x 系列） |
| **torchvision** | 与 PyTorch 对应版本 |
| **timm** | 1.0.28（用于 ConvNeXt V2） |
| **scikit-learn** | 用于 train_test_split、classification_report |
| **PIL (Pillow)** | 图像读取 |
| **训练硬件** | NVIDIA RTX 3080 20GB VRAM |
| **推理硬件（评测）** | 2 核 CPU + 8GB RAM（无 GPU） |

### 网络环境注意事项
本机使用 Clash Verge 代理（TUN/Fake-IP 模式），会强制劫持所有 DNS 和 HTTPS。因此：
- `pip install` 可能遭遇 `SSLEOFError`，需使用阿里云镜像 + `--trusted-host`
- 模型预训练权重下载可能失败 → 代码已内置 fallback 逻辑
- 所有训练/推理脚本均可完全离线运行

---

## 五、训练命令速查

### 5.1 ★ 首选：直接训练官方数据集（ImageNet 预训练权重初始化）
```powershell
# EfficientNetV2-M
.venv\Scripts\python.exe train.py --model efficientnet_v2_m --dataset official --dataset_path ./datasets/official_train --epochs 50 --batch_size 16 --grad_accum 2 --img_size 384 --loss focal --swa_start 40 --patience 5 --num_workers 2 --mixup_alpha 0.2 --warmup_epochs 3

# ConvNeXt V2-Base
.venv\Scripts\python.exe train.py --model convnextv2_base --dataset official --dataset_path ./datasets/official_train --epochs 50 --batch_size 16 --grad_accum 2 --img_size 384 --loss focal --swa_start 40 --patience 5 --num_workers 2 --mixup_alpha 0.2 --warmup_epochs 3

# Swin V2-Base
.venv\Scripts\python.exe train.py --model swin_v2_b --dataset official --dataset_path ./datasets/official_train --epochs 50 --batch_size 32 --img_size 256 --loss focal --swa_start 40 --patience 5 --num_workers 2 --mixup_alpha 0.2 --warmup_epochs 3
```

### 5.2 备用：GTSRB 预训练权重微调（仅当确认官方数据为德国标志风格时尝试）
```powershell
# EfficientNetV2-M（微调）
.venv\Scripts\python.exe train.py --model efficientnet_v2_m --dataset official --dataset_path ./datasets/official_train --epochs 30 --batch_size 16 --grad_accum 2 --img_size 384 --loss focal --swa_start 25 --patience 5 --num_workers 2 --pretrained_path ./results/efficientnet_v2_m_best.pth

# ConvNeXt V2-Base（微调）
.venv\Scripts\python.exe train.py --model convnextv2_base --dataset official --dataset_path ./datasets/official_train --epochs 30 --batch_size 16 --grad_accum 2 --img_size 384 --loss focal --swa_start 25 --patience 5 --num_workers 2 --pretrained_path ./results/convnextv2_base_best.pth

# Swin V2-Base（微调）
.venv\Scripts\python.exe train.py --model swin_v2_b --dataset official --dataset_path ./datasets/official_train --epochs 30 --batch_size 32 --img_size 256 --loss focal --swa_start 25 --patience 5 --num_workers 2 --pretrained_path ./results/swin_v2_b_best.pth
```

### 5.3 ONNX 导出
```powershell
.venv\Scripts\python.exe export_onnx.py --model convnextv2_base --checkpoint ./results/convnextv2_base_best.pth --num_classes 25 --img_size 384
.venv\Scripts\python.exe export_onnx.py --model swin_v2_b --checkpoint ./results/swin_v2_b_best.pth --num_classes 25 --img_size 256
.venv\Scripts\python.exe export_onnx.py --model efficientnet_v2_m --checkpoint ./results/efficientnet_v2_m_best.pth --num_classes 25 --img_size 384
```

---

## 六、已完成的技术储备

### 6.1 GTSRB 预训练模型（Stage 1 已完成）
三个模型均已在 GTSRB 43 类数据集上完成预训练，权重保存在 `./results/` 目录：
- `convnextv2_base_best.pth`（351MB）
- `swin_v2_b_best.pth`（349MB）
- `efficientnet_v2_m_best.pth`（213MB）

这些权重作为**迁移学习的底座**，在官方数据发布后只需替换分类头并微调即可快速收敛。

### 6.2 TT100K 数据集（已清洗备用）
- 从 TT100K 街景大图中流式裁剪出 **26,349 张**精确的中国交通标志小图
- 分类存放在 `D:\MOPRO\datasets\tt100k_cropped\`
- 如果官方数据集确认为中国标志，可作为补充训练数据

### 6.3 训练脚本功能清单
- [x] AMP 自动混合精度训练
- [x] Focal Loss 抑制类别不均衡
- [x] Label Smoothing 0.1
- [x] CosineAnnealingWarmRestarts 学习率调度
- [x] SWA 随机权重平均
- [x] Early Stopping（patience=5，基于 Val Macro-F1）
- [x] 梯度累加（等效大 Batch）
- [x] 分类报告逐类 Precision/Recall/F1 输出
- [x] 两阶段微调支持（--pretrained_path）

---

## 七、待办事项（官方数据发布后）

1. **更新类别**：将官方 25 类名称写入 `selected_25_classes.txt` 和 `main.py` 的 `label` 列表
2. **替换数据**：解压官方训练集到 `D:\MOPRO\datasets\official_train\`（以类别子文件夹组织）
3. **Stage 2 微调**：运行上方 5.2 节的命令
4. **验证推理**：运行 `main.py` 测试推理流程
5. **提交前检查**：确保 `main.py` 中没有使用水平翻转 TTA

---

## 八、关键注意事项

> **[!CAUTION] 绝对禁止水平翻转 TTA**
> 交通标志有方向性！"靠右行驶"翻转后变成"靠左行驶"，"左转弯"变成"右转弯"。初赛（天气分类）用翻转没问题，但国赛必须禁止。

> **[!WARNING] 推理时间预算**
> 3 个 Base 级模型 + 双尺度 TTA = 每张图约 6 次推理。在 2 核 CPU 上，每次约 0.5~1.5 秒。800 张图 × 6 次 × 1 秒 ≈ 80 分钟。如果逼近上限，优先：
> 1. 去掉 TTA（最简单）
> 2. 使用 ONNX Runtime 加速
> 3. 减少集成模型数量（保留 2 个最优）

> **[!IMPORTANT] Stage 2 微调时分类头不匹配处理**
> GTSRB 有 43 类，官方有 25 类。在加载 `--pretrained_path` 时，`strict=False` 会自动忽略分类头权重不匹配的层，只迁移特征提取器部分。
