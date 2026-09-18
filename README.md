# 恶劣天气下的交通标识识别 (Traffic Sign Recognition under Adverse Weather)

<p align="center">
  <img src="https://img.shields.io/badge/Award-National%20Second%20Prize%20(%E5%9B%BD%E8%B5%9B%E4%BA%8C%E7%AD%89%E5%A5%96)-gold?style=for-the-badge&logo=trophy" alt="National Second Prize" />
  <img src="https://img.shields.io/badge/National%20Macro--F1-0.945-brightgreen?style=for-the-badge" alt="Macro F1 0.945" />
  <img src="https://img.shields.io/badge/Preliminary%20Macro--F1-0.922-blue?style=for-the-badge" alt="Macro F1 0.922" />
  <img src="https://img.shields.io/badge/Platform-MoModel%20(2%20Core%20CPU)-orange?style=for-the-badge" alt="CPU Inference" />
  <img src="https://img.shields.io/badge/PyTorch-1.12+-EE4C2C?style=for-the-badge&logo=pytorch" alt="PyTorch" />
</p>

---

## 📖 项目简介

本项目为“**恶劣天气下的交通标识识别**”全国算法竞赛（国赛）完整开源技术方案。赛题要求在浓雾、强降雨、降雪积雪、夜间弱光/反光眩光、运动模糊与遮挡等恶劣场景下，对 25 类细粒度交通标志进行精准分类。

方案在**极度苛刻的评测约束（纯 2 核 CPU、8GB RAM、无 GPU、70 分钟总推理时限）**下，凭借创新的异构三模型加权软投票集成（Weighted Soft Voting）以及安全多尺度 TTA 策略，最终取得 **Macro-F1 = 0.945** 的顶尖成绩，荣获**全国二等奖**！

> 📘 **深度复盘与技术体系总结**：详见 [RETROSPECTIVE.md](RETROSPECTIVE.md)（包含从 0.922 到 0.945 的完整迭代演进、四大实战避坑实录与黑盒算力受限比赛方法论）。

---

## 🏆 方案亮点

- **异构“三剑客”特征互补**：
  - `ConvNeXt V2-Base` (FCMAE 自监督预训练，纯卷积高抗噪，单模 F1=0.938)
  - `Swin Transformer V2-Base` (窗口自注意力机制，256×256 降采样控制 CPU 算力开销，单模 F1=0.916)
  - `EfficientNetV2-M` (极致 CPU 吞吐能力与泛化能力，单模 F1=0.917)
- **非均匀加权软投票 (Weighted Soft Voting)**：
  - 突破传统算术平均的局限，按单模验证能力分配置信度权重：$0.60 \times \text{ConvNeXtV2} + 0.20 \times \text{SwinV2} + 0.20 \times \text{EffNetV2}$。
- **物理先验防错 TTA**：
  - 针对交通标志的强方向性（左转/右转/靠左/靠右），**坚决禁止水平翻转**，采用安全双尺度插值平滑边界噪声。
- **100% 离线自给自足交付**：
  - 评测环境网络隔离，将核心依赖 `timm` 源码直接内置于提交包，杜绝在线安装失败或版本不一致导致的架构降级。

---

## 📂 仓库目录结构

```text
D:\MOPRO/
├── RETROSPECTIVE.md            # ⭐ 核心：比赛全面复盘、技术沉淀与经验教训总结
├── CONTEXT.md                  # 赛题背景与技术规格白皮书
├── inference_optimization_notes.md # CPU 算力平衡与模型选型对比笔记
├── platform_faq.md             # 平台环境限制与排错指引
│
├── train.py                    # 核心训练代码（支持 Focal Loss, SWA, Mixup, Warmup）
├── dataset.py                  # 恶劣天气针对性数据增强管道
├── main.py                     # 线上最终高分推理主程序 (F1 = 0.945)
├── classes.txt                 # 官方 25 类交通标志分类映射表
├── benchmark.py                # 本地 CPU 多线程速度测试工具
├── export_onnx.py              # ONNX 模型导出与一致性验证工具
├── preliminary_round_main.py   # 初赛天气分类推理源码 (初赛 F1 = 0.922)
│
├── run_train_official.bat      # Windows 一键训练官方 3 模型
├── run_train_efficientnet.bat  # 单独训练 EfficientNetV2-M
├── run_finetune_official.bat   # GTSRB 两阶段预训练微调脚本
├── run_test_train.bat          # 5-Epoch 快速连通性验证脚本
│
├── submit/                     # 🚀 线上提交目录（可直接打包或上传至平台）
│   ├── main.py                 # 提交执行入口
│   ├── main_single.py          # 单模型对照测试入口
│   ├── classes.txt             # 标签索引文件
│   └── timm/                   # 离线打包的纯源码 timm 库（免联网）
│
├── saved_versions/             # 历史成绩迭代版本归档
│   ├── 0.929.py                # 初代提分版本 (F1 = 0.929)
│   ├── 0.9418.py               # 等权投票与依赖自动安装版本 (F1 = 0.9418)
│   ├── 0.945.py                # 最终国赛二等奖加权集成版本 (F1 = 0.945)
│   ├── eval_local.py           # 本地离线验证集快速评测脚本
│   └── download_packages.py    # 离线依赖打包辅助工具
│
├── official_guide/             # 官方下发的初始竞赛基线与提交指引
│   ├── submission_guide.ipynb  # 官方提交说明
│   ├── official_baseline_main.py
│   └── official_baseline_train.py
│
└── results/                    # 模型权重保存目录 (*.pth)
    ├── convnextv2_base_best.pth
    ├── efficientnet_v2_m_best.pth
    └── swin_v2_b_best.pth
```

---

## 🚀 快速上手

### 1. 本地环境配置
```bash
git clone https://github.com/Lyumnire/traffic-sign-adverse-weather.git
cd traffic-sign-adverse-weather

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux
# .venv\Scripts\activate   # Windows

# 安装核心依赖
pip install torch torchvision timm scikit-learn pillow opencv-python tqdm
```

### 2. 模型训练
```bash
# 方式 A：Windows 下一键训练三剑客模型
run_train_official.bat

# 方式 B：使用命令行启动单一模型训练
python train.py \
    --model convnextv2_base \
    --dataset official \
    --dataset_path ./datasets/train \
    --classes_file ./classes.txt \
    --epochs 50 \
    --batch_size 16 \
    --grad_accum 2 \
    --img_size 384 \
    --loss focal \
    --swa_start 40 \
    --patience 5 \
    --num_workers 0
```

### 3. 本地与线上推理评测
```bash
# 本地测试单张图片或验证流程
python main.py
```
线上部署时，只需将 `submit/` 文件夹内的内容上传至平台的 `/home/jovyan/work/` 目录即可，系统将自动调用本地内置的 `timm` 库并使用 `results/*.pth` 权重运行。

---

## 📊 成绩演进对比

| 阶段 | 提交版本 / 策略 | Macro-F1 | 关键动作 |
| :--- | :--- | :--- | :--- |
| **初赛** | `preliminary_round_main.py` | **0.9220** | 初赛 4 类天气识别，多模型集成 + 水平翻转 TTA |
| **国赛 v1** | `saved_versions/0.929.py` | **0.9290** | 升级 ConvNeXt V2 + Swin V2 + EffNetV2 三模型，移除水平翻转 |
| **国赛 v3** | `saved_versions/0.9418.py` | **0.9418** | 修复线上 timm 缺失导致的架构降级，架构断言校验，等权软投票 |
| **国赛 v4** | `saved_versions/0.945.py` | **0.9450** | **加权软投票 (0.60/0.20/0.20)** + 移除 JIT，斩获**全国二等奖** 🏆 |

---

## 📜 开源协议

本项目采用 [MIT License](LICENSE) 开源协议。
