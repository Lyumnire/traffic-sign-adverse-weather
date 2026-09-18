@echo off
echo ========================================================
echo  Stage 2: Official Dataset Fine-Tuning Pipeline
echo ========================================================

rem Activate virtual environment
call .venv\Scripts\activate

rem 1. Fine-tune EfficientNetV2-M
echo.
echo [1/3] Fine-tuning EfficientNetV2-M on official dataset...
.venv\Scripts\python.exe train.py --model efficientnet_v2_m --dataset official --dataset_path ./datasets/official_train --epochs 30 --batch_size 16 --grad_accum 2 --img_size 384 --loss focal --swa_start 25 --patience 5 --num_workers 2 --mixup_alpha 0.2 --warmup_epochs 3 --pretrained_path ./results/efficientnet_v2_m_best.pth

rem 2. Fine-tune ConvNeXt V2-Base (load GTSRB pretrained weights)
echo.
echo [2/3] Fine-tuning ConvNeXt V2-Base on official dataset...
.venv\Scripts\python.exe train.py --model convnextv2_base --dataset official --dataset_path ./datasets/official_train --epochs 30 --batch_size 16 --grad_accum 2 --img_size 384 --loss focal --swa_start 25 --patience 5 --num_workers 2 --mixup_alpha 0.2 --warmup_epochs 3 --pretrained_path ./results/convnextv2_base_best.pth

rem 3. Fine-tune Swin V2-Base
echo.
echo [3/3] Fine-tuning Swin V2-Base on official dataset...
.venv\Scripts\python.exe train.py --model swin_v2_b --dataset official --dataset_path ./datasets/official_train --epochs 30 --batch_size 32 --img_size 256 --loss focal --swa_start 25 --patience 5 --num_workers 2 --mixup_alpha 0.2 --warmup_epochs 3 --pretrained_path ./results/swin_v2_b_best.pth

echo.
echo Fine-tuning completed
pause
