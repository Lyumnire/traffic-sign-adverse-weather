@echo off
echo ========================================================
echo  Official Dataset Training Pipeline (25 Classes)
echo  Strategy: ImageNet pretrained -> Official data direct
echo ========================================================

rem 1. Train EfficientNetV2-M
echo.
echo [1/3] Training EfficientNetV2-M...
.venv\Scripts\python.exe train.py --model efficientnet_v2_m --dataset official --dataset_path ./datasets/train --classes_file ./classes.txt --epochs 50 --batch_size 16 --grad_accum 2 --img_size 384 --loss focal --swa_start 40 --patience 5 --num_workers 2 --mixup_alpha 0.2 --warmup_epochs 3

rem 2. Train ConvNeXt V2-Base
echo.
echo [2/3] Training ConvNeXt V2-Base...
.venv\Scripts\python.exe train.py --model convnextv2_base --dataset official --dataset_path ./datasets/train --classes_file ./classes.txt --epochs 50 --batch_size 16 --grad_accum 2 --img_size 384 --loss focal --swa_start 40 --patience 5 --num_workers 2 --mixup_alpha 0.2 --warmup_epochs 3

rem 3. Train Swin V2-Base
echo.
echo [3/3] Training Swin V2-Base...
.venv\Scripts\python.exe train.py --model swin_v2_b --dataset official --dataset_path ./datasets/train --classes_file ./classes.txt --epochs 50 --batch_size 32 --img_size 256 --loss focal --swa_start 40 --patience 5 --num_workers 2 --mixup_alpha 0.2 --warmup_epochs 3

echo.
echo ========================================================
echo  All training completed!
echo  Models saved to: ./results/
echo  Submit files in: ./submit/ (copy .pth to submit/results/)
echo ========================================================
pause
