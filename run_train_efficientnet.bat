@echo off
echo ========================================================
echo  Official Dataset Training: EfficientNetV2-M Only
echo  Strategy: ImageNet pretrained -> Official data direct
echo ========================================================

rem Activate virtual environment
call .venv\Scripts\activate

rem Train EfficientNetV2-M
echo.
echo Training EfficientNetV2-M...
.venv\Scripts\python.exe train.py --model efficientnet_v2_m --dataset official --dataset_path ./datasets/train --classes_file ./classes.txt --epochs 50 --batch_size 16 --grad_accum 2 --img_size 384 --loss focal --swa_start 40 --patience 5 --num_workers 2 --mixup_alpha 0.2 --warmup_epochs 3

echo.
echo ========================================================
echo  EfficientNetV2-M training completed!
echo  Model saved to: ./results/efficientnet_v2_m_best.pth
echo ========================================================
pause
