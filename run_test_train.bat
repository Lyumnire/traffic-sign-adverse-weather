@echo off
echo ========================================================
echo  Quick Single Model Training Test
echo  (EfficientNetV2-M, 5 epochs, verify pipeline works)
echo ========================================================

.venv\Scripts\python.exe train.py --model efficientnet_v2_m --dataset official --dataset_path ./datasets/train --classes_file ./classes.txt --epochs 5 --batch_size 16 --grad_accum 2 --img_size 384 --loss focal --swa_start 99 --patience 99 --num_workers 2 --mixup_alpha 0.0 --warmup_epochs 1

echo.
echo Quick test completed. Check output above for errors.
pause
