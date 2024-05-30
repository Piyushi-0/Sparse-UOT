#!/bin/sh
ktype=imq_v2
export CUDA_VISIBLE_DEVICES=4
python main.py --gamma 0.1 --ktype $ktype --khp -1 --lda3 10 --model MobileNetV2 --save_as prp_may29 --v 0 --epochs 10
