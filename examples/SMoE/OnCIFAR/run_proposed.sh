#!/bin/sh
export CUDA_VISIBLE_DEVICES=1
python main.py --gamma 1 --ktype imq --khp -1 --lda3 10 --model MobileNetV2 --save_as proposed --v 0