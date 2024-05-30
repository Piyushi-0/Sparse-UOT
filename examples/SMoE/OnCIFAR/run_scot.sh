#!/bin/sh

export CUDA_VISIBLE_DEVICES=1
python main.py --epochs 10 --gamma 10 --model MobileNetV2 --save_as scot
