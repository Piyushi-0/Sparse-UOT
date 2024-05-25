#!/bin/sh

export CUDA_VISIBLE_DEVICES=4
python main.py --gamma 10 --model MobileNetV2 --save_as SCOT
