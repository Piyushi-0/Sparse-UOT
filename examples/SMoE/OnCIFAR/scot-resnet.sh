#!/bin/sh

for gamma in 0.1 1 0.01 100
do
export CUDA_VISIBLE_DEVICES=1
python main.py --epochs 10 --gamma $gamma --model resnet18 --save_as scot-rnet
done