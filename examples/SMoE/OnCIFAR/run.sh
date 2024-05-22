#!/bin/sh
ktype=imq

python main.py --epochs 10 --gamma 1 --ktype $ktype --khp -1 --lda3 0.1
