#!/bin/sh
export CUDA_VISIBLE_DEVICE=1
python tst.py --method 'scot' --lda3 10

# export CUDA_VISIBLE_DEVICE=1
# python tst.py --method 'ssot' --lda3 1

# export CUDA_VISIBLE_DEVICE=1
# python tst.py --method 'ot' --lda3 0.1

# export CUDA_VISIBLE_DEVICE=1
# python tst.py --method 'uotmmd' --lda 1 --ktype imq --khp 1

# export CUDA_VISIBLE_DEVICE=1
# python tst.py --method 'uotkl' --lda 10 --lda3 10

export CUDA_VISIBLE_DEVICE=1
python tst.py --method 'prp' --lda 100 --ktype imq --khp 100 --lda3 10
