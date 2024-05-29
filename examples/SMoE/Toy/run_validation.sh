#!/bin/sh

# for lda3 in 0.1 1 10
# do
#     export CUDA_VISIBLE_DEVICES=3
#     python main_lin.py --method 'scot' --lda3 $lda3
# done
for lda3 in 0.01 0.1 1 10
    do
    export CUDA_VISIBLE_DEVICES=2
    python main.py --method 'ssot' --lda3 $lda3
    done

# for lda3 in 0.01 0.1 1 10
#     do
#     export CUDA_VISIBLE_DEVICES=2
#     python main_lin.py --method 'ot' --lda3 $lda3
#     done

# for ktype in 'imq' 'rbf' 'imq_v2'
# do
#     for khp in 1e-2 1 1e+2
#     do
#         for lda in 1 10 100
#         do
#             export CUDA_VISIBLE_DEVICES=2
#             python main_lin.py --method 'uotmmd' --lda $lda --ktype $ktype --khp $khp
#         done
#     done
# done

# for lda3 in 0.1 1 10
# do
#     for lda in 0.01 0.1 1 10
#         do
#         export CUDA_VISIBLE_DEVICES=2
#         python main_lin.py --method 'uotkl' --lda $lda --lda3 $lda3
#         done
# done