khp=1
lda3=0
lda=100
ktype=rbf
val=0

for L in 100 175 250
do
    for seed in 1 2 3 4 5
        do
        for s_eps in 1e-4 1e-3 1e-2
            do
            python main_prp.py --L $L --seed $seed --ktype $ktype --lda $lda --lda3 $lda3 --khp $khp --s_eps $s_eps --val $val
            done
        done
done