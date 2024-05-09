alpha=10
rho=1
for seed in 1 2 3 4 5
    do
    for L in 100 175 250
        do
        python main_gsot.py --L $L --seed $seed --alpha $alpha --rho $rho --val 0
        done
    done