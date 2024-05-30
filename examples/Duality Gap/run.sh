for K in 4
do
    for lda3 in 0.1 1 10
        do
            for lda in 0.1 1 10
            do
                for khp in -1
                do
                    for ktype in rbf imq imq_v2
                    do
                    python dg_proposed.py --khp $khp --ktype $ktype --lda $lda --lda3 $lda3 --K $K --save_as May27_Hcn
                    python dg_scot.py --khp $khp --ktype $ktype --lda $lda --lda3 $lda3 --K $K --save_as May27_Hcn
                    done
                done
            done
        done
done