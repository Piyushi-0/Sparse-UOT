method="prp"
# for ktype in imq_v2
# do
# for khp in 100
# do
# for lda in 100
# do
# for lda3 in 10
# do
# python main.py --method $method --ktype $ktype --khp $khp --lda $lda --lda3 $lda3
# done
# done
# done
# done

for ktype in imq imq_v2 rbf
do
for khp in 100 1000
do
for lda in 100 10 1
do
for lda3 in 10 1 0.1
do
    if [ $ktype == "imq_v2" ] && [ $khp == 100 ] && [ $lda == 100 ] && [ $lda3 == 10 ];
    then
        continue
    fi
python main.py --method $method --ktype $ktype --khp $khp --lda $lda --lda3 $lda3
done
done
done
done