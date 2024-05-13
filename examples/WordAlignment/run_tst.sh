UN_OUTDIR=./out/unsupervised/
SEED=42
OT=prp
WT=uniform
DT=cos
model=bert-base-uncased
layer=-3
max_itr=100
ws=1
s=0


lda3=0.1
ktype=rbf
khp_med=med_8
lda=1
DATA=wiki
python uns_our.py --best_thresh 1 --khp_med $khp_med --ktype $ktype --lda3 $lda3 --lda $lda --data $DATA --centering --pair_encode --ot_type $OT --weight_type $WT --dist_type $DT --seed $SEED --max_itr $max_itr --ws $ws --s $s --save_as test1_${max_itr}_${DATA}
