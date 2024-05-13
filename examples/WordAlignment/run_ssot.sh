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

for DATA in wiki
do
    python uns_ssot.py --data $DATA --centering --pair_encode --ot_type $OT --weight_type $WT --dist_type $DT --seed $SEED --max_itr $max_itr --ws $ws --s $s --save_as logs_ssot
done
