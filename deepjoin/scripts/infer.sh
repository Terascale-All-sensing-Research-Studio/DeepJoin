# !/bin/bash
EXPERIMENT="$1"
UNIFORM_RATIO="0.2"
CHK="2000"
NUMITS="1600"
LREG="0.0001"
LR="0.005"
LAMBDA1="1.0" 
LAMBDA2="0.0" 
LAMBDA3="1.0" 
SPLITNUM="$2"
if [ -z "$SPLITNUM" ]; then
    SPLITNUM="1"
fi
if [ -z "$NUMTHREADS" ]; then
    NUMTHREADS="5"
fi
if [ -z "$NUMTHREADSRENDER" ]; then
    NUMTHREADSRENDER="5"
fi

NME="_test"
echo "Loading from ""$1"
echo "Reconstructing $NME"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

python python/reconstruct.py \
    -e "$1" \
    -c "$CHK" \
    --name "$NME" \
    --threads "$NUMTHREADS" \
    --num_iters "$NUMITS" \
    --lambda_reg "$LREG" \
    --learning_rate "$LR" \
    --render_threads "$NUMTHREADSRENDER" \
    --uniform_ratio "$UNIFORM_RATIO" \
    --lambda1 "$LAMBDA1" \
    --lambda2 "$LAMBDA2" \
    --lambda3 "$LAMBDA3" \
    --out_of_order \
    --stop 10000 \
    --seed 1 \
    --split "$SPLITNUM"
