# !/bin/bash
# Build train/val/test files

ROOTDIR="$1"    # Path to the root of a fractured shapenet dataset
SPLITSFILE="$2" # Path to a splits file
OUTFILE="$3"    # Path to an output file (no extension)
NUMBREAKS="$4"  # Number of break to load

python python/build_n.py \
    "$ROOTDIR" \
    "$SPLITSFILE" \
    --train_out "$OUTFILE"_train.pkl \
    --val_out "$OUTFILE"_val.pkl \
    --test_out "$OUTFILE"_test.pkl \
    --breaks "$NUMBREAKS" \
    --use_pointer \
    --use_normals \
    --use_spline \
    --load_models \
    --fix_distance_normals