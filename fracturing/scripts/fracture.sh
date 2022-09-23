# !/bin/bash
# Fracture the ShapeNet dataset
#

# Set the input arguments
ROOTDIR="$1"         # eg:
if [ -z "$1" ]; then
    echo "Must pass: ROOTDIR"; exit
fi
SPLITSFILE="$2"     # eg:
if [ -z "$2" ]; then
    echo "Must pass: SPLITSFILE"; exit
fi
OPERATION="$3"      # eg:
if [ -z "$3" ]; then
    echo "Must pass: OPERATION"; exit
fi
NUMBREAKS="$4"      # eg:
if [ -z "$4" ]; then
    NUMBREAKS="1"
fi
SUBSAMPLE="10000"
NUMFRACTURETHREADS="3"

# Run 
case $OPERATION in
	"0")
        python -m processor.process \
            "$ROOTDIR" \
            "$SPLITSFILE" \
            process_waterproof2 \
            --instance_subsample "$SUBSAMPLE" \
            --outoforder \
            -t "$NUMFRACTURETHREADS"
        ;;
    
	"1")
        python -m processor.process \
            "$ROOTDIR" \
            "$SPLITSFILE" \
            process_normalize \
            --outoforder \
            -t "$NUMFRACTURETHREADS"
        ;;

	"2")
        python -m processor.process \
            "$ROOTDIR" \
            "$SPLITSFILE" \
            process_break \
            --breaks "$NUMBREAKS" \
            --min_break 0.05 \
            --max_break 0.20 \
            --break_method surface-area \
            --outoforder \
            -t "$NUMFRACTURETHREADS"
        ;;

	"3")
        python -m processor.process \
            "$ROOTDIR" \
            "$SPLITSFILE" \
            process_sample \
            --breaks "$NUMBREAKS" \
            --outoforder \
            -t "$NUMFRACTURETHREADS"
        ;;

	"4")
        python -m processor.process \
            "$ROOTDIR" \
            "$SPLITSFILE" \
            process_sdf2 \
            --breaks "$NUMBREAKS" \
            --outoforder \
            -t "$NUMFRACTURETHREADS"
        ;;

	"5")
        python -m processor.process \
            "$ROOTDIR" \
            "$SPLITSFILE" \
            process_partial_sdf \
            --breaks "$NUMBREAKS" \
            --outoforder \
            --test_only \
            -t "$NUMFRACTURETHREADS"
        ;;

	"6")
        python -m processor.process \
            "$ROOTDIR" \
            "$SPLITSFILE" \
            process_spline \
            --breaks "$NUMBREAKS" \
            --outoforder \
            -t "$NUMFRACTURETHREADS"
        ;;

	"7")
        python -m processor.process \
            "$ROOTDIR" \
            "$SPLITSFILE" \
            process_uniform_occupancy \
            --breaks "$NUMBREAKS" \
            --outoforder \
            --test_only \
            --voxel_dim 256 \
            -t "$NUMFRACTURETHREADS"
        ;;

esac