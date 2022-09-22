# !/bin/bash
# Fracture the ShapeNet dataset
#

# Set the input arguments
ROOTDIR="$1" 
if [ -z "$1" ]; then
    echo "Must pass: ROOTDIR"; exit
fi
SPLITSFILE="$2" 
if [ -z "$2" ]; then
    echo "Must pass: SPLITSFILE"; exit
fi
NUMBREAKS="$3" 
if [ -z "$3" ]; then
    NUMBREAKS="10"
fi

for i in {1..8}
do
    ./scripts/fracture.sh \
        $ROOTDIR \
        $SPLITSFILE \
        $i \
        $NUMBREAKS
done