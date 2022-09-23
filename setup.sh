source env/bin/activate
if [ -f "constants.sh" ]; then
    source constants.sh
fi
if [[ x$DATADIR != "x" ]]; then
    echo "The primary data directory is located at: "$DATADIR
fi

unset PYTHONPATH

# Add local libraries to pythonpath
export PYTHONPATH=$PYTHONPATH:`pwd`/fracturing
export PYTHONPATH=$PYTHONPATH:`pwd`/deepjoin/python

# Add dependancies to pythonpath
if [ -d "libs" ]; then
    export PYTHONPATH=$PYTHONPATH:`pwd`/libs/mesh-fusion
    export PYTHONPATH=$PYTHONPATH:`pwd`/libs/inside_mesh
    export LIBRENDERPATH=`pwd`/libs/mesh-fusion/librender
    export PYMESH_PATH=`pwd`/libs/PyMesh
else
    echo "Library directory not found, cannot add libraries"
fi
