# data-prep

Make sure you've created and activated the python virtual environment. Then install pymesh.
```
./install_pymesh.sh
```

To generate fractured shapes from scratch, download shapenet from https://shapenet.org/. Extract the data into a directory, e.g. "/path/to/above/ShapeNet".

Almost all scripts require that the following environment variable be set to the directory above ShapeNet.
```
export DATADIR="/path/to/above/ShapeNet"
```

## Data preparation
Fracturing is done in several steps:

1) Waterproof meshes.
2) Normalize meshes. 
3) Fracture meshes. 
4) Compute sample points.
5) Compute SDF values. 
6) Compute spline breaks. 
7) Compute OCC values. (for baselines)

The `scripts/fracture.sh` takes an integer corresponding to one of the above processes.

NOTE: waterproofing cannot be done on a PC that does not have a display.

This readme will demo the data processing pipeline for the mugs dataset from shapenet. We assume you've downloaded shapenet to:
```
$DATADIR/ShapeNetCore.v2
```

To perform a single step of data processing use the `scripts/fracture.sh` script. The script takes 4 arguments:
- Path to the directory containing shapenet data
- Path to a .json train/test split file (will be created if does not exist)
- Operation (int from 1 to 7, corresponding to the above list)
- Number of breaks

The first operation that needs to be done is waterproofing:
```
./scripts/fracture.sh \
    $DATADIR/ShapeNetCore.v2/03797390 \
    $DATADIR/ShapeNetCore.v2/mugs_split.json \
    0 1
```

We provide a wrapper, `scripts/run.sh`, to execute all preprocessing steps in order (except waterproofing). The following will perform all preprocessing steps in order, fracturing each object 1 time.
```
./scripts/run.sh \
    $DATADIR/ShapeNetCore.v2/03797390 \
    $DATADIR/ShapeNetCore.v2/mugs_split.json \
    1
```

The next step is to package the data into pkl files for training and testing. Navigate into the `deepjoin` directory.
```
cd ..
cd deepjoin
```

Run the build script with 4 arguments:
- Path to the directory containing shapenet data
- Path to a .json train/test split file
- Path to a train/test database file (will be created and have _train.pkl, _val.pkl, or _test.pkl appended)
- Number of breaks

The following: 
```
./scripts/build.sh \
    $DATADIR/ShapeNetCore.v2 \
    $DATADIR/ShapeNetCore.v2/mugs_split.json \
    $DATADIR/ShapeNetCore.v2/mugs \
    1
```
will create the files `$DATADIR/ShapeNetCore.v2/mugs_train.pkl`, `$DATADIR/ShapeNetCore.v2/mugs_val.pkl`, and `$DATADIR/ShapeNetCore.v2/mugs_test.pkl`.

This will create two pkl files containing the training and testing data. Now you can train according to the training procedure specified in the parent directory. When creating the `specs.json` file, set the `TrainSplit`, `ValSplit`, and `TestSplit` as the files that were created by the build script:
```
{
    "Description" : "deepjoin, mugs dataset",
    "DataSource" : "$DATADIR/ShapeNetCore.v2",
    "TrainSplit" : "$DATADIR/ShapeNetCore.v2/mugs_train.pkl",
    "ValSplit" : "$DATADIR/ShapeNetCore.v2/mugs_val.pkl",
    "TestSplit" : "$DATADIR/ShapeNetCore.v2/mugs_test.pkl",
...
```
