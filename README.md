# DeepJoin
Code for "DeepJoin: Learning a Joint Occupancy, Signed Distance, and Normal Field Function for Shape Repair." \
Published at SIGGRAPH Asia 2022.

| <img src="assets/three_shapes_bef2.gif" alt="example1" width="300"/> | <img src="assets/three_shapes_aft2.gif" alt="example1" width="300"/> |
| :-: | :-: |
| Input | Output |

```
citation placeholder
```

## Installation

Code tested using Ubutnu 18.04 and python 3.8.0.
Note that you need to have the following apt dependencies installed. 
```bash
sudo apt install python3.8-distutils python3.8-dev libgl1 libglew-dev freeglut3-dev
```

Clone the repo.
```bash
https://github.com/Terascale-All-sensing-Research-Studio/DeepJoin.git
cd DeepJoin
```

We recommend using virtualenv. The following snippet will create a new virtual environment, activate it, and install deps.
```bash
sudo apt-get install virtualenv && \
virtualenv -p python3.8 env && \
source env/bin/activate && \
pip install -r requirements.txt && \
./install.sh && \
source setup.sh
```
Issues with compiling pyrender are typically solved by upgrading cython: `pip install --upgrade cython`.

If you want to run the fracturing and sampling code, you'll need to install pymesh dependencies:
```
./install_pymesh.sh
```

## Quickstart Inference

If you just want to try out inference, run the following script with the example file. This will infer a restoration and create a gif.
```
cd deepjoin
./scripts/infer_quick.sh experiments/mugs/specs.json ../example_files/fractured_mug.obj
```

You should get a gif that looks like this one! \
<img src="assets/example_mug.gif" alt="example1" width="200"/>

## Data Preparation

See `fracturing/README.md`.

## Training

Navigate into the `deepjoin` directory.
```
cd deepjoin
```

Each experiment needs a corresponding directory with a "specs.json" file. You can find an example at `deepjoin/experiments/mugs`.

To train, run the training python script with the path to an experiment directory.
```
python python/train.py -e experiments/mugs
```

## Inference

Navigate into the `deepjoin` directory.
```
cd deepjoin
```

Inference (and related operations) is done in four steps:

1) Infer latent codes. 
2) Reconstruct meshes. 
3) Generate renders. 
4) Evaluate meshes.


Each experiment needs a corresponding directory with a "specs.json" file. You can find an example at `deepjoin/experiments/mugs`.

To infer:
```
./scripts/infer.sh experiments/mugs
```

Data is saved in the experiment directory passed to the reconstruction script, under a `Reconstructions` subdirectory. For example, results for the mugs example will be stored in `deepjoin/experiments/mugs/Reconstructions/ours/`. Meshes are stored in the `Meshes` subdirectory. A render of all the results is stored in the top-level reconstruction directory. 
