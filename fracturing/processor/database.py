import os
import logging

import pickle
import trimesh
import numpy as np
from PIL import Image
import torch


def loader(f_in):
    """Multipurpose loader used for all file types"""
    extension = os.path.splitext(f_in)[-1]
    logging.debug("Attempting to load file {}".format(f_in))
    if extension == ".pkl":
        with open(f_in, "rb") as f:
            return pickle.load(f)
    elif extension == ".obj":
        return trimesh.load(f_in, force=True)
    elif extension == ".ply":
        return trimesh.load(f_in, force=True)
    elif extension == ".npz":
        return dict(np.load(f_in, allow_pickle=True))
    elif extension == ".npy":
        return np.load(f_in, allow_pickle=True)
    elif extension == ".png":
        return np.array(Image.open(f_in))
    elif extension == ".pth":
        return torch.load(f_in, map_location=torch.device("cpu"))
    else:
        raise RuntimeError("Loader: Unhandled file type: {}".format(f_in))


def saver(f_out, data):
    """Multipurpose saver used for all file types"""
    logging.debug("Saving file {}".format(f_out))
    extension = os.path.splitext(f_out)[-1]
    if extension == ".obj":
        data.export(f_out)
    if extension == ".ply":
        data.export(f_out)
    elif extension == ".png":
        Image.fromarray(data).save(f_out)
    elif extension == ".pth":
        torch.save(data, f_out)
    elif extension == ".npz":
        np.savez(f_out, data)
    elif extension == ".npy":
        np.save(f_out, data)
    else:
        raise RuntimeError("Saver: Unhandled file type: {}".format(f_out))


def save(path, lst):
    """Save a list of database objects"""
    pickle.dump(
        lst,
        open(path, "wb"),
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def load(path):
    """Load a list of database objects"""
    return pickle.load(open(path, "rb"))


class DatabaseObject:
    def __init__(self, root_dir, class_id, instance_id):
        self._root_dir = root_dir
        self._class_id = class_id
        self._instance_id = instance_id

        self._cache = {}

    def __repr__(self):
        return (
            "DatabaseObject("
            + self._root_dir
            + ", "
            + self._class_id
            + ", "
            + self._instance_id
            + ")"
        )

    def load(self, f_in, skip_cache=False):
        if f_in in self._cache and not skip_cache:
            logging.debug("Pulling file from cache: {}".format(f_in))
            return self._cache[f_in]

        data = loader(f_in)

        if not skip_cache:
            self._cache[f_in] = data
        return data

    def __hash__(self):
        return hash(str(self))

    @property
    def root_dir(self):
        return self._root_dir

    @property
    def class_id(self):
        return self._class_id

    @property
    def instance_id(self):
        return self._instance_id

    def path(self, *args, **kwargs):
        raise NotImplementedError

    def build_dirs(self):
        raise NotImplementedError
