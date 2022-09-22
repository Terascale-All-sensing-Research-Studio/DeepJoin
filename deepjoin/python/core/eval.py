import numpy as np


def add_generated(data):
    """
    Add a binary mask of generated objects.
    """
    data["generated"] = np.logical_not(np.isnan(data["chamfer"])).astype(int)


def shift_column(experiment, column_from, column_to):
    """
    Shifts a column of a given experiment.
    """
    for metric in experiment:
        if hasattr(experiment[metric], "shape"):
            experiment[metric][:, column_to] = experiment[metric][:, column_from]


def trim_column(experiment, num_columns):
    """
    Removes any columns beyond num_columns
    """
    for metric in experiment:
        if hasattr(experiment[metric], "shape"):
            experiment[metric] = experiment[metric][:, :num_columns]


def get_dataset(name):
    """
    return the dataset corresponding to a given path.
    """
    kwds = [
        "jars",
        "bottles",
        "mugs",
        "airplanes",
        "chairs",
        "cars",
        "tables",
        "sofas",
        "hampson",
        "qp",
    ]
    for k in kwds:
        if k in name:
            return k
