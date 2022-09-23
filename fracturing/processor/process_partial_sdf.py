import os, argparse
import logging

import trimesh
import numpy as np
from scipy.spatial import cKDTree as KDTree

from processor.process_sample import sample_points
from processor.process_sdf2 import compute_sdf
import processor.errors as errors
import processor.logger as logger


def intersect_mesh(a, b, sig=5):
    """mask of vertices that a shares with b"""
    av = [frozenset(np.round(v, sig)) for v in a]
    bv = set([frozenset(np.round(v, sig)) for v in b])
    return np.asarray(list(map(lambda v: v in bv, av)))


def get_fracture_mask(broken_mesh, complete_mesh):
    b_pts = broken_mesh.vertices
    num_points = b_pts.shape[0]

    # Indices of the broken object that are very close to the fracture
    b_fracture_mask = np.logical_not(
        intersect_mesh(broken_mesh.vertices, complete_mesh.vertices)
    )
    return b_fracture_mask


def get_fracture_point_mask(
    broken_mesh,
    complete_mesh,
    pts,
):
    """
    Return a mask that identifies points not in the fracture region.
    This simulates high accuracy fracture removal on
    the broken shape. Uses knn search to determine which points to keep.
    This corresponds to the optimal partial view of the broken shape.
    """

    b_pts = broken_mesh.vertices
    num_points = b_pts.shape[0]

    # Indices of the broken object that are very close to the fracture
    b_fracture_mask = np.logical_not(
        intersect_mesh(broken_mesh.vertices, complete_mesh.vertices)
    )
    b_not_fracture_inds = np.where(np.logical_not(b_fracture_mask))[0]
    b_fracture_inds = np.where(b_fracture_mask)[0]

    # Reorganize the vertices such that the fracture points come last
    b_pts = np.vstack((b_pts[b_not_fracture_inds, :], b_pts[b_fracture_inds, :]))
    assert b_pts.shape[0] == num_points

    # Find points that are closer to the fracture than the rest of the broken shape
    _, inds = KDTree(b_pts).query(pts, k=1)
    pts_not_fracture_mask = inds < b_not_fracture_inds.shape[0]

    return pts_not_fracture_mask


def partial_sdf(
    f_in,
    f_c,
    f_out,
    n_points=500000,
    uniform_ratio=0.5,
    padding=0.2,
    sigma=0.01,
    overwrite=False,
):
    # Load meshes
    broken_mesh = trimesh.load(f_in)
    complete_mesh = trimesh.load(f_c)

    # Get sample points
    logging.debug("Sampling points on broken")
    pts = sample_points(
        mesh=broken_mesh,
        n_points=n_points,
        uniform_ratio=uniform_ratio,
        padding=padding,
        sigma=sigma,
    )

    # Compute sdf
    logging.debug("Computing sdf for sample points")
    sdf, normals = compute_sdf(mesh=broken_mesh, points=pts, return_normals=True)
    sdf = np.expand_dims(sdf, axis=1)

    # Compute partial view
    logging.debug("Computing partial view")
    mask = get_fracture_point_mask(
        broken_mesh,
        complete_mesh,
        pts,
    )

    # Save
    if overwrite or not os.path.exists(f_out):
        logging.debug("Saving to: {}".format(f_out))
        np.savez(
            f_out,
            xyz=pts.astype(np.float16),
            sdf=sdf.astype(np.float16),
            mask=mask.astype(bool),
            n=normals.astype(np.float16),
        )


def process(
    obj,
    num_results,
    overwrite,
    executor,
    args,
):

    for idx in range(num_results):
        f_in = obj.path_b(idx)
        f_c = obj.path_c()
        f_out = obj.path_b_partial_sdf(idx)

        if os.path.exists(f_in) and (not os.path.exists(f_out) or overwrite):
            executor.graceful_submit(
                partial_sdf,
                f_in=f_in,
                f_c=f_c,
                f_out=f_out,
                n_points=500000,
                uniform_ratio=0.5,
                padding=0.1,
                sigma=0.01,
                overwrite=overwrite,
            )


def validate_outputs(
    obj,
    num_results,
    args,
):
    outputs = []
    for idx in range(num_results):
        if not os.path.exists(obj.path_b_partial_sdf(idx)):
            outputs.append(False)
            continue
        outputs.append(True)
    return outputs
