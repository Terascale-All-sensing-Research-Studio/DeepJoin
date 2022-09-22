import os
import argparse
import logging

import trimesh
import numpy as np

from libmesh import check_mesh_contains

import processor.errors as errors
import processor.process_sample as sampler


def occupancies_uniform(
    f_in,
    f_out,
    dim=256,
    padding=0.1,
    overwrite=False,
):
    # Load the mesh
    mesh = trimesh.load(f_in)

    # Get uniform points
    points = sampler.uniform_sample_points(dim=dim, padding=padding)

    # Get occupancies
    occupancies = check_mesh_contains(mesh, points)
    logging.debug(
        "Mesh had {}/{} interior points".format(
            occupancies.astype(int).sum(), occupancies.shape[0]
        )
    )

    # Save as boolean values
    if overwrite or not os.path.exists(f_out):
        logging.debug("Saving to: {}".format(f_out))
        np.savez(f_out, occ=occupancies.astype(bool))


def process(
    obj,
    num_results,
    overwrite,
    executor,
    args,
):

    dim = args.voxel_dim

    for idx in range(num_results):
        f_in = obj.path_b(idx)
        f_out = obj.path_b_uniform_occ(idx, dim)

        if os.path.exists(f_in) and (not os.path.exists(f_out) or overwrite):
            executor.graceful_submit(
                occupancies_uniform,
                f_in=f_in,
                f_out=f_out,
                overwrite=overwrite,
                dim=dim,
            )


def validate_outputs(
    obj,
    num_results,
    args,
):

    dim = args.voxel_dim

    outputs = []
    for idx in range(num_results):
        if not os.path.exists(obj.path_b_uniform_occ(idx, dim)):
            outputs.append(False)
            continue
        outputs.append(True)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes the occupancy values for samples points on and "
        + "around an object. Accepts the arguments common for sampling."
    )
    parser.add_argument(dest="f_in", type=str, help="Path to the input file.")
    parser.add_argument(dest="f_out", type=str, help="Path to the output file.")
    parser.add_argument(
        "--dim",
        "-d",
        type=int,
        default=256,
        help="Dimension of point samples.",
    )
    parser.add_argument(
        "--padding",
        "-p",
        type=float,
        default=0.1,
        help="Extra padding to add when performing uniform sampling. eg 0 = "
        + "uniform sampling is done in unit cube.",
    )
    args = parser.parse_args()

    occupancies_uniform(
        f_in=args.f_in,
        f_out=args.f_out,
        dim=args.dim,
        padding=args.padding,
    )
