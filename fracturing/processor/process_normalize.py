import os
import argparse
import logging

import trimesh
import numpy as np

try:
    from sklearn.decomposition import PCA
except ImportError:
    logging.debug("Unable to import PCA")

import processor.logger as logger
import processor.errors as errors
import processor.repair as mesh_repair


def points_maximal_orient(points):
    """Return the transformation matrix that orients a point set by its maximal dimensions"""
    pca = PCA(n_components=3)
    pca.fit(points)
    matrix = pca.components_
    return np.vstack(
        (
            np.hstack(
                (
                    np.expand_dims(matrix[2, :], axis=1),
                    np.expand_dims(matrix[1, :], axis=1),
                    np.expand_dims(matrix[0, :], axis=1),
                    np.zeros((3, 1)),
                )
            ),
            np.array([0, 0, 0, 1]),
        )
    ).T


def normalize_unit_cube(mesh):
    """Normalize a mesh so that it occupies a unit cube"""

    # Get the overall size of the object
    mesh = mesh.copy()
    mesh_min, mesh_max = np.min(mesh.vertices, axis=0), np.max(mesh.vertices, axis=0)
    size = mesh_max - mesh_min

    # Center the object
    mesh.vertices = mesh.vertices - ((size / 2.0) + mesh_min)

    # Normalize scale of the object
    mesh.vertices = mesh.vertices * (1.0 / np.max(size))
    return mesh


def normalize(f_in, f_out, skip_check=True, reorient=False, overwrite=False):
    """Translate and rescale a mesh so that it is centered inside a unit cube"""
    mesh = trimesh.load(f_in)

    if not skip_check:
        mesh_in = mesh
        if not mesh.is_watertight:
            mesh = mesh_repair.repair_self_intersection(mesh_in)
        if not mesh.is_watertight:
            f_out_temp = os.path.splitext(f_out)[0] + ".temp.ply"
            mesh_repair.repair_watertight_handsoff(f_in, f_out_temp, timeout=600)
            trimesh.load(f_out_temp)
            os.remove(f_out_temp)
            # mesh = mesh_repair.repair_watertight(mesh_in)
            mesh = mesh_repair.repair_self_intersection(mesh)
        if not mesh.is_watertight:
            raise errors.MeshNotClosedError

    mesh = normalize_unit_cube(mesh)
    if reorient:
        mesh.apply_transform(points_maximal_orient(mesh.vertices))
        normalize_unit_cube(mesh)
    if overwrite or not os.path.exists(f_out):
        logging.debug("Saving to: {}".format(f_out))
        mesh.export(f_out)


def process(
    obj,
    num_results,
    overwrite,
    executor,
    args,
):
    f_in = obj.path_waterproofed()
    f_out = obj.path_c()
    if os.path.exists(f_in) and (not os.path.exists(f_out) or overwrite):
        executor.graceful_submit(
            normalize,
            f_in=f_in,
            f_out=f_out,
            skip_check=False,
            reorient=False,
            overwrite=overwrite,
        )


def validate_outputs(
    obj,
    num_results,
    args,
):
    if os.path.exists(obj.path_c()):
        return [True]
    else:
        return [False]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(dest="input", type=str, help="Path to the input file.")
    parser.add_argument(dest="output", type=str, help="Path to the output file.")
    parser.add_argument(
        "--reorient",
        action="store_true",
        default=False,
        help="If passed, will reorient the mesh using PCA.",
    )
    parser.add_argument(
        "--skip_check",
        action="store_true",
        default=False,
        help="If passed, will skip watertight check.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="If passed, will overwrite the output, if it exists.",
    )
    logger.add_logger_args(parser)
    args = parser.parse_args()
    logger.configure_logging(args)

    normalize(
        f_in=args.input,
        f_out=args.output,
        skip_check=args.skip_check,
        reorient=args.reorient,
        overwrite=args.overwrite,
    )
