import os, argparse
import logging
import time

# Set the available threads to (1/3)
os.environ["OMP_NUM_THREADS"] = str(int(os.cpu_count() * (1 / 3)))

import trimesh
import numpy as np
from pykdtree.kdtree import KDTree

try:
    from libmesh import check_mesh_contains
except ImportError:
    pass

import processor.process_sample as sampler
import processor.errors as errors
import processor.logger as logger


def compute_sdf(
    mesh,
    points,
    surface_points=10000000,
    compute_sign=True,
    fix_normals=True,
    return_normals=False,
    return_indices=False,
):

    if fix_normals:
        mesh.fix_normals()

    # Get distances
    logging.debug("Computing SDF approximation using {} points".format(surface_points))
    surface_points, face_indices = mesh.sample(count=surface_points, return_index=True)

    t0 = time.time()
    sdf, inds = KDTree(surface_points.astype(np.float64)).query(
        points.astype(np.float64)
    )
    logging.debug("SDF Computation took {} seconds".format(time.time() - t0))

    if return_normals:
        normals = np.array(mesh.face_normals)
        mag = np.linalg.norm(normals, axis=-1, keepdims=True)
        if (mag == 0).sum() > 0:
            logging.warning(
                "{}/{} normals are zero".format((mag == 0).sum(), mag.shape[0])
            )
        mag[mag == 0] = 1
        normals = normals / mag
        # assert np.isclose(
        #     np.linalg.norm(normals, axis=-1, keepdims=True).sum(),
        #     mesh.face_normals.shape[0]
        # ), "Only {} / {} normals are correctly normalized".format(
        #     np.linalg.norm(normals, axis=-1, keepdims=True).sum(),
        #     mesh.face_normals.shape[0]
        # )
        normals = normals[face_indices[inds], :]

    # Apply sign using check mesh contains
    if compute_sign:
        logging.debug("Computing sign")
        t0 = time.time()
        sign = check_mesh_contains(mesh, points).astype(int)
        sdf[sign == 1] = -np.abs(sdf[sign == 1])
        sdf[sign != 1] = np.abs(sdf[sign != 1])
        logging.debug("OCC computation took {} seconds".format(time.time() - t0))

    if return_normals:
        if return_indices:
            return sdf, normals, face_indices[inds]
        return sdf, normals
    if return_indices:
        return sdf, face_indices[inds]
    return sdf


def process_sdf(
    f_in,
    f_out,
    f_samp=None,
    n_points=500000,
    surface_points=10000000,
    uniform_ratio=0.5,
    padding=0.1,
    sigma=0.01,
    min_percent=0.02,
    overwrite=False,
):
    # Load meshes
    mesh = trimesh.load(f_in)

    if f_samp is None:
        # Get sample points
        points = sampler.sample_points(
            mesh=mesh,
            n_points=n_points,
            uniform_ratio=uniform_ratio,
            padding=padding,
            sigma=sigma,
        )
    else:
        logging.debug("Loading sample points from {}".format(f_samp))
        points = np.load(f_samp)["xyz"]
        assert (
            points.shape[0] == n_points
        ), "Loaded sample points were the wrong size {} vs {}".format(
            points.shape[0], n_points
        )

    # Compute sdf
    sdf, normals = compute_sdf(
        mesh=mesh,
        points=points,
        surface_points=surface_points,
        return_normals=True,
    )

    # Must have at least this many points, else is a bad sample
    logging.debug(
        "Mesh had {}/{} interior points".format((sdf < 0).sum(), sdf.shape[0])
    )
    if (sdf < 0).sum() < (n_points * min_percent):
        raise errors.MeshEmptyError

    # Compress
    if overwrite or not os.path.exists(f_out):
        logging.debug("Saving to: {}".format(f_out))
        np.savez(
            f_out,
            sdf=sdf.astype(np.float16),
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
        f_in = obj.path_c()
        f_out = obj.path_c_sdf(idx)
        f_samp = obj.path_sampled(idx)

        if (
            os.path.exists(f_in)
            and os.path.exists(f_samp)
            and (not os.path.exists(f_out) or overwrite)
        ):
            executor.graceful_submit(
                process_sdf,
                f_in=f_in,
                f_out=f_out,
                f_samp=f_samp,
                n_points=500000,
                uniform_ratio=0.5,
                padding=0.1,
                sigma=0.01,
                overwrite=overwrite,
            )

        f_in = obj.path_b(idx)
        f_out = obj.path_b_sdf(idx)

        if (
            os.path.exists(f_in)
            and os.path.exists(f_samp)
            and (not os.path.exists(f_out) or overwrite)
        ):
            executor.graceful_submit(
                process_sdf,
                f_in=f_in,
                f_out=f_out,
                f_samp=f_samp,
                n_points=500000,
                uniform_ratio=0.5,
                padding=0.1,
                sigma=0.01,
                overwrite=overwrite,
            )

        f_in = obj.path_r(idx)
        f_out = obj.path_r_sdf(idx)

        if (
            os.path.exists(f_in)
            and os.path.exists(f_samp)
            and (not os.path.exists(f_out) or overwrite)
        ):
            executor.graceful_submit(
                process_sdf,
                f_in=f_in,
                f_out=f_out,
                f_samp=f_samp,
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
        if not os.path.exists(obj.path_c_sdf(idx)):
            outputs.append(False)
            continue
        if not os.path.exists(obj.path_b_sdf(idx)):
            outputs.append(False)
            continue
        if not os.path.exists(obj.path_r_sdf(idx)):
            outputs.append(False)
            continue
        outputs.append(True)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes the occupancy values for samples points on and "
        + "around an object. Accepts the arguments common for sampling."
    )
    parser.add_argument(dest="input", type=str, help="Path to the input file.")
    parser.add_argument(dest="output", type=str, help="Path to the output file.")
    parser.add_argument(
        "--samples",
        type=str,
        help="Input file that stores sample points to use (.npz).",
    )
    parser.add_argument(
        "--uniform",
        "-r",
        type=float,
        default=0.5,
        help="Uniform ratio. eg 1.0 = all uniform points, no surface points.",
    )
    parser.add_argument(
        "--n_points",
        "-n",
        type=int,
        default=500000,
        help="Total number of sample points.",
    )
    parser.add_argument(
        "--padding",
        "-p",
        type=float,
        default=0.1,
        help="Extra padding to add when performing uniform sampling. eg 0 = "
        + "uniform sampling is done in unit cube.",
    )
    parser.add_argument(
        "--sigma",
        "-s",
        type=float,
        default=0.01,
        help="Sigma used to compute surface points perturbation.",
    )
    logger.add_logger_args(parser)
    args = parser.parse_args()
    logger.configure_logging(args)

    process_sdf(
        f_in=args.input,
        f_out=args.output,
        f_samp=args.samples,
        n_points=args.n_points,
        uniform_ratio=args.uniform,
        padding=args.padding,
        sigma=args.sigma,
    )
