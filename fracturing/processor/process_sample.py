import os
import argparse
import logging

import trimesh
import numpy as np
from scipy.spatial import KDTree

import processor.logger as logger
import processor.errors as errors


def intersect_mesh(a, b, sig=5):
    """get mask of vertices in a occurring in both a and b, to apply to a"""
    av = [frozenset(np.round(v, sig)) for v in a]
    bv = set([frozenset(np.round(v, sig)) for v in b])
    return np.asarray(list(map(lambda v: v in bv, av)))


def uniform_sample_points(
    dim=256,
    padding=0.1,
):
    grid_pts = np.meshgrid(
        *[
            np.linspace(0, 1.0 + (padding * 2), d) - (0.5 + padding)
            for d in (dim, dim, dim)
        ]
    )
    return np.vstack([p.flatten() for p in grid_pts]).T


def sample_points(
    mesh,
    mask=None,
    mask_uniform=False,
    n_points=500000,
    uniform_ratio=0.5,
    padding=0.1,
    sigma=0.01,
    crop_meshes=True,
):
    def apply_mask(points, vertices, fracture_mask):
        # For each surface point, find the closest mesh point
        tree = KDTree(vertices)
        _, ind = tree.query(points, k=1)

        # If that point is on the fracture, throw it out
        del_list = np.in1d(ind.flatten(), fracture_mask)
        return points[np.logical_not(del_list), :]

    assert mask is None, "Argument under development, do not use"
    assert not mask_uniform, "Argument under development, do not use"

    # Compute more sample points than we need, in case we throw some out
    overshoot = 1.0
    if mask is not None:
        raise NotImplementedError()

    # Compute number of surface and uniform points
    n_points_uniform = int(n_points * float(uniform_ratio))
    n_points_surface = n_points - n_points_uniform

    if crop_meshes:
        cube = trimesh.creation.box(
            extents=[1.0 + padding, 1.0 + padding, 1.0 + padding]
        )
        if isinstance(mesh, list):
            for idx in range(len(mesh)):
                mesh[idx] = mesh[idx].slice_plane(
                    cube.facets_origin, -cube.facets_normal
                )
        else:
            mesh = mesh.slice_plane(cube.facets_origin, -cube.facets_normal)

    points_surface, points_uniform = np.empty((0, 3)), np.empty((0, 3))
    while (points_surface.shape[0] < n_points_surface) or (
        points_uniform.shape[0] < n_points_uniform
    ):

        # Generate uniform points
        boxsize = 1 + padding
        pts_to_sample = int(n_points_uniform * overshoot)
        points_uniform = np.vstack(
            (points_uniform, boxsize * (np.random.rand(pts_to_sample, 3) - 0.5))
        )

        # Handle multiple meshes
        if isinstance(mesh, list):
            pts_to_sample = int((n_points_surface + 1 / len(mesh)) * overshoot)
            points_surface = np.vstack(
                (points_surface, np.vstack([m.sample(pts_to_sample) for m in mesh]))
            )
            vertices = np.concatenate([m.vertices for m in mesh], axis=0)
        else:
            pts_to_sample = int(n_points_surface * overshoot)
            points_surface = np.vstack((points_surface, mesh.sample(pts_to_sample)))
            vertices = mesh.vertices

        # Remove any unwanted faces
        if mask is not None:
            # Obtain a mask by directly comparing the vertex values
            fracture_mask = np.where(
                intersect_mesh(vertices, np.load(mask)["fracture_vertices"])
            )[0]

            # Apply the mask
            points_surface = apply_mask(points_surface, vertices, fracture_mask)
            if mask_uniform:
                points_uniform = apply_mask(points_uniform, vertices, fracture_mask)

    # Finally, apply sigma to the surface points
    points_surface += sigma * np.random.randn(points_surface.shape[0], 3)

    # Randomize the points so that excess points are removed fairly
    points_surface = points_surface[np.random.permutation(points_surface.shape[0]), :]
    points_uniform = points_uniform[np.random.permutation(points_uniform.shape[0]), :]

    # If we have too many points, throw them out
    points_surface = points_surface[:n_points_surface, :]
    points_uniform = points_uniform[:n_points_uniform, :]
    logging.debug(
        "Sampled {} surface points, {} uniform points.".format(
            points_surface.shape[0], points_uniform.shape[0]
        )
    )

    return np.vstack([points_surface, points_uniform])


def process_sample(f_in, f_out, n_points=500000, overwrite=False):
    # Load meshes
    mesh = [trimesh.load(f) for f in f_in]
    if any([not m.is_watertight for m in mesh]):
        raise errors.MeshNotClosedError

    # Sample points
    points = sample_points(
        mesh,
        mask=None,
        mask_uniform=False,
        n_points=n_points,
        uniform_ratio=0.5,
        padding=0.1,
        sigma=0.01,
        crop_meshes=True,
    )

    # Save
    if overwrite or not os.path.exists(f_out):
        logging.debug("Saving to: {}".format(f_out))
        if os.path.splitext(f_out)[-1] == ".ply":
            trimesh.PointCloud(points).export(f_out)
        else:
            np.savez(f_out, xyz=points.astype(np.float16))


def process(
    obj,
    num_results,
    overwrite,
    executor,
    args,
):

    for idx in range(num_results):
        f_in = [obj.path_c(), obj.path_b(idx), obj.path_r(idx)]
        f_out = obj.path_sampled(idx)

        if all([os.path.exists(f) for f in f_in]) and (
            not os.path.exists(f_out) or overwrite
        ):
            executor.graceful_submit(
                process_sample,
                f_in=f_in,
                f_out=f_out,
                n_points=500000,
                overwrite=overwrite,
            )


def validate_outputs(
    obj,
    num_results,
    args,
):
    outputs = []
    for idx in range(num_results):
        if not os.path.exists(obj.path_sampled(idx)):
            outputs.append(False)
            continue
        outputs.append(True)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample surface and uniform points for a mesh or list of "
        + "meshes."
    )
    parser.add_argument(
        dest="input", type=str, nargs="+", help="Path to the input file."
    )
    parser.add_argument(dest="output", type=str, help="Path to the output file.")
    parser.add_argument(
        "--n_points",
        "-n",
        type=int,
        default=500000,
        help="Total number of sample points.",
    )
    logger.add_logger_args(parser)
    args = parser.parse_args()
    logger.configure_logging(args)

    process_sample(f_in=args.input, f_out=args.output, n_points=args.n_points)
