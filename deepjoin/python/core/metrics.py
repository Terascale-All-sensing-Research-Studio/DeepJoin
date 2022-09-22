import logging
import trimesh
import numpy as np

from scipy.spatial import cKDTree as KDTree

try:
    from libmesh import check_mesh_contains
except ImportError:
    pass

import core
import core.errors as errors


def chamfer(gt_shape, pred_shape, num_mesh_samples=2000):
    """
    Compute the chamfer distance for two 3D meshes.
    This function computes a symmetric chamfer distance, i.e. the mean chamfers.
    Based on the code provided by DeepSDF.

    Args:
        gt_shape (trimesh object or points): Ground truth shape.
        pred_shape (trimesh object): Predicted shape.
        num_mesh_samples (points): Number of points to sample from the predicted
            shape. Must be the same number of points as were computed for the
            ground truth shape.
    """

    if pred_shape.vertices.shape[0] == 0:
        raise core.errors.MeshEmptyError
    assert gt_shape.vertices.shape[0] != 0, "gt shape has no vertices"

    try:
        gt_pts = trimesh.sample.sample_surface(gt_shape, num_mesh_samples)[0]
    except AttributeError:
        gt_pts = gt_shape
        assert (
            gt_pts.shape[0] == num_mesh_samples
        ), "Wrong number of gt points, expected {} got {}".format(
            num_mesh_samples, gt_pts.shape[0]
        )
    pred_pts = trimesh.sample.sample_surface(pred_shape, num_mesh_samples)[0]

    # one direction
    one_distances, _ = KDTree(pred_pts).query(gt_pts)
    gt_to_pred_chamfer = np.mean(np.square(one_distances))

    # other direction
    two_distances, _ = KDTree(gt_pts).query(pred_pts)
    pred_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_pred_chamfer + pred_to_gt_chamfer


def normal_consistency(gt_shape, pred_shape, num_mesh_samples=30000):
    """
    Compute the normal alignment for two 3d meshes.
    Based on the code provided by Occupancy Networks.
    Args:
        gt_shape (trimesh object): Ground truth shape.
        pred_shape (trimesh object): Predicted shape.
    """
    if pred_shape.vertices.shape[0] == 0:
        raise core.errors.MeshEmptyError
    assert gt_shape.vertices.shape[0] != 0, "gt shape has no vertices"

    def normal_diff(obj_from, obj_to):
        obj_to.face_normals
        obj_from.face_normals

        verts_from, face_indices_from = obj_from.sample(
            count=num_mesh_samples, return_index=True
        )
        verts_to, face_indices_to = obj_to.sample(
            count=num_mesh_samples, return_index=True
        )

        _, idx = KDTree(verts_to).query(verts_from)

        normals_from = obj_from.face_normals[face_indices_from, :]
        normals_to = obj_to.face_normals[face_indices_to[idx], :]

        # Normalize the normals, sometimes trimesh doesn't do this
        normals_from = normals_from / np.linalg.norm(
            normals_from, axis=-1, keepdims=True
        )
        normals_to = normals_to / np.linalg.norm(normals_to, axis=-1, keepdims=True)

        # Compute the dot product
        return (normals_to * normals_from).sum(axis=-1)

    return np.hstack(
        (normal_diff(gt_shape, pred_shape), normal_diff(pred_shape, gt_shape))
    ).mean()


def connected_artifacts_score2(
    gt_complete,
    gt_broken,
    gt_restoration,
    pd_restoration,
    max_dist=0.02,
    num_mesh_samples=30000,
):

    if pd_restoration.vertices.shape[0] == 0:
        raise core.errors.MeshEmptyError
    assert gt_complete.vertices.shape[0] != 0, "gt shape has no vertices"
    assert gt_broken.vertices.shape[0] != 0, "gt shape has no vertices"
    assert gt_restoration.vertices.shape[0] != 0, "gt shape has no vertices"

    # Get the fracture and exterior vertices
    exterior_verts = np.ones(gt_broken.vertices.shape[0]).astype(bool)
    exterior_verts[core.get_fracture_points(gt_broken, gt_restoration)] = False

    # Get the associated faces
    # fracture_faces = fracture_verts[gt_broken.faces].all(axis=1)
    exterior_faces = exterior_verts[gt_broken.faces].all(axis=1)

    # Sample the broken
    gt_broken_points, face_inds = trimesh.sample.sample_surface(
        gt_broken, num_mesh_samples
    )
    _, exterior_inds, _ = np.intersect1d(
        face_inds, np.where(exterior_faces)[0], return_indices=True
    )
    exterior_points = gt_broken_points[exterior_inds, :]

    # Sample the restorations
    gt_restoration_points = trimesh.sample.sample_surface(
        gt_restoration, num_mesh_samples
    )[0]
    pd_restoration_points = trimesh.sample.sample_surface(
        pd_restoration, num_mesh_samples
    )[0]

    # Throw out exterior points that have a close point in the gt restoration
    d = KDTree(gt_restoration_points).query(exterior_points)[0]
    exterior_points = exterior_points[d > max_dist, :]

    # What percentage of exterior points DO have close neighbors when they SHOULDNT?
    return (
        KDTree(pd_restoration_points).query(exterior_points)[0] < max_dist
    ).sum() / exterior_points.shape[0]


def connected_components(mesh):
    """
    Return number of connected components.
    """
    if mesh.vertices.shape[0] == 0:
        raise core.errors.MeshEmptyError
    return len(core.utils_3d.trimesh2vedo(mesh).splitByConnectivity())


def component_error(gt_shape, pred_shape):
    """
    Return difference of connected components.
    """
    return abs(connected_components(pred_shape) - connected_components(gt_shape))


def connected_protrusion_error(
    gt_shape, pred_shape, num_mesh_samples=30000, max_dist=0.02
):
    """
    How many points, when moved back by max_dist, are still inside of the shape?
    """

    # Sample and get normals
    points_pred, face_indices_pred = pred_shape.sample(
        count=num_mesh_samples, return_index=True
    )
    points_gt, face_indices_gt = gt_shape.sample(
        count=num_mesh_samples, return_index=True
    )
    normals_pred = pred_shape.face_normals[face_indices_pred, :]
    normals_gt = gt_shape.face_normals[face_indices_gt, :]

    # Renormalize
    normals_pred = normals_pred / np.linalg.norm(normals_pred, axis=-1, keepdims=True)
    normals_gt = normals_gt / np.linalg.norm(normals_gt, axis=-1, keepdims=True)

    # Get the offset points
    points_pred -= normals_pred * max_dist
    points_gt -= normals_gt * max_dist

    try:
        return (
            abs(
                check_mesh_contains(pred_shape, points_pred).sum()
                - check_mesh_contains(gt_shape, points_gt).sum()
            )
            / num_mesh_samples
        )
    except RuntimeError:
        return np.nan
