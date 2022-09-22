import os
import logging

import trimesh
import numpy as np
import scipy.interpolate as interpolator
from sklearn.decomposition import PCA
from matplotlib import tri

from processor.process_sdf2 import compute_sdf
import processor.errors as errors


def points_maximal_orient(points):
    """Return the transformation matrix that orients a point set by its maximal dimensions"""
    pca = PCA(n_components=3)
    pca.fit(points)
    matrix = pca.components_
    return np.vstack(
        (
            np.hstack(
                (
                    np.expand_dims(
                        matrix[0, :], axis=1
                    ),  # X corresponds to largest component
                    np.expand_dims(matrix[1, :], axis=1),  #
                    np.expand_dims(
                        matrix[2, :], axis=1
                    ),  # Z corresponds to smallest component
                    np.zeros((3, 1)),
                )
            ),
            np.array([0, 0, 0, 1]),
        )
    ).T


def normalize_transform(v):
    """Return matrix that centers vertices"""
    return trimesh.transformations.translation_matrix(-v.mean(axis=0))


def points_transform_matrix(vs, mat):
    """Apply a transformation matrix to a set of points"""
    return np.dot(mat, np.hstack((vs, np.ones((vs.shape[0], 1)))).T).T[:, :3]


def intersect_mesh(a, b, sig=5):
    """mask of vertices that a shares with b"""
    av = [frozenset(np.round(v, sig)) for v in a]
    bv = set([frozenset(np.round(v, sig)) for v in b])
    return np.asarray(list(map(lambda v: v in bv, av)))


def get_faces_from_vertices(vertex_mask, faces, inclusive=False):
    """Get faces containting vertices"""
    vertex_index = set(np.nonzero(vertex_mask)[0])
    face_mask = np.zeros((faces.shape[0],))
    for idx, f in enumerate(faces):
        if inclusive:
            if f[0] in vertex_index and f[1] in vertex_index and f[2] in vertex_index:
                face_mask[idx] = 1
        else:
            if f[0] in vertex_index or f[1] in vertex_index or f[2] in vertex_index:
                face_mask[idx] = 1
    return face_mask.astype(bool)


def get_fracture_points(b, r):
    """Get points on the fracture"""
    vb, vr = b.vertices, r.vertices
    logging.debug(
        "Computing fracture points for meshes with size {} and {} ...".format(
            vb.shape[0], vr.shape[0]
        )
    )
    return intersect_mesh(vb, vr)


def plot_3d(f, inv_orienter, density=128, limit=0.5):
    x, y = np.meshgrid(
        np.linspace(-limit, limit, density),
        np.linspace(-limit, limit, density),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    z = f(x, y)

    pts = np.hstack(
        (
            np.expand_dims(x.flatten(), axis=1),
            np.expand_dims(y.flatten(), axis=1),
            np.expand_dims(z.flatten(), axis=1),
        )
    )

    return inv_orienter(pts)


def fit_3d(b, r, method="thin_plate", smoothing=0):
    # Get the fracture region
    frac_points = b.vertices[get_fracture_points(b, r), :]
    # assert frac_points.shape[0] > 200, "Too few fracture points"

    # Orient the fracture region and extract the corresponding matrix
    mat1 = normalize_transform(frac_points)
    norm_frac_points = points_transform_matrix(
        frac_points,
        mat1,
    )
    mat2 = points_maximal_orient(norm_frac_points)
    oriented_norm_frac_points = points_transform_matrix(
        norm_frac_points,
        mat2,
    )
    mat = mat2 @ mat1
    mat_inv = np.linalg.inv(mat)

    logging.debug(
        "Fitting 3D function on {} points ...".format(
            oriented_norm_frac_points.shape[0]
        )
    )

    fit_function = interpolator.Rbf(
        oriented_norm_frac_points[:, 0],
        oriented_norm_frac_points[:, 1],
        oriented_norm_frac_points[:, 2],
        function=method,
        smoothing=smoothing,
    )

    def orienter(pts):
        return points_transform_matrix(pts, mat)

    def inv_orienter(pts):
        return points_transform_matrix(pts, mat_inv)

    return fit_function, orienter, inv_orienter


def fit_quality(function, orienter, b, r):
    ptsb = orienter(b.vertices)
    zb = function(ptsb[:, 0], ptsb[:, 1])
    ptsr = orienter(r.vertices)
    zr = function(ptsr[:, 0], ptsr[:, 1])
    occupancy = np.hstack(
        (
            zb >= ptsb[:, 2],
            zr <= ptsr[:, 2],
        )
    ).astype(bool)
    if 1 - occupancy.mean() > occupancy.mean():
        occupancy = ~occupancy
    return occupancy.astype(int)


def batch_eval(function, pts, batch_size=50000):
    accumulator = []
    for start in range(0, pts.shape[0], batch_size):
        end = min(start + batch_size, pts.shape[0])
        accumulator.append(function(pts[start:end, 0], pts[start:end, 1]))
    return np.hstack(accumulator).flatten()


def compute_fitted_occupancy(
    input_pts, function, orienter, b, r, return_accuracy=False
):
    logging.debug("Computing sample occupancy from spline ...")
    # Evaluate broken
    oriented_b_pts = orienter(b.vertices)
    zb = batch_eval(function, oriented_b_pts)

    # Evaluate restoration
    oriented_r_pts = orienter(r.vertices)
    zr = batch_eval(function, oriented_r_pts)

    # Evaluate sample points
    oriented_input_pts = orienter(input_pts)
    zpts = batch_eval(function, oriented_input_pts)

    # Compute if the point is above or below the plane
    occupancy = np.hstack(
        (
            zb >= oriented_b_pts[:, 2],
            zr <= oriented_r_pts[:, 2],
        )
    ).astype(bool)
    pt_occupancy = (zpts < oriented_input_pts[:, 2]).astype(bool)

    # We don't know which way the plane is oriented, so we may need to flip it
    if (1 - occupancy.mean()) > occupancy.mean():
        occupancy = ~occupancy

    if return_accuracy:
        return pt_occupancy, occupancy.mean()
    return pt_occupancy


def triangulate_plane(plane):
    try:
        vertices = plane.vertices
    except AttributeError:
        vertices = plane
    x, y = np.meshgrid(
        np.arange(int(np.sqrt(vertices.shape[0]))),
        np.arange(int(np.sqrt(vertices.shape[0]))),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()
    return trimesh.Trimesh(vertices, tri.Triangulation(x, y).triangles)


def get_boundary_inds(mesh):
    index = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    return np.unique(mesh.edges[index].flatten())


def cube_cut(mesh, side_length=1.2):
    vertex_mask = np.vstack(
        (
            mesh.vertices[:, 0] < (side_length / 2),
            mesh.vertices[:, 0] > -(side_length / 2),
            mesh.vertices[:, 1] < (side_length / 2),
            mesh.vertices[:, 1] > -(side_length / 2),
            mesh.vertices[:, 2] < (side_length / 2),
            mesh.vertices[:, 2] > -(side_length / 2),
        )
    ).all(axis=0)

    # Make sure all boundary points are removed
    vertex_inds = np.where(vertex_mask)[0]
    boundary_indx = get_boundary_inds(mesh)
    assert len(set(vertex_inds).intersection(set(boundary_indx))) == 0

    # Remove faces outside of the cube
    face_mask = get_faces_from_vertices(vertex_mask, mesh.faces, inclusive=False)
    mesh.update_faces(face_mask)

    return mesh


def spliner(
    f_in,
    f_sdf,
    f_rest,
    f_rest_sdf,
    f_samp,
    f_out,
    f_plane,
    f_spline_mesh,
    accuracy_threshold=0.95,
    overwrite=False,
    full_spline=False,
):
    # Load meshes
    broken_mesh = trimesh.load(f_in)
    if not broken_mesh.is_watertight:
        raise errors.MeshNotClosedError
    restoration_mesh = trimesh.load(f_rest)
    if not restoration_mesh.is_watertight:
        raise errors.MeshNotClosedError
    pts = np.load(f_samp)["xyz"]

    methods = [
        "thin_plate",
        "linear",
        "inverse",
    ]

    for m in methods:
        for _ in range(1):
            # Fit
            f, orienter, inv_orienter = fit_3d(broken_mesh, restoration_mesh, m)

            # This gives you the points corresponding to the fit spline
            plane_points = plot_3d(f, inv_orienter)

            # Compute point occupancy
            mask, accuracy = compute_fitted_occupancy(
                pts, f, orienter, broken_mesh, restoration_mesh, return_accuracy=True
            )
            logging.debug(
                "Computed {} spline fit with accuracy: {}".format(m, accuracy)
            )

            if accuracy > accuracy_threshold:
                break

        if accuracy > accuracy_threshold:
            break

    if accuracy <= accuracy_threshold:
        raise errors.SplineFitError

    b_sdf = np.load(f_sdf)["sdf"]
    r_sdf = np.load(f_rest_sdf)["sdf"]

    inside_b = b_sdf.squeeze() <= 0.0
    inside_r = r_sdf.squeeze() < 0.0

    # The mask is 1 where it intersects with the restoration
    correct = (mask[inside_b] == False).sum() + (mask[inside_r] == True).sum()
    correct_flipped = (~mask[inside_b] == False).sum() + (~mask[inside_r] == True).sum()
    if correct_flipped > correct:
        mask = ~mask

    # Get break surface
    if full_spline:
        # Get the plane that bisects the unit cube
        plane_points = plot_3d(f, inv_orienter, density=256, limit=1.2)
        spline_mesh = triangulate_plane(plane_points)
        spline_mesh = cube_cut(spline_mesh, side_length=1.2)
    else:
        fracture_vertex_mask = intersect_mesh(
            broken_mesh.vertices, restoration_mesh.vertices
        )
        fracture_face_mask = get_faces_from_vertices(
            fracture_vertex_mask, broken_mesh.faces, inclusive=True
        )
        spline_mesh = broken_mesh.copy()
        spline_mesh.update_faces(fracture_face_mask)

    # Get break surface sdf
    sdf, normals = compute_sdf(
        mesh=spline_mesh,
        points=pts,
        return_normals=True,
        compute_sign=False,
        fix_normals=False,
    )

    # Mask should include no points in the fracture
    mask[inside_b] = False
    # Mask should include all points in the restoration
    mask[inside_r] = True

    # Apply the sign based on the mask
    sdf[mask] = -np.abs(sdf[mask])
    sdf[np.logical_not(mask)] = np.abs(sdf[np.logical_not(mask)])

    # Export
    if overwrite or not os.path.exists(f_out):
        logging.debug("Saving to: {}".format(f_out))
        np.savez(
            f_out,
            sdf=sdf.astype(np.float16),
            n=normals.astype(np.float16),
            method=m,
        )

    if overwrite or not os.path.exists(f_plane):
        logging.debug("Saving to: {}".format(f_plane))
        trimesh.PointCloud(plane_points).export(f_plane)

    if overwrite or not os.path.exists(f_spline_mesh):
        logging.debug("Saving to: {}".format(f_spline_mesh))
        spline_mesh.export(f_spline_mesh)


def process(
    obj,
    num_results,
    overwrite,
    executor,
    args,
):

    for idx in range(num_results):
        if args.full_spline:
            f_out = obj.path_full_spline_sdf(idx)
        else:
            f_out = obj.path_spline_sdf(idx)

        f_in = [
            obj.path_b(idx),
            obj.path_b_sdf(idx),
            obj.path_r(idx),
            obj.path_r_sdf(idx),
            obj.path_sampled(idx),
        ]

        if all([os.path.exists(f) for f in f_in]) and (
            not os.path.exists(f_out) or overwrite
        ):
            executor.graceful_submit(
                spliner,
                f_in=obj.path_b(idx),
                f_sdf=obj.path_b_sdf(idx),
                f_rest=obj.path_r(idx),
                f_rest_sdf=obj.path_r_sdf(idx),
                f_samp=obj.path_sampled(idx),
                f_out=f_out,
                f_plane=obj.path_spline_plane(idx),
                f_spline_mesh=obj.path_spline_mesh(idx),
                full_spline=args.full_spline,
            )


def validate_outputs(
    obj,
    num_results,
    args,
):
    outputs = []
    for idx in range(num_results):

        if args.full_spline:
            f_out = obj.path_full_spline_sdf(idx)
        else:
            f_out = obj.path_spline_sdf(idx)

        if not os.path.exists(f_out):
            outputs.append(False)
            continue
        outputs.append(True)
    return outputs
