import os
import logging
import pickle5 as pickle

import torch
import torch.utils.data
import trimesh
import numpy as np
from scipy.spatial import cKDTree as KDTree

import core
import core.errors as errors


def select_samples(pts, values, num_samples=None, uniform_ratio=0.5):
    """
    Randomly select a subset of points such that each shape will have at least
    num_samples / (len(shapes) * 2) interior and exterior points.

    Args:
        pts: Points in n dimensional space.
        values: Value at each point. The number of columns gives the number
            of shapes.
        num_samples: Number of samples to return.
        uniform_ratio: Ratio between uniform and surface sampled points.
            Set to 0.5 to disable.
    """
    if uniform_ratio is not None:
        pts, values = select_uniform_ratio_samples(pts, values, uniform_ratio)

    num_shapes = values.shape[1]
    if num_samples is None:
        num_samples = values.shape[0]

    # We want to sample each object about equally
    if num_shapes == 1:
        num_samples = [num_samples]
    else:
        num_samples = [(num_samples // num_shapes) for _ in range(num_shapes - 1)] + [
            num_samples - ((num_shapes - 1) * (num_samples // num_shapes))
        ]

    # Pick the samples fairly
    idx_accumulator = []
    for d, s in zip(values.T, num_samples):
        d = np.expand_dims(d, axis=1)
        pos_inds = np.where(d[:, 0] > 0)[0]
        neg_inds = np.where(d[:, 0] <= 0)[0]
        pos_num, neg_num = (s // 2), (s - (s // 2))

        # Pick negative samples
        if len(neg_inds) > neg_num:
            start_ind = np.random.randint(len(neg_inds) - neg_num)
            neg_picked = neg_inds[start_ind : start_ind + neg_num]
        else:
            try:
                neg_picked = np.random.choice(neg_inds, neg_num)
            except ValueError:
                raise errors.NoSamplesError

        # Pick positive samples
        if len(pos_inds) > pos_num:
            start_ind = np.random.randint(len(pos_inds) - pos_num)
            pos_picked = pos_inds[start_ind : start_ind + pos_num]
        else:
            try:
                pos_picked = np.random.choice(pos_inds, pos_num)
            except ValueError:
                raise errors.NoSamplesError
        idx_accumulator.extend([pos_picked, neg_picked])
    idx_accumulator = np.concatenate(idx_accumulator)

    return pts[idx_accumulator, :], values[idx_accumulator, :]


def select_partial_samples(pts, values, mask, num_samples=None, uniform_ratio=0.5):
    """
    Return a point value pair such that each shape will have at least
    num_samples / (len(shapes) * 2) interior and exterior points. Applies a
    mask first.

    Args:
        pts: Points in n dimensional space.
        values: Value at each point. The number of columns gives the number
            of shapes.
        mask: Mask corresponding to those points to not sample from.
        num_samples: Number of samples to return.
        uniform_ratio: Ratio between uniform and surface sampled points.
            Set to 0.5 to disable.
    """

    # Add an indexing row to pts
    pts = np.hstack((np.expand_dims(np.arange(pts.shape[0]), axis=1), pts))

    # Get a random sampling of the data with the correct uniform ratio
    pts, values = select_uniform_ratio_samples(
        pts, values, uniform_ratio, randomize=False
    )

    # Get non-masked points
    _, intersection_inds, _ = np.intersect1d(
        pts[:, 0], np.where(mask)[0], return_indices=True
    )

    # Remove the indexing row
    pts = pts[:, 1:]

    # Discard non masked points
    pts = pts[intersection_inds, :]
    values = values[intersection_inds, :]

    # Now we need to re-randomize the order
    num_dims = pts.shape[1]
    data = np.hstack((pts, values))
    data = data[np.random.permutation(data.shape[0])]
    pts, values = data[:, :num_dims], data[:, num_dims:]

    # Return fairly selected samples
    return select_samples(pts, values, num_samples, uniform_ratio=None)


def select_uniform_ratio_samples(pts, values, uniform_ratio=0.5, randomize=True):
    """
    Randomly select a subset of points with a specific uniform ratio.

    Args:
        pts: N-dimensional sample points.
        values: Sdf or occupancy values, same size as pts.
        uniform_ratio: Ratio between uniform and surface sampled points.
            Set to 0.5 to disable.
    """

    # Points will come in split half between uniform and surface
    def adjust_uniform_ratio(data, n_pts, bad_surface=0, bad_uniform=0):
        max_can_select = int(n_pts / 2) - max(bad_surface, bad_uniform)
        surface_ends_at = int(n_pts / 2) - bad_surface

        # We can balance the number of uniform and surface points here
        if uniform_ratio > 0.5:
            select_n_pts = int((max_can_select * (1 - uniform_ratio)) / uniform_ratio)
            selected_pts = np.random.choice(
                max_can_select, size=(select_n_pts), replace=False
            )
            data = np.vstack(
                (
                    data[selected_pts, :],
                    data[surface_ends_at : surface_ends_at + max_can_select, :],
                )
            )
        elif uniform_ratio < 0.5:
            select_n_pts = int((max_can_select * uniform_ratio) / (1 - uniform_ratio))
            selected_pts = (
                np.random.choice(max_can_select, size=(select_n_pts), replace=False)
                + surface_ends_at
            )
            data = np.vstack((data[:max_can_select, :], data[selected_pts, :]))
        else:
            data = np.vstack(
                (
                    data[:max_can_select, :],
                    data[surface_ends_at : surface_ends_at + max_can_select, :],
                )
            )
        return data

    num_dims = pts.shape[1]
    data = np.hstack((pts, values))

    # Adjust the ratio of points if necessary
    if uniform_ratio != 0.5:
        data = adjust_uniform_ratio(data, data.shape[0])

    # Shuffle
    if randomize:
        data = data[np.random.permutation(data.shape[0]), :]
    return data[:, :num_dims], data[:, num_dims:]


def sdf_to_occ_grid_threshold(data, thresh=0, flip=False):
    """
    Given a grid, convert sdf values to occupancy values using a specific threshold.
    """
    data = data.copy()
    mask = data >= thresh
    if flip:
        data[mask] = 1
        data[~mask] = 0
    else:
        data[mask] = 0
        data[~mask] = 1
    return data.astype(int)


def sdf_to_occ(data, skip_cols=0):
    """
    Given a sample, convert sdf values to occupancy values.
    """
    if data.shape[1] == 0:
        data[data >= 0] = 0.0
        data[data < 0] = 1.0
    else:
        data[:, skip_cols:][data[:, skip_cols:] >= 0] = 0.0
        data[:, skip_cols:][data[:, skip_cols:] < 0] = 1.0
    return data


def clamp_samples(data, clamp_dist, skip_cols=0):
    """
    Given a sample, clamp that sample to +/- clamp_dist.
    """
    data[:, skip_cols:] = np.clip(data[:, skip_cols:], -clamp_dist, clamp_dist)
    return data


def get_uniform(data):
    """
    Given a sample, return uniform points.
    """
    return data[int(data.shape[0] / 2) :, :]


def get_surface(data):
    """
    Given a sample, return surface points.
    """
    return data[: int(data.shape[0] / 2), :]


def intersect_mesh(a, b, sig=5):
    """mask of vertices that a shares with b"""
    av = [frozenset(np.round(v, sig)) for v in a]
    bv = set([frozenset(np.round(v, sig)) for v in b])
    return np.asarray(list(map(lambda v: v in bv, av)))


def get_faces_from_vertices2(vertex_mask, faces):
    """Get faces containting vertices"""
    vertex_index = set(np.nonzero(vertex_mask)[0])
    face_mask = np.zeros((faces.shape[0],))
    for idx, f in enumerate(faces):
        if f[0] in vertex_index or f[1] in vertex_index or f[2] in vertex_index:
            face_mask[idx] = 1
    return face_mask.astype(bool)


def get_nonfracture_vertex_mask(complete_gt, broken_gt, return_mask=True):
    """
    Return a mask of which points on the broken object that correspond to points not on the fracture
    """
    b_pts = broken_gt.vertices
    num_points = b_pts.shape[0]

    # Indices of the broken object that are very close to the fracture
    b_nonfracture_mask = intersect_mesh(broken_gt.vertices, complete_gt.vertices)
    if return_mask:
        return b_nonfracture_mask
    return np.where(b_nonfracture_mask)[0]


def get_nonfracture_mesh(complete_gt, broken_gt, inverse=False):
    """
    Return a nonfracture submesh of the broken mesh
    """

    not_fracture_inds = get_nonfracture_vertex_mask(complete_gt, broken_gt)
    mesh = broken_gt.copy()
    not_fracture_faces = core.get_faces_from_vertices(
        not_fracture_inds, broken_gt.faces, inclusive=False
    )
    if inverse:
        mesh.update_faces(np.logical_not(not_fracture_faces))
    else:
        mesh.update_faces(not_fracture_faces)

    return mesh


def get_fracture_point_mask(complete_gt, broken_gt, pts):
    """
    Return a mask that identifies points not in the fracture region.
    This simulates high accuracy fracture removal on
    the broken shape. Uses knn search to determine which points to keep.
    """

    b_pts = broken_gt.vertices
    num_points = b_pts.shape[0]

    # Indices of the broken object that are very close to the fracture
    b_fracture_mask = np.logical_not(
        intersect_mesh(broken_gt.vertices, complete_gt.vertices)
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


def partial_sample_with_random_noise(complete_gt, broken_gt, pts, mask, percent=0.15):
    """
    Update a mask corresponding to the nonfracture region, and add random noise.
    """

    if percent == 1.0:
        mask[:] = True
        return mask
    elif percent == 0.0:
        return mask

    # Get the fracture verts (this will work because B comes from C)
    d, _ = KDTree(complete_gt.vertices).query(broken_gt.vertices)
    fracture_vert_mask = d > 0.001

    # How many are we flipping
    num_to_flip = int(fracture_vert_mask.sum() * percent)

    # Pick this many from the mask
    idxs_to_flip = np.random.choice(
        np.where(fracture_vert_mask)[0], num_to_flip, replace=False
    )
    fracture_vert_mask[idxs_to_flip] = False

    # Find the closest point on the broken object for all the query points
    _, ind = KDTree(broken_gt.vertices).query(pts)

    no_sample_zone_mask = np.logical_not(mask)
    eroded_no_sample_zone_mask = np.logical_and(
        no_sample_zone_mask, fracture_vert_mask[ind]
    )
    return np.logical_not(eroded_no_sample_zone_mask)


def partial_sample_with_classification_noise(
    complete_gt, broken_gt, pts, mask, percent=0.15
):
    """
    Update a mask corresponding to the nonfracture region, and add noise at the
    edge of the nonfracture region.
    """

    if percent == 1.0:
        mask[:] = True
        return mask
    elif percent == 0.0:
        return mask

    # Get the fracture verts (this will work because B comes from C)
    d, _ = KDTree(complete_gt.vertices).query(broken_gt.vertices)
    fracture_vertices_index = set(np.where(d > 0.001)[0])
    exterior_vertices_index = set(np.where(d < 0.001)[0])
    start_size = len(fracture_vertices_index)

    while True:
        # Find adjacencies of the exterior surface
        accumulator = set()
        for v1, v2 in broken_gt.edges:
            if v1 in exterior_vertices_index:
                accumulator.add(v2)
            elif v2 in exterior_vertices_index:
                accumulator.add(v1)

        # Subtract them
        fracture_vertices_index = fracture_vertices_index.difference(accumulator)
        exterior_vertices_index = exterior_vertices_index.union(accumulator)

        if len(fracture_vertices_index) < start_size * (1 - percent):
            break

    # Add some vertices back
    add_back = int(start_size * (1 - percent)) - len(fracture_vertices_index)
    fracture_vertices_index = fracture_vertices_index.union(
        set(list(accumulator)[: min(len(accumulator), add_back)])
    )

    # Update the mask
    fracture_vert_mask = np.zeros((broken_gt.vertices.shape[0])).astype(bool)
    fracture_vert_mask[np.array(list(fracture_vertices_index))] = True

    # Find the closest point on the broken object for all the query points
    _, ind = KDTree(broken_gt.vertices).query(pts)

    no_sample_zone_mask = np.logical_not(mask)
    eroded_no_sample_zone_mask = np.logical_and(
        no_sample_zone_mask, fracture_vert_mask[ind]
    )
    return np.logical_not(eroded_no_sample_zone_mask)


def compute_roughness_mask(mesh, threshold=0.01, lamb=0.5, iterations=10):
    """
    Compute a vertex mask that identifies rough vertices on the surface of the mesh.
    optimal parameters:
        lamb = 0.5
        iterations = 10
        threshold = 0.01
    """
    # Perform laplacian smoothing
    smoothed_mesh = mesh.copy()
    trimesh.smoothing.filter_laplacian(smoothed_mesh, lamb=lamb, iterations=iterations)

    # Get indices where roughness is above the threshold
    roughness = np.linalg.norm(mesh.vertices - smoothed_mesh.vertices, axis=1)
    return roughness > threshold


def partial_sample_analytical(
    broken_gt,
    pts,
    percent=0.15,
    threshold=0.01,
    lamb=0.5,
    iterations=10,
):
    """
    Create a mask corresponding to the nonfracture region based on surface
    roughness.
    """

    assert percent == 0.0
    fracture_vert_mask = compute_roughness_mask(
        broken_gt, threshold=threshold, lamb=lamb, iterations=iterations
    )
    not_fracture_vert_mask = np.logical_not(fracture_vert_mask)

    # Find the closest point on the broken object for all the query points,
    # and return the mask at that value
    _, ind = KDTree(broken_gt.vertices).query(pts)
    return not_fracture_vert_mask[ind]


def partial_sample_pointnet(
    broken_gt,
    pts,
    label,
    sample=2**15,
    votes=3,
):
    """
    Create a mask corresponding to the nonfracture region using pointnet.
    """

    # Densely sample the surface of the mesh
    vertices, face_indices = broken_gt.sample(count=sample, return_index=True)
    normals = broken_gt.face_normals[face_indices, :]

    # Get the fracture mask
    mask = core.pointnet.predict(vertices, normals, label, votes)
    fracture_vert_mask = mask.astype(bool)
    not_fracture_vert_mask = np.logical_not(fracture_vert_mask)

    # Find the closest point on the broken object for all the query points,
    # and return the mask at that value
    _, ind = KDTree(vertices).query(pts)
    return not_fracture_vert_mask[ind]


class SamplesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split,  # The dictionary object holding sample or shape data
        use_occ=True,  # Use occupancy samples
        subsample=None,  # How many samples to use during training
        learned_breaks=False,
        uniform_ratio=0.5,  # Ratio of uniform to surface samples
        root=None,  # Location of the top level data directory on disk
        validate_occ=True,  # Will perform validation on the occupancy values
        clamp_dist=None,  # If using sdf, clamp the values
        load_values=True,
        one_code_per_complete=False,
        train_columns=[0, 1, 2],
        use_normals=False,
    ):
        self._use_occ = use_occ
        self._subsample = subsample
        self._learned_breaks = learned_breaks
        self._uniform_ratio = uniform_ratio
        self._clamp_dist = clamp_dist
        self._one_code_per_complete = one_code_per_complete
        self._train_columns = train_columns
        self._use_normals = use_normals

        self._num_dims = 3  # This is always constant for the time being
        self._num_components = None

        if self._clamp_dist is False:
            self._clamp_dist = None
        if not self._use_occ:
            assert self._clamp_dist is not None
            logging.info("Using SDF samples")
        else:
            logging.info("Using OCC samples")

        logging.info("Using point sampling: {}".format(self._subsample))
        logging.info("Using learned breaks: {}".format(self._learned_breaks))
        logging.info("Using uniform ratio:  {}".format(self._uniform_ratio))
        logging.info("Using clamp distance: {}".format(self._clamp_dist))
        logging.info("Using one code per C: {}".format(self._one_code_per_complete))
        logging.info("Using train columns:  {}".format(self._train_columns))
        logging.info("Using normals:        {}".format(self._use_normals))

        # Train columns need to be adjusted because they're stored with points
        self._train_columns = [t + self._num_dims for t in self._train_columns]

        # Data class supports passing a dictionary containing the data directly,
        # if desired. Note that if not passed a dictionary, can pass a string
        if not isinstance(split, dict):
            logging.info("Loading data from: {}".format(split))
            self._raw_data = pickle.load(open(split, "rb"))
        else:
            self._raw_data = split

        # The input data could be in OCC mode to save on disk space. If this is
        # the case, then we cannot convert back to sdf.
        input_data_in_occ = False
        if ("use_occ" in self._raw_data) and self._raw_data["use_occ"]:
            assert use_occ, "Cannot generate sdf samples"
            input_data_in_occ = True

        # Pick apart the input data
        train_mode = self._raw_data.get("train", False)
        self._indexer = self._raw_data["indexer"]
        self._complete_indexer = self._raw_data["complete_indexer"]
        self._objects = self._raw_data["objects"]

        # Record the number of shapes and the number of individual instances
        self._num_shapes = len(set([o[0] for o in self._indexer]))
        self._num_instances = len(self._indexer)
        logging.info(
            "Loaded {} shapes, {} instances".format(self.num_shapes, self.num_instances)
        )

        # Optionally update the stored root location. Note the double check so
        # that root is not set to the empty string. This could be considered
        # a bug, but I don't think it'll ever be used.
        if (root is not None) and root:
            for idx in range(len(self._objects)):
                self._objects[idx]._root_dir = root

        # The data can also be loaded without sdf values. This dramatically
        # reduces size on disk.
        if not load_values:
            return
        self._data = self._raw_data["sdf"]
        try:
            if self._use_normals:
                self._normals = self._raw_data["n"]
        except KeyError:
            pass

        if isinstance(self._data, str):
            self._data = self._data.replace("$DATADIR", os.environ["DATADIR"])
            logging.info("Loading raw sdf data from: {}".format(self._data))

            try:
                self._data = np.load(self._data)["sdf"]
            except FileNotFoundError:
                self._data = np.load(
                    os.path.join(os.path.dirname(split), os.path.basename(self._data))
                )["sdf"]

            if self._use_normals:
                logging.info("Loading raw normal data from: {}".format(self._normals))

                try:
                    self._normals = np.load(self._normals)["n"]
                except FileNotFoundError:
                    self._normals = np.load(
                        os.path.join(
                            os.path.dirname(split), os.path.basename(self._normals)
                        )
                    )["n"]

                assert len(self._data) == len(self._normals)

        if self._use_occ:
            logging.debug("Converting sdf samples to occupancy samples ...")

            # Data is in test mode
            if isinstance(self._data[0], dict):
                if input_data_in_occ:
                    for idx in range(len(self._data)):
                        self._data[idx]["sdf"] = self._data[idx]["sdf"].astype(float)

                else:
                    for idx in range(len(self._data)):
                        self._data[idx]["sdf"] = sdf_to_occ(
                            self._data[idx]["sdf"].astype(float)
                        )

            # Data is in train mode
            else:
                if not input_data_in_occ:
                    for idx in range(len(self._data)):
                        self._data[idx] = sdf_to_occ(
                            self._data[idx].astype(float), skip_cols=self._num_dims
                        )

                # Do a quick check that the values are correct
                if validate_occ:
                    for idx, d in enumerate(self._data):
                        assert (
                            d[:, self._num_dims]
                            == d[:, self._num_dims + 1] + d[:, self._num_dims + 2]
                        ).all(), "Occupancy values are incorrect for object {}".format(
                            idx
                        )

        else:
            if self._clamp_dist is not None:
                logging.debug(
                    "Clamping sdf samples to +/-{} ...".format(self._clamp_dist)
                )

                # Data is in test mode
                if isinstance(self._data[0], dict):
                    for idx in range(len(self._data)):
                        self._data[idx]["sdf"] = clamp_samples(
                            self._data[idx]["sdf"], self._clamp_dist
                        ).astype(float)

                # Data is in train mode
                else:
                    for idx in range(len(self._data)):
                        self._data[idx] = clamp_samples(
                            self._data[idx], self._clamp_dist, skip_cols=self._num_dims
                        )

    @property
    def num_dims(self):
        return self._num_dims

    @property
    def is_occ(self):
        return self._use_occ

    @property
    def num_shapes(self):
        """Number of complete objects"""
        return self._num_shapes

    @property
    def num_instances(self):
        """Number of training/testing instances"""
        return self._num_instances

    @property
    def objects(self):
        return self._objects.copy()

    @property
    def data(self):
        return self._data.copy()

    @property
    def indexer(self):
        return self._indexer.copy()

    @property
    def complete_indexer(self):
        return self._complete_indexer.copy()

    @property
    def num_components(self):
        """Return an array with the number of components associated with each restoration shape"""
        if self._num_components is None:
            self._num_components = np.array(
                [
                    core.metrics.connected_components(self.get_mesh(idx, 2))
                    for idx in range(len(self))
                ]
            )
        return self._num_components.copy()

    def get_complete_index(self, idx):
        """
        Return the means to access a specific object.

        obj index | break index | overall index
        0 : 0 : 0
        0 : 1 : 1
        1 : 0 : 2
        1 : 1 : 3
        """
        obj_idx, _ = self._indexer[idx]
        return obj_idx

    def get_broken_index(self, idx):
        """
        Return the means to access a specific object.

        obj index | break index | overall index
        0 : 0 : 0
        0 : 1 : 1
        1 : 0 : 2
        1 : 1 : 3
        """
        _, break_idx = self._indexer[idx]
        return break_idx

    def get_object(self, idx):
        """Return a specific object"""
        obj_idx, _ = self._indexer[idx]
        return self._objects[obj_idx]

    def get_sample(self, idx, shape_idx):
        """
        Return a complete, broken, or restoration samples. Each sample returned
        will be of the form: (pts, samples).

        Args:
            idx: index of the break to return.
            shape_idx: index of the shape to return in ["complete", "broken",
                "restoration"].
        """
        assert 0 <= shape_idx <= 4
        assert not isinstance(
            self._data[0], dict
        ), "Dataloader is in test mode, use 'get_broken_sample()' instead"

        return tuple(
            (
                self._data[idx][:, : self._num_dims].copy(),
                np.expand_dims(
                    self._data[idx][:, self._num_dims + shape_idx].copy(), axis=1
                ),
            )
        )

    def get_broken_sample(self, idx, return_mask=False):
        """
        Return broken samples (for testing). Each sample returned
        will be of the form: (pts, samples). Optionally return a mask for partial
        view synthesis.

        Args:
            idx: index of the break to return.
            return_mask: return a partial view mask.
        """
        assert isinstance(
            self._data[0], dict
        ), "Dataloader is in training mode, use 'get_sample()' instead"

        if self._use_normals:
            if return_mask:
                return (
                    self._data[idx]["xyz"].copy(),
                    self._data[idx]["sdf"].copy(),
                    self._data[idx]["n"].copy(),
                    self._data[idx]["mask"].copy(),
                )
            return (
                self._data[idx]["xyz"].copy(),
                self._data[idx]["sdf"].copy(),
                self._data[idx]["n"].copy(),
            )

        if return_mask:
            return (
                self._data[idx]["xyz"].copy(),
                self._data[idx]["sdf"].copy(),
                self._data[idx]["mask"].copy(),
            )
        return self._data[idx]["xyz"].copy(), self._data[idx]["sdf"].copy()

    def path_mesh(self, idx, shape_idx):
        """
        Return the path to the mesh for a specific sample.
        """
        assert 0 <= shape_idx <= 3
        obj_idx, break_idx = self._indexer[idx]
        obj = self._objects[obj_idx]
        if shape_idx == 0:
            return obj.path_c()
        elif shape_idx == 1:
            return obj.path_b(break_idx)
        elif shape_idx == 2:
            return obj.path_r(break_idx)
        elif shape_idx == 3:
            return obj.path_tool(break_idx)

    def get_mesh(self, idx, shape_idx):
        """
        Return the mesh for a specific sample.
        """
        assert 0 <= shape_idx <= 3
        obj_idx, break_idx = self._indexer[idx]
        obj = self._objects[obj_idx]
        if shape_idx == 0:
            return obj.load(obj.path_c())
        elif shape_idx == 1:
            return obj.load(obj.path_b(break_idx))
        elif shape_idx == 2:
            return obj.load(obj.path_r(break_idx))
        elif shape_idx == 3:
            return obj.load(obj.path_tool(break_idx))

    def get_tool(self, idx):
        """
        Return the mesh for a specific sample.
        """
        obj_idx, break_idx = self._indexer[idx]
        obj = self._objects[obj_idx]
        return obj.load(obj.path_tool(break_idx))

    def get_render(self, idx, shape_idx, angle=0, resolution=(200, 200), save=True):
        """
        Return render for a specific sample.
        """
        assert 0 <= shape_idx <= 2
        obj_idx, break_idx = self._indexer[idx]
        obj = self._objects[obj_idx]

        try:
            if shape_idx == 0:
                path = obj.path_c_rendered(angle=angle, resolution=resolution)
            elif shape_idx == 1:
                path = obj.path_b_rendered(
                    idx=break_idx, angle=angle, resolution=resolution
                )
            elif shape_idx == 2:
                path = obj.path_r_rendered(
                    idx=break_idx, angle=angle, resolution=resolution
                )
            try:
                return obj.load(path)
            except FileNotFoundError:
                pass

            logging.debug(
                "Render ({}, {}, {}) with filename {} could not be found, generating".format(
                    idx,
                    shape_idx,
                    angle,
                    path,
                )
            )
        except (errors.PathAccessError, PermissionError):
            save = False  # Cannot access the path, so cannot load or save

            logging.debug(
                "Render ({}, {}, {}) could not be accessed, generating".format(
                    idx,
                    shape_idx,
                    angle,
                )
            )

        try:
            render = core.utils_3d.render_mesh(
                self.get_mesh(idx, shape_idx),
                yrot=angle,
                resolution=resolution,
                remove_texture=True,
            )
        except ValueError:
            return np.ones(list(resolution) + [3])

        if save:
            try:
                core.handler.saver(path, render)
            except PermissionError:
                pass
        return render

    def get_composite(
        self,
        idx,
        angle=0,
        resolution=(200, 200),
        self_color=(128, 64, 64, 255),
        gt_color=(64, 128, 64, 128),
        save=True,
    ):
        obj_idx, break_idx = self._indexer[idx]
        obj = self._objects[obj_idx]

        try:
            path = obj.path_composite(idx=break_idx, angle=angle, resolution=resolution)
            try:
                return obj.load(path)
            except FileNotFoundError:
                pass

            logging.debug(
                "Composite ({}, {}) with filename {} could not be found, generating".format(
                    idx,
                    angle,
                    path,
                )
            )
        except (errors.PathAccessError, PermissionError):
            save = False  # Cannot access the path, so cannot load or save

            logging.debug(
                "Composite ({}, {}) could not be accessed, generating".format(
                    idx,
                    angle,
                )
            )

        # Get the meshes
        r_mesh = self.get_mesh(idx, 2)
        b_mesh = self.get_mesh(idx, 1)

        # Update the colors
        r_mesh.visual = trimesh.visual.color.ColorVisuals(
            r_mesh, vertex_colors=self_color
        )
        b_mesh.visual = trimesh.visual.color.ColorVisuals(
            b_mesh, vertex_colors=gt_color
        )

        # Render
        render = core.utils_3d.render_mesh(
            [b_mesh, r_mesh],
            yrot=angle,
            resolution=resolution,
        )

        if save:
            try:
                core.handler.saver(path, render)
            except PermissionError:
                pass
        return render

    def get_classname(self, idx):
        """
        Return classname for a given shape.
        """
        return self._objects[self._indexer[idx][0]].class_id

    def __len__(self):
        """
        Return the number of discrete shapes in the dataset.
        """
        return self.num_instances

    def __getitem__(self, idx):
        """
        Return
            [[pts, c, b, r, t], [idx, b_idx]]
            OR
            [[pts, c, b, r], [idx]]
        """

        if self._learned_breaks:
            # Get the corresponding complete object index
            c_idx = idx
            if self._one_code_per_complete:
                c_idx = self.get_complete_index(idx)
            b_idx = idx

            indices = tuple(
                [
                    torch.tensor(c_idx).repeat(1, self._subsample).T,  # Complete idx
                    torch.tensor(b_idx).repeat(1, self._subsample).T,  # Break idx
                ]
            )

        else:
            indices = tuple([torch.tensor(idx).repeat(1, self._subsample).T])

        # Select the columns to use for training
        pts = self._data[idx][:, : self._num_dims]
        values = self._data[idx][:, self._train_columns]
        if len(values.shape) == 1:
            values = np.expand_dims(values, axis=1)

        if self._use_normals:
            normals = self._normals[idx]
            pts = np.hstack((pts, normals))

        # Will return [pts], [c, b, r, t], etc.
        pts, values = core.data.select_samples(
            pts,
            values,
            self._subsample,
            self._uniform_ratio,
        )

        if self._use_normals:
            pts, normals = pts[:, : self._num_dims], pts[:, self._num_dims :]
            sdf_data = tuple(
                [torch.from_numpy(pts)]
                + [
                    torch.from_numpy(
                        normals[:, (c * self._num_dims) : ((c + 1) * self._num_dims)]
                    )
                    for c in [t - self._num_dims for t in self._train_columns]
                ]
                + [torch.from_numpy(v).unsqueeze(1) for v in values.T]
            )
        else:
            sdf_data = tuple(
                [torch.from_numpy(pts)]
                + [torch.from_numpy(v).unsqueeze(1) for v in values.T]
            )

        return sdf_data, indices
