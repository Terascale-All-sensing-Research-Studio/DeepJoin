import math
import sys
import argparse, os
import logging
import subprocess

import mcubes
import numpy as np
from scipy import ndimage
import trimesh
from trimesh import grouping, util

# This is required to get librender and pyrender to play nice
assert os.environ[
    "LIBRENDERPATH"
], "Need to set LIBRENDERPATH to the top level directory of librender"
sys.path.insert(0, os.environ["LIBRENDERPATH"])

import librender

try:
    import libfusiongpu as libfusion
    from libfusiongpu import tsdf_gpu as compute_tsdf
except (ImportError, OSError):
    logging.debug("Not able to import GPU libfusion, falling back to CPU.")
    import libfusioncpu as libfusion
    from libfusioncpu import tsdf_cpu as compute_tsdf

import processor.errors as errors
import processor.logger as logger


def subdivide_to_size(vertices, faces, max_edge, max_iter=10, return_index=False):
    """
    Subdivide a mesh until every edge is shorter than a
    specified length.
    Will return a triangle soup, not a nicely structured mesh.
    Parameters
    ------------
    vertices : (n, 3) float
      Vertices in space
    faces : (m, 3) int
      Indices of vertices which make up triangles
    max_edge : float
      Maximum length of any edge in the result
    max_iter : int
      The maximum number of times to run subdivision
    return_index : bool
      If True, return index of original face for new faces
    Returns
    ------------
    vertices : (j, 3) float
      Vertices in space
    faces : (q, 3) int
      Indices of vertices
    index : (q, 3) int
      Only returned if `return_index`, index of
      original face for each new face.
    """
    # store completed
    done_face = []
    done_vert = []
    done_idx = []

    # copy inputs and make sure dtype is correct
    current_faces = np.array(faces, dtype=np.int64, copy=True)
    current_vertices = np.array(vertices, dtype=np.float64, copy=True)
    current_index = np.arange(len(faces))

    # loop through iteration cap
    for i in range(max_iter + 1):
        # compute the length of every triangle edge
        edge_length = (
            np.diff(current_vertices[current_faces[:, [0, 1, 2, 0]], :3], axis=1) ** 2
        ).sum(axis=2) ** 0.5
        # check edge length against maximum
        too_long = (edge_length > max_edge).any(axis=1)
        # faces that are OK
        face_ok = ~too_long

        # clean up the faces a little bit so we don't
        # store a ton of unused vertices
        unique, inverse = grouping.unique_bincount(
            current_faces[face_ok].flatten(), return_inverse=True
        )

        # store vertices and faces meeting criteria
        done_vert.append(current_vertices[unique])
        done_face.append(inverse.reshape((-1, 3)))
        done_idx.append(current_index[face_ok])

        # met our goals so exit
        if not too_long.any():
            break

        current_index = np.tile(current_index[too_long], (4, 1)).T.ravel()
        # run subdivision again
        (current_vertices, current_faces) = trimesh.remesh.subdivide(
            current_vertices, current_faces[too_long]
        )

    if i >= max_iter:
        util.log.warning(
            "subdivide_to_size reached maximum iterations before exit criteria!"
        )

    # stack sequence into nice (n, 3) arrays
    final_vertices, final_faces = util.append_faces(done_vert, done_face)

    if return_index:
        final_index = np.concatenate(done_idx)
        assert len(final_index) == len(final_faces)
        return final_vertices, final_faces, final_index

    return final_vertices, final_faces


def subdivide_to_size_textured(
    vertices, faces, uv, max_edge, max_iter=10, return_index=False
):

    vertices, faces = subdivide_to_size(
        vertices=np.hstack((vertices, uv)),
        faces=faces,
        max_edge=max_edge,
        max_iter=max_iter,
        return_index=return_index,
    )

    # Returns vertices, faces, uv
    return vertices[:, :3], faces, vertices[:, 3:]


def subdivide_scene_to_size_textured(mesh, max_edge):
    """
    Does three things:
        - upsamples scene objects to improve texture quality.
        - converts textures to vertex colors.
        - converts a trimesh scene object to a mesh.
    """

    logging.debug("Attempting to subdivide mesh")
    if isinstance(mesh, trimesh.Scene):
        for k in mesh.geometry.keys():
            if isinstance(
                mesh.geometry[k].visual, trimesh.visual.texture.TextureVisuals
            ):
                logging.debug("Input mesh is multi mesh with TextureVisuals")
                if mesh.geometry[k].visual.uv is not None:
                    (
                        mesh.geometry[k].vertices,
                        mesh.geometry[k].faces,
                        mesh.geometry[k].visual.uv,
                    ) = subdivide_to_size_textured(
                        vertices=mesh.geometry[k].vertices,
                        faces=mesh.geometry[k].faces,
                        uv=mesh.geometry[k].visual.uv,
                        max_edge=max_edge,
                    )
                    mesh.geometry[k].visual = mesh.geometry[k].visual.to_color()
                else:
                    (
                        mesh.geometry[k].vertices,
                        mesh.geometry[k].faces,
                    ) = trimesh.remesh.subdivide_to_size(
                        vertices=mesh.geometry[k].vertices,
                        faces=mesh.geometry[k].faces,
                        max_edge=max_edge,
                    )
                    mesh.geometry[k].visual = trimesh.visual.color.ColorVisuals(
                        mesh=mesh.geometry[k],
                        vertex_colors=np.ones(
                            (mesh.geometry[k].vertices.shape[0], 4)
                        ).astype(np.uint8)
                        * 255,
                    )
            else:
                logging.debug("Input mesh is multi mesh with ColorVisuals")
                (
                    mesh.geometry[k].vertices,
                    mesh.geometry[k].faces,
                ) = trimesh.remesh.subdivide_to_size(
                    vertices=mesh.geometry[k].vertices,
                    faces=mesh.geometry[k].faces,
                    max_edge=max_edge,
                )

        # Turn into one single mesh
        mesh = trimesh.util.concatenate(
            [
                trimesh.Trimesh(
                    vertices=m.vertices,
                    faces=m.faces,
                    vertex_colors=m.visual.vertex_colors,
                )
                for m in mesh.geometry.values()
            ]
        )

    else:
        if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
            logging.debug("Input mesh is single mesh with TextureVisuals")
            if mesh.visual.uv is not None:
                mesh.vertices, mesh.faces, mesh.visual.uv = subdivide_to_size_textured(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    uv=mesh.visual.uv,
                    max_edge=max_edge,
                )

                mesh.visual = mesh.visual.to_color()
            else:
                mesh.vertices, mesh.faces = trimesh.remesh.subdivide_to_size(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    max_edge=max_edge,
                )
                mesh.visual = trimesh.visual.color.ColorVisuals(
                    mesh=mesh,
                    vertex_colors=np.ones((mesh.vertices.shape[0], 4)).astype(np.uint8)
                    * 255,
                )
        else:
            logging.debug("Input mesh is single mesh with ColorVisuals")
            mesh.vertices, mesh.faces = trimesh.remesh.subdivide_to_size(
                vertices=mesh.vertices,
                faces=mesh.faces,
                max_edge=max_edge,
            )
    return mesh


class Mesh:
    """
    Represents a mesh.
    """

    def __init__(self, vertices=[[]], faces=[[]]):
        """
        Construct a mesh from vertices and faces.

        :param vertices: list of vertices, or numpy array
        :type vertices: [[float]] or numpy.ndarray
        :param faces: list of faces or numpy array, i.e. the indices of the corresponding vertices per triangular face
        :type faces: [[int]] fo rnumpy.ndarray
        """

        self.vertices = np.array(vertices, dtype=float)
        """ (numpy.ndarray) Vertices. """

        self.faces = np.array(faces, dtype=int)
        """ (numpy.ndarray) Faces. """

        assert self.vertices.shape[1] == 3
        assert self.faces.shape[1] == 3

    def extents(self):
        """
        Get the extents.

        :return: (min_x, min_y, min_z), (max_x, max_y, max_z)
        :rtype: (float, float, float), (float, float, float)
        """

        min = [0] * 3
        max = [0] * 3

        for i in range(3):
            min[i] = np.min(self.vertices[:, i])
            max[i] = np.max(self.vertices[:, i])

        return tuple(min), tuple(max)

    def switch_axes(self, axis_1, axis_2):
        """
        Switch the two axes, this is usually useful for switching y and z axes.

        :param axis_1: index of first axis
        :type axis_1: int
        :param axis_2: index of second axis
        :type axis_2: int
        """

        temp = np.copy(self.vertices[:, axis_1])
        self.vertices[:, axis_1] = self.vertices[:, axis_2]
        self.vertices[:, axis_2] = temp

    def mirror(self, axis):
        """
        Mirror given axis.

        :param axis: axis to mirror
        :type axis: int
        """

        self.vertices[:, axis] *= -1

    def scale(self, scales):
        """
        Scale the mesh in all dimensions.

        :param scales: tuple of length 3 with scale for (x, y, z)
        :type scales: (float, float, float)
        """

        assert len(scales) == 3

        for i in range(3):
            self.vertices[:, i] *= scales[i]

    def translate(self, translation):
        """
        Translate the mesh.

        :param translation: translation as (x, y, z)
        :type translation: (float, float, float)
        """

        assert len(translation) == 3

        for i in range(3):
            self.vertices[:, i] += translation[i]

    def _rotate(self, R):

        self.vertices = np.dot(R, self.vertices.T)
        self.vertices = self.vertices.T

    def rotate(self, rotation):
        """
        Rotate the mesh.

        :param rotation: rotation in (angle_x, angle_y, angle_z); angles in radians
        :type rotation: (float, float, float
        :return:
        """

        assert len(rotation) == 3

        x = rotation[0]
        y = rotation[1]
        z = rotation[2]

        # rotation around the x axis
        R = np.array(
            [[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]]
        )
        self._rotate(R)

        # rotation around the y axis
        R = np.array(
            [[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]]
        )
        self._rotate(R)

        # rotation around the z axis
        R = np.array(
            [[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]]
        )
        self._rotate(R)

    def inv_rotate(self, rotation):
        """
        Rotate the mesh.

        :param rotation: rotation in (angle_x, angle_y, angle_z); angles in radians
        :type rotation: (float, float, float
        :return:
        """

        assert len(rotation) == 3

        x = rotation[0]
        y = rotation[1]
        z = rotation[2]

        # rotation around the x axis
        R = np.array(
            [[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]]
        )
        R = R.T
        self._rotate(R)

        # rotation around the y axis
        R = np.array(
            [[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]]
        )
        R = R.T
        self._rotate(R)

        # rotation around the z axis
        R = np.array(
            [[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]]
        )
        R = R.T
        self._rotate(R)

    def copy(self):
        """
        Copy the mesh.

        :return: copy of the mesh
        :rtype: Mesh
        """

        mesh = Mesh(self.vertices.copy(), self.faces.copy())

        return mesh


class Fusion:
    """
    Performs TSDF fusion.
    """

    def __init__(
        self,
        focal_length_x,
        focal_length_y,
        principal_point_x,
        principal_point_y,
        image_height,
        image_width,
        resolution,
        truncation_factor,
        n_views,
        depth_offset_factor,
        padding,
    ):
        """
        Constructor.
        """
        self.focal_length_x = focal_length_x
        self.focal_length_y = focal_length_y
        self.principal_point_x = principal_point_x
        self.principal_point_y = principal_point_y
        self.image_height = image_height
        self.image_width = image_width
        self.resolution = resolution
        self.truncation_factor = truncation_factor
        self.n_views = n_views
        self.depth_offset_factor = depth_offset_factor
        self.padding = padding

        self.render_intrinsics = np.array(
            [
                self.focal_length_x,
                self.focal_length_y,
                self.principal_point_x,
                self.principal_point_x,
            ],
            dtype=float,
        )
        # Essentially the same as above, just a slightly different format.
        self.fusion_intrisics = np.array(
            [
                [self.focal_length_x, 0, self.principal_point_x],
                [0, self.focal_length_y, self.principal_point_y],
                [0, 0, 1],
            ]
        )
        self.image_size = np.array(
            [
                self.image_height,
                self.image_width,
            ],
            dtype=np.int32,
        )
        # Mesh will be centered at (0, 0, 1)!
        self.znf = np.array([1 - 0.75, 1 + 0.75], dtype=float)
        # Derive voxel size from resolution.
        self.voxel_size = 1.0 / self.resolution
        self.truncation = self.truncation_factor * self.voxel_size

    def get_points(self):
        """
        See https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere.

        :param n_points: number of points
        :type n_points: int
        :return: list of points
        :rtype: numpy.ndarray
        """

        rnd = 1.0
        points = []
        offset = 2.0 / self.n_views
        increment = math.pi * (3.0 - math.sqrt(5.0))

        for i in range(self.n_views):
            y = ((i * offset) - 1) + (offset / 2)
            r = math.sqrt(1 - pow(y, 2))

            phi = ((i + rnd) % self.n_views) * increment

            x = math.cos(phi) * r
            z = math.sin(phi) * r

            points.append([x, y, z])

        # visualization.plot_point_cloud(np.array(points))
        return np.array(points)

    def get_views(self):
        """
        Generate a set of views to generate depth maps from.

        :param n_views: number of views per axis
        :type n_views: int
        :return: rotation matrices
        :rtype: [numpy.ndarray]
        """

        Rs = []
        points = self.get_points()

        for i in range(points.shape[0]):
            # https://math.stackexchange.com/questions/1465611/given-a-point-on-a-sphere-how-do-i-find-the-angles-needed-to-point-at-its-ce
            longitude = -math.atan2(points[i, 0], points[i, 1])
            latitude = math.atan2(
                points[i, 2], math.sqrt(points[i, 0] ** 2 + points[i, 1] ** 2)
            )

            R_x = np.array(
                [
                    [1, 0, 0],
                    [0, math.cos(latitude), -math.sin(latitude)],
                    [0, math.sin(latitude), math.cos(latitude)],
                ]
            )
            R_y = np.array(
                [
                    [math.cos(longitude), 0, math.sin(longitude)],
                    [0, 1, 0],
                    [-math.sin(longitude), 0, math.cos(longitude)],
                ]
            )

            R = R_y.dot(R_x)
            Rs.append(R)

        return Rs

    def render(self, mesh, Rs):
        """
        Render the given mesh using the generated views.

        :param base_mesh: mesh to render
        :type base_mesh: mesh.Mesh
        :param Rs: rotation matrices
        :type Rs: [numpy.ndarray]
        :return: depth maps
        :rtype: numpy.ndarray
        """

        depthmaps = []
        for i in range(len(Rs)):
            np_vertices = Rs[i].dot(mesh.vertices.astype(np.float64).T)
            np_vertices[2, :] += 1

            np_faces = mesh.faces.astype(np.float64)
            np_faces += 1

            depthmap, mask, img = librender.render(
                np_vertices.copy(),
                np_faces.T.copy(),
                self.render_intrinsics,
                self.znf,
                self.image_size,
            )

            # This is mainly result of experimenting.
            # The core idea is that the volume of the object is enlarged slightly
            # (by subtracting a constant from the depth map).
            # Dilation additionally enlarges thin structures (e.g. for chairs).
            depthmap -= self.depth_offset_factor * self.voxel_size
            depthmap = ndimage.morphology.grey_erosion(depthmap, size=(3, 3))

            depthmaps.append(depthmap)

        return depthmaps

    def fusion(self, depthmaps, Rs):
        """
        Fuse the rendered depth maps.

        :param depthmaps: depth maps
        :type depthmaps: numpy.ndarray
        :param Rs: rotation matrices corresponding to views
        :type Rs: [numpy.ndarray]
        :return: (T)SDF
        :rtype: numpy.ndarray
        """

        Ks = self.fusion_intrisics.reshape((1, 3, 3))
        Ks = np.repeat(Ks, len(depthmaps), axis=0).astype(np.float32)

        Ts = []
        for i in range(len(Rs)):
            Rs[i] = Rs[i]
            Ts.append(np.array([0, 0, 1]))

        Ts = np.array(Ts).astype(np.float32)
        Rs = np.array(Rs).astype(np.float32)

        depthmaps = np.array(depthmaps).astype(np.float32)
        views = libfusion.PyViews(depthmaps, Ks, Rs, Ts)

        # Note that this is an alias defined as libfusiongpu.tsdf_gpu or libfusioncpu.tsdf_cpu!
        return compute_tsdf(
            views,
            self.resolution,
            self.resolution,
            self.resolution,
            self.voxel_size,
            self.truncation,
            False,
        )

    def fuse(
        self, f_in, f_out, normalize=True, overwrite=False, texture=True, smooth=True
    ):
        """
        Run fusion.
        """

        # Load mesh
        input_mesh = trimesh.load(f_in)

        # Turn into one single mesh
        if isinstance(input_mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [
                    trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                    for m in input_mesh.geometry.values()
                ]
            )
        else:
            mesh = input_mesh.copy()
        vertices, faces = [tuple(v) for v in mesh.vertices.copy()], [
            tuple(f) for f in mesh.faces.copy()
        ]
        mesh = Mesh(vertices, faces)

        # Perform scaling
        min, max = mesh.extents()
        total_min = np.min(np.array(min))
        total_max = np.max(np.array(max))

        # Set the center (although this should usually be the origin already).
        centers = ((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2)
        # Scales all dimensions equally.
        sizes = (total_max - total_min, total_max - total_min, total_max - total_min)
        translation = (-centers[0], -centers[1], -centers[2])
        scales = (
            1 / (sizes[0] + 2 * self.padding * sizes[0]),
            1 / (sizes[1] + 2 * self.padding * sizes[1]),
            1 / (sizes[2] + 2 * self.padding * sizes[2]),
        )

        mesh.translate(translation)
        mesh.scale(scales)
        input_mesh.apply_transform(
            trimesh.transformations.translation_matrix(translation)
        )
        input_mesh.apply_transform(np.eye(4) * np.array(list(scales) + [1]))

        # Get the depth images
        logging.debug("Fusion is getting views")
        Rs = self.get_views()
        logging.debug("Fusion is rendering")
        depths = self.render(mesh, Rs)

        # Fuse!
        logging.debug("Fusion is fusing")
        tsdf = self.fusion(depths, Rs)
        tsdf = tsdf[0]

        # Create smoothed marching cubes version of mesh
        if smooth:
            logging.debug("Smoothing result")
            tsdf = mcubes.smooth(tsdf)
        vertices, faces = mcubes.marching_cubes(tsdf, 0.5)

        # mcubes reverses the x and z dimensions
        vertices = vertices[:, [2, 1, 0]]
        vertices /= self.resolution
        vertices -= 0.5

        watertight_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Transfer color
        if texture:
            # Multiple repeated vertices cause recursion errors. This prevents that.
            input_mesh = subdivide_scene_to_size_textured(input_mesh, max_edge=0.01)
            input_mesh.vertices += np.random.random(input_mesh.vertices.shape) * 0.0001

            # Prioritize vertices with similar normals
            logging.debug("Attempting vertex color remap")
            v_idx = input_mesh.kdtree.query(watertight_mesh.vertices.copy(), k=2)[1]
            for idx, (vn, ind) in enumerate(zip(watertight_mesh.vertex_normals, v_idx)):
                if np.dot(vn, input_mesh.vertex_normals[ind[0]]) > np.dot(
                    vn, input_mesh.vertex_normals[ind[1]]
                ):
                    watertight_mesh.visual.vertex_colors[
                        idx, :
                    ] = input_mesh.visual.vertex_colors[ind[0], :]
                else:
                    watertight_mesh.visual.vertex_colors[
                        idx, :
                    ] = input_mesh.visual.vertex_colors[ind[1], :]

        # Undo the normalization
        if not normalize:
            logging.debug("Reversing normalization")
            watertight_mesh.apply_transform(
                np.eye(4) * np.array([1 / s for s in scales] + [1])
            )
            watertight_mesh.apply_transform(
                trimesh.transformations.translation_matrix([-t for t in translation])
            )

        # Write out
        if overwrite or not os.path.exists(f_out):
            logging.debug("Saving to: {}".format(f_out))
            watertight_mesh.export(f_out)


def waterproofer(
    f_in,
    f_out,
    focal_length_x=640,
    focal_length_y=640,
    principal_point_x=320,
    principal_point_y=320,
    image_height=640,
    image_width=640,
    resolution=256,
    truncation_factor=10,
    n_views=100,
    depth_offset_factor=1.5,
    padding=0.1,
    normalize=False,
    overwrite=False,
    texture=False,
    smooth=True,
):

    app = Fusion(
        focal_length_x,
        focal_length_y,
        principal_point_x,
        principal_point_y,
        image_height,
        image_width,
        resolution,
        truncation_factor,
        n_views,
        depth_offset_factor,
        padding,
    )

    app.fuse(
        f_in,
        f_out,
        normalize=normalize,
        overwrite=overwrite,
        texture=texture,
        smooth=smooth,
    )


def handsoff_waterproofer(
    f_in,
    f_out,
    focal_length_x=None,
    focal_length_y=None,
    principal_point_x=None,
    principal_point_y=None,
    image_height=None,
    image_width=None,
    resolution=None,
    truncation_factor=None,
    n_views=None,
    depth_offset_factor=None,
    padding=None,
    normalize=False,
    smooth=True,
    overwrite=False,
    verbose=False,
):

    cmd = ["python " + os.path.abspath(__file__) + " " + f_in + " " + f_out]

    # Badness, but prevents segfault
    for arg, flag in zip(
        [
            focal_length_x,
            focal_length_y,
            principal_point_x,
            principal_point_y,
            image_height,
            image_width,
            resolution,
            truncation_factor,
            n_views,
            depth_offset_factor,
            padding,
        ],
        [
            "--focal_length_x",
            "--focal_length_y",
            "--principal_point_x",
            "--principal_point_y",
            "--image_height",
            "--image_width",
            "--resolution",
            "--truncation_factor",
            "--n_views",
            "--depth_offset_factor",
            "--padding",
        ],
    ):
        if arg is not None:
            cmd[0] += flag + " " + str(arg) + " "

    if normalize:
        cmd[0] += " --normalize"

    if verbose:
        cmd[0] += " --debug"

    if smooth:
        cmd[0] += " --smooth"

    if not overwrite:
        cmd[0] += " --no_overwrite"

    logging.debug("Executing command in the shell: \n{}".format(cmd))

    # Call the pcl script
    if subprocess.call(cmd, shell=True) != 0:
        raise errors.MeshFusionError


def process(
    obj,
    num_results,
    overwrite,
    executor,
    args,
):
    """Waterproof a mesh using the watershed method"""
    f_in = obj.path_normalized()
    f_out = obj.path_waterproofed()
    if os.path.exists(f_in) and (not os.path.exists(f_out) or overwrite):
        executor.graceful_submit(
            waterproofer,
            f_in=f_in,
            f_out=f_out,
            smooth=True,
            overwrite=overwrite,
        )


def validate_outputs(
    obj,
    num_results,
    args,
):
    if os.path.exists(obj.path_waterproofed()):
        return [True]
    else:
        return [False]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Waterproof a mesh.")
    parser.add_argument(dest="input", type=str, help="Path to the input file.")
    parser.add_argument(dest="output", type=str, help="Path to the output file.")
    parser.add_argument(
        "--n_views",
        type=int,
        default=100,
        help="Number of " + "views to use during rendering.",
    )
    parser.add_argument(
        "--image_height", type=int, default=640, help="Depth " + "image height."
    )
    parser.add_argument(
        "--image_width", type=int, default=640, help="Depth " + "image width."
    )
    parser.add_argument(
        "--focal_length_x", type=float, default=640, help="Focal length in x direction."
    )
    parser.add_argument(
        "--focal_length_y", type=float, default=640, help="Focal length in y direction."
    )
    parser.add_argument(
        "--principal_point_x",
        type=float,
        default=320,
        help="Principal point location in x direction.",
    )
    parser.add_argument(
        "--principal_point_y",
        type=float,
        default=320,
        help="Principal point location in y direction.",
    )
    parser.add_argument(
        "--depth_offset_factor",
        type=float,
        default=1.5,
        help="The depth maps are offsetted using depth_offset_factor*voxel_size.",
    )
    parser.add_argument(
        "--resolution", type=float, default=256, help="Resolution for fusion."
    )
    parser.add_argument(
        "--truncation_factor",
        type=float,
        default=10,
        help="Truncation for fusion is derived as truncation_factor*voxel_size.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.1,
        help="Relative padding applied on each side.",
    )
    parser.add_argument(
        "--handsoff",
        default=False,
        action="store_true",
        help="Handsoff will execute the script in a separate shell instance. "
        + "This isn't really recommended but when calling this script through "
        + "another python function this will circumvent segfaults.",
    )
    parser.add_argument(
        "--normalize",
        default=False,
        action="store_true",
        help="Will normalize the scale and translation of the input object.",
    )
    parser.add_argument(
        "--smooth",
        default=False,
        action="store_true",
        help="Apply smoothing before marching cubes.",
    )
    parser.add_argument(
        "--no_overwrite",
        default=False,
        action="store_true",
        help="If passed will not overwrite existing files.",
    )
    logger.add_logger_args(parser)
    args = parser.parse_args()
    logger.configure_logging(args)

    if args.handsoff:
        handsoff(
            args.input,
            args.output,
            args.focal_length_x,
            args.focal_length_y,
            args.principal_point_x,
            args.principal_point_y,
            args.image_height,
            args.image_width,
            args.resolution,
            args.truncation_factor,
            args.n_views,
            args.depth_offset_factor,
            args.padding,
            args.normalize,
        )
    else:
        app = Fusion(
            args.focal_length_x,
            args.focal_length_y,
            args.principal_point_x,
            args.principal_point_y,
            args.image_height,
            args.image_width,
            args.resolution,
            args.truncation_factor,
            args.n_views,
            args.depth_offset_factor,
            args.padding,
        )
        app.fuse(
            args.input,
            args.output,
            normalize=args.normalize,
            smooth=args.smooth,
            overwrite=args.no_overwrite,
        )
