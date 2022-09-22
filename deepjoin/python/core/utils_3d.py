import os
import logging

import vedo
import trimesh
import imageio
from matplotlib import tri
import numpy as np
from PIL import Image

try:
    import pyrender
except ImportError:
    pass
try:
    import pymesh
except ImportError:
    pass
try:
    import pymeshfix
except ImportError:
    pass

import core


os.environ["PYOPENGL_PLATFORM"] = "egl"


def trimesh2vedo(mesh, **kwargs):
    return vedo.Mesh([mesh.vertices, mesh.faces], **kwargs)


def vedo2trimesh(mesh):
    return trimesh.Trimesh(
        vertices=mesh.points(),
        faces=mesh.faces(),
    )


def pymesh2trimesh(m):
    return trimesh.Trimesh(m.vertices, m.faces)


def trimesh2pymesh(m):
    return pymesh.form_mesh(m.vertices, m.faces)


def view_normals(m, n):
    vec = np.column_stack((m.vertices, m.vertices + (n * 10.0)))
    v = np.random.choice(np.arange(vec.shape[0]), 500)
    vec = vec[v]
    m.vertices = m.vertices[v]
    path = trimesh.load_path(vec.reshape((-1, 2, 3)))
    trimesh.Scene([m, path]).show(smooth=False)


def repair_self_intersection(mt):
    if mt.is_watertight:
        return mt

    m = trimesh2pymesh(mt)
    m, _ = pymesh.remove_degenerated_triangles(m)
    mt = pymesh2trimesh(m)
    if mt.is_watertight:
        return mt

    m, _ = pymesh.remove_duplicated_vertices(m)
    mt = pymesh2trimesh(m)
    if mt.is_watertight:
        return mt

    m, _ = pymesh.remove_duplicated_faces(m)
    mt = pymesh2trimesh(m)
    if mt.is_watertight:
        return mt

    m = pymesh.resolve_self_intersection(m)
    return pymesh2trimesh(m)


def repair_watertight(mesh):
    """Attempt to repair a mesh using the default pymeshfix procedure"""
    mesh = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
    mesh.repair(joincomp=True, remove_smallest_components=False)
    return trimesh.Trimesh(mesh.v, mesh.f)


def trimesh_boolean(mesh1, mesh2, operation="difference", **kwargs):
    return pymesh2trimesh(
        pymesh.boolean(
            trimesh2pymesh(mesh1), trimesh2pymesh(mesh2), operation, **kwargs
        )
    )


def points_transform(points, mat):
    """Apply a transformation matrix to a set of points"""
    points = np.dot(mat, np.hstack((points, np.ones((points.shape[0], 1)))).T).T[:, :3]
    return points


def trimesh_transform(mesh, mat):
    """Apply a transformation matrix to a trimesh mesh"""
    mesh = mesh.copy()
    mesh.vertices = np.dot(
        mat, np.hstack((mesh.vertices, np.ones((mesh.vertices.shape[0], 1)))).T
    ).T[:, :3]
    return mesh


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


def normalize_unit_cube(mesh, scale=True):
    """Normalize a mesh so that it occupies a unit cube"""

    # Get the overall size of the object
    mesh = mesh.copy()
    mesh_min, mesh_max = np.min(mesh.vertices, axis=0), np.max(mesh.vertices, axis=0)
    size = mesh_max - mesh_min

    # Center the object
    mesh.vertices = mesh.vertices - ((size / 2.0) + mesh_min)

    # Normalize scale of the object
    if scale:
        mesh.vertices = mesh.vertices * (1.0 / np.max(size))
    try:
        mesh.fix_normals()
    except AttributeError:
        pass
    return mesh


def mesh_split_componenets(
    mesh,
):
    return [
        core.vedo2trimesh(m)
        for m in core.utils_3d.trimesh2vedo(mesh).splitByConnectivity()
    ]


def mesh_discard_components(
    pred_mesh,
    thresh=0.001,
):
    """
    Remove all connected components smaller than a threshold.
    """

    # Split a mesh into connected components
    meshes = core.utils_3d.trimesh2vedo(pred_mesh).splitByConnectivity()

    # Compute the volumes
    volumes = np.array([m.volume() for m in meshes])

    # If no threshold was passed, return the largest
    if thresh is None:
        return core.utils_3d.vedo2trimesh(meshes[np.argmax(volumes)])

    # Get meshes larger than the threshold
    large_meshes = [m for m in meshes if m.volume() > thresh]

    # If they're all smaller, return the largest
    if len(large_meshes) == 0:
        return core.utils_3d.vedo2trimesh(meshes[np.argmax(volumes)])

    # Merge the large meshes into a single trimesh
    return trimesh.util.concatenate(
        [core.utils_3d.vedo2trimesh(m) for m in large_meshes]
    )


def colorize_mesh(mesh, color):
    """
    Applies a color to a mesh
    """
    mesh = mesh.copy()
    mesh.visual = trimesh.visual.color.ColorVisuals(mesh, vertex_colors=color)
    return mesh


def colorize_mesh_from_index(mesh, shape_idx, color_strength=0, transparent=False):
    """
    Applies a predefined color to a mesh based on its shape index, in place
    """

    # Define the colors
    color_red = [c - color_strength for c in [228, 129, 129]] + [255]
    color_grey = [c - color_strength for c in [210, 210, 210]] + [255]

    # Apply transparency
    if transparent:
        color_grey[-1] = 128

    # Apply colors
    if shape_idx >= 2:
        mesh.visual = trimesh.visual.color.ColorVisuals(mesh, vertex_colors=color_red)
    else:
        mesh.visual = trimesh.visual.color.ColorVisuals(mesh, vertex_colors=color_grey)
    return mesh


def colorize_mesh_from_index_auto(
    mesh, shape_idx, color_strength=60, transparent=False
):
    """
    Applies a predefined color to a mesh based on its shape index, in place
    """

    # Define the colors
    color_red = [c - color_strength for c in [228, 129, 129]] + [255]
    color_grey = [c - color_strength for c in [150, 150, 150]] + [255]

    # Apply transparency
    if transparent:
        color_grey[-1] = 128

    # Apply colors
    if shape_idx >= 2:
        mesh.visual = trimesh.visual.color.ColorVisuals(mesh, vertex_colors=color_red)
    else:
        mesh.visual = trimesh.visual.color.ColorVisuals(mesh, vertex_colors=color_grey)
    return mesh


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


def get_intersection_points(a, b, sig=5):
    """get mask of vertices in a occurring in both a and b, corresponding to a"""
    av = [frozenset(np.round(v, sig)) for v in a]
    bv = set([frozenset(np.round(v, sig)) for v in b])
    return np.asarray(list(map(lambda v: v in bv, av)))


def get_fracture_points(b, r):
    """Get points on the fracture"""
    try:
        vb, vr = b.vertices, r.vertices
    except AttributeError:
        vb, vr = b, r

    logging.debug(
        "Computing fracture points for meshes with size {} and {} ...".format(
            vb.shape[0], vr.shape[0]
        )
    )
    return get_intersection_points(vb, vr)


def create_sphere_pointcloud(pointcloud, radius=None):
    if radius is None:
        radius = max(pointcloud.extents) / 35
    return trimesh.util.concatenate(
        [
            trimesh.primitives.Sphere(
                radius=4,
                center=v,
                subdivisions=2,
            )
            for v in pointcloud.vertices
        ]
    )


def create_gif_rot(
    models,
    zoom=2.0,
    num_renders=24,
    start_angle=0,
    **kwargs,
):
    """Return a list of images that can be used to save a gif"""
    img_list = []
    for angle in range(0, 360, int(360 / num_renders)):
        img_list.append(
            np.expand_dims(
                core.utils_3d.render_mesh(
                    models, yrot=start_angle + angle, ztrans=zoom, **kwargs
                ),
                axis=3,
            )
        )
    return np.concatenate(img_list, axis=3)


def create_gif_tf(
    models,
    transforms,
    zoom=2.0,
    **kwargs,
):
    """Return a list of images that can be used to save a gif"""

    if not isinstance(models, list):
        models = [models]

    img_list = []
    for tf in transforms:
        img_list.append(
            np.expand_dims(
                core.utils_3d.render_mesh(
                    [trimesh_transform(m, t) for m, t in zip(models, tf)],
                    ztrans=zoom,
                    **kwargs,
                ),
                axis=3,
            )
        )
    return np.concatenate(img_list, axis=3)


def save_gif(
    f_out,
    data,
    **kwargs,
):
    """Save a list of images as a gif"""
    data = [Image.fromarray(data[:, :, :, i]) for i in range(data.shape[3])]
    data[0].save(f_out, save_all=True, append_images=data[1:], **kwargs)


def load_gif(
    f_in,
    return_first=None,
):
    gif = imageio.get_reader(f_in)
    frames = []
    for idx, f in enumerate(gif):
        if (return_first is not None) and idx >= return_first:
            break
        frames.append(np.array(f))
    if len(frames[0].shape) == 2:
        frames = [np.expand_dims(f, axis=2) for f in frames]
    frames = [np.expand_dims(f, axis=3) for f in frames]
    return np.concatenate(frames, axis=3)


def force_trimesh(mesh, remove_texture=False):
    """
    Forces a mesh or list of meshes to be a single trimesh object.
    """

    if isinstance(mesh, list):
        return [force_trimesh(m) for m in mesh]

    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            mesh = trimesh.Trimesh()
        else:
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in mesh.geometry.values()
                )
            )
    if remove_texture:
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    return mesh


def render_mesh(
    mesh,
    objects=None,
    mode="RGB",
    remove_texture=False,
    yfov=(np.pi / 4.0),
    resolution=(1280, 720),
    xtrans=0.0,
    ytrans=0.0,
    ztrans=2.0,
    xrot=-25.0,
    yrot=45.0,
    zrot=0.0,
    spotlight_intensity=8.0,
    bg_color=255,
):
    assert len(resolution) == 2

    def _force_trimesh(mesh, remove_texture=False):
        """
        Forces a mesh or list of meshes to be a single trimesh object.
        """

        if isinstance(mesh, list):
            return [_force_trimesh(m) for m in mesh]

        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                mesh = trimesh.Trimesh()
            else:
                mesh = trimesh.util.concatenate(
                    tuple(
                        trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                        for g in mesh.geometry.values()
                    )
                )
        if remove_texture:
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        return mesh

    mesh = _force_trimesh(mesh, remove_texture)

    # Create a pyrender scene with ambient light
    scene = pyrender.Scene(ambient_light=np.ones(3), bg_color=bg_color)

    if objects is not None:
        for o in objects:
            o = o.subdivide_to_size(max_edge=0.05)
            n = o.vertices.shape[0]
            o.visual = trimesh.visual.create_visual(
                vertex_colors=np.hstack(
                    (
                        np.ones((n, 1)) * 0,
                        np.ones((n, 1)) * 0,
                        np.ones((n, 1)) * 255,
                        np.ones((n, 1)) * 50,
                    )
                )
            )
            scene.add(pyrender.Mesh.from_trimesh(o, wireframe=True))

    if not isinstance(mesh, list):
        mesh = [mesh]
    for m in mesh:
        if isinstance(m, trimesh.points.PointCloud):
            scene.add(pyrender.Mesh.from_points(m.vertices, colors=m.colors))
        else:
            scene.add(pyrender.Mesh.from_trimesh(m))

    camera = pyrender.PerspectiveCamera(
        yfov=yfov, aspectRatio=resolution[0] / resolution[1]
    )

    # Apply translations
    trans = np.array(
        [
            [1.0, 0.0, 0.0, xtrans],
            [0.0, 1.0, 0.0, ytrans],
            [0.0, 0.0, 1.0, ztrans],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # Apply rotations
    xrotmat = trimesh.transformations.rotation_matrix(
        angle=np.radians(xrot), direction=[1, 0, 0], point=(0, 0, 0)
    )
    camera_pose = np.dot(xrotmat, trans)
    yrotmat = trimesh.transformations.rotation_matrix(
        angle=np.radians(yrot), direction=[0, 1, 0], point=(0, 0, 0)
    )
    camera_pose = np.dot(yrotmat, camera_pose)
    zrotmat = trimesh.transformations.rotation_matrix(
        angle=np.radians(zrot), direction=[0, 0, 1], point=(0, 0, 0)
    )
    camera_pose = np.dot(zrotmat, camera_pose)

    # Insert the camera
    scene.add(camera, pose=camera_pose)

    # Insert a splotlight to give contrast
    spot_light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=spotlight_intensity,
        innerConeAngle=np.pi / 16.0,
        outerConeAngle=np.pi / 6.0,
    )
    scene.add(spot_light, pose=camera_pose)

    # Render!
    r = pyrender.OffscreenRenderer(*resolution, point_size=5)
    color, _ = r.render(scene)
    return np.array(Image.fromarray(color).convert(mode))
