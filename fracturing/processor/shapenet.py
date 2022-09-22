import os
import json

import processor.utils as utils
import processor.database as database


SHAPENET_CLASSES = None


class ShapeNetObject(database.DatabaseObject):
    def build_dirs(self):
        """Build any required subdirectories"""
        # List of all the directories to build
        dir_list = [
            os.path.join(self._root_dir, self._class_id, self._instance_id),
            os.path.join(self._root_dir, self._class_id, self._instance_id, "models"),
            os.path.join(self._root_dir, self._class_id, self._instance_id, "renders"),
        ]

        # Build the directories
        for d in dir_list:
            if not os.path.exists(d):
                os.mkdir(d)

    def path(self, subdir, basename):
        """Return the path to a subdir + basename"""
        return os.path.join(
            self._root_dir,
            self._class_id,
            self._instance_id,
            subdir,
            basename,
        )

    # == model path shortcuts ==
    def path_normalized(self):
        return self.path("models", "model_normalized.obj")

    def path_waterproofed(self):
        return self.path("models", "model_waterproofed.obj")

    def path_sampled(self, idx=0):
        return self.path("models", "model_{}_sampled.npz".format(idx))

    # == reorient path shortcuts ==
    def path_random_reorient(self, idx=0):
        return self.path("models", "model_reoriented_{}.obj".format(idx))

    def path_samples_reorient(self, idx=0):
        return self.path("models", "samples_reoriented_{}.npz".format(idx))

    def path_sdf_reorient(self, idx=0):
        return self.path("models", "sdf_reoriented_{}.npz".format(idx))

    # == complete model path shortcuts ==
    def path_c(self):
        return self.path("models", "model_c.obj")

    def path_c_occ(self, idx=0):
        return self.path("models", "model_c_{}_occ.npz".format(idx))

    def path_c_uniform_occ(self, size=256):
        if size == 256:
            return self.path("models", "model_c_uniform_occ.npz")
        return self.path("models", "model_c_uniform_occ_{}.npz".format(size))

    def path_c_sdf(self, idx=0):
        return self.path("models", "model_c_{}_sdf.npz".format(idx))

    def path_c_uniform_sdf(self, size=256):
        if size == 256:
            return self.path("models", "model_c_uniform_sdf.npz")
        return self.path("models", "model_c_uniform_sdf_{}.npz".format(size))

    def path_c_voxel(self, size=32):
        return self.path(
            "models", "model_c_{}_uniform_padded_occ_{}.npz".format(0, size)
        )

    # == tool model path shortcuts
    def path_tool(self, idx=0):
        return self.path("models", "model_tool_{}.obj".format(idx))

    def path_tool_sdf(self, idx=0):
        return self.path("models", "model_tool_{}_sdf.npz".format(idx))

    # == spline shortcuts ==
    def path_spline_sdf(self, idx=0):
        return self.path("models", "model_spline_{}_sdf.npz".format(idx))

    def path_full_spline_sdf(self, idx=0):
        return self.path("models", "model_full_spline_{}_sdf.npz".format(idx))

    def path_spline_plane(self, idx=0):
        return self.path("models", "model_spline_{}_plane.ply".format(idx))

    def path_spline_mesh(self, idx=0):
        return self.path("models", "model_spline_{}_mesh.ply".format(idx))

    def path_spline_interpolator(self, idx=0):
        return self.path("models", "model_spline_{}_interpolator.ply".format(idx))

    # == primitive model path shortcuts
    def path_primitive(self, idx=0):
        return self.path("models", "model_primitive_{}.ply".format(idx))

    def path_primitive_sdf(self, idx=0):
        return self.path("models", "model_primitive_{}_sdf.npz".format(idx))

    def path_primitive_zone_occ(self, idx=0):
        return self.path("models", "model_primitive_zone_{}_occ.npz".format(idx))

    # == broken model path shortcuts ==
    def path_b(self, idx=0):
        return self.path("models", "model_b_{}.obj".format(idx))

    def path_b_occ(self, idx=0):
        return self.path("models", "model_b_{}_occ.npz".format(idx))

    def path_b_uniform_occ(self, idx=0, size=256):
        if size == 256:
            return self.path("models", "model_b_{}_uniform_occ.npz".format(idx))
        return self.path("models", "model_b_{}_uniform_occ_{}.npz".format(idx, size))

    def path_b_sdf(self, idx=0):
        return self.path("models", "model_b_{}_sdf.npz".format(idx))

    def path_b_uniform_sdf(self, idx=0, size=256):
        if size == 256:
            return self.path("models", "model_b_{}_uniform_sdf.npz".format(idx))
        return self.path("models", "model_b_{}_uniform_sdf_{}.npz".format(idx, size))

    def path_b_partial_sdf(self, idx=0):
        return self.path("models", "model_b_{}_partial_sdf.npz".format(idx))

    def path_b_pointnet_mask(self, idx=0):
        return self.path("models", "model_b_{}_pointnet_mask.npz".format(idx))

    def path_b_uniform_sdf_partial(self, idx=0, size=256):
        if size == 256:
            return self.path("models", "model_b_{}_uniform_sdf_partial.npz".format(idx))
        return self.path(
            "models", "model_b_{}_uniform_sdf_{}_partial.npz".format(idx, size)
        )

    def path_b_uniform_sdf_pointnet_partial(self, idx=0, size=256):
        if size == 256:
            return self.path(
                "models", "model_b_{}_uniform_sdf_pointnet_partial.npz".format(idx)
            )
        return self.path(
            "models", "model_b_{}_uniform_sdf_{}_pointnet_partial.npz".format(idx, size)
        )

    def path_b_voxel(self, idx=0, size=32):
        return self.path(
            "models", "model_b_{}_uniform_padded_occ_{}.npz".format(idx, size)
        )

    # == restoration model path shortcuts ==
    def path_r(self, idx=0):
        return self.path("models", "model_r_{}.obj".format(idx))

    def path_r_occ(self, idx=0):
        return self.path("models", "model_r_{}_occ.npz".format(idx))

    def path_r_uniform_occ(self, idx=0):
        return self.path("models", "model_r_{}_uniform_occ.npz".format(idx))

    def path_r_sdf(self, idx=0):
        return self.path("models", "model_r_{}_sdf.npz".format(idx))

    def path_r_uniform_sdf(self, idx=0):
        return self.path("models", "model_r_{}_uniform_sdf.npz".format(idx))

    def path_r_voxel(self, idx=0, size=32):
        return self.path(
            "models", "model_r_{}_uniform_padded_occ_{}.npz".format(idx, size)
        )

    # == render path shortcuts ==
    def path_c_rendered(self, angle=0, resolution=None):
        if resolution is None:
            return self.path("renders", "model_c_rendered{}.png".format(angle))
        else:
            return self.path(
                "renders", "model_c_rendered_{}_{}_{}.png".format(*resolution, angle)
            )

    def path_b_rendered(self, idx=0, angle=0, resolution=None):
        if resolution is None:
            return self.path("renders", "model_b_{}_rendered{}.png".format(idx, angle))
        else:
            return self.path(
                "renders",
                "model_b_{}_rendered_{}_{}_{}.png".format(idx, *resolution, angle),
            )

    def path_r_rendered(self, idx=0, angle=0, resolution=None):
        if resolution is None:
            return self.path("renders", "model_r_{}_rendered{}.png".format(idx, angle))
        else:
            return self.path(
                "renders",
                "model_r_{}_rendered_{}_{}_{}.png".format(idx, *resolution, angle),
            )

    def path_composite(self, idx=0, angle=0, resolution=(200, 200)):
        if resolution == (200, 200):
            return self.path("renders", "model_{}_composite{}.png".format(idx, angle))
        else:
            return self.path(
                "renders",
                "model_{}_composite{}_{}_{}.png".format(
                    idx, resolution[0], resolution[1], angle
                ),
            )


def class2string(class_id):
    """
    Given the shapenet class id, will return the human readable class label.
    If the class id is not found, will return "UNKOWNN"
    """
    global SHAPENET_CLASSES
    if SHAPENET_CLASSES is None:
        SHAPENET_CLASSES = json.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "shapenet_classes.json"
                )
            )
        )
    return SHAPENET_CLASSES[class_id]


def shapenet_search_all(root, return_toplevel=False):
    """
    Return a list of all objects in the dataset.
    Searches in a smart way, and will always return all objects in shapenet,
    regardless of which directory it is pointed to.
    """
    objects = []
    try:
        lowest_level = next(utils.get_file(root, [".obj"], "model_normalized"))
    except StopIteration:
        return
    classes_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(lowest_level)))
    )
    for f in os.listdir(classes_dir):
        f = os.path.join(classes_dir, f)
        if os.path.isdir(f):
            class_id = f.split("/")[-1]
            for fprime in os.listdir(f):
                fprime = os.path.join(f, fprime)
                if os.path.isdir(fprime):
                    object_id = fprime.split("/")[-1]
                    objects.append(ShapeNetObject(classes_dir, class_id, object_id))
    if return_toplevel:
        return objects, classes_dir
    return objects


def shapenet_search(root, return_toplevel=False):
    """
    Return a list of objects in the dataset.
    Searches in a simplified way, will just find all files called
    model_normalized in any subdirectories of root. Use this if you'd like a
    list containing objects from just one class.
    """
    objects = []
    if os.path.exists(root):
        for f in utils.get_file(root, [".obj"], "model_normalized"):
            classes_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(f)))
            )
            class_id = os.path.basename(
                os.path.dirname(os.path.dirname(os.path.dirname(f)))
            )
            instance_id = os.path.basename(os.path.dirname(os.path.dirname(f)))
            objects.append(ShapeNetObject(classes_dir, class_id, instance_id))
    if return_toplevel:
        return objects, classes_dir
    return objects


def shapenet_toplevel(root, steps=4):
    """
    Will return the toplevel directory, regardless of directory passed. Be careful.
    """
    try:
        lowest_level = next(utils.get_file(root, [".obj"], "model_normalized"))
    except StopIteration:
        return

    parent_path = lowest_level
    for _ in range(steps):
        parent_path = os.path.dirname(parent_path)
    return parent_path
