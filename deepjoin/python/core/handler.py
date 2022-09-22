import logging
import os
import pickle
import json
import time
import traceback
import multiprocessing

import tqdm
import torch
import trimesh
import numpy as np
from PIL import Image

try:
    import pickle5
except ImportError:
    pass

import core
import core.workspace as ws
import core.errors as errors


# This prevents torch file errors when handling thousands of files.
torch.multiprocessing.set_sharing_strategy("file_system")


def path_exists(p, p_list=None):
    if (p_list is not None) and (p in p_list):
        return True
    return os.path.exists(p)


def loader(f_in):
    """Multipurpose loader used for all file types"""
    extension = os.path.splitext(f_in)[-1]
    logging.debug("Attempting to load file {}".format(f_in))

    try:
        if extension == ".pkl":
            with open(f_in, "rb") as f:
                try:
                    return pickle.load(f)
                except ValueError:
                    return pickle5.load(f)
        elif extension == ".json":
            return json.load(open(f_in, "r"))
        elif extension == ".obj":
            return trimesh.load(f_in, force=True)
        elif extension == ".ply":
            return trimesh.load(f_in, force=True)
        elif extension == ".npz":
            return dict(np.load(f_in, allow_pickle=True))
        elif extension == ".npy":
            return np.load(f_in, allow_pickle=True)
        elif extension == ".png":
            return np.array(Image.open(f_in))
        elif extension == ".jpg":
            return np.array(Image.open(f_in))
        elif extension == ".gif":
            return core.load_gif(f_in)
        elif extension == ".pth":
            try:
                return torch.load(f_in, map_location=torch.device("cpu"))
            except EOFError as e:
                # raise e
                os.remove(f_in)
                print("Removed: ", f_in)
        else:
            raise RuntimeError("Loader: Unhandled file type: {}".format(f_in))
    except PermissionError:
        input(
            "Permission error with file, press enter to continue once fixed: {}".format(
                f_in
            )
        )
        return loader(f_in)


def saver(f_out, data, all_access=True, **kwargs):
    """Multipurpose saver used for all file types"""
    logging.debug("Saving file {}".format(f_out))
    extension = os.path.splitext(f_out)[-1]
    if all_access:
        os.open(
            f_out, os.O_CREAT | os.O_WRONLY, 0o777
        )  # Create the file WITH THE MOST GENERAL PERMISSIONS

    try:
        if extension == ".pkl":
            pickle.dump(
                data,
                open(f_out, "wb"),
                pickle.HIGHEST_PROTOCOL,
            )
        elif extension == ".obj":
            data.export(f_out)
        elif extension == ".json":
            return json.dump(data, open(f_out, "w"))
        elif extension == ".ply":
            data.export(f_out)
        elif extension == ".npz":
            np.savez(f_out, data)
        elif extension == ".npy":
            np.save(f_out, data)
        elif extension == ".png":
            Image.fromarray(data).save(f_out)
        elif extension == ".jpg":
            Image.fromarray(data).save(f_out)
        elif extension == ".gif":
            core.utils_3d.save_gif(f_out, data, **kwargs)
        elif extension == ".pth":
            torch.save(data, f_out)
        else:
            raise RuntimeError("Saver: Unhandled file type: {}".format(f_out))
    except PermissionError:
        input(
            "Permission error with file, press enter to continue once fixed: {}".format(
                f_out
            )
        )
        return saver(f_out, data, all_access)


def execute_and_save(
    path,
    fn,
    overwrite,
    **kwargs,
):
    """executes a function and then saves the result"""
    # logging.debug("running {} with {}".format(path, fn))
    val = None
    try:
        if overwrite or not os.path.exists(path) or core.utils.file_is_old(path):
            val = fn(**kwargs)
            saver(path, val)
    except (errors.MeshEmptyError, errors.MeshContainsError):
        logging.error("Mesh empty error {} with save path: {}".format(str(fn), path))
    except (PermissionError):
        input(
            "Permission error with file, press enter to continue once fixed: {}".format(
                path
            )
        )
        execute_and_save(path, fn, overwrite, **kwargs)
    except (ValueError, IndexError, TypeError, FileExistsError, AttributeError) as e:
        logging.debug(traceback.format_exc())
        # raise e
        logging.error(
            "Problem running: {}[{}] with save path: {}".format(str(fn), e, path)
        )
    return val


def render_engine(
    reconstruct_list,
    outputs,
    threads,
    reconstruction_handler,
    overwrite=False,
    num_renders=3,
    resolution=(200, 200),
    composite=False,  # Reconstruction shape, gt shape
    data_handler=None,
    render_gt=False,
    angle_list=None,
    zoom=2.0,
    dynamic_path_check=False,
    transparent_composite=False,
):
    """the handlers are great, but they can't be easily combined with multiprocessing"""

    if composite:
        assert data_handler is not None
        if not isinstance(composite, list):
            composite = [composite]

    # Compile a list of all the shape indices
    all_shapes = list(outputs)
    if composite:
        for c in composite:
            all_shapes.extend(list(c))
    all_shapes = sorted(list(set(all_shapes)))

    # Build the angle list
    if angle_list is None:
        angle_list = range(0, 360, int(360 / num_renders))

    p_list = None
    if not dynamic_path_check:
        root_dir = reconstruction_handler.dir_renders(create=True)
        p_list = set([os.path.join(root_dir, f) for f in os.listdir(root_dir)])

    futures_list = []
    args_list = []
    start = time.time()
    logging.info("Spawning {} threads".format(threads))
    logging.info("Loading files ...")

    with multiprocessing.Pool(threads) as pool:
        for idx in tqdm.tqdm(reconstruct_list):
            for shape_idx in all_shapes:

                for angle in angle_list:

                    # Build a composite
                    if composite and (reconstruction_handler is not None):
                        for (gt_compi, pd_compi) in composite:
                            if shape_idx == pd_compi:
                                path = reconstruction_handler.path_composite(
                                    idx,
                                    shape_idx,
                                    gt_compi,
                                    angle,
                                    resolution,
                                    create=True,
                                )
                                if not path_exists(path, p_list) or overwrite:
                                    reconstruction_handler.delete_from_cache(path)

                                    # Load the predicted mesh
                                    try:
                                        mesh = reconstruction_handler.get_mesh(
                                            idx, shape_idx
                                        )
                                        mesh = mesh.copy()
                                        # if shape_idx == 3:
                                        #     mesh.invert()
                                    except errors.IsosurfaceExtractionError:
                                        continue

                                    # Load the gt mesh
                                    gt_b_mesh = data_handler.get_mesh(idx, 1)
                                    gt_b_mesh = gt_b_mesh.copy()

                                    # Update the colors
                                    core.utils_3d.colorize_mesh_from_index_auto(
                                        mesh, shape_idx
                                    )
                                    core.utils_3d.colorize_mesh_from_index_auto(
                                        gt_b_mesh, 1, transparent=transparent_composite
                                    )

                                    # Set the thread working
                                    futures_list.append(
                                        pool.apply_async(
                                            execute_and_save,
                                            tuple(
                                                (
                                                    path,
                                                    core.utils_3d.render_mesh,
                                                    overwrite,
                                                )
                                            ),  # args
                                            dict(
                                                mesh=[gt_b_mesh, mesh],
                                                yrot=angle,
                                                resolution=resolution,
                                                ztrans=zoom,
                                            ),  # kwargs
                                        )
                                    )

                                    # Save the arguments so that we can set the render later
                                    args_list.append(
                                        tuple((idx, shape_idx, angle, resolution))
                                    )

                                # Render gt composite with itself
                                if render_gt and (shape_idx == 2):
                                    # Get gt path
                                    obj_idx, break_idx = data_handler._indexer[idx]
                                    obj = data_handler.objects[obj_idx]
                                    path = obj.path_composite(
                                        idx=break_idx,
                                        angle=angle,
                                        resolution=resolution,
                                    )

                                    if not path_exists(path, p_list) or overwrite:
                                        gt_r_mesh = data_handler.get_mesh(
                                            idx, shape_idx
                                        )
                                        gt_r_mesh = gt_r_mesh.copy()

                                        gt_b_mesh = data_handler.get_mesh(idx, 1)
                                        gt_b_mesh = gt_b_mesh.copy()

                                        # Update the colors
                                        core.utils_3d.colorize_mesh_from_index_auto(
                                            gt_r_mesh, shape_idx
                                        )
                                        core.utils_3d.colorize_mesh_from_index_auto(
                                            gt_b_mesh,
                                            1,
                                            transparent=transparent_composite,
                                        )

                                        # Set the thread working
                                        futures_list.append(
                                            pool.apply_async(
                                                execute_and_save,
                                                tuple(
                                                    (
                                                        path,
                                                        core.utils_3d.render_mesh,
                                                        overwrite,
                                                    )
                                                ),  # args
                                                dict(
                                                    mesh=[gt_b_mesh, gt_r_mesh],
                                                    yrot=angle,
                                                    resolution=resolution,
                                                    ztrans=zoom,
                                                ),  # kwargs
                                            )
                                        )

                    # Render the mesh, plan and simple
                    if shape_idx in outputs:

                        # Render gt
                        if render_gt:
                            if shape_idx in [0, 1, 2] and data_handler is not None:
                                # Get gt path
                                obj_idx, break_idx = data_handler._indexer[idx]
                                obj = data_handler._objects[obj_idx]
                                if shape_idx == 0:
                                    path = obj.path_c_rendered(
                                        angle=angle, resolution=resolution
                                    )
                                elif shape_idx == 1:
                                    path = obj.path_b_rendered(
                                        idx=break_idx,
                                        angle=angle,
                                        resolution=resolution,
                                    )
                                elif shape_idx == 2:
                                    path = obj.path_r_rendered(
                                        idx=break_idx,
                                        angle=angle,
                                        resolution=resolution,
                                    )

                                if not path_exists(path, p_list) or overwrite:
                                    mesh = data_handler.get_mesh(idx, shape_idx)
                                    mesh = mesh.copy()

                                    # Update the colors
                                    core.utils_3d.colorize_mesh_from_index_auto(
                                        mesh, shape_idx
                                    )

                                    # Set the thread working
                                    futures_list.append(
                                        pool.apply_async(
                                            execute_and_save,
                                            tuple(
                                                (
                                                    path,
                                                    core.utils_3d.render_mesh,
                                                    overwrite,
                                                )
                                            ),  # args
                                            dict(
                                                mesh=mesh,
                                                yrot=angle,
                                                resolution=resolution,
                                                ztrans=zoom,
                                            ),  # kwargs
                                        )
                                    )

                                    # Save the arguments so that we can set the render later
                                    args_list.append(
                                        tuple((idx, shape_idx, angle, resolution))
                                    )

                        if reconstruction_handler is not None:

                            path = reconstruction_handler.path_render(
                                idx, shape_idx, angle, resolution, create=True
                            )
                            if not path_exists(path, p_list) or overwrite:
                                reconstruction_handler.delete_from_cache(path)
                                try:
                                    mesh = reconstruction_handler.get_mesh(
                                        idx, shape_idx
                                    )
                                    mesh = mesh.copy()
                                except errors.IsosurfaceExtractionError:
                                    continue

                                # Update the colors
                                core.utils_3d.colorize_mesh_from_index_auto(
                                    mesh, shape_idx
                                )

                                # Set the thread working
                                futures_list.append(
                                    pool.apply_async(
                                        execute_and_save,
                                        tuple(
                                            (
                                                path,
                                                core.utils_3d.render_mesh,
                                                overwrite,
                                            )
                                        ),  # args
                                        dict(
                                            mesh=mesh,
                                            yrot=angle,
                                            resolution=resolution,
                                            ztrans=zoom,
                                        ),  # kwargs
                                    )
                                )

                                # Save the arguments so that we can set the render later
                                args_list.append(
                                    tuple((idx, shape_idx, angle, resolution))
                                )

        logging.info("File loading took {}s".format(time.time() - start))

        start = time.time()
        # Wait on threads and display a progress bar
        logging.info("Queued {} jobs".format(len(futures_list)))
        for f, args in tqdm.tqdm(zip(futures_list, args_list)):
            val = f.get()
            # reconstruction_handler.set_render(f.get(), *args)  # Set the render
        logging.info("Collection took {}s".format(time.time() - start))


def eval_engine(
    reconstruct_list,
    output_pairs,
    threads,
    metric,
    reconstruction_handler,
    data_handler,
    overwrite=False,
    dynamic_path_check=False,
):
    """the handlers are great, but they can't be easily combined with multiprocessing"""

    if metric == "chamfer":
        fn = core.metrics.chamfer
    elif metric == "connected_artifacts_score2":  # NFRE in the paper
        fn = core.metrics.connected_artifacts_score2
    elif metric == "normal_consistency":
        fn = core.metrics.normal_consistency
    elif metric == "connected_protrusion_error":  # CAE in the paper
        fn = core.metrics.connected_protrusion_error
    else:
        raise RuntimeError("Unknown metric: {}".format(metric))

    p_list = None
    if not dynamic_path_check:
        root_dir = reconstruction_handler.dir_codes(create=True)
        p_list = set([os.path.join(root_dir, f) for f in os.listdir(root_dir)])

    futures_list = []
    args_list = []
    start = time.time()
    logging.info("Spawning {} threads ...".format(threads))
    with multiprocessing.Pool(threads) as pool:
        for idx in tqdm.tqdm(reconstruct_list):

            for (gt_idx, pd_idx) in output_pairs:

                path = reconstruction_handler.path_eval(
                    idx, pd_idx, gt_idx, metric, create=True
                )

                # Restoration-only metrics
                if (
                    metric
                    in [
                        "connected_artifacts_score2",
                        "connected_protrusion_error",
                    ]
                    and pd_idx < 2
                ):
                    continue

                if not overwrite and path_exists(path, p_list):
                    continue

                reconstruction_handler.delete_from_cache(path)

                # Load pred mesh
                try:
                    pred_mesh = reconstruction_handler.get_mesh(idx, pd_idx)
                except errors.IsosurfaceExtractionError:
                    continue

                if metric in ["connected_artifacts_score2"]:

                    # Only valid for restoration meshes
                    if pd_idx >= 2:
                        try:
                            gt_complete = data_handler.get_mesh(idx, 0)
                            gt_broken = data_handler.get_mesh(idx, 1)
                            gt_restoration = data_handler.get_mesh(idx, 2)
                        except FileNotFoundError:
                            continue

                        kwargs = dict(
                            gt_complete=gt_complete,
                            gt_broken=gt_broken,
                            gt_restoration=gt_restoration,
                            pd_restoration=pred_mesh,
                        )
                    else:
                        continue

                else:
                    # Load gt mesh
                    try:
                        gt_mesh = data_handler.get_mesh(
                            idx,
                            gt_idx,
                        )
                    except FileNotFoundError:
                        continue

                    kwargs = dict(
                        gt_shape=gt_mesh,
                        pred_shape=pred_mesh,
                    )

                futures_list.append(
                    pool.apply_async(
                        execute_and_save,
                        tuple(
                            (
                                path,
                                fn,
                                overwrite,
                            )
                        ),  # args
                        kwargs,  # kwargs
                    )
                )

                # Save the arguments so that we can set the eval later
                args_list.append(tuple((idx, pd_idx, gt_idx, metric)))

        logging.info("File loading took {}s".format(time.time() - start))

        start = time.time()
        # Wait on threads and display a progress bar
        logging.info("Queued {} jobs".format(len(futures_list)))
        for f, args in tqdm.tqdm(zip(futures_list, args_list)):
            val = f.get()
            # reconstruction_handler.set_eval(val, *args)  # Set the eval
        logging.info("Collection took {}s".format(time.time() - start))


def load_knittable(path):
    d = json.load(open(path, "r"))
    if "experiment_directory" not in d:
        d["experiment_directory"] = os.path.dirname(os.path.dirname(path))
    return ReconstructionHandler(**d)


def load_handler(path):
    with open(path, "rb") as f:
        h = pickle.load(f)
    h._loaded_from = os.sep + os.path.join(*os.path.normpath(path).split(os.sep)[:-4])
    return h


def quick_load_handler(path, json=True):
    # If it was passed a dir, then find the pkl
    if os.path.isdir(path):
        # Load from json
        if json:
            loadable_files = [f for f in os.listdir(path) if ".json" in f]
            assert (
                len(loadable_files) <= 1
            ), "Found more than one json file in knit directory {}".format(path)
            if len(loadable_files) == 1:
                logging.info("Loaded from json")
                return core.handler.load_knittable(
                    os.path.join(path, loadable_files[0])
                )

        # Load from pkl
        loadable_files = [f for f in os.listdir(path) if ".pkl" in f]
        assert (
            len(loadable_files) <= 1
        ), "Found more than one pkl file in knit directory {}".format(path)
        if len(loadable_files) == 1:
            logging.info("Loaded from pkl")
            return core.handler.load_handler(os.path.join(path, loadable_files[0]))
        raise RuntimeError(
            "Couldn't find any loadable files in knit directory {}".format(path)
        )
    else:
        if os.path.splitext(path)[-1] == ".json":
            logging.info("Loaded from pkl")
            return core.handler.load_knittable(path)

    logging.info("Loaded from pkl")
    return core.handler.load_handler(path)


def iterify_path(path, i):
    p, e = os.path.splitext(path)
    return p + "__iter-" + str(i) + e


def lossify_log_path(path):
    p, e = os.path.splitext(path)
    return p + "__loss" + e


class ReconstructionHandler:
    def __init__(
        self,
        experiment_directory,
        signiture,
        name,
        checkpoint,
        lat_signiture=None,
        use_occ=False,
        decoder=None,
        dims=(256, 256, 256),
        overwrite=False,
    ):
        """
        Handles saving and loading reconstructed objects in a fast way for
        evaluation.

        Args:
            handler: ModelHandler, used to get filepaths.
            signiture: List of identifying info that will differentiate this
                run from others (lr, num epochs, etc.)
            decoder: Autodecoder,  will be used to reconstruct a mesh if
                necessary.
            dims: List of dimensions, will be used to reconstruct a mesh if
                necessary.
        """
        assert os.path.exists(experiment_directory), "{}".format(experiment_directory)

        self._lat_signiture = lat_signiture
        if self._lat_signiture is None:
            self._lat_signiture = signiture
        self._knittable = {
            "experiment_directory": experiment_directory,
            "signiture": signiture,
            "lat_signiture": lat_signiture,
            "name": name,
            "checkpoint": checkpoint,
            "use_occ": use_occ,
            "decoder": decoder,
            "dims": dims,
            "overwrite": overwrite,
        }
        self._experiment_directory = os.path.abspath(experiment_directory)
        self._signiture = "".join([str(s) + "_" for s in signiture])
        self._lat_signiture = "".join([str(s) + "_" for s in self._lat_signiture])
        self._decoder = decoder
        self._dims = dims
        self._name = name
        self._checkpoint = checkpoint
        self._overwrite = overwrite
        self._cache = {}
        self._use_occ = use_occ
        self._loaded_from = self._experiment_directory
        if self._use_occ:
            self._level = 0.5
        else:
            self._level = 0.0

    @property
    def overwrite(self):
        return self._overwrite

    @property
    def signiture(self):
        return self._signiture

    @property
    def experiment_name(self):
        return self._experiment_directory.split("/")[-1]

    @property
    def root(self):
        return self._loaded_from

    @property
    def name(self):
        return self._name

    def save(self):
        del self._decoder
        path = self.path_loadable(create=True)
        pickle.dump(
            self,
            open(path, "wb"),
            pickle.HIGHEST_PROTOCOL,
        )

    def save_knittable(self):
        """
        When called, this creates a list of arguments that can be used to regenerate
        this reconstruction handler, similar to __repr__, and saves it to a specific file.
        Use this path to rebuild the reconstruction handler for knitting renders.

        e.g.
            path = old_handler.save_knittable()
            new_handler = core.handler.load_knittable(path)
        """

        path = self.path_knittable(create=True)
        json.dump(self._knittable, open(path, "w"), indent=2)
        return path

    def set_decoder(self, decoder):
        self._decoder = decoder

    def path_reconstruction(self, create=False):
        return ws.get_reconstructions_dir(
            self._experiment_directory,
            name=self._name,
            checkpoint=self._checkpoint,
            create=create,
        )

    def path_metrics(self, metrics_file, create=False):
        return os.path.join(
            ws.get_reconstructions_dir(
                self._experiment_directory,
                name=self._name,
                checkpoint=self._checkpoint,
                create=create,
            ),
            metrics_file,
        )

    def path_stats(self, create=False):
        return ws.get_reconstructions_stats_dir(
            self._experiment_directory,
            name=self._name,
            checkpoint=self._checkpoint,
            create=create,
        )

    def path_knittable(self, create=False):
        return os.path.join(
            ws.get_reconstructions_dir(
                self._experiment_directory,
                name=self._name,
                checkpoint=self._checkpoint,
                create=create,
            ),
            "knittable_" + self._signiture + ".json",
        )

    def path_loadable(self, create=False):
        return os.path.join(
            ws.get_reconstructions_dir(
                self._experiment_directory,
                name=self._name,
                checkpoint=self._checkpoint,
                create=create,
            ),
            "handler_" + self._signiture + ".pkl",
        )

    def path_summary_render(self, create=False):
        return os.path.join(
            ws.get_reconstructions_dir(
                self._experiment_directory,
                name=self._name,
                checkpoint=self._checkpoint,
                create=create,
            ),
            "summary_" + self._signiture + ".png",
        )

    def path_code(self, idx, create=False):
        return os.path.join(
            ws.get_reconstructions_codes_dir(
                self._experiment_directory,
                name=self._name,
                checkpoint=self._checkpoint,
                create=create,
            ),
            str(idx) + "_" + self._lat_signiture + ".pth",
        )

    def dir_codes(self, create=False):
        return ws.get_reconstructions_codes_dir(
            self._experiment_directory,
            name=self._name,
            checkpoint=self._checkpoint,
            create=create,
        )

    def path_loss(self, idx, create=False):
        return lossify_log_path(self.path_code(idx, create))

    def path_mesh(self, idx, shape_idx, create=False):
        return os.path.join(
            ws.get_reconstructions_meshes_dir(
                self._experiment_directory,
                name=self._name,
                checkpoint=self._checkpoint,
                create=create,
            ),
            str(idx) + "_" + str(shape_idx) + "_" + self._signiture + ".ply",
        )

    def path_mesh_interp(
        self, idx_from, idx_to, step, shape_idx, interp_type, create=False
    ):
        return os.path.join(
            ws.get_reconstructions_meshes_dir(
                self._experiment_directory,
                name=self._name,
                checkpoint=self._checkpoint,
                create=create,
            ),
            "interp"
            + str(idx_from)
            + "_"
            + str(idx_to)
            + "_"
            + str(step)
            + "_"
            + str(shape_idx)
            + "_"
            + str(interp_type)
            + "_"
            + self._signiture
            + ".ply",
        )

    def path_values(self, idx, shape_idx, create=False):
        return os.path.join(
            ws.get_reconstructions_meshes_dir(
                self._experiment_directory,
                name=self._name,
                checkpoint=self._checkpoint,
                create=create,
            ),
            str(idx)
            + "_"
            + str(shape_idx)
            + "_"
            + self._signiture
            + "_values"
            + ".npy",
        )

    def path_render(self, idx, shape_idx, angle, resolution=(200, 200), create=False):
        # Backwards compatability
        if tuple(resolution) == (200, 200):
            return os.path.join(
                ws.get_reconstructions_renders_dir(
                    self._experiment_directory,
                    name=self._name,
                    checkpoint=self._checkpoint,
                    create=create,
                ),
                str(idx)
                + "_"
                + str(shape_idx)
                + "_"
                + str(angle)
                + "_"
                + self._signiture
                + ".png",
            )
        return os.path.join(
            ws.get_reconstructions_renders_dir(
                self._experiment_directory,
                name=self._name,
                checkpoint=self._checkpoint,
                create=create,
            ),
            str(idx)
            + "_"
            + str(shape_idx)
            + "_"
            + str(angle)
            + "_"
            + str(resolution[0])
            + "_"
            + str(resolution[1])
            + "_"
            + self._signiture
            + ".png",
        )

    def dir_renders(self, create=False):
        return ws.get_reconstructions_renders_dir(
            self._experiment_directory,
            name=self._name,
            checkpoint=self._checkpoint,
            create=create,
        )

    def path_composite(
        self, idx, shape_idx, gt_shape_idx, angle, resolution=(200, 200), create=False
    ):
        return os.path.join(
            ws.get_reconstructions_renders_dir(
                self._experiment_directory,
                name=self._name,
                checkpoint=self._checkpoint,
                create=create,
            ),
            str(idx)
            + "_"
            + str(shape_idx)
            + "_"
            + str(gt_shape_idx)
            + "_"
            + str(angle)
            + "_"
            + str(resolution[0])
            + "_"
            + str(resolution[1])
            + "_"
            + "composite"
            + "_"
            + self._signiture
            + ".png",
        )

    def path_eval(self, idx, shape_idx, gt_shape_idx, metric, create=False):
        return os.path.join(
            ws.get_reconstructions_codes_dir(
                self._experiment_directory,
                name=self._name,
                checkpoint=self._checkpoint,
                create=create,
            ),
            str(idx)
            + "_"
            + str(shape_idx)
            + "_"
            + str(gt_shape_idx)
            + "_"
            + str(metric)
            + "_"
            + self._signiture
            + ".npy",
        )

    def delete_from_cache(self, path):
        if path in self._cache:
            del self._cache[path]

    def load(self, f_in, skip_cache=False):
        if f_in in self._cache and not skip_cache:
            logging.debug("Pulling file from cache: {}".format(f_in))
            return self._cache[f_in]

        data = loader(f_in)

        if not skip_cache:
            self._cache[f_in] = data
        return data

    def set_code(self, idx, code, save=False, cache=True):
        """
        Set a code. Generating codes requires running lengthly reconstructions,
        which I don't want to offload to a handler function. Thus codes must be
        manually set by the user.
        """
        path = self.path_code(idx, create=True)
        if save:
            saver(path, code)
        if cache:
            self._cache[path] = code

    def get_code(self, idx):
        """
        Get a code. If code does not exist will throw FileNotFoundError.
        """
        return self.load(self.path_code(idx, create=True))

    def set_values(self, idx, shape_idx, values, save=False, cache=True):
        """
        Set the raw values predicted by the network.
        """
        path = self.path_values(idx, shape_idx, create=True)
        if save:
            saver(path, values)
        if cache:
            self._cache[path] = values

    def get_values(self, idx, shape_idx):
        """
        Get the raw values predicted by the network. May throw FileNotFoundError.
        """
        return self.load(self.path_values(idx, shape_idx, create=True))

    def set_mesh(
        self,
        mesh,
        idx,
        shape_idx,
        save=False,
    ):
        path = self.path_mesh(idx, shape_idx, create=True)
        if save:
            saver(path, mesh)
        self._cache[path] = mesh

    def get_mesh(self, idx, shape_idx):
        """
        Get a mesh. The mesh will be loaded from the cache. If it cannot be loaded
        from the cache, it will be read from disk.
        """
        path = self.path_mesh(idx, shape_idx, create=True)
        if not self._overwrite:
            try:
                mesh = core.utils_3d.force_trimesh(self.load(path))
                if (mesh is None) or (mesh.vertices.shape[0] == 0):
                    raise errors.IsosurfaceExtractionError
                return mesh

            # If the file doesn't exist, this will be thrown
            except ValueError:
                raise errors.IsosurfaceExtractionError

            # This has been happening for a few reconstructions - meshes are just crashing the viewer
            except IndexError:
                raise errors.IsosurfaceExtractionError

    def set_mesh_interp(
        self,
        mesh,
        idx_from,
        idx_to,
        step,
        shape_idx,
        interp_type=None,
        save=False,
    ):
        path = self.path_mesh_interp(
            idx_from, idx_to, step, shape_idx, interp_type, create=True
        )
        if save:
            saver(path, mesh)
        self._cache[path] = mesh

    def get_mesh_interp(
        self, idx_from, idx_to, step, shape_idx, interp_type=None, cache=True
    ):
        path = self.path_mesh_interp(
            idx_from, idx_to, step, shape_idx, interp_type, create=True
        )
        mesh = core.utils_3d.force_trimesh(self.load(path))
        if cache:
            self._cache[path] = mesh
        return mesh

    def set_composite(
        self,
        render,
        idx,
        shape_idx,
        gt_shape_idx,
        angle=0,
        resolution=(200, 200),
        save=False,
    ):
        path = self.path_composite(
            idx, shape_idx, gt_shape_idx, angle, resolution, create=True
        )
        if save:
            saver(path, render)
        self._cache[path] = render

    def get_composite(
        self,
        idx,
        shape_idx,
        gt_shape,
        gt_shape_idx,
        angle=0,
        resolution=(200, 200),
        self_color=(128, 64, 64, 255),
        gt_color=(64, 128, 64, 128),
        save=True,
        cache=True,
    ):
        path = self.path_composite(
            idx, shape_idx, gt_shape_idx, angle, resolution, create=True
        )
        if not self._overwrite:
            try:
                return self.load(path)
            except FileNotFoundError:
                pass

        try:
            mesh = self.get_mesh(idx, shape_idx)
        except errors.IsosurfaceExtractionError as e:
            if save:
                saver(path, core.utils_2d.space_image(list(resolution)))
            raise e

        # Update the colors
        mesh.visual = trimesh.visual.color.ColorVisuals(mesh, vertex_colors=self_color)
        gt_shape.visual = trimesh.visual.color.ColorVisuals(
            gt_shape, vertex_colors=gt_color
        )

        # Render
        render = core.utils_3d.render_mesh(
            [gt_shape, mesh],
            yrot=angle,
            resolution=resolution,
        )

        if save:
            saver(path, render)
        if cache:
            self._cache[path] = render
        return render

    def set_render(
        self,
        render,
        idx,
        shape_idx,
        angle=0,
        resolution=(200, 200),
        save=False,
    ):
        path = self.path_render(idx, shape_idx, angle, resolution, create=True)
        if save:
            saver(path, render)
        self._cache[path] = render

    def get_render(
        self, idx, shape_idx, angle=0, resolution=(200, 200), save=True, cache=True
    ):
        """
        Get a render. The render will be loaded from the cache. If it cannot be
        loaded from the cache, it will be read from disk. If it cannot be read
        from disk it will be generated.
        """
        path = self.path_render(
            idx, shape_idx, angle, resolution=resolution, create=True
        )
        if not self._overwrite:
            try:
                return self.load(path)
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

        try:
            mesh = self.get_mesh(idx, shape_idx)
        except errors.IsosurfaceExtractionError as e:
            if save:
                saver(path, core.utils_2d.space_image(list(resolution)))
            raise e

        render = core.utils_3d.render_mesh(
            mesh,
            yrot=angle,
            resolution=resolution,
        )
        if save:
            saver(path, render)
        if cache:
            self._cache[path] = render
        return render

    def set_eval(
        self,
        eval,
        idx,
        shape_idx,
        gt_shape_idx=None,
        metric="chamfer",
        save=True,
    ):
        if gt_shape_idx is None:
            gt_shape_idx = shape_idx
        path = self.path_eval(idx, shape_idx, gt_shape_idx, metric, create=True)
        if save:
            saver(path, eval)
        self._cache[path] = eval

    def get_eval(
        self,
        gt_shape,
        idx,
        shape_idx,
        gt_shape_idx=None,
        metric="chamfer",
        save=True,
        cache=True,
    ):
        """
        Evaluate a reconstructed mesh given a ground truth sample.
        """
        if gt_shape_idx is None:
            gt_shape_idx = shape_idx
        path = self.path_eval(idx, shape_idx, gt_shape_idx, metric, create=True)
        if not self._overwrite:
            try:
                dist = self.load(path)
                if np.isnan(dist):
                    raise errors.IsosurfaceExtractionError
                return dist
            except FileNotFoundError:
                pass

        logging.debug(
            "Eval ({}, {}) with filename {} could not be found, generating".format(
                idx,
                shape_idx,
                path,
            )
        )

        try:
            pred_shape = self.get_mesh(idx, shape_idx)
        except errors.IsosurfaceExtractionError as e:
            # if save:
            #     saver(path, np.nan)
            raise e

        if metric == "chamfer":
            dist = core.metrics.chamfer(gt_shape=gt_shape, pred_shape=pred_shape)
        elif metric == "normal_consistency":
            dist = core.metrics.normal_consistency(
                gt_shape=gt_shape, pred_shape=pred_shape
            )
        else:
            raise RuntimeError("Unknown metric: {}".format(metric))

        if save:
            saver(path, dist)
        if cache:
            self._cache[path] = dist
        return dist


# === for backwards compatability ===
class ShapeNetHandler:
    def __init__(self, root_dir, class_id, instance_id) -> None:
        self._root_dir = root_dir
        self._class_id = class_id
        self._instance_id = instance_id

        self._cache = {}

    def __repr__(self) -> str:
        return (
            self.__name__
            + "("
            + self._root_dir
            + ", "
            + self._class_id
            + ", "
            + self._instance_id
            + ")"
        )

    def __hash__(self) -> int:
        return hash(self._class_id + "-" + self._instance_id)

    @property
    def root_dir(self) -> str:
        return self._root_dir

    @property
    def class_id(self) -> str:
        return self._class_id

    @property
    def instance_id(self) -> str:
        return self._instance_id

    def path(self, subdir, basename) -> str:
        return os.path.join(
            self._root_dir,
            self._class_id,
            self._instance_id,
            subdir,
            basename,
        )

    def build_dirs(self) -> None:
        dir_path = os.path.join(
            self._root_dir, self._class_id, self._instance_id, "renders"
        )
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        dir_path = os.path.join(
            self._root_dir, self._class_id, self._instance_id, "evals"
        )
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    def load(self, f_in, skip_cache=False):
        if f_in in self._cache and not skip_cache:
            logging.debug("Pulling file from cache: {}".format(f_in))
            return self._cache[f_in]

        data = loader(f_in)

        if not skip_cache:
            self._cache[f_in] = data
        return data

    # == model path shortcuts ==
    def path_normalized(self):
        return self.path("models", "model_normalized.obj")

    def path_waterproofed(self):
        return self.path("models", "model_waterproofed.obj")

    def path_sampled(self, idx=0):
        return self.path("models", "model_{}_sampled.npz".format(idx))

    def path_c(self):
        return self.path("models", "model_c.obj")

    def path_c_voxelized(self):
        return self.path("models", "model_c_voxelized.obj")

    def path_c_sdf(self, idx=0):
        return self.path("models", "model_c_{}_sdf.npz".format(idx))

    def path_c_uniform_sdf(self, idx=0):
        return self.path("models", "model_c_{}_uniform_padded_sdf.npz".format(idx))

    def path_c_uniform_occ(self, idx=0):
        return self.path("models", "model_c_{}_uniform_padded_occ.npz".format(idx))

    def path_c_voxel(self, size=32):
        assert size == 32
        return self.path(
            "models", "model_c_{}_uniform_padded_occ_{}.npz".format(0, size)
        )

    def path_c_uniform_occ_64(self, idx=0):
        return self.path("models", "model_c_{}_uniform_padded_occ_64.npz".format(idx))

    def path_b(self, idx=0):
        return self.path("models", "model_b_{}.obj".format(idx))

    def path_b_nofrac(self, idx=0):
        return self.path("models", "model_b_{}_nofrac.obj".format(idx))

    def path_b_voxelized(self, idx=0):
        return self.path("models", "model_b_{}_voxelized.obj".format(idx))

    def path_b_sdf(self, idx=0):
        return self.path("models", "model_b_{}_sdf.npz".format(idx))

    def path_b_partial_sdf(self, idx=0):
        return self.path("models", "model_b_{}_partial_sdf.npz".format(idx))

    def path_b_uniform_sdf(self, idx=0):
        return self.path("models", "model_b_{}_uniform_padded_sdf.npz".format(idx))

    def path_b_uniform_occ(self, idx=0):
        return self.path("models", "model_b_{}_uniform_padded_occ.npz".format(idx))

    def path_b_uniform_occ_64(self, idx=0):
        return self.path("models", "model_b_{}_uniform_padded_occ_64.npz".format(idx))

    def path_b_voxel(self, idx=0, size=32):
        assert size == 32
        return self.path(
            "models", "model_b_{}_uniform_padded_occ_{}.npz".format(idx, size)
        )

    def path_r(self, idx=0):
        return self.path("models", "model_r_{}.obj".format(idx))

    def path_r_voxelized(self, idx=0):
        return self.path("models", "model_r_{}_voxelized.obj".format(idx))

    def path_r_sdf(self, idx=0):
        return self.path("models", "model_r_{}_sdf.npz".format(idx))

    def path_r_uniform_sdf(self, idx=0):
        return self.path("models", "model_r_{}_uniform_padded_sdf.npz".format(idx))

    def path_r_uniform_occ(self, idx=0):
        return self.path("models", "model_r_{}_uniform_padded_occ.npz".format(idx))

    def path_r_uniform_occ_64(self, idx=0):
        return self.path("models", "model_r_{}_uniform_padded_occ_64.npz".format(idx))

    def path_r_nofrac(self, idx=0):
        return self.path("models", "model_r_{}_nofrac.obj".format(idx))

    def path_r_voxel(self, idx=0, size=32):
        assert size == 32
        return self.path(
            "models", "model_r_{}_uniform_padded_occ_{}.npz".format(idx, size)
        )

    # === eval path shortcuts ===
    def path_c_upsampled(self):
        return self.path("evals", "model_c_upsampled.npz")

    def path_b_upsampled(self, idx=0):
        return self.path("evals", "model_b_{}_upsampled.npz".format(idx))

    def path_r_upsampled(self, idx=0):
        return self.path("evals", "model_r_{}_upsampled.npz".format(idx))

    # === eval path shortcuts ===
    def path_c_upsampled(self):
        return self.path("evals", "model_c_upsampled.npz")

    def path_b_upsampled(self, idx=0):
        return self.path("evals", "model_b_{}_upsampled.npz".format(idx))

    def path_r_upsampled(self, idx=0):
        return self.path("evals", "model_r_{}_upsampled.npz".format(idx))

    # == render path shortcuts ==
    def path_c_rendered(self, angle=0, resolution=(200, 200)):
        if resolution == (200, 200):
            return self.path("renders", "model_c_rendered{}.png".format(angle))
        else:
            return self.path(
                "renders",
                "model_c_rendered_{}_{}_{}.png".format(
                    resolution[0], resolution[1], angle
                ),
            )

    def path_c_depth(self, angle=0, resolution=(200, 200)):
        if resolution == (200, 200):
            return self.path("renders", "model_c_depth{}.png".format(angle))
        else:
            return self.path(
                "renders",
                "model_c_depth{}_{}_{}.png".format(resolution[0], resolution[1], angle),
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

    def path_b_rendered(self, idx=0, angle=0, resolution=(200, 200)):
        if resolution == (200, 200):
            return self.path("renders", "model_b_{}_rendered{}.png".format(idx, angle))
        else:
            return self.path(
                "renders",
                "model_b_{}_rendered{}_{}_{}.png".format(
                    idx, resolution[0], resolution[1], angle
                ),
            )

    def path_b_depth(self, angle, idx=0, resolution=(200, 200)):
        if resolution == (200, 200):
            return self.path("renders", "model_b_{}_depth{}.png".format(idx, angle))
        else:
            return self.path(
                "renders",
                "model_b_{}_depth{}_{}_{}.png".format(
                    idx, resolution[0], resolution[1], angle
                ),
            )

    def path_r_rendered(self, idx=0, angle=0, resolution=(200, 200)):
        if resolution == (200, 200):
            return self.path("renders", "model_r_{}_rendered{}.png".format(idx, angle))
        else:
            return self.path(
                "renders",
                "model_r_{}_rendered{}_{}_{}.png".format(
                    idx, resolution[0], resolution[1], angle
                ),
            )

    def path_r_depth(self, angle, idx=0, resolution=(200, 200)):
        if resolution == (200, 200):
            return self.path("renders", "model_r_{}_depth{}.png".format(idx, angle))
        else:
            return self.path(
                "renders",
                "model_r_{}_depth{}_{}_{}.png".format(
                    idx, resolution[0], resolution[1], angle
                ),
            )
