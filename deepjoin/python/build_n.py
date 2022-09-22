import os
import argparse
import logging
import json
import random
import pickle

import tqdm
import trimesh
import zipfile
import numpy as np

import core
from processor.shapenet import ShapeNetObject


def distance_normals_fixer(pts, c_sdf, b_sdf, r_sdf, t_sdf, c_n, b_n, r_n, t_n):

    # c_occ = core.data.sdf_to_occ(c_sdf.copy())
    t_occ = core.data.sdf_to_occ(t_sdf.copy())

    b_idxs = (
        np.logical_or(
            t_occ,  # inside break
            c_sdf < -t_sdf,  # argmax(sdf_C, -sdf_B)
        )
        .squeeze()
        .astype(int)
    )
    b_sdf_fixed = np.concatenate(
        (
            np.expand_dims(c_sdf, axis=0),
            -np.expand_dims(t_sdf, axis=0),  # note the negative sign
        ),
        axis=0,
    )[b_idxs, np.arange(c_sdf.shape[0]), :]
    b_n_fixed = np.concatenate(
        (
            np.expand_dims(c_n, axis=0),
            np.expand_dims(t_n, axis=0),
        ),
        axis=0,
    )[b_idxs, np.arange(c_sdf.shape[0]), :]

    r_idxs = (
        np.logical_or(
            np.logical_not(t_occ),  # Outside break
            c_sdf < t_sdf,  # argmax(sdf_C, sdf_B)
        )
        .squeeze()
        .astype(int)
    )
    r_sdf_fixed = np.concatenate(
        (
            np.expand_dims(c_sdf, axis=0),
            np.expand_dims(t_sdf, axis=0),
        ),
        axis=0,
    )[r_idxs, np.arange(c_sdf.shape[0]), :]
    r_n_fixed = np.concatenate(
        (
            np.expand_dims(c_n, axis=0),
            -np.expand_dims(t_n, axis=0),  # note the negative sign
        ),
        axis=0,
    )[r_idxs, np.arange(c_sdf.shape[0]), :]

    return (
        c_sdf,
        b_sdf_fixed,
        r_sdf_fixed,
        t_sdf,
        c_n,
        b_n_fixed,
        r_n_fixed,
        t_n,
    )


def build_train(
    id_list,
    train_out,
    train_c_out,
    train_check,
    num_breaks,
    use_primitive,
    use_spline,
    use_occ,
    use_pointer,
    use_normals,
    full_spline,
    use_pointclouds,
    fix_distance_normals,
    z_rot,
):

    if (train_out is not None) or (train_c_out is not None):
        # System to double check that two train databases are the same
        if train_check is not None:
            logging.info("Loading saved data from check file {}".format(train_check))
            check_data = pickle.load(open(train_check, "rb"))
            check_indexer = check_data["indexer"]
            check_data.clear()
            del check_data

        logging.info("Will save train data to: {}".format(train_out))
        sdf_list = []
        encode_list = []
        norm_list = []
        occupancy_list = []
        index_list = []
        complete_index_list = []
        flipped_list = []
        logging.info("Loading train list")
        for obj_idx, obj in tqdm.tqdm(enumerate(id_list)):

            breaks_loaded = []
            for break_idx in range(num_breaks):

                # Here's all the data we're going to load
                path_list = [
                    obj.path_sampled(break_idx),
                    obj.path_c_sdf(break_idx),
                    obj.path_b_sdf(break_idx),
                    obj.path_r_sdf(break_idx),
                ]
                if use_primitive:
                    path_list.append(obj.path_primitive_sdf(break_idx))
                elif use_spline:
                    if full_spline:
                        path_list.append(obj.path_full_spline_sdf(break_idx))
                    else:
                        path_list.append(obj.path_spline_sdf(break_idx))
                else:
                    path_list.append(obj.path_tool_sdf(break_idx))

                if all([os.path.exists(p) for p in path_list]):
                    try:
                        data = [np.load(d) for d in path_list]
                        pts, c, b, r, t, nc, nb, nr, nt = (
                            np.load(path_list[0])["xyz"],
                            np.expand_dims(data[1]["sdf"], axis=1),
                            np.expand_dims(data[2]["sdf"], axis=1),
                            np.expand_dims(data[3]["sdf"], axis=1),
                            np.expand_dims(data[4]["sdf"], axis=1),
                            data[1]["n"],
                            data[2]["n"],
                            data[3]["n"],
                            data[4]["n"],
                        )
                        sdf_sample_orig = np.hstack(
                            (
                                pts,
                                c,
                                b,
                                r,
                                t,
                            )
                        )
                        if fix_distance_normals:
                            c, b, r, t, nc, nb, nr, nt = distance_normals_fixer(
                                pts, c, b, r, t, nc, nb, nr, nt
                            )
                        sdf_sample = np.hstack(
                            (
                                pts,
                                c,
                                b,
                                r,
                                t,
                            )
                        )
                        if use_normals:
                            normals_sample = np.hstack((nc, nb, nr, nt))

                        if use_pointclouds:
                            c_obj = trimesh.load(obj.path_c())
                            pointcloud_data = {
                                None: c_obj.vertices,
                                "normals": c_obj.vertex_normals,
                            }

                    except (zipfile.BadZipFile):
                        logging.warning(
                            "Sample ({}, {}) is corrupted, skipping".format(
                                obj_idx, break_idx
                            )
                        )
                        continue

                    index_tuple = (obj_idx, break_idx)

                    if z_rot:
                        # Flip a coin
                        if np.random.randint(0, 2):
                            sdf_sample[:, :3] = core.points_transform(
                                sdf_sample[:, :3],
                                trimesh.transformations.rotation_matrix(
                                    angle=np.radians(90),
                                    direction=[0, 0, 1],
                                    point=(0, 0, 0),
                                ),
                            )
                            flipped_list.append(True)
                        else:
                            flipped_list.append(False)

                    occ_sample = core.data.sdf_to_occ(
                        sdf_sample_orig.astype(np.float16), skip_cols=3
                    )

                    # This is a sanity check
                    if (
                        not (
                            occ_sample[:, 3] == occ_sample[:, 4] + occ_sample[:, 5]
                        ).all()
                        or not (
                            occ_sample[:, 4]
                            == np.clip(occ_sample[:, 3] - occ_sample[:, 6], 0, 1)
                        ).all()
                        or not (
                            occ_sample[:, 5]
                            == np.clip(occ_sample[:, 3] + occ_sample[:, 6], 1, 2) - 1
                        ).all()
                    ):
                        logging.warning(
                            "Sample ({}, {}) is invalid, skipping.".format(*index_tuple)
                        )
                        continue

                    # Double check against the input train check
                    if train_check is not None:
                        if not index_tuple in check_indexer:
                            logging.info(
                                "Sample ({}, {}) not in train check, skipping.".format(
                                    *index_tuple
                                )
                            )
                            continue

                    # Add all the data to the corresponding lists
                    if use_pointclouds:
                        encode_list.append(pointcloud_data)
                        occ_sample = occ_sample[:, :4]

                    if use_occ:
                        pts, values = occ_sample[:, :3], occ_sample[:, 3:]

                        sdf_sample = pts
                        occupancy_list.append(values.astype(bool))
                        sdf_list.append(sdf_sample[:, :3].astype(np.float16))
                        if use_normals:
                            norm_list.append(normals_sample.astype(np.float16))
                    else:
                        sdf_list.append(sdf_sample.astype(np.float16))
                        if use_normals:
                            norm_list.append(normals_sample.astype(np.float16))

                    breaks_loaded.append(len(sdf_list))
                    index_list.append(index_tuple)

            if len(breaks_loaded) > 0:
                complete_index_list.append(breaks_loaded)

            # if len(sdf_list) > 10:
            #     break

            # Make sure the cache is empty
            obj._cache = {}

        if use_pointclouds:
            logging.info("num samples loaded: {}".format(len(encode_list)))
        else:
            logging.info("num samples loaded: {}".format(len(sdf_list)))
        logging.info("Saving data ...")

        # Double check against the input train check
        if train_check is not None:
            logging.info("Checking that all samples were added")
            for index_tuple in check_indexer:
                if index_tuple not in index_list:
                    logging.info("Sample ({}, {}) not added.".format(*index_tuple))
            logging.info("Loaded samples: {}".format(len(index_list)))
            logging.info("Check samples:  {}".format(len(check_indexer)))

        if train_out is not None:
            if os.path.exists(train_out):
                input(
                    "File: {} already exists, are you sure you want to overwrite?".format(
                        train_out
                    )
                )
            data_dict = {
                "indexer": index_list,
                "complete_indexer": complete_index_list,
                "objects": id_list,
                "use_occ": use_occ,
                "train": True,
                "flipped_list": flipped_list,
            }
            tr_out, _ = os.path.splitext(train_out)

            if use_pointer:
                path = tr_out + "_sdf.npz"
                logging.info("Saving sdf values to: {}".format(path))

                data_dict["sdf"] = path
                np.savez(path, sdf=sdf_list)

                if use_normals:
                    path = tr_out + "_n.npz"
                    logging.info("Saving values to: {}".format(path))

                    data_dict["n"] = path
                    np.savez(path, n=norm_list)

                if use_pointclouds:
                    path = tr_out + "_encoding.npz"
                    logging.info("Saving values to: {}".format(path))

                    data_dict["encode_list"] = path
                    np.savez(path, encode_list=encode_list)

                if use_occ:
                    path = tr_out + "_occ.npz"
                    logging.info("Saving values to: {}".format(path))

                    data_dict["occ_values"] = path
                    np.savez(path, occupancy_list=occupancy_list)
            else:
                data_dict["sdf"] = sdf_list
                if use_normals:
                    data_dict["n"] = norm_list
                if use_pointclouds:
                    data_dict["encode_list"] = encode_list
                if use_occ:
                    data_dict["occ_values"] = occupancy_list

            logging.info("Saving data_dict to: {}".format(train_out))
            pickle.dump(
                data_dict,
                open(train_out, "wb"),
                pickle.HIGHEST_PROTOCOL,
            )
        if train_c_out is not None:
            if os.path.exists(train_c_out):
                input(
                    "File: {} already exists, are you sure you want to overwrite?".format(
                        train_c_out
                    )
                )
            for idx in range(len(sdf_list)):
                sdf_list[idx] = sdf_list[idx][:, :4]
            pickle.dump(
                {
                    "sdf": sdf_list,
                    "indexer": index_list,
                    "complete_indexer": complete_index_list,
                    "objects": id_list,
                    "use_occ": use_occ,
                    "train": True,
                },
                open(train_c_out, "wb"),
                pickle.HIGHEST_PROTOCOL,
            )


def main(
    root_dir,
    train_out,
    train_c_out,
    train_check,
    val_out,
    test_out,
    splits_file,
    num_breaks,
    load_models,
    use_primitive,
    use_spline,
    use_occ,
    use_pointer,
    use_normals,
    full_spline,
    use_pointclouds,
    recompute_mask,
    fix_distance_normals,
    z_rot,
):
    logging.info("Loading saved data from splits file {}".format(splits_file))

    object_id_dict = json.load(open(splits_file, "r"))
    id_train_list = [
        ShapeNetObject(root_dir, o[0], o[1]) for o in object_id_dict["id_train_list"]
    ]
    try:
        id_val_list = [
            ShapeNetObject(root_dir, o[0], o[1]) for o in object_id_dict["id_val_list"]
        ]
    except KeyError:
        id_val_list = None
    id_test_list = [
        ShapeNetObject(root_dir, o[0], o[1]) for o in object_id_dict["id_test_list"]
    ]

    if train_c_out is not None:
        assert not use_occ

    if train_out is not None or train_c_out is not None:
        build_train(
            id_list=id_train_list,
            train_out=train_out,
            train_c_out=train_c_out,
            train_check=train_check,
            num_breaks=num_breaks,
            use_primitive=use_primitive,
            use_spline=use_spline,
            use_occ=use_occ,
            use_pointer=use_pointer,
            use_normals=use_normals,
            full_spline=full_spline,
            use_pointclouds=use_pointclouds,
            fix_distance_normals=fix_distance_normals,
            z_rot=z_rot,
        )
    if val_out is not None and id_val_list is not None:
        build_train(
            id_list=id_val_list,
            train_out=val_out,
            train_c_out=None,
            train_check=None,
            num_breaks=num_breaks,
            use_primitive=use_primitive,
            use_spline=use_spline,
            use_occ=use_occ,
            use_pointer=use_pointer,
            use_normals=use_normals,
            full_spline=full_spline,
            use_pointclouds=use_pointclouds,
            fix_distance_normals=fix_distance_normals,
            z_rot=False,
        )

    if test_out is not None:
        logging.info("Will save test data to: {}".format(test_out))
        encode_list = []
        sdf_list = []
        index_list = []
        complete_index_list = []
        logging.info("Loading test list")
        for obj_idx, obj in tqdm.tqdm(enumerate(id_test_list)):

            breaks_loaded = []
            for break_idx in range(num_breaks):
                if os.path.exists(obj.path_b_partial_sdf(break_idx)):
                    try:
                        sdf_sample = obj.load(
                            obj.path_b_partial_sdf(break_idx), skip_cache=True
                        )

                        if recompute_mask:
                            c_obj = trimesh.load(obj.path_c())
                            b_obj = trimesh.load(obj.path_b(break_idx))
                            sdf_sample["mask"] = core.get_fracture_point_mask(
                                c_obj, b_obj, sdf_sample["xyz"]
                            )

                        if use_pointclouds:

                            ext_obj = core.get_nonfracture_mesh(c_obj, b_obj)
                            pointcloud_data = {
                                None: ext_obj.vertices,
                                "normals": ext_obj.vertex_normals,
                            }

                    except (zipfile.BadZipFile):
                        logging.warning(
                            "Sample ({}, {}) is corrupted, skipping".format(
                                obj_idx, break_idx
                            )
                        )
                        continue

                    if use_pointclouds:
                        encode_list.append(pointcloud_data)

                    breaks_loaded.append(len(sdf_list))
                    sdf_list.append(sdf_sample)
                    index_list.append(
                        (
                            obj_idx,
                            break_idx,
                        )
                    )

                    if load_models:
                        obj.load(obj.path_b(break_idx))
                        obj.load(obj.path_r(break_idx))

            if len(breaks_loaded) > 0:
                complete_index_list.append(breaks_loaded)
                if load_models:
                    obj.load(obj.path_c())

        if os.path.exists(test_out):
            input(
                "File: {} already exists, are you sure you want to overwrite?".format(
                    test_out
                )
            )
        logging.info("num samples loaded: {}".format(len(sdf_list)))
        logging.info("Saving data ...")
        pickle.dump(
            {
                "sdf": sdf_list,
                "indexer": index_list,
                "complete_indexer": complete_index_list,
                "objects": id_test_list,
                "use_occ": False,
                "train": False,
                "encode_list": encode_list,
            },
            open(test_out, "wb"),
            pickle.HIGHEST_PROTOCOL,
        )
        f_out = os.path.join(
            os.path.dirname(test_out),
            os.path.splitext(os.path.basename(test_out))[0] + "_index.json",
        )
        print(f_out)
        core.saver(
            f_out,
            index_list,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a train and test pkl file from a splits file. The "
        + "pkl files will only contain valid samples. Optionally preload "
        + "upsampled points and models to accelerate evaluation."
    )
    parser.add_argument(
        dest="input",
        type=str,
        help="Location of the database. Pass the top level directory. For "
        + "ShapeNet this would be ShapeNet.v2",
    )
    parser.add_argument(
        dest="splits",
        type=str,
        help=".json file path, this file will be created and will store the "
        + "ids of all objects in the training and testing split.",
    )
    parser.add_argument(
        "--train_check",
        default=None,
        type=str,
        help="Train database file to check against. Should be a .pkl",
    )
    parser.add_argument(
        "--train_out",
        default=None,
        type=str,
        help="Where to save the resulting train database file. Should be a .pkl",
    )
    parser.add_argument(
        "--val_out",
        default=None,
        type=str,
        help="Where to save the resulting val database file. Should be a .pkl",
    )
    parser.add_argument(
        "--train_c_out",
        default=None,
        type=str,
        help="Where to save the resulting train database file. Should be a .pkl. "
        + "This file is a simplified version of the training database file that "
        + "only contains the complete shape.",
    )
    parser.add_argument(
        "--test_out",
        default=None,
        type=str,
        help="Where to save the resulting test database file. Should be a .pkl",
    )
    parser.add_argument(
        "--breaks",
        "-b",
        type=int,
        default=1,
        help="Number of breaks to generate for each object. This will only be "
        + "used if BREAK is passed.",
    )
    parser.add_argument(
        "--load_models",
        action="store_true",
        default=False,
        help="If passed, will preload object models. Note this is only applicable "
        + "to the test database file.",
    )
    parser.add_argument(
        "--use_primitive",
        action="store_true",
        default=False,
        help="If passed, will use the primitive instead of the tool.",
    )
    parser.add_argument(
        "--use_spline",
        action="store_true",
        default=False,
        help="If passed, will use the spline instead of the tool.",
    )
    parser.add_argument(
        "--full_spline",
        action="store_true",
        default=False,
        help="If passed, will use full spline.",
    )
    parser.add_argument(
        "--use_occ",
        action="store_true",
        default=False,
        help="If passed, will load data in occ mode.",
    )
    parser.add_argument(
        "--use_pointer",
        action="store_true",
        default=False,
        help="If passed, will save a pointer to the file.",
    )
    parser.add_argument(
        "--use_normals",
        action="store_true",
        default=False,
        help="If passed, will load normals.",
    )
    parser.add_argument(
        "--use_pointclouds", action="store_true", default=False, help=""
    )
    parser.add_argument("--recompute_mask", action="store_true", default=False, help="")
    parser.add_argument(
        "--fix_distance_normals", action="store_true", default=False, help=""
    )
    parser.add_argument("--z_rot", action="store_true", default=False, help="")
    core.add_common_args(parser)
    args = parser.parse_args()
    core.configure_logging(args)
    logging.info(args)

    main(
        args.input,
        args.train_out,
        args.train_c_out,
        args.train_check,
        args.val_out,
        args.test_out,
        args.splits,
        args.breaks,
        args.load_models,
        args.use_primitive,
        args.use_spline,
        args.use_occ,
        args.use_pointer,
        args.use_normals,
        args.full_spline,
        args.use_pointclouds,
        args.recompute_mask,
        args.fix_distance_normals,
        args.z_rot,
    )
