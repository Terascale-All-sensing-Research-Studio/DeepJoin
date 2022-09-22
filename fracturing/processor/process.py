import os
import argparse
import logging
import random
import json
import time
import importlib

import tqdm

import processor.utils as utils
import processor.shapenet as shapenet
import processor.logger as logger
from processor.utils import (
    GracefulProcessPoolExecutor,
    GracefulProcessPoolExecutorDebug,
)


def main(
    root_dir,
    ops,
    threads,
    overwrite,
    num_breaks,
    class_subsample,
    instance_subsample,
    splits_file,
    train_ratio,
    debug,
    outoforder,
    train_only,
    test_only,
    count,
    args,
):

    logging.info("Performing the following operations: {}".format(ops))
    logging.info("Using {} thread(s)".format(threads))
    logging.info("Using splits file: {}".format(splits_file))

    if (splits_file is not None) and os.path.exists(splits_file):
        logging.info("Loading saved data from splits file {}".format(splits_file))
        object_id_dict = json.load(open(splits_file, "r"))

        root_dir = shapenet.shapenet_toplevel(root_dir)
        id_train_list = [
            shapenet.ShapeNetObject(root_dir, o[0], o[1])
            for o in object_id_dict["id_train_list"]
        ]
        id_test_list = [
            shapenet.ShapeNetObject(root_dir, o[0], o[1])
            for o in object_id_dict["id_test_list"]
        ]
        if "id_val_list" in object_id_dict:
            id_val_list = [
                shapenet.ShapeNetObject(root_dir, o[0], o[1])
                for o in object_id_dict["id_val_list"]
            ]
            id_test_list += id_val_list
        object_id_list = id_test_list + id_train_list
    else:
        # Obtain a list of all the objects in the shapenet dataset
        logging.info("Searching for objects ...")
        object_id_list, root_dir = shapenet.shapenet_search(
            root_dir, return_toplevel=True
        )
        logging.info("Found {} objects".format(len(object_id_list)))

        # Subsample this list, if required
        if (class_subsample is not None) or (instance_subsample is not None):
            class_list = list(set([o.class_id for o in object_id_list]))

            # Sample classes
            if class_subsample is not None:
                try:
                    class_list = random.sample(class_list, class_subsample)
                except ValueError:
                    raise ValueError(
                        "Requested too many samples, there are only {} objects".format(
                            len(class_list)
                        )
                    )

            # Group object by class
            id_by_class = []
            for c in class_list:
                id_by_class.append([o for o in object_id_list if o.class_id == c])

            # Sample instances
            if instance_subsample is not None:
                for idx in range(len(id_by_class)):
                    # It's often the case that
                    if len(id_by_class[idx]) < instance_subsample:
                        logging.warning(
                            "Only {} samples in class {}, adding all".format(
                                len(id_by_class[idx]), class_list[idx]
                            )
                        )
                    try:
                        id_by_class[idx] = random.sample(
                            id_by_class[idx],
                            min(instance_subsample, len(id_by_class[idx])),
                        )
                    except ValueError:
                        raise ValueError(
                            "Requested too many samples, there are only {} objects".format(
                                len(class_list)
                            )
                        )

            # Flatten list
            object_id_list = []
            for e in id_by_class:
                object_id_list.extend(e)

        # Split into a train test list
        id_train_list, id_test_list = utils.split_train_test(
            object_id_list, train_ratio
        )
        # This should have a passable ratio but I dont have time
        id_train_list, id_val_list, id_test_list = utils.rebalance_split_train_test_val(
            id_train_list, id_test_list
        )

        logging.info("Reduced to {} objects after sampling".format(len(object_id_list)))

        # Save the list
        logging.info("Saving data to splits file {}".format(splits_file))
        json.dump(
            {
                "id_train_list": [[o.class_id, o.instance_id] for o in id_train_list],
                "id_val_list": [[o.class_id, o.instance_id] for o in id_val_list],
                "id_test_list": [[o.class_id, o.instance_id] for o in id_test_list],
            },
            open(splits_file, "w"),
        )
        id_test_list += id_val_list

    logging.info("Building subdirectories ...")
    for o in object_id_list:
        o.build_dirs()

    if outoforder:
        random.shuffle(id_train_list)
        random.shuffle(id_test_list)
        object_id_list = id_test_list + id_train_list

    logging.info("Root dir at {}".format(root_dir))
    logging.info("Processing {} objects".format(len(object_id_list)))
    logging.info(
        "{} train objects, {} test objects".format(
            len(id_train_list), len(id_test_list)
        )
    )
    logging.info("{} classes".format(len(set([o.class_id for o in object_id_list]))))

    global GracefulProcessPoolExecutor
    # This will completely disable the pool
    if (threads == 1) and (debug):
        GracefulProcessPoolExecutor = GracefulProcessPoolExecutorDebug

    object_counter = {o: [0, 0] for o in ops}

    with GracefulProcessPoolExecutor(max_workers=threads) as executor:
        # Seed randomizer
        if not ((threads == 1) and (debug)):
            executor.map(
                utils.random_seeder, [int(time.time()) + i for i in range(threads)]
            )

        for op in ops:
            module = importlib.import_module("processor." + op)
            list_to_run = object_id_list
            if train_only:
                list_to_run = id_train_list
            elif test_only:
                list_to_run = id_test_list

            pbar = tqdm.tqdm(list_to_run, desc="Running {}".format(op))
            try:
                for obj in pbar:
                    if not count:
                        pbar.write(
                            "[{}|{}] {}".format(op, num_breaks, obj.path_normalized())
                        )
                        module.process(
                            obj=obj,
                            num_results=num_breaks,
                            overwrite=overwrite,
                            executor=executor,
                            args=args,
                        )
                executor.graceful_finish()

            except KeyboardInterrupt:
                logging.info("Waiting for running processes ...")
                executor.graceful_finish()

            _success_list = [
                module.validate_outputs(obj, num_breaks, args) for obj in list_to_run
            ]
            success_list = []
            for s in _success_list:  # Flatten
                success_list.extend(s)
            success_list = [int(s) for s in success_list]
            object_counter[op] = [sum(success_list), len(success_list)]

    # Print out any errors encountered
    if len(executor.exceptions_log) > 0:
        logging.info("SUMMARY: The following errors were encountered ...")
        for k, v in executor.exceptions_log.items():
            logging.info("{}: {}".format(k, v))
    else:
        logging.info("SUMMARY: All operations completed successfully.")

    for o in ops:
        logging.info(
            "{} successfully processed {} out of {} breaks".format(
                o, object_counter[o][0], object_counter[o][1]
            )
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Applies a sequence of "
        + "transforms to all objects in a database in parallel. Upon "
        + "completion prints a summary of errors encountered during runtime."
    )
    parser.add_argument(
        dest="input",
        type=str,
        help="Location of the database. Pass the top level directory. For "
        + 'ShapeNet this would be "ShapeNet.v2". Models will be extracted '
        + "by name and by extension (ENSURE THERE ARE NO OTHER .obj FILES IN "
        + "THIS DIRECTORY).",
    )
    parser.add_argument(
        dest="splits",
        type=str,
        help=".json file path, this file will be created and will store the "
        + "ids of all objects in the training and testing split. Will be used "
        + "if preprocessing is restarted to accelerate initial steps.",
    )
    parser.add_argument(
        dest="ops",
        type=str,
        nargs="+",
        help="List of operations to apply",
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=1,
        help="Number of threads to use. This script uses multiprocessing so "
        + "it is not recommended to set this number higher than the number of "
        + "physical cores in your computer.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of training samples to testing samples that will be saved "
        + "to the split file.",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="If passed will overwrite existing files. Else will skip existing "
        + "files.",
    )
    parser.add_argument(
        "--break_handle",
        action="store_true",
        default=False,
        help="If passed, will fracture 2/10 mug handles.",
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
        "--break_method",
        type=str,
        default="surface-area",
        help="Which breaking method to use.",
    )
    parser.add_argument(
        "--use_tool",
        default=False,
        action="store_true",
        help="Save and compute the SDF values for the breaking tool.",
    )
    parser.add_argument(
        "--class_subsample",
        default=None,
        type=int,
        help="If passed, will randomly sample this many classes from the "
        + "dataset. Will override subsample flag.",
    )
    parser.add_argument(
        "--voxel_dim",
        default=256,
        type=int,
        help="Voxel dimension.",
    )
    parser.add_argument(
        "--instance_subsample",
        default=None,
        type=int,
        help="If passed, will randomly sample this many instances from each "
        + "class from the dataset. Will override subsample flag.",
    )
    parser.add_argument(
        "--reorient",
        action="store_true",
        default=False,
        help="If passed, will use PCA to reorient the model.",
    )
    parser.add_argument(
        "--max_break",
        default=0.5,
        type=float,
        help="Max amount (percentage based) of the source model to remove in a "
        + "given break. Breaks will be retried if they remove more than this "
        + "amount.",
    )
    parser.add_argument(
        "--min_break",
        default=0.3,
        type=float,
        help="Min amount (percentage based) of the source model to remove in a "
        + "given break. Breaks will be retried if they remove less than this "
        + "amount.",
    )
    parser.add_argument(
        "--outoforder",
        default=False,
        action="store_true",
        help="If passed, will shuffle the dataset before processing. Note this "
        + "will not alter the cotents of the split file. Use this option if "
        + "you plan to process data simultaneously with multiple scripts or PCS.",
    )
    parser.add_argument(
        "--train_only",
        default=False,
        action="store_true",
        help="If passed, will only run on train set.",
    )
    parser.add_argument(
        "--test_only",
        default=False,
        action="store_true",
        help="If passed, will only run on test set.",
    )
    parser.add_argument(
        "--count",
        default=False,
        action="store_true",
        help="If passed, will only count.",
    )
    parser.add_argument(
        "--full_spline",
        default=False,
        action="store_true",
        help="If passed, will use a full spline representation.",
    )
    logger.add_logger_args(parser)
    args = parser.parse_args()
    logger.configure_logging(args)

    assert os.path.isdir(args.input), "Input directory does not exist: {}".format(
        args.input
    )
    if args.input[-1] == "/":
        args.input = args.input[:-1]

    assert not (args.use_tool and args.use_primitive)

    main(
        args.input,
        args.ops,
        args.threads,
        args.overwrite,
        args.breaks,
        args.class_subsample,
        args.instance_subsample,
        args.splits,
        args.train_ratio,
        args.debug,
        args.outoforder,
        args.train_only,
        args.test_only,
        args.count,
        args,
    )
