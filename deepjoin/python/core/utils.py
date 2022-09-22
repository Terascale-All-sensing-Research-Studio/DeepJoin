import os
import logging
import random


def file_is_old(path, overwrite_time=1636410000):
    """
    Returns true if a file was last edited before the input time.
    """
    if overwrite_time is None:
        return False
    if (
        os.path.exists(path)
        and overwrite_time is not None
        and os.path.getmtime(path) < overwrite_time
    ):
        return True
    return False


def add_common_args(arg_parser):
    arg_parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action="store_true",
        help="If set, debugging messages will be printed",
    )
    arg_parser.add_argument(
        "--quiet",
        "-q",
        dest="quiet",
        default=False,
        action="store_true",
        help="If set, only warnings will be printed",
    )
    arg_parser.add_argument(
        "--log",
        dest="logfile",
        default=None,
        help="If set, the log will be saved using the specified filename.",
    )


def configure_logging(args):
    logger = logging.getLogger()
    if not logger.handlers:
        if args.debug:
            logger.setLevel(logging.DEBUG)
        elif args.quiet:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)
        logger_handler = logging.StreamHandler()
        formatter = logging.Formatter("AutoDecoder - %(levelname)s - %(message)s")
        logger_handler.setFormatter(formatter)
        logger.addHandler(logger_handler)

        if args.logfile is not None:
            file_logger_handler = logging.FileHandler(args.logfile)
            file_logger_handler.setFormatter(formatter)
            logger.addHandler(file_logger_handler)


def get_file(d, fe, fn=None):
    """
    Given a root directory and a list of file extensions, recursively
    return all files in that directory that have that extension.
    """
    for f in os.listdir(d):
        fp = os.path.join(d, f)
        if os.path.isdir(fp):
            yield from get_file(fp, fe, fn)
        elif os.path.splitext(fp)[-1] in fe:
            if fn is None:
                yield fp
            elif fn == os.path.splitext(f)[0]:
                yield fp


def find_specs(p):
    if os.path.basename(p) == "specs.json":
        return p
    p = p.replace("$DATADIR", os.environ["DATADIR"])
    if not os.path.exists(os.path.join(p, "specs.json")):
        p = os.path.dirname(os.path.dirname(p))
    assert os.path.isfile(
        os.path.join(p, "specs.json")
    ), "Could not find specs file in directory: {}".format(p)
    logging.info("Found specs at: {}".format(os.path.join(p, "specs.json")))
    return os.path.join(p, "specs.json")


def quick_load_path(path):
    """Converts a normal path to a quickload path"""
    return os.path.splitext(path)[0] + "_quickload" + os.path.splitext(path)[1]


def reshuffle_split_class(data, train_ratio):
    """
    Takes an existing splits dictionary and reshuffles it so that
    classes are split between the train and test split.
    """
    data_new = {
        "id_train_list": [],
        "id_test_list": [],
    }

    data_list = data["id_test_list"] + data["id_train_list"]

    class_counter = {}
    for class_id, instance_id in data_list:
        if class_id not in class_counter:
            class_counter[class_id] = list()
        class_counter[class_id].append(instance_id)

    for class_id in class_counter.keys():
        num_instances = len(class_counter[class_id])

        num_train = int(num_instances * train_ratio)

        instance_list = class_counter[class_id]
        random.shuffle(instance_list)

        data_new["id_train_list"].extend(
            [[class_id, inst] for inst in instance_list[:num_train]]
        )
        data_new["id_test_list"].extend(
            [[class_id, inst] for inst in instance_list[num_train:]]
        )

    return data_new
