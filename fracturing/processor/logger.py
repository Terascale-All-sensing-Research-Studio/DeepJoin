import logging


def add_logger_args(arg_parser):
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
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        logger_handler.setFormatter(formatter)
        logger.addHandler(logger_handler)

        if args.logfile is not None:
            file_logger_handler = logging.FileHandler(args.logfile)
            file_logger_handler.setFormatter(formatter)
            logger.addHandler(file_logger_handler)
