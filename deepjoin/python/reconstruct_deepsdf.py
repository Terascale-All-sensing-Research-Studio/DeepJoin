import argparse
import json
import logging
import os
import random
import math
import multiprocessing

import torch
import numpy as np

import core

STATUS_INDICATOR = None


def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    stat,
    clamp_dist,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
    code_reg_lambda=1e-4,
    slippage_method=None,
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    # Create the latent vector
    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()
    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_l1 = torch.nn.L1Loss()
    loss_dict = {
        "epoch": [],
        "data_loss": [],
        "reg_loss": [],
        "mag": [],
    }

    # Get the points
    pts, data_sdf, sample_kwargs = test_sdf

    # Update the mask with some noise, if required
    logging.debug("using slippage_method: {}".format(slippage_method))
    if slippage_method == "fake-classifier":
        complete_gt = core.loader(sample_kwargs["complete_gt"])
        broken_gt = core.loader(sample_kwargs["broken_gt"])

        # Get a perfect mask
        mask = core.data.get_fracture_point_mask(
            complete_gt=complete_gt,
            broken_gt=broken_gt,
            pts=pts,
        )

        # Simulate a non-perfect classifier
        if sample_kwargs["percent"] != 0.0:
            mask = core.data.partial_sample_with_classification_noise(
                complete_gt=complete_gt,
                broken_gt=broken_gt,
                pts=pts,
                mask=mask,
                percent=sample_kwargs["percent"],
            )

    elif slippage_method == "classifier":
        mask = core.data.partial_sample_pointnet(
            broken_gt=core.loader(sample_kwargs["broken_gt"]),
            pts=pts,
            label=sample_kwargs["label"],
        )
    elif slippage_method == "analytical":
        mask = core.data.partial_sample_analytical(
            broken_gt=core.loader(sample_kwargs["broken_gt"]),
            pts=pts,
            percent=sample_kwargs["percent"],
        )
    else:
        raise RuntimeError("Unknown slippage method: {}".format(slippage_method))

    for e in range(num_iterations):

        decoder.eval()

        xyz, sdf_gt = core.data.select_partial_samples(
            pts=pts,
            values=data_sdf,
            mask=mask,
            uniform_ratio=0.2,
            num_samples=num_samples,
        )

        # Convert to tensors
        xyz = torch.from_numpy(xyz).type(torch.float).cuda()
        sdf_gt = torch.from_numpy(sdf_gt).type(torch.float).cuda()

        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)

        inputs = torch.cat([latent_inputs, xyz], 1).cuda()

        pred_sdf = decoder(inputs)

        # TODO: why is this needed?
        if e == 0:
            pred_sdf = decoder(inputs)

        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        data_loss = loss_l1(pred_sdf, sdf_gt)
        loss = data_loss
        if l2reg:
            reg_loss = code_reg_lambda * torch.mean(latent.pow(2))
            loss = loss + reg_loss
        loss.backward()
        optimizer.step()

        if e % 10 == 0:
            loss_dict["epoch"].append(e)
            loss_dict["data_loss"].append(data_loss.item())
            loss_dict["reg_loss"].append(reg_loss.item())
            loss_dict["mag"].append(torch.norm(latent).item())
            logging.debug(
                "epoch: {:4d} | data loss: {} reg loss: {}".format(
                    loss_dict["epoch"][-1],
                    loss_dict["data_loss"][-1],
                    loss_dict["reg_loss"][-1],
                )
            )

    return loss_dict, latent.cpu()


def callback():
    global STATUS_INDICATOR
    STATUS_INDICATOR.increment()


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--threads",
        default=6,
        type=int,
        help="Number of threads to use for reconstruction.",
    )
    arg_parser.add_argument(
        "--render_threads",
        default=6,
        type=int,
        help="Number of threads to use for rendering.",
    )
    arg_parser.add_argument(
        "--num_iters",
        default=800,
        type=int,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--num_samples",
        default=8000,
        type=int,
        help="Number of samples to use.",
    )
    arg_parser.add_argument(
        "--stop",
        default=None,
        type=int,
        help="Stop inference after x samples.",
    )
    arg_parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Randomized seed.",
    )
    arg_parser.add_argument(
        "--out_of_order",
        default=False,
        action="store_true",
        help="Randomize the order of inference.",
    )
    arg_parser.add_argument(
        "--name",
        default="deepsdf_",
        type=str,
        help="",
    )
    arg_parser.add_argument(
        "--slippage",
        default=0.0,
        type=float,
        help="Number of fracture samples to let 'slip through'.",
    )
    arg_parser.add_argument(
        "--slippage_method",
        default="fake-classifier",
        type=str,
        help="Method for computing slippage.",
    )
    arg_parser.add_argument(
        "--overwrite_codes",
        action="store_true",
        default=False,
        help="",
    )
    arg_parser.add_argument(
        "--overwrite_meshes",
        action="store_true",
        default=False,
        help="",
    )
    arg_parser.add_argument(
        "--overwrite_evals",
        action="store_true",
        default=False,
        help="",
    )
    arg_parser.add_argument(
        "--overwrite_renders",
        action="store_true",
        default=False,
        help="",
    )
    arg_parser.add_argument(
        "--save_values",
        action="store_true",
        default=False,
        help="",
    )
    arg_parser.add_argument(
        "--skip_render",
        action="store_true",
        default=False,
        help="",
    )
    arg_parser.add_argument(
        "--skip_eval",
        action="store_true",
        default=False,
        help="",
    )
    arg_parser.add_argument(
        "--skip_export_eval",
        action="store_true",
        default=False,
        help="",
    )
    arg_parser.add_argument(
        "--render_only",
        action="store_true",
        default=False,
        help="",
    )
    arg_parser.add_argument(
        "--split",
        default=None,
        type=int,
        help="",
    )
    core.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    core.configure_logging(args)
    logging.info(args)

    assert os.environ["DATADIR"], "environment variable $DATADIR must be defined"

    specs_filename = core.find_specs(args.experiment_directory)
    specs = json.load(open(specs_filename))
    args.experiment_directory = os.path.dirname(specs_filename)

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    # >>> begin update: parsing arguments
    num_samples = args.num_samples
    num_iterations = args.num_iters
    slippage = args.slippage
    slippage_method = args.slippage_method
    name = args.name
    render_threads = args.render_threads
    overwrite_codes = args.overwrite_codes
    overwrite_meshes = args.overwrite_meshes
    overwrite_evals = args.overwrite_evals
    overwrite_renders = args.overwrite_renders
    threads = args.threads
    save_values = args.save_values

    clamp_dist = specs["ClampingDistance"]
    test_split_file = specs["TestSplit"]

    network_outputs = 0
    total_outputs = (0, 2, 9)
    eval_pairs = [
        (0, 0),
        (1, 1),
        (2, 2),
        (2, 9),
    ]  # ORDER MATTERS
    metrics_version = "v3.metrics.npy"
    metrics = [
        "chamfer",
        "break_error",
        "connected_artifacts_score2",
        "normal_consistency",
    ]
    composite = [(1, 2), (1, 9)]
    render_resolution = (200, 200)
    lr = 5e-3
    do_code_regularization = True
    code_reg_lambda = 1e-4

    assert (
        specs["NetworkArch"] == "decoder_deepsdf"
    ), "Only supports decoder_deepsdf network arch."

    # >>> begin update: need to offload the decoder to the threads
    network_kwargs = dict(
        decoder_kwargs=dict(
            latent_size=latent_size,
            **specs["NetworkSpecs"],
        ),
        decoder_constructor=arch.Decoder,
        experiment_directory=args.experiment_directory,
        checkpoint=args.checkpoint,
    )
    assert slippage_method in ["analytical", "classifier", "fake-classifier"]
    reconstruction_kwargs = dict(
        num_iterations=num_iterations,
        latent_size=latent_size,
        stat=0.01,  # [emp_mean,emp_var],
        clamp_dist=clamp_dist,
        num_samples=num_samples,
        lr=lr,
        l2reg=do_code_regularization,
        code_reg_lambda=code_reg_lambda,
        slippage_method=slippage_method,
    )
    mesh_kwargs = dict(
        N=256,
        max_batch=int(2**16),
    )

    # >>> begin update: using our dataloader
    # Get the data directory from environment variable
    test_split_file = test_split_file.replace("$DATADIR", os.environ["DATADIR"])
    data_source = specs["DataSource"].replace("$DATADIR", os.environ["DATADIR"])

    # Create and load the dataset
    reconstruction_handler = core.handler.ReconstructionHandler(
        experiment_directory=args.experiment_directory,
        dims=[256, 256, 256],
        name=name,
        checkpoint=args.checkpoint,
        overwrite=False,
        use_occ=specs["UseOccupancy"],
        signiture=[
            num_samples,
            slippage,
        ],
    )
    sdf_dataset = core.data.SamplesDataset(
        test_split_file,
        subsample=num_samples,
        uniform_ratio=specs["UniformRatio"],
        use_occ=specs["UseOccupancy"],
        clamp_dist=clamp_dist,
        root=data_source,
    )

    # >>> begin update: using our reconstruction handler
    reconstruct_list = list(range(len(sdf_dataset)))
    unseed = random.randint(0, 10000)
    if args.seed is not None:
        random.seed(args.seed)
        random.shuffle(reconstruct_list)
    if args.stop is not None:
        reconstruct_list = reconstruct_list[: args.stop]
    if args.out_of_order:
        random.seed(unseed)  # Unseed the random generator
        random.shuffle(reconstruct_list)
        random.seed(args.seed)  # Reseed the random generator

    # I hate this
    if args.split is not None:
        reconstruct_list = reconstruct_list[: int(len(reconstruct_list) / args.split)]

    # >>> begin update: changed everything
    input_list, path_list = [], []
    for ii in reconstruct_list:

        # Generate the code if necessary
        path_code = reconstruction_handler.path_code(ii, create=True)
        if (not os.path.exists(path_code)) or overwrite_codes:
            pts, data_sdf = sdf_dataset.get_broken_sample(ii)
            input_list.append(
                dict(
                    test_sdf=[
                        pts,
                        data_sdf,
                        dict(
                            complete_gt=sdf_dataset.path_mesh(ii, 0),
                            broken_gt=sdf_dataset.path_mesh(ii, 1),
                            percent=slippage,
                            label=sdf_dataset.get_object(ii).class_id,
                        ),
                    ],
                )
            )
            path_list.append(path_code)

    # # Spawn a threadpool to do reconstruction
    num_tasks = len(input_list)
    STATUS_INDICATOR = core.utils_multiprocessing.MultiprocessBar(num_tasks)

    with multiprocessing.Pool(threads) as pool:
        logging.info("Starting {} threads".format(threads))
        if num_tasks != 0:
            logging.info("Reconstructing {} codes".format(num_tasks))
            STATUS_INDICATOR.reset(num_tasks)
            futures_list = []

            # Cut the work into chunks
            step_size = math.ceil(num_tasks / threads)
            for idx in range(0, num_tasks, step_size):
                start, end = idx, min(idx + step_size, num_tasks)

                futures_list.append(
                    pool.apply_async(
                        core.utils_multiprocessing.reconstruct_chunk,
                        tuple(
                            (
                                input_list[start:end],
                                path_list[start:end],
                                reconstruct,
                                network_kwargs,
                                reconstruction_kwargs,
                                overwrite_codes,
                                callback,
                            )
                        ),
                    )
                )

            # Wait on threads and display a progress bar
            for f in futures_list:
                f.get()

        # Create the complete meshes
        input_list, path_list = [], []
        for ii in reconstruct_list:

            # Generate the mesh if necessary
            path_mesh = reconstruction_handler.path_mesh(ii, 0, create=True)
            path_values = reconstruction_handler.path_values(ii, 0, create=True)
            if (
                not os.path.exists(path_mesh)
                or not os.path.exists(path_values)
                or overwrite_meshes
            ):
                input_list.append(
                    dict(
                        latent_vec=reconstruction_handler.get_code(ii),
                        save_values=reconstruction_handler.path_values(ii, 0),
                    )
                )
                path_list.append(path_mesh)

        # Spawn a threadpool to do mesh generation
        num_tasks = len(input_list)
        if num_tasks != 0:
            logging.info("Reconstructing {} complete meshes".format(num_tasks))
            STATUS_INDICATOR.reset(num_tasks)
            futures_list = []

            # Cut the work into chunks
            step_size = math.ceil(num_tasks / threads)
            for idx in range(0, num_tasks, step_size):
                start, end = idx, min(idx + step_size, num_tasks)

                futures_list.append(
                    pool.apply_async(
                        core.utils_multiprocessing.mesh_chunk,
                        tuple(
                            (
                                input_list[start:end],
                                path_list[start:end],
                                core.utils_deepsdf.create_mesh,
                                network_kwargs,
                                mesh_kwargs,
                                True,  # This is set to always true so that the values are always generated
                                callback,
                            )
                        ),
                    )
                )

            # Wait on threads and display a progress bar
            for f in futures_list:
                f.get()

    logging.info("Staring {} threads".format(render_threads))
    with multiprocessing.Pool(render_threads) as pool:
        # Create the restoration meshes
        input_list, path_list = [], []
        for ii in reconstruct_list:

            # Generate the mesh if necessary
            shape_idx = 2
            path_mesh = reconstruction_handler.path_mesh(ii, shape_idx, create=True)
            path_output_values = reconstruction_handler.path_values(
                ii, shape_idx, create=True
            )
            if (
                not os.path.exists(path_mesh)
                or not os.path.exists(path_output_values)
                or overwrite_meshes
            ):
                try:
                    pred_path = reconstruction_handler.path_values(ii, 0)
                    if not os.path.exists(pred_path):
                        continue
                    oidx, bidx = sdf_dataset._indexer[ii]
                    gt_path = sdf_dataset.objects[oidx].path_b_uniform_occ(bidx)
                    if not os.path.exists(gt_path):
                        raise RuntimeError(
                            "GT Uniform samples do not exist: {}".format(gt_path)
                        )
                except FileNotFoundError:
                    continue
                input_list.append(
                    dict(
                        pred_values=pred_path,
                        gt_values=gt_path,
                        save_values=path_output_values,
                        thresh=0.0,
                    )
                )
                path_list.append(path_mesh)

        # Spawn a threadpool to do mesh generation
        num_tasks = len(input_list)
        if num_tasks != 0:
            logging.info("Reconstructing {} restoration meshes".format(num_tasks))
            STATUS_INDICATOR.reset(num_tasks)
            futures_list = []

            # Cut the work into chunks
            step_size = math.ceil(num_tasks / threads)
            for idx in range(0, num_tasks, step_size):
                start, end = idx, min(idx + step_size, num_tasks)

                futures_list.append(
                    pool.apply_async(
                        core.utils_multiprocessing.mesh_chunk_no_decoder,
                        tuple(
                            (
                                input_list[start:end],
                                path_list[start:end],
                                core.utils_deepsdf.create_restoration,
                                dict(),
                                overwrite_meshes,
                                callback,
                            )
                        ),
                    )
                )

            # Wait on threads and display a progress bar
            for f in futures_list:
                f.get()

        # Create the restoration meshes, without a threshold
        input_list, path_list = [], []
        for ii in reconstruct_list:

            # Generate the mesh if necessary
            shape_idx = 9
            path_mesh = reconstruction_handler.path_mesh(ii, shape_idx, create=True)
            if not os.path.exists(path_mesh) or overwrite_meshes:
                try:
                    pred_mesh = reconstruction_handler.get_mesh(ii, 2)
                except core.errors.IsosurfaceExtractionError:
                    continue
                input_list.append(
                    dict(
                        pred_mesh=pred_mesh,
                        thresh=0.01,
                    )
                )
                path_list.append(path_mesh)

        # Spawn a threadpool to do mesh generation
        num_tasks = len(input_list)
        if num_tasks != 0:
            logging.info("Reconstructing {} meshes".format(num_tasks))
            STATUS_INDICATOR.reset(num_tasks)
            futures_list = []

            # Cut the work into chunks
            step_size = math.ceil(num_tasks / threads)
            for idx in range(0, num_tasks, step_size):
                start, end = idx, min(idx + step_size, num_tasks)

                futures_list.append(
                    pool.apply_async(
                        core.utils_multiprocessing.mesh_chunk_no_decoder,
                        tuple(
                            (
                                input_list[start:end],
                                path_list[start:end],
                                core.utils_3d.mesh_discard_components,
                                dict(),
                                overwrite_meshes,
                                callback,
                            )
                        ),
                    )
                )

            # Wait on threads and display a progress bar
            for f in futures_list:
                f.get()

    STATUS_INDICATOR.close()

    reconstruct_list.sort()

    if not args.skip_render:
        path = os.path.join(
            reconstruction_handler.path_reconstruction(), "summary_img_{}.jpg"
        )
        if not os.path.exists(path.format(0)):
            # Spins up a multiprocessed renderer
            logging.info("Rendering results ...")
            core.handler.render_engine(
                data_handler=sdf_dataset,
                reconstruct_list=reconstruct_list,
                reconstruction_handler=reconstruction_handler,
                outputs=total_outputs,
                num_renders=3,
                resolution=render_resolution,
                composite=composite,
                overwrite=overwrite_renders,
                threads=render_threads,
            )

            logging.info("Building summary render")
            img = core.vis.image_results(
                data_handler=sdf_dataset,
                reconstruct_list=reconstruct_list,
                reconstruction_handler=reconstruction_handler,
                outputs=total_outputs,
                num_renders=3,
                resolution=render_resolution,
                composite=composite,
                knit_handlers=[],
            )
            logging.info(
                "Saving summary render to: {}".format(
                    path.replace(os.environ["DATADIR"], "$DATADIR")
                )
            )
            core.vis.save_image_block(img, path)

    if not args.skip_eval:
        for metric in metrics:
            logging.info("Computing {} ...".format(metric))
            core.handler.eval_engine(
                reconstruct_list=reconstruct_list,
                output_pairs=eval_pairs,
                threads=render_threads,
                overwrite=overwrite_evals,
                reconstruction_handler=reconstruction_handler,
                data_handler=sdf_dataset,
                metric=metric,
            )

    if not args.skip_export_eval:
        exclusion_list = sdf_dataset.num_components != 1

        out_folder = reconstruction_handler.path_stats(create=True)
        logging.info("Saving metrics to: {}".format(out_folder))
        out_metrics = reconstruction_handler.path_metrics(metrics_version)
        core.statistics.export_report(
            out_metrics=out_metrics,
            reconstruction_handler=reconstruction_handler,
            reconstruct_list=reconstruct_list,
            output_pairs=eval_pairs,
            metrics=metrics,
        )

    # Save
    logging.info("Saving reconstruction handler, please wait ...")
    reconstruction_handler.save_knittable()
    logging.info(
        "To knit with this experiment, pass the following path: \n{}".format(
            reconstruction_handler.path_loadable().replace(
                os.environ["DATADIR"], "$DATADIR"
            )
        )
    )
