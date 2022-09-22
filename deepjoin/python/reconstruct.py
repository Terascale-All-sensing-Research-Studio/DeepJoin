import argparse
import json
import logging
import os
import random
import math
import multiprocessing

import torch
import numpy as np
from collections import defaultdict

import core

STATUS_INDICATOR = None
STATUS_COUNTER = 0


def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    break_latent_size,
    test_sdf,
    stat,
    lambda1,
    lambda2,
    lambda3,
    clamp_dist,
    sampling_method=None,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
    code_reg_lambda=1e-4,
    iter_path=None,
    loss_version=None,
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
        break_latent = torch.ones(1, break_latent_size).normal_(mean=0, std=stat).cuda()
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()
        break_latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()

    latent.requires_grad = True
    break_latent.requires_grad = True

    optimizer = torch.optim.Adam([latent, break_latent], lr=lr)

    if sampling_method == "surface_interior_only":
        # Get only external samples
        pts_gt, b_sdf_gt = test_sdf
        test_sdf = np.hstack((pts_gt, b_sdf_gt))
        assert test_sdf.shape[1] == 4
        pts_samples_uni = core.data.get_uniform(test_sdf)
        pts_samples_surf = core.data.get_surface(test_sdf)
        pts_samples_uni = pts_samples_uni[pts_samples_uni[:, -1] < 0, :]

        def get_samples():
            num_uni = int(num_samples * 0.2)
            num_surf = int(num_samples - num_uni)

            # Get uniform points
            try:
                uni_inds = np.random.choice(
                    pts_samples_uni.shape[0], num_uni, replace=False
                )
            except ValueError:
                uni_inds = np.random.choice(
                    pts_samples_uni.shape[0], num_uni, replace=True
                )

            pts_samples_uni_cur = pts_samples_uni[uni_inds, :]
            pts_uni, values_uni = pts_samples_uni_cur[:, :-1], np.expand_dims(
                pts_samples_uni_cur[:, -1], axis=1
            )

            # Get surface points
            pts_surf, values_surf = core.data.select_samples(
                pts=pts_samples_surf[:, :-1],
                values=np.expand_dims(pts_samples_surf[:, -1], axis=1),
                num_samples=num_surf,
                uniform_ratio=None,
            )
            pts = np.vstack((pts_uni, pts_surf))
            values = np.vstack((values_uni, values_surf))
            return pts, values

    loss_l1 = torch.nn.L1Loss()
    loss_bce = torch.nn.BCELoss()
    loss_l2 = torch.nn.MSELoss()
    loss_dict = defaultdict(lambda: [])

    sdf_data_loss = torch.Tensor([0]).cuda()
    norm_loss = torch.Tensor([0]).cuda()

    for e in range(num_iterations):

        decoder.eval()

        if sampling_method == "surface_interior_only":
            pts_gt, b_sdf_gt = get_samples()
        else:
            pts_gt, b_sdf_gt, b_n_gt = test_sdf
            pts_gt = np.hstack((pts_gt, b_n_gt))
            pts_gt, b_sdf_gt = core.data.select_samples(
                pts_gt, b_sdf_gt, num_samples, uniform_ratio=0.2
            )

        pts_gt, b_n_gt = pts_gt[:, :3], pts_gt[:, 3:]

        pts_gt = torch.from_numpy(pts_gt).type(torch.float).cuda()
        b_n_gt = torch.from_numpy(b_n_gt).type(torch.float)
        b_sdf_gt = torch.from_numpy(b_sdf_gt).type(torch.float)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)
        break_latent_inputs = break_latent.expand(num_samples, -1)
        inputs = torch.cat([latent_inputs, break_latent_inputs, pts_gt], dim=1).cuda()

        (
            c_occ,
            b_occ,
            r_occ,
            t_occ,
            c_x,
            b_x,
            r_x,
            t_x,
            c_n,
            b_n,
            r_n,
            t_n,
        ) = decoder(inputs)

        if e == 0:
            (
                c_occ,
                b_occ,
                r_occ,
                t_occ,
                c_x,
                b_x,
                r_x,
                t_x,
                c_n,
                b_n,
                r_n,
                t_n,
            ) = decoder(inputs)

        # Clamp sdf values
        b_x = torch.clamp(b_x, -clamp_dist, clamp_dist)

        b_sdf_gt = b_sdf_gt
        b_occ_gt = core.reconstruct.tensor_sdf_to_occ(b_sdf_gt)

        if lambda3 != 0:
            sdf_data_loss = (
                loss_l1(
                    b_x,
                    b_sdf_gt.cuda(),
                )
                * lambda3
            )

        occ_data_loss = loss_bce(
            b_occ,
            b_occ_gt.cuda(),
        )

        if lambda1 != 0:
            norm_loss = (
                loss_l2(
                    b_n,
                    b_n_gt.cuda(),
                )
                * lambda1
            )

        loss = sdf_data_loss + occ_data_loss + norm_loss

        # Regularization loss
        if l2reg:
            reg_loss = torch.mean(latent.pow(2))
            break_reg_loss = torch.mean(break_latent.pow(2))
            reg_loss = (reg_loss + break_reg_loss) * code_reg_lambda
            loss = loss + reg_loss

        if e % 10 == 0:
            loss_dict["epoch"].append(e)
            loss_dict["loss"].append(loss.item())
            loss_dict["sdf_data_loss"].append(sdf_data_loss.item())
            loss_dict["occ_data_loss"].append(occ_data_loss.item())
            loss_dict["norm_loss"].append(norm_loss.item())
            loss_dict["reg_loss"].append(reg_loss.item())
            loss_dict["mag"].append(torch.norm(latent).item())
            loss_dict["h_mag"].append(torch.norm(break_latent).item())
            logging.debug(
                "epoch: {:4d} | loss: {:1.5e} sdf_data_loss: {:1.5e} occ_data_loss: {:1.5e} norm_loss: {:1.5e} reg_loss: {:1.5e}".format(
                    loss_dict["epoch"][-1],
                    loss_dict["loss"][-1],
                    loss_dict["sdf_data_loss"][-1],
                    loss_dict["occ_data_loss"][-1],
                    loss_dict["norm_loss"][-1],
                    loss_dict["reg_loss"][-1],
                )
            )

        loss.backward()
        optimizer.step()

    return dict(loss_dict), torch.cat([latent, break_latent], dim=1)


def callback():
    global STATUS_INDICATOR
    global STATUS_COUNTER
    try:
        STATUS_INDICATOR.increment()
    except AttributeError:
        print("Completed: {}".format(STATUS_COUNTER))
        STATUS_COUNTER += 1


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
        default="ours_",
        type=str,
        help="",
    )
    arg_parser.add_argument(
        "--sampling_method",
        default=None,
        type=str,
        help="",
    )
    arg_parser.add_argument(
        "--loss_version",
        default=None,
        type=str,
        help="",
    )
    arg_parser.add_argument(
        "--lambda_reg",
        default=1e-4,
        type=float,
        help="Regularization lambda value.",
    )
    arg_parser.add_argument(
        "--learning_rate",
        default=5e-3,
        type=float,
        help="Regularization lambda value.",
    )
    arg_parser.add_argument(
        "--lambda1",
        default=0.0,
        type=float,
        help="",
    )
    arg_parser.add_argument(
        "--lambda2",
        default=0.0,
        type=float,
        help="",
    )
    arg_parser.add_argument(
        "--lambda3",
        default=0.0,
        type=float,
        help="",
    )
    arg_parser.add_argument(
        "--uniform_ratio",
        default=None,
        type=float,
        help="Uniform Ratio.",
    )
    arg_parser.add_argument(
        "--reconstruct_with_occ",
        action="store_true",
        default=False,
        help="",
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
        "--save_iter",
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
        "--mesh_only",
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

    lambda1 = args.lambda1
    lambda2 = args.lambda2
    lambda3 = args.lambda3
    logging.info("Using lambda1: {}".format(lambda1))
    logging.info("Using lambda2: {}".format(lambda2))
    logging.info("Using lambda3: {}".format(lambda3))

    num_samples = args.num_samples
    num_iterations = args.num_iters
    code_reg_lambda = args.lambda_reg
    lr = args.learning_rate
    name = args.name
    render_threads = args.render_threads
    overwrite_codes = args.overwrite_codes
    overwrite_meshes = args.overwrite_meshes
    overwrite_evals = args.overwrite_evals
    overwrite_renders = args.overwrite_renders
    threads = args.threads
    save_iter = args.save_iter
    uniform_ratio = args.uniform_ratio
    mesh_only = args.mesh_only
    sampling_method = args.sampling_method
    reconstruct_with_occ = args.reconstruct_with_occ
    assert sampling_method in [None, "surface_interior_only"]
    if uniform_ratio is None:
        uniform_ratio = specs["UniformRatio"]
    assert specs["UseNormals"]
    assert not specs["UseOccupancy"]

    clamp_dist = specs["ClampingDistance"]
    test_split_file = specs["TestSplit"]
    break_latent_size = specs["BreakCodeLength"]

    network_outputs = (0, 1, 2, 3)
    total_outputs = (0, 1, 2, 3)
    eval_pairs = [(0, 0), (1, 1), (2, 2)]  # ORDER MATTERS
    metrics_version = "v3.metrics.npy"
    metrics = [
        "chamfer",
        "connected_artifacts_score2",
        "normal_consistency",
        "connected_protrusion_error",
    ]
    composite = [(1, 2), (1, 3)]
    render_resolution = (200, 200)
    do_code_regularization = True

    assert specs["NetworkArch"] in [
        "decoder_z_lb_both_leaky_n_sdf"
    ], "wrong arch: {}".format(specs["NetworkArch"])

    network_kwargs = dict(
        decoder_kwargs=dict(
            latent_size=latent_size,
            tool_latent_size=break_latent_size,
            num_dims=3,
            do_code_regularization=do_code_regularization,
            **specs["NetworkSpecs"],
            **specs["SubnetSpecs"],
            return_occ=reconstruct_with_occ,
        ),
        decoder_constructor=arch.Decoder,
        experiment_directory=args.experiment_directory,
        checkpoint=args.checkpoint,
    )
    reconstruction_kwargs = dict(
        num_iterations=num_iterations,
        latent_size=latent_size,
        break_latent_size=break_latent_size,
        sampling_method=sampling_method,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        stat=0.01,  # [emp_mean,emp_var],
        clamp_dist=clamp_dist,
        num_samples=num_samples,
        lr=lr,
        l2reg=do_code_regularization,
        code_reg_lambda=code_reg_lambda,
        loss_version=args.loss_version,
    )

    # Get the data directory from environment variable
    test_split_file = test_split_file.replace("$DATADIR", os.environ["DATADIR"])
    data_source = specs["DataSource"].replace("$DATADIR", os.environ["DATADIR"])

    # Create and load the dataset
    if reconstruct_with_occ:
        signiture = [
            num_samples,
            num_iterations,
            lr,
            code_reg_lambda,
            lambda1,
            lambda2,
            lambda3,
            reconstruct_with_occ,
        ]
        lat_signiture = [
            num_samples,
            num_iterations,
            lr,
            code_reg_lambda,
            lambda1,
            lambda2,
            lambda3,
        ]
        mesh_kwargs = dict(
            dims=[256, 256, 256],
            level=0.5,
            gradient_direction="descent",
            batch_size=2**14,
        )
    else:
        signiture = [
            num_samples,
            num_iterations,
            lr,
            code_reg_lambda,
            lambda1,
            lambda2,
            lambda3,
        ]
        lat_signiture = None
        mesh_kwargs = dict(
            dims=[256, 256, 256],
            level=0.0,
            gradient_direction="ascent",
            batch_size=2**14,
        )

    reconstruction_handler = core.handler.ReconstructionHandler(
        experiment_directory=args.experiment_directory,
        dims=[256, 256, 256],
        name=name,
        checkpoint=args.checkpoint,
        overwrite=False,
        use_occ=specs["UseOccupancy"],
        signiture=signiture,
        lat_signiture=lat_signiture,
    )
    sdf_dataset = core.data.SamplesDataset(
        test_split_file,
        subsample=num_samples,
        uniform_ratio=uniform_ratio,
        use_occ=specs["UseOccupancy"],
        clamp_dist=clamp_dist,
        root=data_source,
        use_normals=specs["UseNormals"],
    )

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

    input_list, path_list = [], []
    if not mesh_only:
        for ii in reconstruct_list:

            # Generate the code if necessary
            path_code = reconstruction_handler.path_code(ii, create=True)
            if (not os.path.exists(path_code)) or overwrite_codes:
                if save_iter:
                    input_list.append(
                        dict(
                            test_sdf=sdf_dataset.get_broken_sample(ii),
                            iter_path=reconstruction_handler.path_values(
                                ii, 1, create=True
                            ),
                        )
                    )
                else:
                    input_list.append(
                        dict(
                            test_sdf=sdf_dataset.get_broken_sample(ii),
                        )
                    )
                path_list.append(path_code)

    # Spawn a threadpool to do reconstruction
    num_tasks = len(input_list)
    STATUS_INDICATOR = core.utils_multiprocessing.MultiprocessBar(num_tasks)

    with multiprocessing.Pool(threads) as pool:
        if not mesh_only:
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

        input_list, path_list = [], []
        for ii in reconstruct_list:

            # Generate the mesh if necessary
            for shape_idx in network_outputs:
                path_mesh = reconstruction_handler.path_mesh(ii, shape_idx, create=True)
                path_values = reconstruction_handler.path_values(ii, shape_idx)
                if (
                    not os.path.exists(path_mesh)
                    # or not os.path.exists(path_values)
                    or overwrite_meshes
                ):
                    sigmoid = False
                    if reconstruct_with_occ:
                        sigmoid = True
                        if shape_idx in [1, 2]:
                            sigmoid = False
                    if os.path.exists(reconstruction_handler.path_code(ii)):
                        input_list.append(
                            dict(
                                vec=reconstruction_handler.get_code(ii),
                                use_net=shape_idx,
                                # save_values=path_values,
                                sigmoid=sigmoid,
                            )
                        )
                        path_list.append(path_mesh)

        # Reconstruct meshes
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
                        core.utils_multiprocessing.mesh_chunk,
                        tuple(
                            (
                                input_list[start:end],
                                path_list[start:end],
                                core.reconstruct.create_mesh,
                                network_kwargs,
                                mesh_kwargs,
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
            try:
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
            except multiprocessing.pool.MaybeEncodingError:
                logging.error("Render could not be built, skipping")

    # Spins up a multiprocessed evaluator
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
        exclusion_list = np.array(sdf_dataset.num_components != 1)[
            np.array(reconstruct_list)
        ]

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
