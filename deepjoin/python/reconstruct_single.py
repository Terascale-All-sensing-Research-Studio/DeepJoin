import argparse
import json
import logging
import os
import random
import math
import multiprocessing

import trimesh
import torch
import numpy as np
from collections import defaultdict

import core

STATUS_INDICATOR = None


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
    STATUS_INDICATOR.increment()


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--input_mesh",
        required=True,
        help="gt fractured object, used only for rendering.",
    )
    arg_parser.add_argument(
        "--input_points",
        required=True,
        help="Sample points, specified as a .npz file.",
    )
    arg_parser.add_argument(
        "--input_sdf",
        required=True,
        help="Sample sdf, specified as a .npz file.",
    )
    arg_parser.add_argument(
        "--output_meshes",
        required=True,
        help="Path template to save the meshes to.",
    )
    arg_parser.add_argument(
        "--output_code",
        required=True,
        help="Path template to save the code to.",
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
        "--eq_warmup",
        default=1,
        type=float,
        help="Equality lambda value.",
    )
    arg_parser.add_argument(
        "--sup_cooldown",
        default=800,
        type=float,
        help="Equality lambda value.",
    )
    arg_parser.add_argument(
        "--lambda_eq",
        default=0.0001,
        type=float,
        help="Equality lambda value.",
    )
    arg_parser.add_argument(
        "--lambda_sup",
        default=0.0001,
        type=float,
        help="Equality lambda value.",
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
        "--uniform_ratio",
        default=None,
        type=float,
        help="Uniform Ratio.",
    )
    arg_parser.add_argument(
        "--ncol_method",
        default=None,
        type=str,
        help="Method by which to apply noncollapse constraint.",
    )
    arg_parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite code.",
    )
    arg_parser.add_argument(
        "--gif",
        action="store_true",
        default=False,
        help="",
    )
    core.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    core.configure_logging(args)

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
    threads = args.threads
    uniform_ratio = args.uniform_ratio
    if uniform_ratio is None:
        uniform_ratio = specs["UniformRatio"]
    assert specs["UseNormals"]
    assert not specs["UseOccupancy"]

    clamp_dist = specs["ClampingDistance"]
    test_split_file = specs["TestSplit"]
    break_latent_size = specs["BreakCodeLength"]

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
        ),
        decoder_constructor=arch.Decoder,
        experiment_directory=args.experiment_directory,
        checkpoint=args.checkpoint,
    )
    reconstruction_kwargs = dict(
        num_iterations=num_iterations,
        latent_size=latent_size,
        break_latent_size=break_latent_size,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        stat=0.01,  # [emp_mean,emp_var],
        clamp_dist=clamp_dist,
        num_samples=num_samples,
        lr=lr,
        l2reg=do_code_regularization,
        code_reg_lambda=code_reg_lambda,
    )
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
        sigmoid=False,
    )

    assert os.path.splitext(args.output_code)[-1] == ".pth"
    assert os.path.splitext(args.output_meshes)[-1] == ".obj"

    # Load the data
    xyz = np.load(args.input_points)["xyz"]
    sdf = np.expand_dims(np.load(args.input_sdf)["sdf"], axis=1)
    n = np.load(args.input_sdf)["n"]
    # sdf = core.sdf_to_occ(np.expand_dims(sdf, axis=1))
    assert len(xyz.shape) == 2 and len(sdf.shape) == 2 and len(sdf.shape) == 2

    # # Load the network
    try:
        decoder = core.load_network(**network_kwargs)
        decoder.eval()
    except RuntimeError:
        pass

    # Reconstruct the code
    if not os.path.exists(args.output_code) or args.overwrite:
        losses, code = reconstruct(
            test_sdf=[xyz, sdf, n],
            decoder=decoder,
            **reconstruction_kwargs,
        )
        core.saver(args.output_code, code)
    else:
        code = core.loader(args.output_code)

    mesh_path_list = [
        os.path.splitext(args.output_meshes)[0]
        + str(shape_idx)
        + os.path.splitext(args.output_meshes)[-1]
        for shape_idx in range(3)
    ]

    # Reconstruct the meshes
    mesh_list = []
    for shape_idx, path in enumerate(mesh_path_list):
        if not os.path.exists(path) or args.overwrite:
            try:
                mesh = core.reconstruct.create_mesh(
                    vec=code,
                    decoder=decoder,
                    use_net=shape_idx,
                    **mesh_kwargs,
                )
                mesh.export(path)
            except core.errors.IsosurfaceExtractionError:
                logging.info(
                    "Isosurface extraction error for shape: {}".format(shape_idx)
                )
                mesh = None
        else:
            mesh = core.loader(path)
        mesh_list.append(mesh)

    # Create a render of the the restoration object with gt fractured mesh

    DURATION = 5  # in seconds
    FRAME_RATE = 15
    RESOLUTION = (400, 400)
    ZOOM = 2.0
    num_renders = DURATION * FRAME_RATE

    if mesh_list[2] is not None:
        gt_mesh = core.loader(args.input_mesh)
        gt_mesh.fix_normals()
        if args.gif:
            core.saver(
                f_out=os.path.splitext(args.output_meshes)[0] + "_f_r.gif",
                data=core.create_gif_rot(
                    [
                        core.colorize_mesh_from_index_auto(gt_mesh, 1),
                        core.colorize_mesh_from_index_auto(mesh_list[2], 2),
                    ],
                    num_renders=num_renders,
                    resolution=RESOLUTION,
                    zoom=ZOOM,
                    bg_color=255,
                ),
                loop=0,
                duration=(1 / num_renders) * DURATION * 1000,
            )
        else:
            core.saver(
                f_out=os.path.splitext(args.output_meshes)[0] + "_f_r.png",
                data=core.render_mesh(
                    [
                        core.colorize_mesh_from_index_auto(gt_mesh, 1),
                        core.colorize_mesh_from_index_auto(mesh_list[2], 2),
                    ],
                    resolution=RESOLUTION,
                    ztrans=ZOOM,
                    bg_color=255,
                ),
            )
