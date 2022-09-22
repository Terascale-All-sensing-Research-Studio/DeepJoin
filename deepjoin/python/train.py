import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import math
import time
import datetime
from collections import defaultdict

import socket
import tqdm

try:
    import neptune
except ImportError:
    pass
import numpy as np
from sklearn.metrics import accuracy_score
import core

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

import core.workspace as ws


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def dd():
    return []


def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules


def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = ws.get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = ws.get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        ws.get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


# TODO: duplicated in workspace
def load_latent_vectors(experiment_directory, filename, lat_vecs):

    full_filename = os.path.join(
        ws.get_latent_codes_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["latent_codes"])

    return data["epoch"]


def save_log_dict(experiment_directory, log_dict):
    np.save(
        os.path.join(experiment_directory, ws.logdict_filename),
        log_dict,
    )


def load_log_dict(experiment_directory):
    return np.load(
        os.path.join(experiment_directory, ws.logdict_filename),
        allow_pickle=True,
    ).item()


def save_logs(
    experiment_directory,
    loss_log,
    lr_log,
    timing_log,
    lat_mag_log,
    param_mag_log,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "learning_rate": lr_log,
            "timing": timing_log,
            "latent_magnitude": lat_mag_log,
            "param_magnitude": param_mag_log,
        },
        os.path.join(experiment_directory, ws.logs_filename),
    )


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, ws.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["learning_rate"],
        data["timing"],
        data["latent_magnitude"],
        data["param_magnitude"],
        data["epoch"],
    )


def clip_logs(loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch):

    iters_per_epoch = len(loss_log) // len(lr_log)

    loss_log = loss_log[: (iters_per_epoch * epoch)]
    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]
    lat_mag_log = lat_mag_log[:epoch]
    for n in param_mag_log:
        param_mag_log[n] = param_mag_log[n][:epoch]

    return (loss_log, lr_log, timing_log, lat_mag_log, param_mag_log)


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


# >>> begin update: get mean magnitude from a list of lat vecs
def get_mean_latent_vector_magnitude_list(latent_vector_list):
    mag_list = [torch.mean(torch.norm(lv, dim=1)) for lv in latent_vector_list]
    return np.array(mag_list).mean()


# >>> end update


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


def main_function(experiment_directory, continue_from, batch_split):

    logging.debug("running " + experiment_directory)

    specs = ws.load_experiment_specifications(experiment_directory)

    logging.info("Experiment description: \n" + specs["Description"])

    data_source = specs["DataSource"]

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    logging.debug(specs["NetworkSpecs"])

    # >>> begin update: added break lat vec
    latent_size = specs["CodeLength"]
    tool_latent_size = specs["BreakCodeLength"]
    use_break_loss = specs.get("BreakLoss", True)
    # >>> end update

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch):

        save_model(experiment_directory, "latest.pth", decoder, epoch)

        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)
        save_latent_vectors(
            experiment_directory, "latest_tool.pth", tool_lat_vecs, epoch
        )

        save_optimizer(experiment_directory, "latest_val.pth", optimizer_all_val, epoch)
        save_latent_vectors(experiment_directory, "latest_val.pth", lat_vecs_val, epoch)
        save_latent_vectors(
            experiment_directory, "latest_tool_val.pth", tool_lat_vecs_val, epoch
        )

    def save_checkpoints(epoch):

        save_model(experiment_directory, str(epoch) + ".pth", decoder, epoch)

        save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, str(epoch) + ".pth", lat_vecs, epoch)
        save_latent_vectors(
            experiment_directory, str(epoch) + "_tool.pth", tool_lat_vecs, epoch
        )

        save_optimizer(
            experiment_directory, str(epoch) + "_val.pth", optimizer_all_val, epoch
        )
        save_latent_vectors(
            experiment_directory, str(epoch) + "_val.pth", lat_vecs_val, epoch
        )
        save_latent_vectors(
            experiment_directory, str(epoch) + "_tool_val.pth", tool_lat_vecs_val, epoch
        )

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    use_rb_loss = specs["BRLoss"]
    lambda_sdf_loss = specs["LambdaSDFLoss"]
    lambda_norm_loss = specs["LambdaNormLoss"]
    sdf_norm_warmup = specs["SDFNormWarmup"]
    assert specs["UseNormals"]
    assert not specs["UseOccupancy"]
    # >>> begin update: these are no longer needed
    clamp_dist = specs["ClampingDistance"]
    val_every = specs.setdefault("ValEvery", None)
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True
    # >>> end update

    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)

    code_bound = get_spec_with_default(specs, "CodeBound", None)

    # >>> begin update: added a few passable arguments
    assert specs["NetworkArch"] in [
        "decoder_z_lb_both_leaky_n_sdf",
    ], "wrong arch: {}".format(specs["NetworkArch"])
    one_code_per_complete = get_spec_with_default(specs, "OneCodePerComplete", False)
    reg_loss_warmup = get_spec_with_default(specs, "CodeRegularizationWarmup", 100)
    # >>> end update

    # >>> begin update: we need to pass a few more things to the network
    decoder = arch.Decoder(
        latent_size=latent_size,
        tool_latent_size=tool_latent_size,
        num_dims=3,
        do_code_regularization=do_code_regularization,
        **specs["NetworkSpecs"],
        **specs["SubnetSpecs"],
    ).cuda()
    # >>> end update

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

    # if torch.cuda.device_count() > 1:
    decoder = torch.nn.DataParallel(decoder)

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)

    # >>> begin update: using our dataloader
    sdf_dataset = core.data.SamplesDataset(
        specs["TrainSplit"].replace("$DATADIR", os.environ["DATADIR"]),
        learned_breaks=True,
        subsample=num_samp_per_scene,
        uniform_ratio=specs["UniformRatio"],
        use_occ=specs["UseOccupancy"],
        one_code_per_complete=one_code_per_complete,
        train_columns=[0, 1, 2, 3],  # Need the complete and break
        clamp_dist=clamp_dist,
        use_normals=specs["UseNormals"],
    )
    if val_every is not None:
        val_dataset = core.data.SamplesDataset(
            specs["ValSplit"].replace("$DATADIR", os.environ["DATADIR"]),
            learned_breaks=True,
            subsample=num_samp_per_scene,
            uniform_ratio=specs["UniformRatio"],
            use_occ=specs["UseOccupancy"],
            one_code_per_complete=one_code_per_complete,
            train_columns=[0, 1, 2, 3],  # Need the complete and break
            clamp_dist=clamp_dist,
            use_normals=specs["UseNormals"],
        )
    # >>> end update

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )

    if val_every is not None:
        val_loader = data_utils.DataLoader(
            val_dataset,
            batch_size=scene_per_batch,
            shuffle=True,
            num_workers=num_data_loader_threads,
            drop_last=True,
        )

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(sdf_dataset)

    logging.info("There are {} scenes".format(num_scenes))

    logging.debug(decoder)

    # >>> begin update: added break lat vec
    lat_vecs = torch.nn.Embedding(
        sdf_dataset.num_instances, latent_size, max_norm=code_bound
    )
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )
    logging.debug(
        "initialized with mean magnitude {}".format(
            get_mean_latent_vector_magnitude(lat_vecs)
        )
    )

    lat_vecs_val = torch.nn.Embedding(
        val_dataset.num_instances, latent_size, max_norm=code_bound
    )
    torch.nn.init.normal_(
        lat_vecs_val.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )

    tool_lat_vecs = torch.nn.Embedding(
        sdf_dataset.num_instances, tool_latent_size, max_norm=code_bound
    )
    torch.nn.init.normal_(
        tool_lat_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0)
        / math.sqrt(tool_latent_size),
    )
    logging.debug(
        "initialized with mean break magnitude {}".format(
            get_mean_latent_vector_magnitude(tool_lat_vecs)
        )
    )

    tool_lat_vecs_val = torch.nn.Embedding(
        val_dataset.num_instances, tool_latent_size, max_norm=code_bound
    )
    torch.nn.init.normal_(
        tool_lat_vecs_val.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0)
        / math.sqrt(tool_latent_size),
    )

    # >>> end update

    # >>> begin update: using the bceloss
    loss_bce_with_logits = torch.nn.BCEWithLogitsLoss(reduction="sum")
    loss_bce = torch.nn.BCELoss(reduction="sum")
    loss_l1 = torch.nn.L1Loss(reduction="sum")
    loss_l2 = torch.nn.MSELoss(reduction="sum")
    # >>> end update

    # >>> begin update: added break lat vec
    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
            {
                "params": tool_lat_vecs.parameters(),
                "lr": lr_schedules[2].get_learning_rate(0),
            },
        ]
    )
    optimizer_all_val = torch.optim.Adam(
        [
            {
                "params": lat_vecs_val.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
            {
                "params": tool_lat_vecs_val.parameters(),
                "lr": lr_schedules[2].get_learning_rate(0),
            },
        ]
    )
    # >>> end update

    loss_log = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}
    log_dict = defaultdict(dd)

    start_epoch = 1

    # >>> begin update: added network backup
    backup_location = specs.setdefault("NetBackupLocation", None)
    core.train.network_backup(experiment_directory, backup_location)
    # >>> end update

    # >>> begin update: neptune logging
    neptune_name = specs.setdefault("NeptuneName", None)
    test_every = specs.setdefault("TestEvery", None)
    stop_netptune_after = get_spec_with_default(specs, "StopNeptuneAfter", 200)
    if neptune_name is not None:
        logging.info("Logging to neptune project: {}".format(neptune_name))
        neptune.init(
            project_qualified_name=neptune_name,
            api_token=os.getenv("NEPTUNE_API_TOKEN"),
        )
        params = specs
        params.update(
            {
                "hostname": str(socket.gethostname()),
                "experiment_dir": os.path.basename(experiment_directory),
                "device count": str(int(torch.cuda.device_count())),
                "loader threads": str(int(num_data_loader_threads)),
                "torch threads": str(int(torch.get_num_threads())),
            }
        )
        neptune.create_experiment(params=params)
    # >>> end update

    if continue_from is not None:

        logging.info('continuing from "{}"'.format(continue_from))

        lat_epoch = load_latent_vectors(
            experiment_directory, continue_from + ".pth", lat_vecs
        )
        lat_epoch = load_latent_vectors(
            experiment_directory, continue_from + "_tool.pth", tool_lat_vecs
        )
        lat_epoch = load_latent_vectors(
            experiment_directory, continue_from + "_val.pth", lat_vecs_val
        )
        lat_epoch = load_latent_vectors(
            experiment_directory, continue_from + "_tool_val.pth", tool_lat_vecs_val
        )

        optimizer_epoch = load_optimizer(
            experiment_directory, continue_from + ".pth", optimizer_all
        )
        optimizer_epoch = load_optimizer(
            experiment_directory, continue_from + "_val.pth", optimizer_all_val
        )

        model_epoch = ws.load_model_parameters(
            experiment_directory, continue_from, decoder
        )

        loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, log_epoch = load_logs(
            experiment_directory
        )

        log_dict = load_log_dict(experiment_directory)
        try:
            log_dict["epoch"].append(None)
            log_dict["epoch"].pop()
        except TypeError:
            log_dict = defaultdict(dd)

        if not log_epoch == model_epoch:
            loss_log, lr_log, timing_log, lat_mag_log, param_mag_log = clip_logs(
                loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, model_epoch
            )

        if not (model_epoch == optimizer_epoch and model_epoch == lat_epoch):
            raise RuntimeError(
                "epoch mismatch: {} vs {} vs {} vs {}".format(
                    model_epoch, optimizer_epoch, lat_epoch, log_epoch
                )
            )

        start_epoch = model_epoch + 1

        logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))

    logging.info(
        "Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())
        )
    )
    logging.info(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings,
            lat_vecs.embedding_dim,
        )
    )
    # >>> being update: added break vecs
    logging.info(
        "Number of tool shape code parameters: {} (# codes {}, code dim {})".format(
            tool_lat_vecs.num_embeddings * tool_lat_vecs.embedding_dim,
            tool_lat_vecs.num_embeddings,
            tool_lat_vecs.embedding_dim,
        )
    )
    # >>> end update

    # >>> being update: added tqdm indicator

    pbar = tqdm.tqdm(
        range(start_epoch, num_epochs + 1), initial=start_epoch, total=num_epochs
    )

    # >>> being update: added tqdm indicator
    for epoch in pbar:
        date_time = datetime.datetime.now().strftime("%m/%d/%Y-%H:%M:%S")
        pbar.set_description(date_time)
        start = time.time()

        # decoder.train()

        adjust_learning_rate(lr_schedules, optimizer_all, epoch)

        # >>> being update: data is in a slightly different format

        for phase, loader, lvecs, tvecs, optim in zip(
            ["train", "val"],
            [sdf_loader, val_loader],
            [lat_vecs, lat_vecs_val],
            [tool_lat_vecs, tool_lat_vecs_val],
            [optimizer_all, optimizer_all_val],
        ):

            if phase == "val":
                decoder.eval()
            else:
                decoder.train()

            ticker = 0

            for data, inds in loader:

                normal_inds = 5

                for d in range(len(data)):
                    data[d] = data[d].reshape(-1, data[d].shape[-1])
                num_sdf_samples = data[0].shape[0]

                # Disambiguate pts
                pts = data[0]
                pts.requires_grad = False
                normals = data[1:normal_inds]
                gts = data[normal_inds:]
                for d in range(len(normals)):
                    normals[d].requires_grad = False
                    gts[d].requires_grad = False

                # Disambiguate indices
                indices = inds[0]
                tool_indices = inds[1]

                # Chunk points
                xyz = pts.type(torch.float)
                xyz = torch.chunk(xyz, batch_split)

                # Chunk points
                xyz = pts.type(torch.float)
                xyz = torch.chunk(xyz, batch_split)
                for d in range(len(gts)):
                    normals[d] = normals[d].type(torch.float)
                    normals[d] = torch.chunk(normals[d], batch_split)
                    gts[d] = gts[d].type(torch.float)
                    gts[d] = torch.chunk(gts[d], batch_split)

                # Chunk indices
                indices = torch.chunk(
                    indices.flatten(),
                    batch_split,
                )
                tool_indices = torch.chunk(
                    tool_indices.flatten(),
                    batch_split,
                )
                # >>> end update

                batch_loss = 0.0

                optim.zero_grad()

                for i in range(batch_split):

                    # c_occ, b_occ, r_occ, t_occ = [
                    #     core.reconstruct.tensor_sdf_to_occ(g[i]) for g in gts
                    # ]
                    # c_sdf, b_sdf, r_sdf, t_sdf = [
                    #     g[i] for g in gts
                    # ]
                    # c_n, b_n, r_n, t_n = [
                    #     g[i] for g in normals
                    # ]

                    # pts = xyz[i]

                    # core.vis.plot_samples(
                    #     (pts[:num_samp_per_scene, :], b_sdf[:num_samp_per_scene, :]),
                    # n_plots=16, min_max=(-0.1, 0.1)).savefig("b_sdf_gt.png")
                    # core.vis.plot_samples(
                    #     (pts[:num_samp_per_scene, :], r_sdf[:num_samp_per_scene, :]),
                    # n_plots=16, min_max=(-0.1, 0.1)).savefig("r_sdf_gt.png")
                    # core.vis.plot_samples(
                    #     (pts[:num_samp_per_scene, :], b_n[:num_samp_per_scene, :]),
                    # n_plots=16, min_max=(-0.1, 0.1)).savefig("b_n_gt.png")
                    # core.vis.plot_samples(
                    #     (pts[:num_samp_per_scene, :], r_n[:num_samp_per_scene, :]),
                    # n_plots=16, min_max=(-0.1, 0.1)).savefig("r_n_gt.png")

                    # idxs = torch.logical_or(
                    #     t_occ > 0.5, # inside break
                    #     c_sdf < -t_sdf, # max(sdf_C, -sdf_T)
                    # ).type(torch.long).squeeze()
                    # sdfs = torch.cat(
                    #     (c_sdf.unsqueeze(0), -t_sdf.unsqueeze(0)
                    # ), axis=0)
                    # b_sdf = sdfs[idxs, torch.arange(num_sdf_samples), :] # note the negative sign
                    # ns = torch.cat(
                    #     (c_n.unsqueeze(0), t_n.unsqueeze(0)
                    # ), axis=0)
                    # b_n = ns[idxs, torch.arange(num_sdf_samples), :]

                    # core.vis.plot_samples(
                    #     (pts[:num_samp_per_scene, :], b_sdf[:num_samp_per_scene, :]),
                    # n_plots=16, min_max=(-0.1, 0.1)).savefig("b_sdf_pred.png")
                    # core.vis.plot_samples(
                    #     (pts[:num_samp_per_scene, :], b_n[:num_samp_per_scene, :]),
                    # n_plots=16, min_max=(-0.1, 0.1)).savefig("b_n_pred.png")

                    # idxs = torch.logical_or(
                    #     t_occ <= 0.5, # outside break
                    #     c_sdf < t_sdf,  # max(sdf_C, sdf_T)
                    # ).type(torch.long).squeeze()
                    # sdfs = torch.cat(
                    #     (c_sdf.unsqueeze(0), t_sdf.unsqueeze(0)
                    # ), axis=0)
                    # r_sdf = sdfs[idxs, torch.arange(num_sdf_samples), :]
                    # ns = torch.cat(
                    #     (c_n.unsqueeze(0), -t_n.unsqueeze(0)
                    # ), axis=0)
                    # r_n = ns[idxs, torch.arange(num_sdf_samples), :]

                    # core.vis.plot_samples(
                    #     (pts[:num_samp_per_scene, :], r_sdf[:num_samp_per_scene, :]),
                    # n_plots=16, min_max=(-0.1, 0.1)).savefig("r_sdf_pred.png")
                    # core.vis.plot_samples(
                    #     (pts[:num_samp_per_scene, :], r_n[:num_samp_per_scene, :]),
                    # n_plots=16, min_max=(-0.1, 0.1)).savefig("r_n_pred.png")
                    # exit()

                    batch_vecs = lvecs(indices[i])
                    tool_batch_vecs = tvecs(tool_indices[i])

                    input = torch.cat([batch_vecs, tool_batch_vecs, xyz[i]], dim=1)

                    (
                        c_x,
                        b_x,
                        r_x,
                        t_x,
                        c_sdf,
                        b_sdf,
                        r_sdf,
                        t_sdf,
                        c_n,
                        b_n,
                        r_n,
                        t_n,
                    ) = decoder(input.cuda())

                    if enforce_minmax:
                        c_sdf = torch.clamp(c_sdf, minT, maxT)
                        b_sdf = torch.clamp(b_sdf, minT, maxT)
                        r_sdf = torch.clamp(r_sdf, minT, maxT)
                        t_sdf = torch.clamp(t_sdf, minT, maxT)

                    cn_gt, bn_gt, rn_gt, tn_gt = [n[i].cuda() for n in normals]

                    c_gt, b_gt, r_gt, t_gt = [
                        core.reconstruct.tensor_sdf_to_occ(g[i]).cuda() for g in gts
                    ]

                    loss_c = (loss_bce_with_logits(c_x, c_gt)) / num_sdf_samples
                    chunk_loss_occ = loss_c

                    c_n_loss = (
                        (loss_l2(c_n, cn_gt)) / num_sdf_samples * lambda_norm_loss
                    )
                    norm_loss = c_n_loss

                    if use_rb_loss:
                        loss_rb = (
                            loss_bce(b_x, b_gt) + loss_bce(r_x, r_gt)
                        ) / num_sdf_samples
                        chunk_loss_occ = chunk_loss_occ + loss_rb

                        rb_n_loss = (
                            (loss_l2(b_n, bn_gt) + loss_l2(r_n, rn_gt))
                            / num_sdf_samples
                            * lambda_norm_loss
                        )
                        norm_loss = norm_loss + rb_n_loss

                    if use_break_loss:
                        loss_t = loss_bce_with_logits(t_x, t_gt) / num_sdf_samples
                        chunk_loss_occ = chunk_loss_occ + loss_t

                        t_n_loss = (
                            (loss_l2(t_n, tn_gt)) / num_sdf_samples * lambda_norm_loss
                        )
                        norm_loss = norm_loss + t_n_loss

                    c_gt, b_gt, r_gt, t_gt = [g[i].cuda() for g in gts]

                    loss_c_sdf = (
                        (loss_l1(c_sdf, c_gt)) / num_sdf_samples * lambda_sdf_loss
                    )
                    chunk_loss_sdf = loss_c_sdf

                    if use_rb_loss:
                        loss_rb_sdf = (
                            (loss_l1(b_sdf, b_gt) + loss_l1(r_sdf, r_gt))
                            / num_sdf_samples
                            * lambda_sdf_loss
                        )
                        chunk_loss_sdf = chunk_loss_sdf + loss_rb_sdf

                    if use_break_loss:
                        loss_t_sdf = (
                            loss_l1(t_sdf, t_gt) / num_sdf_samples * lambda_sdf_loss
                        )
                        chunk_loss_sdf = chunk_loss_sdf + loss_t_sdf

                    # Apply warmups
                    chunk_loss_sdf = chunk_loss_sdf * min(1, epoch / sdf_norm_warmup)
                    norm_loss = norm_loss * min(1, epoch / sdf_norm_warmup)

                    chunk_loss = chunk_loss_occ + chunk_loss_sdf + norm_loss

                    if do_code_regularization:
                        l1_size_loss = torch.sum(
                            torch.norm(batch_vecs, dim=1)
                        ) + torch.sum(torch.norm(tool_batch_vecs, dim=1))

                        reg_loss = (
                            code_reg_lambda
                            * min(1, epoch / reg_loss_warmup)
                            * l1_size_loss
                        ) / num_sdf_samples

                        chunk_loss = chunk_loss + reg_loss.cuda()

                    log_dict["epoch"].append(epoch)
                    log_dict[phase + "_loss_total"].append(chunk_loss.item())
                    log_dict[phase + "_loss_occ"].append(chunk_loss_occ.item())
                    log_dict[phase + "_loss_sdf"].append(chunk_loss_sdf.item())
                    log_dict[phase + "_loss_n"].append(norm_loss.item())
                    log_dict[phase + "_loss_reg"].append(reg_loss.item())
                    log_dict[phase + "_loss_c_occ"].append(loss_c.item())
                    log_dict[phase + "_loss_c_sdf"].append(loss_c_sdf.item())
                    log_dict[phase + "_loss_c_n"].append(c_n_loss.item())
                    if use_rb_loss:
                        log_dict[phase + "_loss_rb_occ"].append(loss_rb.item())
                        log_dict[phase + "_loss_rb_sdf"].append(loss_rb_sdf.item())
                        log_dict[phase + "_loss_rb_n"].append(rb_n_loss.item())
                    if use_break_loss:
                        log_dict[phase + "_loss_t_occ"].append(loss_t.item())
                        log_dict[phase + "_loss_t_sdf"].append(loss_t_sdf.item())
                        log_dict[phase + "_loss_t_n"].append(t_n_loss.item())

                    if (neptune_name is not None) and (ticker == 0):
                        neptune.log_metric(phase + "_loss_total", chunk_loss.item())

                        neptune.log_metric(phase + "_loss_occ", chunk_loss_occ.item())
                        neptune.log_metric(phase + "_loss_sdf", chunk_loss_sdf.item())
                        neptune.log_metric(phase + "_loss_n", norm_loss.item())
                        neptune.log_metric(phase + "_loss_reg", reg_loss.item())

                        neptune.log_metric(phase + "_loss_c_occ", loss_c.item())
                        neptune.log_metric(phase + "_loss_c_sdf", loss_c_sdf.item())
                        neptune.log_metric(phase + "_loss_c_n", c_n_loss.item())

                        if use_rb_loss:
                            neptune.log_metric(phase + "_loss_rb_occ", loss_rb.item())
                            neptune.log_metric(
                                phase + "_loss_rb_sdf", loss_rb_sdf.item()
                            )
                            neptune.log_metric(phase + "_loss_rb_n", rb_n_loss.item())

                        if use_break_loss:
                            neptune.log_metric(phase + "_loss_t_occ", loss_t.item())
                            neptune.log_metric(phase + "_loss_t_sdf", loss_t_sdf.item())
                            neptune.log_metric(phase + "_loss_t_n", t_n_loss.item())

                        _error_cn = torch.abs(
                            c_n.cpu().detach() - normals[0][i],
                        ).mean()
                        neptune.log_metric(phase + "_error_cn", _error_cn)
                        _error_bn = torch.abs(
                            b_n.cpu().detach() - normals[1][i],
                        ).mean()
                        neptune.log_metric(phase + "_error_bn", _error_bn)
                        _error_rn = torch.abs(
                            r_n.cpu().detach() - normals[2][i],
                        ).mean()
                        neptune.log_metric(phase + "_error_rn", _error_rn)
                        _error_tn = torch.abs(
                            t_n.cpu().detach() - normals[3][i],
                        ).mean()
                        neptune.log_metric(phase + "_error_tn", _error_tn)

                        c_gt, b_gt, r_gt, t_gt = [
                            core.reconstruct.tensor_sdf_to_occ(g[i]) for g in gts
                        ]
                        _accuracy_c_occ = accuracy_score(
                            torch.sigmoid(c_x).cpu().detach().round().numpy(),
                            c_gt.numpy(),
                        )
                        neptune.log_metric(phase + "_accuracy_c_occ", _accuracy_c_occ)
                        _accuracy_b_occ = accuracy_score(
                            b_x.cpu().detach().round().numpy(),
                            b_gt.numpy(),
                        )
                        neptune.log_metric(phase + "_accuracy_b_occ", _accuracy_b_occ)
                        _accuracy_r_occ = accuracy_score(
                            r_x.cpu().detach().round().numpy(),
                            r_gt.numpy(),
                        )
                        neptune.log_metric(phase + "_accuracy_r_occ", _accuracy_r_occ)
                        _accuracy_t_occ = accuracy_score(
                            torch.sigmoid(t_x).cpu().detach().round().numpy(),
                            t_gt.numpy(),
                        )
                        neptune.log_metric(phase + "_accuracy_t_occ", _accuracy_t_occ)

                        c_gt, b_gt, r_gt, t_gt = [g[i] for g in gts]
                        _error_c_sdf = torch.abs(
                            c_sdf.cpu().detach() - c_gt.numpy(),
                        ).mean()
                        neptune.log_metric(phase + "_error_c_sdf", _error_c_sdf)
                        _error_b_sdf = torch.abs(
                            b_sdf.cpu().detach() - b_gt.numpy(),
                        ).mean()
                        neptune.log_metric(phase + "_error_b_sdf", _error_b_sdf)
                        _error_r_sdf = torch.abs(
                            r_sdf.cpu().detach() - r_gt.numpy(),
                        ).mean()
                        neptune.log_metric(phase + "_error_r_sdf", _error_r_sdf)
                        _error_t_sdf = torch.abs(
                            t_sdf.cpu().detach() - t_gt.numpy(),
                        ).mean()
                        neptune.log_metric(phase + "_error_t_sdf", _error_t_sdf)
                    # >>> end update

                    chunk_loss.backward()
                    batch_loss += chunk_loss.item()

                    ticker += 1

                logging.debug("loss = {}".format(batch_loss))
                loss_log.append(batch_loss)

                if (grad_clip is not None) and (phase != "val"):
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

                optim.step()

        # Log some images
        if (neptune_name is not None) and (test_every is not None):
            if (epoch % test_every) == 0:
                pts = xyz[-1][:num_samp_per_scene, :].cpu().detach().numpy()
                neptune.log_image(
                    "c_occ.jpg".format(epoch),
                    core.plt2numpy(
                        core.vis.plot_samples(
                            (
                                pts,
                                torch.sigmoid(c_x)[:num_samp_per_scene, :]
                                .cpu()
                                .detach()
                                .numpy(),
                            ),
                            n_plots=16,
                        )
                    ).astype(float)
                    / 255,
                )
                neptune.log_image(
                    "r_occ.jpg".format(epoch),
                    core.plt2numpy(
                        core.vis.plot_samples(
                            (
                                pts,
                                r_x[:num_samp_per_scene, :].cpu().detach().numpy(),
                            ),
                            n_plots=16,
                        )
                    ).astype(float)
                    / 255,
                )
                neptune.log_image(
                    "b_occ.jpg".format(epoch),
                    core.plt2numpy(
                        core.vis.plot_samples(
                            (
                                pts,
                                b_x[:num_samp_per_scene, :].cpu().detach().numpy(),
                            ),
                            n_plots=16,
                        )
                    ).astype(float)
                    / 255,
                )
                neptune.log_image(
                    "t_occ.jpg".format(epoch),
                    core.plt2numpy(
                        core.vis.plot_samples(
                            (
                                pts,
                                torch.sigmoid(t_x)[:num_samp_per_scene, :]
                                .cpu()
                                .detach()
                                .numpy(),
                            ),
                            n_plots=16,
                        )
                    ).astype(float)
                    / 255,
                )
                neptune.log_image(
                    "c_sdf.jpg".format(epoch),
                    core.plt2numpy(
                        core.vis.plot_samples(
                            (
                                pts,
                                c_sdf[:num_samp_per_scene, :].cpu().detach().numpy(),
                            ),
                            n_plots=16,
                        )
                    ).astype(float)
                    / 255,
                )
                neptune.log_image(
                    "r_sdf.jpg".format(epoch),
                    core.plt2numpy(
                        core.vis.plot_samples(
                            (
                                pts,
                                r_sdf[:num_samp_per_scene, :].cpu().detach().numpy(),
                            ),
                            n_plots=16,
                        )
                    ).astype(float)
                    / 255,
                )
                neptune.log_image(
                    "b_sdf.jpg".format(epoch),
                    core.plt2numpy(
                        core.vis.plot_samples(
                            (
                                pts,
                                b_sdf[:num_samp_per_scene, :].cpu().detach().numpy(),
                            ),
                            n_plots=16,
                        )
                    ).astype(float)
                    / 255,
                )
                neptune.log_image(
                    "t_sdf.jpg".format(epoch),
                    core.plt2numpy(
                        core.vis.plot_samples(
                            (
                                pts,
                                t_sdf[:num_samp_per_scene, :].cpu().detach().numpy(),
                            ),
                            n_plots=16,
                        )
                    ).astype(float)
                    / 255,
                )
                plt.close("all")

        end = time.time()

        seconds_elapsed = end - start
        timing_log.append(seconds_elapsed)

        lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])

        lat_mag_log.append(get_mean_latent_vector_magnitude(lat_vecs))

        append_parameter_magnitudes(param_mag_log, decoder)

        # >>> begin update: neptune logging
        if neptune_name is not None:
            neptune.log_metric("z mag", get_mean_latent_vector_magnitude(lat_vecs))
            neptune.log_metric("t mag", get_mean_latent_vector_magnitude(tool_lat_vecs))
            neptune.log_metric("time", seconds_elapsed)
        # >>> end update

        if epoch in checkpoints:
            save_checkpoints(epoch)

            # >>> begin update: added network backup
            if backup_location is not None:
                core.train.network_backup(experiment_directory, backup_location)
            # >>> end update

        if epoch % log_frequency == 0:

            save_latest(epoch)
            save_logs(
                experiment_directory,
                loss_log,
                lr_log,
                timing_log,
                lat_mag_log,
                param_mag_log,
                epoch,
            )
            save_log_dict(experiment_directory, log_dict)


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )
    core.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    core.configure_logging(args)
    logging.info(args)

    main_function(args.experiment_directory, args.continue_from, int(args.batch_split))
