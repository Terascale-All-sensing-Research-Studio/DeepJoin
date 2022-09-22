import os, sys
import subprocess
import logging

import torch
from sklearn.metrics import accuracy_score

import core

# == lr helpers ==
class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor, startat):
        self.initial = initial
        self.interval = interval
        self.factor = factor
        self.startat = startat

    def get_learning_rate(self, epoch):
        if epoch >= self.startat:
            return self.initial * (
                self.factor ** ((epoch - self.startat) // self.interval)
            )
        else:
            return self.initial


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


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
                    schedule_specs.get("StartAt", 0),
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


def adjust_learning_rate(lr_schedules, optimizer, epoch):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)


# == Eval helpers ==
def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


def append_performances(
    use_occ, logger, pred, gt, name, batch_size, use_round=True, use_sigmoid=True
):
    if use_occ:
        if use_sigmoid:
            pred = torch.sigmoid(pred.detach().cpu().type(torch.float))
        else:
            pred = pred.detach().cpu().type(torch.float)
        if use_round:
            pred = torch.round(pred)
        gt = gt.detach().cpu().type(torch.float)

        logger[name] += accuracy_score(pred.numpy(), gt.numpy()) / batch_size
    else:
        logger[name] += (
            torch.sum(torch.norm(pred - gt, dim=1)).detach().cpu().numpy() / batch_size
        )


def empirical_stat(latent_vecs, indices):
    lat_mat = torch.zeros(0).cuda()
    for ind in indices:
        lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
    mean = torch.mean(lat_mat, 0)
    var = torch.var(lat_mat, 0)
    return mean, var


def neptune_log_gt_samples(
    neptune_hook, num_samples, train_dataset=None, test_dataset=None
):
    for i in range(num_samples):
        for logname, name in zip(
            ["{}_c_gt", "{}_b_gt", "{}_r_gt"], ["complete", "broken", "restoration"]
        ):
            if train_dataset is not None:
                neptune_hook.log_image(
                    "train_" + logname.format(str(i)),
                    core.utils_2d.shape_to_pil(train_dataset.data[name][i]),
                )
            if test_dataset is not None:
                neptune_hook.log_image(
                    "test_" + logname.format(str(i)),
                    core.utils_2d.shape_to_pil(test_dataset.data[name][i]),
                )


def neptune_log_pred_samples(
    neptune_hook, num_samples, train_shape_list=None, test_shape_list=None
):
    for i in range(num_samples):
        for logname, idx in zip(["{}_c_pred", "{}_b_pred", "{}_r_pred"], range(3)):
            if test_shape_list is not None:
                neptune_hook.log_image(
                    "train_" + logname.format(str(i)),
                    core.utils_2d.shape_to_pil(test_shape_list[i][idx]),
                )
            if train_shape_list is not None:
                neptune_hook.log_image(
                    "test_" + logname.format(str(i)),
                    core.utils_2d.shape_to_pil(train_shape_list[i][idx]),
                )


def neptune_log_dict(neptune_hook, loss_logger, names=None):
    if names is None:
        for k, i in loss_logger.items():
            neptune_hook.log_metric(k, float(i))
    else:
        for k, i in zip(names, loss_logger.values()):
            neptune_hook.log_metric(k, float(i))


# == Logging helpers ==
def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


# == Other ==
def signal_handler(sig, frame):
    logging.info("Stopping early...")
    sys.exit(0)


def network_backup(experiment_directory, ssh_address):
    if ssh_address is None:
        return
    dir = os.path.abspath(experiment_directory)
    logging.info("Backing up to server: {}".format(["rsync", "-r", dir, ssh_address]))
    process = subprocess.Popen(["rsync", "-r", dir, ssh_address])
    process.wait()


def get_checkpoints(specs):
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
    return checkpoints
