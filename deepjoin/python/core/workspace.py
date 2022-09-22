import os
import json
import logging

import torch


model_params_subdir = "ModelParameters"
optimizer_params_subdir = "OptimizerParameters"
grad_scalar_params_subdir = "GradScalarParameters"
latent_codes_subdir = "LatentCodes"

reconstructions_subdir = "Reconstructions"
reconstruction_meshes_subdir = "Meshes"
reconstruction_renders_subdir = "Renders"
reconstruction_codes_subdir = "Codes"
reconstruction_stats_subdir = "Stats"

logs_filename = "Logs.pth"
logdict_filename = "LogDict.npy"
specifications_filename = "specs.json"


def load_experiment_specifications(experiment_directory):
    filename = os.path.join(experiment_directory, specifications_filename)
    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"specs.json"'.format(experiment_directory)
        )
    return json.load(open(filename))


def load_model_parameters(experiment_directory, checkpoint, decoder):
    filename = os.path.join(
        experiment_directory, model_params_subdir, checkpoint + ".pth"
    )
    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))
    data = torch.load(filename)
    decoder.load_state_dict(data["model_state_dict"])
    return data["epoch"]


def build_decoder(experiment_directory, experiment_specs):
    arch = __import__(
        "networks." + experiment_specs["NetworkArch"], fromlist=["Decoder"]
    )

    latent_size = experiment_specs["CodeLength"]
    decoder = arch.Decoder(latent_size, **experiment_specs["NetworkSpecs"]).cuda()
    return decoder


def load_decoder(
    experiment_directory, experiment_specs, checkpoint, data_parallel=True
):
    decoder = build_decoder(experiment_directory, experiment_specs)
    if data_parallel:
        decoder = torch.nn.DataParallel(decoder)
    epoch = load_model_parameters(experiment_directory, checkpoint, decoder)
    return (decoder, epoch)


def load_latent_vectors(experiment_directory, checkpoint):

    filename = os.path.join(
        experiment_directory, latent_codes_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include a latent code file"
            + " for checkpoint '{}'".format(experiment_directory, checkpoint)
        )

    data = torch.load(filename)

    if isinstance(data["latent_codes"], torch.Tensor):
        num_vecs = data["latent_codes"].size()[0]
        lat_vecs = []
        for i in range(num_vecs):
            lat_vecs.append(data["latent_codes"][i].cuda())
        return lat_vecs
    else:
        num_embeddings, embedding_dim = data["latent_codes"]["weight"].shape
        lat_vecs = torch.nn.Embedding(num_embeddings, embedding_dim)
        lat_vecs.load_state_dict(data["latent_codes"])
        return lat_vecs.weight.data.detach()


def get_model_params_dir(experiment_dir, create_if_nonexistent=False):
    dir = os.path.join(experiment_dir, model_params_subdir)
    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)
    return dir


def get_optimizer_params_dir(experiment_dir, create_if_nonexistent=False):
    dir = os.path.join(experiment_dir, optimizer_params_subdir)
    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)
    return dir


def get_latent_codes_dir(experiment_dir, create_if_nonexistent=False):
    dir = os.path.join(experiment_dir, latent_codes_subdir)
    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)
    return dir


def get_reconstructions_dir(
    experiment_directory, name=None, checkpoint="latest", create=False
):
    if name is None:
        dir = os.path.join(
            experiment_directory,
            reconstructions_subdir,
            str(checkpoint),
        )
    else:
        dir = os.path.join(
            experiment_directory,
            reconstructions_subdir,
            name + "@" + str(checkpoint),
        )
    if create and not os.path.isdir(dir):
        logging.debug("Creating reconstruction directory: {}".format(dir))
        os.makedirs(dir)
    return dir


def get_reconstructions_codes_dir(
    experiment_directory, name=None, checkpoint="latest", create=False
):
    dir = os.path.join(
        get_reconstructions_dir(
            experiment_directory=experiment_directory,
            name=name,
            checkpoint=checkpoint,
            create=create,
        ),
        reconstruction_codes_subdir,
    )
    if create and not os.path.isdir(dir):
        logging.debug("Creating reconstruction codes directory: {}".format(dir))
        os.makedirs(dir)
    return dir


def get_reconstructions_stats_dir(
    experiment_directory, name=None, checkpoint="latest", create=False
):
    dir = os.path.join(
        get_reconstructions_dir(
            experiment_directory=experiment_directory,
            name=name,
            checkpoint=checkpoint,
            create=create,
        ),
        reconstruction_stats_subdir,
    )
    if create and not os.path.isdir(dir):
        logging.debug("Creating reconstruction stats directory: {}".format(dir))
        os.makedirs(dir)
    return dir


def get_reconstructions_renders_dir(
    experiment_directory, name=None, checkpoint="latest", create=False
):
    dir = os.path.join(
        get_reconstructions_dir(
            experiment_directory=experiment_directory,
            name=name,
            checkpoint=checkpoint,
            create=create,
        ),
        reconstruction_renders_subdir,
    )
    if create and not os.path.isdir(dir):
        logging.debug("Creating reconstruction renders directory: {}".format(dir))
        os.makedirs(dir)
    return dir


def get_reconstructions_meshes_dir(
    experiment_directory, name=None, checkpoint="latest", create=False
):
    dir = os.path.join(
        get_reconstructions_dir(
            experiment_directory=experiment_directory,
            name=name,
            checkpoint=checkpoint,
            create=create,
        ),
        reconstruction_meshes_subdir,
    )
    if create and not os.path.isdir(dir):
        logging.debug("Creating reconstruction meshes directory: {}".format(dir))
        os.makedirs(dir)
    return dir
