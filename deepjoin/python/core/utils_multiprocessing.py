import logging
import os
import torch
import datetime
import multiprocessing

import numpy as np
import tqdm
import trimesh

import core
import core.errors as errors
import core.workspace as ws


def load_network(
    decoder_kwargs,
    decoder_constructor,
    experiment_directory,
    checkpoint,
):
    decoder = decoder_constructor(**decoder_kwargs)
    decoder = torch.nn.DataParallel(decoder)
    saved_model_state = torch.load(
        os.path.join(experiment_directory, ws.model_params_subdir, checkpoint + ".pth")
    )
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda()
    decoder.eval()
    return decoder


def reconstruct_chunk(
    chunk_inputs,
    chunk_paths,
    reconstruct_fn,
    network_kwargs,
    reconstruction_kwargs,
    overwrite=False,
    callback=None,
    save=True,
):
    decoder = load_network(**network_kwargs)
    decoder.eval()
    for input, path in zip(chunk_inputs, chunk_paths):

        if overwrite or not os.path.exists(path) or core.utils.file_is_old(path):
            unpacker = reconstruct_fn(
                decoder=decoder,
                **input,
                **reconstruction_kwargs,
            )
            loss, latent = unpacker
            try:
                if save:
                    # Save the latent vector
                    logging.debug("Reconstruct chunker saving to: {}".format(path))
                    torch.save(latent, path)

                    # Save the loss log
                    np.save(
                        core.handler.lossify_log_path(path),
                        loss,
                    )
            except PermissionError:
                logging.warning("Could not save: {}".format(path))

        else:
            logging.debug("Skipping {}".format(path))
        if callback is not None:
            callback()


def mesh_chunk(
    chunk_inputs,
    chunk_paths,
    mesh_fn,
    network_kwargs,
    mesh_kwargs,
    overwrite=False,
    callback=None,
    save=True,
):
    decoder = load_network(**network_kwargs)
    decoder.eval()
    for input, path in zip(chunk_inputs, chunk_paths):

        if overwrite or not os.path.exists(path) or core.utils.file_is_old(path):
            try:
                with torch.no_grad():
                    mesh = mesh_fn(
                        decoder=decoder,
                        **input,
                        **mesh_kwargs,
                    )
            except errors.IsosurfaceExtractionError:
                mesh = trimesh.Trimesh()
                logging.debug("Isosurface extraction error on sample: {}".format(path))
            except PermissionError:
                mesh = None
                logging.warning("Could not save: {}".format(path))

            try:
                if save:
                    if mesh is not None:
                        logging.debug("Mesh chunker saving to: {}".format(path))
                        mesh.export(path)
            except PermissionError:
                logging.warning("Could not save: {}".format(path))

        else:
            logging.debug("Skipping {}".format(path))
        if callback is not None:
            callback()


def mesh_chunk_no_decoder(
    chunk_inputs,
    chunk_paths,
    mesh_fn,
    mesh_kwargs,
    overwrite=False,
    callback=None,
    save=True,
):
    for input, path in zip(chunk_inputs, chunk_paths):

        if overwrite or not os.path.exists(path) or core.utils.file_is_old(path):
            try:
                mesh = mesh_fn(
                    **input,
                    **mesh_kwargs,
                )
            except errors.IsosurfaceExtractionError:
                mesh = trimesh.Trimesh()
                logging.debug("Isosurface extraction error on sample: {}".format(path))
            except PermissionError:
                mesh = None
                logging.warning("Could not save: {}".format(path))

            try:
                if mesh is not None:
                    if save:
                        logging.debug("Mesh chunker saving to: {}".format(path))
                        mesh.export(path)
            except PermissionError:
                logging.warning("Could not save: {}".format(path))
        else:
            logging.debug("Skipping {}".format(path))
        if callback is not None:
            callback()


class MultiprocessBar(object):
    def __init__(self, total, show_time=True):
        self.val = multiprocessing.Value("i", 0)
        self.show_time = show_time
        if self.show_time:
            date_time = datetime.datetime.now().strftime("%m/%d/%Y-%H:%M:%S")
            self.pbar = tqdm.tqdm(total=total, desc=date_time)

    def increment(self, n=1):
        date_time = datetime.datetime.now().strftime("%m/%d/%Y-%H:%M:%S")
        with self.val.get_lock():
            self.val.value += n
            self.pbar.n = self.val.value
            self.pbar.last_print_n = self.val.value
            if self.show_time:
                self.pbar.set_description(date_time)
            self.pbar.refresh()

    def close(self):
        self.pbar.close()

    def reset(self, n):
        with self.val.get_lock():
            self.val.value = 0
            self.pbar.n = 0
            self.pbar.reset(n)
