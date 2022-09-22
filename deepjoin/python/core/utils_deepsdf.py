import logging

import torch
import trimesh
import numpy as np
import skimage.measure

import core


def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]
    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat.cuda(), queries.cuda()], 1)

    sdf = decoder(inputs)
    return sdf


def create_restoration_occ(
    pred_values,
    gt_values,
    save_values=None,
    padding=0.1,
    N=256,
    befN=None,
    level=0.5,
    downscale=None,
):
    if isinstance(gt_values, str):
        gt_values = np.load(gt_values)["occ"]
    if isinstance(pred_values, str):
        pred_values = np.load(pred_values)

    dims = (N, N, N)
    if downscale is not None:
        gt_values = gt_values.reshape((befN, befN, befN))
        gt_values = gt_values[
            int(downscale / 2) :: downscale,
            int(downscale / 2) :: downscale,
            int(downscale / 2) :: downscale,
        ]
        pred_values = pred_values.reshape((N, N, N))

        qp = core.get_query_points((befN, befN, befN))
        vgo = abs(qp[int(downscale / 2), 2])
        voxel_grid_origin = [-vgo, -vgo, -vgo]
        voxel_size = abs(
            qp[int(downscale / 2), 2] - qp[int(downscale / 2) + downscale, 2]
        )
    else:
        pred_values, gt_values = pred_values.reshape((N, N, N)), gt_values.reshape(
            (N, N, N)
        )
        voxel_grid_origin = [-(0.5 + (padding)) for _ in dims]
        voxel_size = (1 + (padding * 2)) / N

    # These values have to be known and have to be correct

    pred_values = np.clip(pred_values, 0, 1)

    restoration_values = pred_values - gt_values

    if save_values is not None:
        logging.debug("Saving values to {}".format(save_values))
        np.save(save_values, restoration_values)

    return values2mesh(
        restoration_values,
        voxel_grid_origin,
        voxel_size,
        gradient_direction="descent",
        level=level,
        flipxy=True,
    )


def create_restoration(
    pred_values,
    gt_values,
    flip_threshold=False,
    save_values=None,
    thresh=-0.01,
    padding=0.1,
    N=256,
    level=0.5,
):
    if isinstance(gt_values, str):
        gt_values = np.load(gt_values)["occ"]
    if isinstance(pred_values, str):
        pred_values = np.load(pred_values)
    pred_values, gt_values = pred_values.reshape((N, N, N)), gt_values.reshape(
        (N, N, N)
    )

    # These values have to be known and have to be correct
    dims = (N, N, N)
    voxel_grid_origin = [-(0.5 + (padding)) for _ in dims]
    voxel_size = (1 + (padding * 2)) / N

    pred_values = core.data.sdf_to_occ_grid_threshold(
        pred_values, thresh, flip_threshold
    )
    restoration_values = pred_values - gt_values

    if save_values is not None:
        logging.debug("Saving values to {}".format(save_values))
        np.save(save_values, restoration_values)

    return values2mesh(
        restoration_values,
        voxel_grid_origin,
        voxel_size,
        gradient_direction="descent",
        level=level,
        flipxy=True,
    )


def create_mesh(
    decoder,
    latent_vec,
    N=256,
    max_batch=32**3,
    padding=0.1,
    save_values=None,
):
    decoder.eval()

    # Build the grid of points
    dims = (N, N, N)
    grid_pts = np.meshgrid(
        *[np.linspace(0, 1.0 + (padding * 2), d) - (0.5 + padding) for d in dims]
    )
    samples = np.vstack([p.flatten() for p in grid_pts]).T

    # Add another dimension to hold the predicted values
    num_samples = samples.shape[0]
    samples = np.hstack((samples, np.zeros((num_samples, 1))))

    # Compute the origin and size
    voxel_grid_origin = [-(0.5 + (padding)) for _ in dims]
    voxel_size = (1 + (padding * 2)) / N

    # Decode samples
    samples = torch.from_numpy(samples).type(torch.float)
    samples.requires_grad = False
    head = 0
    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        samples[head : min(head + max_batch, num_samples), 3] = (
            decode_sdf(decoder, latent_vec, sample_subset).squeeze(1).detach().cpu()
        )
        head += max_batch
    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)
    sdf_values = sdf_values.data.cpu().numpy()

    # Optionally save the raw values
    if save_values is not None:
        logging.debug("Saving values to {}".format(save_values))
        np.save(save_values, sdf_values)

    # Convert to a mesh
    return values2mesh(
        sdf_values,
        voxel_grid_origin,
        voxel_size,
        gradient_direction="ascent",
        flipxy=True,
    )


def values2mesh(
    sdf_values,
    voxel_grid_origin,
    voxel_size,
    gradient_direction="descent",
    level=0.0,
    flipxy=False,
):
    try:
        verts, faces, _, _ = skimage.measure.marching_cubes(
            sdf_values,
            level=level,
            spacing=[voxel_size] * 3,
            gradient_direction=gradient_direction,
        )
    except (ValueError, RuntimeError):
        raise core.errors.IsosurfaceExtractionError

    if flipxy:
        verts = np.hstack(
            [np.expand_dims(v, axis=1) for v in (verts[:, 1], verts[:, 0], verts[:, 2])]
        )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    return trimesh.Trimesh(vertices=mesh_points, faces=faces)
