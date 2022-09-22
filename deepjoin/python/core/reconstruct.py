import logging

import torch
import trimesh
import numpy as np
import skimage.measure

import core
import core.errors as errors


def get_latvecs(decoder, lat_vec):
    """
    Extract latent vectors (h and g).
    """
    xyz = torch.zeros((1, 3))
    try:
        net_input = torch.cat([lat_vec.cuda(), xyz.cuda()], 1).cuda()
    except IndexError:
        net_input = torch.cat([lat_vec.unsqueeze(0).cuda(), xyz.cuda()], 1).cuda()
    _, _, _, h_code, g_code = decoder(net_input)
    return h_code, g_code


def get_query_points(dims=(256, 256, 256), padding=0.1):
    """Return a grid of query points"""
    grid_pts = np.meshgrid(
        *[np.linspace(0, 1.0 + (padding * 2), d) - (0.5 + padding) for d in dims]
    )
    return np.vstack([p.flatten() for p in grid_pts]).T


def tensor_sdf_to_occ(data):
    return torch.clamp(torch.sgn(-data), -1, 0) + 1


def tensor_sdf_to_udf(data):
    return torch.abs(data)


def tensor_udf_to_sdf(udf, occ):
    return udf * (((occ.round() * 2) - 1) * -1)


def decode_shape(
    vec,
    decoder,
    dims,
    use_net=1,
    batch_size=2**16,
    padding=0.0,
    reshape=True,
    sigmoid=True,
):
    """
    Return a decoded shape over a grid of query points.

    Args:
        vec: Latent vector.
        decoder: Decoder model.
        dims: Tuple of dimensions.
        net: Net to use for inference {0, 1, 2}.
        batch_size: Max points that can be loaded into the gpu at a time.
        padding: Expand the grid by this much outside of a unit cube.
        reshape: Reshape the resulting values into a grid, ie for a voxel.
    """
    # Get an array of query points
    grid_pts = np.meshgrid(
        *[np.linspace(0, 1.0 + (padding * 2), d) - (0.5 + padding) for d in dims]
    )
    query_pts = np.vstack([p.flatten() for p in grid_pts]).T

    # Get the values
    values = decode_samples(
        vec=vec,
        decoder=decoder,
        pts=query_pts,
        use_net=use_net,
        batch_size=batch_size,
        sigmoid=sigmoid,
    )

    # Reshape the values into an n-dimensional grid (image, voxel, etc)
    if reshape:
        return values.reshape([d for d in dims]).T
    return values


def decode_samples(vec, decoder, pts, use_net=1, batch_size=2**16, sigmoid=True):
    """
    Return a decoded shape over a vector of query points

    Args:
        vec: Latent vector.
        decoder: Decoder model.
        pts: Points to feed to the decoder.
        use_net: Net to use for inference {0, 1, 2}.
        batch_size: Max points that can be loaded into the gpu at a time.
        sigmoid: If passed, will apply sigmoid to the output.
    """

    # Get an array of query points
    num_pts = pts.shape[0]

    # Expand the lat vec to match with the query points
    if len(vec.shape) != 1:
        vec = vec.flatten()
    if num_pts > batch_size:
        vec = vec.expand(batch_size, -1)
    else:
        vec = vec.expand(num_pts, -1)

    # Compute the output values for each batch
    value_accumulator = []
    for batch_start in range(0, num_pts, batch_size):
        batch_query_pts = torch.from_numpy(
            pts[batch_start : batch_start + batch_size, :]
        ).type(torch.float)

        values = decoder(
            torch.cat(
                [vec[: batch_query_pts.shape[0], :].cuda(), batch_query_pts.cuda()],
                dim=1,
            ),
            use_net=use_net,
        )
        if sigmoid:
            values = torch.sigmoid(values)
        value_accumulator.append(values.detach().cpu().numpy())
    values = np.vstack(value_accumulator).astype(np.float)

    return values


def create_mesh(
    vec,
    decoder,
    use_net,
    sigmoid=True,
    level=0.5,
    gradient_direction="descent",
    f_out=None,
    save_values=None,
    dims=(256, 256, 256),
    batch_size=2**16,
    padding=0.1,
):
    """
    Create a mesh from a grid of values.
    """

    values = core.reconstruct.decode_shape(
        vec=vec,
        decoder=decoder,
        dims=dims,
        use_net=use_net,
        padding=padding,
        batch_size=batch_size,
        reshape=False,
        sigmoid=sigmoid,
    )
    values = values.reshape([d for d in dims])

    if save_values is not None:
        logging.debug("Saving values to {}".format(save_values))
        np.save(save_values, values)

    try:
        vertices, faces, _, _ = skimage.measure.marching_cubes(
            values,
            level=level,
            spacing=[(1 + (padding * 2)) / d for d in dims],
            gradient_direction=gradient_direction,
        )
    except (ValueError, RuntimeError):
        raise errors.IsosurfaceExtractionError

    # The x and y channels are flipped in marching cubes
    vertices = np.hstack(
        [
            np.expand_dims(v, axis=1)
            for v in (vertices[:, 1], vertices[:, 0], vertices[:, 2])
        ]
    )

    # Center the shape
    vertices -= 0.5 + padding

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    if f_out is not None:
        mesh.export(f_out)
    return mesh
