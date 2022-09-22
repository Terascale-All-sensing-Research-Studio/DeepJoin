import os
import math
import itertools
import logging

import tqdm
import trimesh
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import PercentFormatter
from PIL import Image

import core
import core.errors as errors


def save_image_block(img, name, max_h=50000):
    """some images are too big to save, so this cuts them up"""
    img = np.array(img)
    h = img.shape[0]
    if h > max_h:
        for idx, y in enumerate(range(0, h, max_h)):
            n = name.format(idx)
            bn, ext = os.path.splitext(os.path.basename(n))
            if len(bn) > 255:
                n = os.path.join(os.path.dirname(n), bn[:250] + str(idx) + ext)
                logging.info("Save name truncated: {}".format(n))
            core.handler.saver(n, img[y : min(y + max_h, h), ...])
    else:
        n = name.format(0)
        bn, ext = os.path.splitext(os.path.basename(n))
        if len(bn) > 255:
            n = os.path.join(os.path.dirname(n), bn[:250] + ext)
            logging.info("Save name truncated: {}".format(n))
        core.handler.saver(n, img)


def plt2numpy(fig):
    fig.canvas.draw()
    return np.reshape(
        np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
    )


def colorize_values(values, cmap=None, min_max=None):

    # Get the colormap
    if cmap is None:
        cmap = cm.jet

    # Get the min and max
    if min_max is None:
        min_, max_ = values.min(), values.max()
    else:
        min_, max_ = min_max

    # Normalize and clip
    values = np.clip((values - min_) / (max_ - min_), 0.0, 1.0)

    # Can handle inputs that are 1-d vectors, or 2-d arrays with one column
    if len(values.shape) == 1:
        values = (cmap(values)[:, :3] * 255).astype(np.uint8)
    elif values.shape[1] == 1:
        values = (cmap(values.flatten())[:, :3] * 255).astype(np.uint8)

    # Else you already have a colormap
    elif values.shape[1] == 3:
        values = (values * 255).astype(np.uint8)
    else:
        raise RuntimeError(
            "Could not resolve pts colors with shape".format(values.shape[1])
        )
    return values


def parse_samples(pts_values):
    """
    Validates points and values. Also transforms them into a standard representation.

    Args:
        pts_values: Accepts inputs of the following types:
            [pts (tensor), c/b/r (tensor)]
            [pts (tensor), c (tensor), b (tensor), r (tensor)]
            [pts (numpy), c/b/r (numpy)]
            [pts (numpy), c (numpy), b (numpy), r (numpy)]
            [pts+c/b/r (numpy)]
            [pts+c+b+r (numpy)]

    Returns:
        pts (numpy), c+b+r (numpy)
    """
    if isinstance(pts_values, tuple):
        # Was passed a tuple of tensors (output of dataset[idx])
        try:
            if len(pts_values) == 4:
                pts = pts_values[0].numpy()
                assert pts.shape[1] == 3
                c = np.expand_dims(pts_values[1].numpy(), axis=1)
                b = np.expand_dims(pts_values[2].numpy(), axis=1)
                r = np.expand_dims(pts_values[3].numpy(), axis=1)
                values = np.expand_dims(np.hstack((c, b, r)), axis=1)
            elif len(pts_values) == 2:
                pts = pts_values[0].numpy()
                assert pts.shape[1] == 3
                values = pts_values[1].numpy()
                if len(values.shape) == 1:
                    values = np.expand_dims(pts_values[1].numpy(), axis=1)
            else:
                raise RuntimeError(
                    "Could not resolve pts values tuple with length".format(
                        len(pts_values)
                    )
                )
        except AttributeError:
            if len(pts_values) == 4:
                pts = pts_values[0]
                assert pts.shape[1] == 3

                c = np.expand_dims(pts_values[1], axis=1)
                b = np.expand_dims(pts_values[2], axis=1)
                r = np.expand_dims(pts_values[3], axis=1)
                values = np.expand_dims(np.hstack((c, b, r)), axis=1)
            elif len(pts_values) == 2:
                pts = pts_values[0]
                assert pts.shape[1] == 3
                values = pts_values[1]
                if len(values.shape) == 1:
                    values = np.expand_dims(pts_values[1], axis=1)
            else:
                raise RuntimeError(
                    "Could not resolve pts values tuple with length".format(
                        len(pts_values)
                    )
                )
    else:
        if pts_values.shape[1] == 6:
            pts = pts_values[:, :3]
            c = np.expand_dims(pts_values[:, 3], axis=1)
            b = np.expand_dims(pts_values[:, 4], axis=1)
            r = np.expand_dims(pts_values[:, 5], axis=1)
            values = np.expand_dims(np.hstack((c, b, r)), axis=1)
        elif pts_values.shape[1] == 4:
            pts = pts_values[:, :3]
            values = np.expand_dims(pts_values[:, 3], axis=1)
        else:
            raise RuntimeError(
                "Could not resolve pts values array with shape".format(pts_values.shape)
            )

    return pts, values


def image_results_row(
    idx,
    reconstruction_handler=None,
    data_handler=None,
    outputs=(0, 1, 2),
    knit_handlers=None,
    resolution=(200, 200),
    num_renders=3,
    bar_width=10,
    fontsize=0.5,
    bars=False,
    composite=False,
    save_data_renders=True,
    annotate=True,
    annotate_indices=True,
    annotate_metric=None,
    gt_composite=False,
    angle_list=None,
):
    """
    Build an image summarizing the results of the network.
    Includes ground-truth, predicted, and comparison (knit) for a single
    object, specified by index.
    Returns a numpy array.
    """

    assert not (data_handler is None and reconstruction_handler is None)
    if composite:
        assert data_handler is not None
        if not isinstance(composite, list):
            composite = [composite]

    render_h, _ = resolution

    # This will hold the rendered images
    col_stacker = []

    bar_size = (render_h, bar_width, 3)
    sidebar_size = (render_h, 60, 3)
    space_size = list(resolution) + [3]

    # Add a left edge with the object index
    if annotate_indices:
        img = core.utils_2d.annotation_image(sidebar_size, str(idx), fontsize=fontsize)
        col_stacker.append(img)

    # Append a vertical bar
    if bars:
        img = core.utils_2d.bar_image(bar_size)
        col_stacker.append(img)

    # Compile a list of all the shape indices
    all_shapes = list(outputs)
    if composite:
        for c in composite:
            all_shapes.extend(list(c))
    all_shapes = sorted(list(set(all_shapes)))

    if angle_list is None:
        angle_list = range(0, 360, int(360 / num_renders))

    for shape_idx in all_shapes:
        for angle in angle_list:

            # Get knitted obejct(s)
            if knit_handlers is not None:
                for k in knit_handlers:
                    if shape_idx in outputs:
                        # if shape_idx != 2:

                        # Get render
                        try:
                            img = k.get_render(
                                idx,
                                shape_idx,
                                angle=angle,
                                resolution=resolution,
                            )
                            if len(img.shape) == 2:
                                img = np.array(Image.fromarray(img).convert("RGB"))
                        except (
                            errors.IsosurfaceExtractionError,
                            errors.DecoderNotLoadedError,
                        ):
                            img = core.utils_2d.space_image(space_size)

                        # Annotate
                        if annotate:
                            img = core.utils_2d.annotate_image(
                                img, k.name, fontsize=fontsize
                            )
                        if annotate_metric is not None:
                            try:
                                val = np.load(
                                    k.path_eval(
                                        idx,
                                        shape_idx,
                                        min(2, shape_idx),
                                        annotate_metric,
                                        create=True,
                                    )
                                )
                                if not np.isnan(val):
                                    val = val.round(6)
                                img = core.utils_2d.annotate_image(
                                    img, str(val), hspace=40, fontsize=fontsize
                                )
                            except FileNotFoundError:
                                pass

                        # Force image to have 3 channels
                        if len(img.shape) == 2:
                            img = np.dstack((img, img, img))
                        col_stacker.append(img)

                    # Include a composite render, if desired
                    if composite:
                        if shape_idx in outputs:
                            for (gt_compi, pd_compi) in composite:
                                if shape_idx == pd_compi:

                                    # Get render
                                    try:
                                        img = k.get_composite(
                                            idx,
                                            shape_idx,
                                            data_handler.get_mesh(idx, 1),
                                            gt_compi,
                                            angle,
                                            resolution=resolution,
                                        )
                                        if len(img.shape) == 2:
                                            img = np.array(
                                                Image.fromarray(img).convert("RGB")
                                            )
                                    except (
                                        errors.IsosurfaceExtractionError,
                                        errors.DecoderNotLoadedError,
                                    ):
                                        img = core.utils_2d.space_image(space_size)

                                    # Force image to have 3 channels
                                    if len(img.shape) == 2:
                                        img = np.dstack((img, img, img))
                                    col_stacker.append(img)

            # Get the gt object
            if data_handler is not None:
                if shape_idx in outputs:
                    if shape_idx in [0, 1, 2]:

                        # Get the render
                        img = data_handler.get_render(
                            idx,
                            shape_idx,
                            angle=angle,
                            resolution=resolution,
                            save=save_data_renders,
                        )
                        if annotate:
                            img = core.utils_2d.annotate_image(
                                img, "gt", fontsize=fontsize
                            )

                        # Force image to have 3 channels
                        if len(img.shape) == 2:
                            img = np.dstack((img, img, img))
                        col_stacker.append(img)

                        # Get gt composite
                        if gt_composite and composite:
                            for (gt_compi, pd_compi) in composite:

                                # Get the render
                                if shape_idx == pd_compi:
                                    try:
                                        img = data_handler.get_composite(
                                            idx, angle, resolution=resolution
                                        )
                                    except (
                                        errors.IsosurfaceExtractionError,
                                        errors.DecoderNotLoadedError,
                                    ):
                                        img = core.utils_2d.space_image(space_size)

                                    # Force image to have 3 channels
                                    if len(img.shape) == 2:
                                        img = np.dstack((img, img, img))
                                    col_stacker.append(img)

            # Get the reconstructed object, if the network supports it
            if reconstruction_handler is not None:
                if shape_idx in outputs:

                    # Get the render
                    try:
                        img = reconstruction_handler.get_render(
                            idx, shape_idx, angle=angle, resolution=resolution
                        )
                    except errors.IsosurfaceExtractionError:
                        img = core.utils_2d.space_image(space_size)

                    # Annotate
                    if annotate:
                        img = core.utils_2d.annotate_image(
                            img, reconstruction_handler.name, fontsize=fontsize
                        )
                    if annotate_metric is not None:
                        try:
                            val = np.load(
                                reconstruction_handler.path_eval(
                                    idx,
                                    shape_idx,
                                    min(2, shape_idx),
                                    annotate_metric,
                                    create=True,
                                )
                            )
                            if not np.isnan(val):
                                val = val.round(6)
                            img = core.utils_2d.annotate_image(
                                img, str(val), hspace=40, fontsize=fontsize
                            )
                        except FileNotFoundError:
                            pass

                    # Force image to have 3 channels
                    if len(img.shape) == 2:
                        img = np.dstack((img, img, img))
                    col_stacker.append(img)

                # Include a composite render, if desired
                if composite:
                    if shape_idx in outputs:
                        for (gt_compi, pd_compi) in composite:
                            if shape_idx == pd_compi:

                                # Get the render
                                try:
                                    img = reconstruction_handler.get_composite(
                                        idx,
                                        shape_idx,
                                        data_handler.get_mesh(idx, 1),
                                        gt_compi,
                                        angle,
                                        resolution=resolution,
                                    )
                                except (
                                    errors.IsosurfaceExtractionError,
                                    errors.DecoderNotLoadedError,
                                ):
                                    img = core.utils_2d.space_image(space_size)

                                # Force image to have 3 channels
                                if len(img.shape) == 2:
                                    img = np.dstack((img, img, img))
                                col_stacker.append(img)

        # Append a vertical bar
        if bars:
            img = core.utils_2d.bar_image(bar_size)
            col_stacker.append(img)

    # img1, img2 = col_stacker[-2], col_stacker[-1]
    # col_stacker[-1] = img1
    # col_stacker[-2] = img2

    return np.hstack(col_stacker)


def image_results(
    data_handler,
    reconstruction_handler,
    reconstruct_list,
    knit_handlers=None,
    outputs=(0, 1, 2),
    resolution=(200, 200),
    num_renders=3,
    bar_width=5,
    fontsize=0.5,
    composite=False,
    annotate=True,
    annotate_indices=True,
    annotate_metric=None,
    gt_composite=False,
    angle_list=None,
    bars=True,
):
    """
    Build an image summarizing the results of the network.
    Includes ground-truth, predicted, and comparison (knit) objects.
    Returns a PIL image.
    """

    row_stacker = []
    for idx in tqdm.tqdm(reconstruct_list):
        img = image_results_row(
            reconstruction_handler=reconstruction_handler,
            data_handler=data_handler,
            knit_handlers=knit_handlers,
            idx=idx,
            outputs=outputs,
            fontsize=fontsize,
            resolution=resolution,
            num_renders=num_renders,
            bar_width=bar_width,
            bars=bars,
            composite=composite,
            annotate=annotate,
            annotate_indices=annotate_indices,
            annotate_metric=annotate_metric,
            gt_composite=gt_composite,
            angle_list=angle_list,
            save_data_renders=True,
        )
        row_stacker.append(img)
    return Image.fromarray(np.vstack(row_stacker)).convert("RGB")


def write_ptcld_ply(
    f_out, vertices, color=None, cmap=None, min_max=None, method="trimesh"
):
    """
    Write out a colorized point cloud as ascii.
    """

    if color is None:
        if method == "trimesh":
            trimesh.Trimesh(vertices=vertices).export(f_out)
        elif method == "ascii":
            with open(f_out, "w") as f:
                f.write("ply \n")
                f.write("format ascii 1.0 \n")
                f.write("element vertex {} \n".format(vertices.shape[0]))
                f.write("property float x \n")
                f.write("property float y \n")
                f.write("property float z \n")
                f.write("element face 0 \n")
                f.write("property list uchar int vertex_indices \n")
                f.write("end_header \n")
                for vertex in vertices:
                    f.write(
                        "{} {} {} \n".format(
                            round(vertex[0], 6),
                            round(vertex[1], 6),
                            round(vertex[2], 6),
                        )
                    )
        else:
            raise RuntimeError("Write method not supported {}".format(method))
    else:
        # Get colors
        color = colorize_values(color, cmap=cmap, min_max=min_max)

        if method == "trimesh":
            color = np.hstack((color, np.ones((color.shape[0], 1))))
            ptcld = trimesh.points.PointCloud(vertices=vertices, colors=color)
            ptcld.export(f_out)
        elif method == "ascii":
            with open(f_out, "w") as f:
                f.write("ply \n")
                f.write("format ascii 1.0 \n")
                f.write(
                    "element vertex {} \n".format(
                        min(vertices.shape[0], color.shape[0])
                    )
                )
                f.write("property float x \n")
                f.write("property float y \n")
                f.write("property float z \n")
                f.write("property uchar red \n")
                f.write("property uchar green \n")
                f.write("property uchar blue \n")
                f.write("element face 0 \n")
                f.write("property list uchar int vertex_indices \n")
                f.write("end_header \n")
                for (r, g, b), vertex in zip(color, vertices):
                    f.write(
                        "{} {} {} {} {} {} \n".format(
                            round(vertex[0], 6),
                            round(vertex[1], 6),
                            round(vertex[2], 6),
                            r,
                            g,
                            b,
                        )
                    )
        else:
            raise RuntimeError("Write method not supported {}".format(method))


def ply_samples(f_out, pts_values, cmap=None, min_max=None, method="trimesh"):
    """
    Write out a training sample as colorized ply.
    """
    pts, values = parse_samples(pts_values)
    write_ptcld_ply(f_out, pts, values, cmap=cmap, min_max=min_max, method=method)


def plot_samples(
    pts_values,
    slice_dim=0,
    n_plots=4,
    cmap=None,
    min_max=None,
    figsize=3,
    padding=0.05,
    swap_yz=True,
):
    """
    Plot samples in several slices.
    """

    size = int(math.sqrt(n_plots))
    assert size**2 == n_plots, "n_plots must be a perfect square"

    # Get points and values
    pts, values = parse_samples(pts_values)
    if swap_yz:
        pts = pts[:, [0, 2, 1]]
    if min_max is None:
        min_, max_ = values.min(), values.max()
    else:
        min_, max_ = min_max
    if cmap is None:
        cmap = cm.jet
    colors = colorize_values(values, cmap, min_max).astype(float)

    fig, axes = plt.subplots(
        nrows=size, ncols=size, figsize=(figsize * n_plots, figsize * n_plots)
    )
    step_size = (1.0 + (2 * padding)) / n_plots
    for (i, j), r in zip(
        itertools.product(list(range(size)), list(range(size))),
        np.linspace(-0.5 - padding, 0.5 + padding, n_plots, endpoint=False),
    ):
        mask = np.logical_and(
            (pts[:, slice_dim] > r), (pts[:, slice_dim] < (r + step_size))
        )
        p = np.delete(pts[mask, :], slice_dim, axis=1)
        axes[i, j].scatter(p[:, 0], p[:, 1], c=(colors[mask, :] / 255.0))
        axes[i, j].set_xlim(-0.5 - padding, 0.5 + padding)
        axes[i, j].set_ylim(-0.5 - padding, 0.5 + padding)

    # Make colorbar
    cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=mpl.colors.Normalize(vmin=min_, vmax=max_)
    )
    sm.set_array([])
    fig.colorbar(sm, cax=cax, **kw)

    # Resquare axes
    for ax in axes.flat:
        ax.set_aspect("equal", "box")

    return fig


def plot_out_raw(raw, dimensions=(128, 128), n_plots=4):
    """
    Create a plot of the raw values outputted by the network.
    """
    raw = raw.astype(float)
    fig, axes = plt.subplots(nrows=1, ncols=n_plots, figsize=(5 * n_plots, 5))
    size = dimensions[0] * dimensions[1]
    if n_plots == 1:
        pcm = axes.imshow(raw[:size].reshape(dimensions))
        fig.colorbar(pcm, ax=axes)
    else:
        for p, ax in enumerate(axes):
            pcm = ax.imshow(raw[p * size : p * size + size].reshape(dimensions))
            fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    return fig


def plot_lat_raw_value_dist(v, bins=100):
    """
    Create a histogram of latent vector values.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(v.flatten(), bins=bins)
    ax.set_ylabel("Number of Raw Values")
    ax.set_xlabel("Value Magnitude")
    ax.set_title("Distribution of Raw Latent Vector Values")
    return fig


def plot_lat_mean_over_dims(v):
    """
    Create a bar chart of the mean values for each latent dimension.
    """
    means = np.mean(v, axis=0)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(range(means.shape[0]), means)
    ax.set_ylabel("Mean")
    ax.set_xlabel("Dimension")
    ax.set_title("Dimension-wise Mean With norm {:.3}".format(np.linalg.norm(means)))
    return fig


def plot_lat_std_over_dims(v):
    """
    Create a bar chart of the std of values for each latent dimension.
    """
    stds = np.std(v, axis=0)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(range(stds.shape[0]), stds)
    ax.set_ylabel("STD")
    ax.set_xlabel("Dimension")
    ax.set_title("Dimension-wise STDs")
    return fig


def plot_lat_std_dist(v, bins=50):
    """
    Create a histogram of the distribution the std over all latent dimensions.
    """
    stds = np.std(v, axis=0)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(stds, bins=bins)
    ax.set_ylabel("Number of Dimensions")
    ax.set_xlabel("Magnitude of STD")
    ax.set_title("Distribution of STDs")
    return fig


def plot_metric(
    data_handler,
    reconstruction_handler,
    reconstruct_list,
    outputs=(0, 1, 2),
    bins=20,
    metric="chamfer",
):
    """
    Plot distribution of a given metric (chamfer, normal const, etc.).
    """

    # Get the metric
    values = [list() for _ in range(3)]
    for idx in reconstruct_list:
        for shape_idx in outputs:
            try:
                value = reconstruction_handler.get_eval(
                    data_handler.get_upsampled(idx, shape_idx),
                    idx,
                    input,
                    gt_shape_idx=shape_idx,
                    metric=metric,
                )
            except errors.IsosurfaceExtractionError:
                continue
            values[shape_idx].append(value)

    min_, max_ = (
        np.array(np.hstack([np.array(v) for v in values])).min(),
        np.array(np.hstack([np.array(v) for v in values])).max(),
    )

    num_subfigs = len(outputs)
    fig = plt.figure(figsize=(10, num_subfigs * 5))
    fig_ptr = 1
    for out_idx, name in zip(outputs, ["complete", "broken", "restoration"]):
        ax = fig.add_subplot(num_subfigs, 1, fig_ptr)
        if len(values[out_idx]) > 0:
            ax.hist(
                np.array(values[out_idx]),
                bins=bins,
                weights=np.ones(len(values[out_idx])) / len(values[out_idx]),
                # range=(min_, max_),
            )
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylabel("Percent of Objects")
            ax.set_xlabel("{} distance".format(metric))
            ax.set_title(
                "Distribution of {} Distance for {} Objects With Mean: {}".format(
                    metric, name, round(np.array(values[out_idx]).mean(), 5)
                )
            )
            # ax.set_xlim(min_, max_)
            fig_ptr += 1
    return fig


def plot_break_percent(data_handler, bins=20, method="surface_area"):
    """
    Create a histogram of the distribution of object breaks by the percent
    of the object discarded.
    """

    # Get the break percentages
    break_precentages = []
    for idx in range(len(data_handler)):
        # Get all of the meshes to do the computations
        gt_complete = data_handler.get_mesh(idx, 0)
        gt_restoration = data_handler.get_mesh(idx, 2)
        break_precentages.append(
            core.metrics.compute_break_percent(
                gt_restoration, gt_complete, method=method
            )
        )

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(
        break_precentages,
        bins=bins,
        weights=np.ones(len(break_precentages)) / len(break_precentages),
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Percent of Objects")
    ax.set_xlabel("Break Percent")
    ax.set_title("Distribution of Object Breaks")
    return fig
