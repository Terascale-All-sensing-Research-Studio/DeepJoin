import os, argparse
import itertools

import trimesh
import numpy as np
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt

import processor.process_sample as sampler


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


def plot_samples(
    pts_values,
    slice_dim=0,
    n_plots=4,
    cmap=None,
    min_max=None,
    figsize=3,
    padding=0.05,
):
    """
    Plot samples in several slices.
    """

    size = int(np.sqrt(n_plots))
    assert size**2 == n_plots, "n_plots must be a perfect square"

    # Get points and values
    pts, values = parse_samples(pts_values)
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


def view_points(
    f_in,
    f_out,
    f_points=None,
    subsample=None,
    key=None,
    dim=None,
    min_max=None,
):

    data = np.load(f_in)
    print("Loading values from {}".format(f_in))

    # (Try to) get the break index
    if f_points is None:

        if "xyz" in data:
            xyz = data["xyz"]
            print("Loading points from {}".format(f_in))
        elif dim is not None:
            xyz = sampler.uniform_sample_points(dim=dim, padding=0.1)
        else:
            break_idx = [
                p for p in os.path.splitext(f_in)[0].split("_") if p.isnumeric()
            ][0]

            # Try and find the sample points
            instance_dir = os.path.dirname(f_in)
            sample_paths = [f for f in os.listdir(instance_dir) if "sample" in f]
            for sp in sample_paths:
                cur_break_idx = [
                    p for p in os.path.splitext(sp)[0].split("_") if p.isnumeric()
                ][0]
                if cur_break_idx == break_idx:
                    break

            # Create the full path
            sp = os.path.join(instance_dir, sp)
            print("Loading points from {}".format(sp))

            # Load
            xyz = np.load(sp)["xyz"]
    else:
        xyz = np.load(f_points)["xyz"]

    # This will trigger if you pass the sample points
    if len(data) == 1 and "xyz" in data:
        trimesh.points.PointCloud(vertices=xyz).export(f_out)
        return

    if key is None:
        for k in ["occ", "sdf"]:
            try:
                values = data[k].astype(float)
                break
            except KeyError:
                pass
        else:
            raise RuntimeError("Could not load from: {}".format(f_in))
    else:
        try:
            values = data[key].astype(float)
        except KeyError:
            raise RuntimeError("Could not load {} from: {}".format(key, f_in))

    if subsample is not None:
        idxs = np.random.randint(0, len(xyz), size=subsample)
        xyz, values = xyz[idxs, :], values[idxs, ...]

    # group up the points and values
    pts_values = tuple((xyz, values))

    if os.path.splitext(f_out)[-1] != ".ply":
        plot_samples(
            pts_values,
            n_plots=16,
            min_max=min_max,
        ).savefig(f_out)
    else:
        # Parse the points and values
        xyz, values = parse_samples(pts_values)

        # Get colors
        color = colorize_values(values, cmap=None, min_max=min_max)
        color = np.hstack((color, np.ones((color.shape[0], 1))))

        # Show the pointcloud
        trimesh.points.PointCloud(vertices=xyz, colors=color).export(f_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tool for visualizing broken object occ/sdf samples. Pass "
        + "the full path to an occ/sdf sample file and an output file to dump "
        + "the visualization to. The sample points will be loaded automatically "
        + "by inferring the break index from the file path."
    )
    parser.add_argument(dest="input_values", type=str, help="Path to the input file.")
    parser.add_argument(dest="output", type=str, help="Path to the output file.")
    parser.add_argument(
        "--input_points", type=str, default=None, help="Path to the input file."
    )
    parser.add_argument(
        "--subsample", type=int, default=None, help="Subsample x points."
    )
    parser.add_argument(
        "--key",
        default=None,
        type=str,
        help="The key from the data file that you'd like to display.",
    )
    parser.add_argument(
        "--dim",
        default=None,
        type=int,
        help="Will associate values with uniform points.",
    )
    parser.add_argument(
        "--minmax",
        default=None,
        nargs="+",
        type=float,
        help="Min and max values to display.",
    )
    args = parser.parse_args()

    view_points(
        f_in=args.input_values,
        f_out=args.output,
        f_points=args.input_points,
        subsample=args.subsample,
        key=args.key,
        dim=args.dim,
        min_max=args.minmax,
    )
