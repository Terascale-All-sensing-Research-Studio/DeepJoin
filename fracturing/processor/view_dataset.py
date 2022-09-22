import os, argparse
import json
import tqdm

import trimesh
import vedo
import numpy as np
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt

import processor.shapenet as shapenet


def trimesh2vedo(mesh, **kwargs):
    return vedo.Mesh([mesh.vertices, mesh.faces], **kwargs)


def vedo2trimesh(mesh):
    return trimesh.Trimesh(
        vertices=mesh.points(),
        faces=mesh.faces(),
    )


def connected_components(mesh):
    """
    Return number of connected components.
    """
    return len(trimesh2vedo(mesh).splitByConnectivity())


def intersect_mesh(a, b, sig=5):
    """mask of vertices that a shares with b"""
    av = [frozenset(np.round(v, sig)) for v in a]
    bv = set([frozenset(np.round(v, sig)) for v in b])
    return np.asarray(list(map(lambda v: v in bv, av)))


def compute_break_percent(gt_restoration, gt_complete, method="volume"):
    """Compute the percent of an object removed by a break"""

    if method == "volume":
        return gt_restoration.volume / gt_complete.volume
    elif method == "surface_area":
        return (
            intersect_mesh(gt_complete.vertices, gt_restoration.vertices).sum()
            / gt_complete.vertices.shape[0]
        )
    else:
        raise RuntimeError("Unknown method {}".format(method))


def report(root_dir, splits_file, num_breaks=1, spline_method=False):
    object_id_dict = json.load(open(splits_file, "r"))
    id_train_list = [
        shapenet.ShapeNetObject(root_dir, o[0], o[1])
        for o in object_id_dict["id_train_list"]
    ]
    id_test_list = [
        shapenet.ShapeNetObject(root_dir, o[0], o[1])
        for o in object_id_dict["id_test_list"]
    ]

    test_report = {
        "surface_area": [],
        "volume": [],
        "restoration_components": [],
        "full_spline_method": [],
        "spline_method": [],
    }
    train_report = {
        "surface_area": [],
        "volume": [],
        "restoration_components": [],
        "full_spline_method": [],
        "spline_method": [],
    }

    for report, id_list in zip(
        [train_report, test_report], [id_train_list, id_test_list]
    ):
        for obj in tqdm.tqdm(id_list):

            path = obj.path_c()
            if not os.path.exists(path):
                continue

            if spline_method:
                for idx in range(num_breaks):
                    path = obj.path_full_spline_sdf(idx)
                    if os.path.exists(path):
                        report["full_spline_method"].append(
                            str(np.load(path)["method"])
                        )
                    path = obj.path_spline_sdf(idx)
                    if os.path.exists(path):
                        report["spline_method"].append(str(np.load(path)["method"]))
            else:
                gt_complete = trimesh.load(path)

                for idx in range(num_breaks):

                    path = obj.path_r(idx)
                    if not os.path.exists(path):
                        continue
                    gt_restoration = trimesh.load(path)

                    report["surface_area"].append(
                        compute_break_percent(
                            gt_restoration, gt_complete, method="surface_area"
                        )
                    )
                    report["volume"].append(
                        compute_break_percent(
                            gt_restoration, gt_complete, method="volume"
                        )
                    )
                    report["restoration_components"].append(
                        connected_components(gt_restoration)
                    )

    if spline_method:
        num_plots = 2
    else:
        num_plots = 4
    fig, axes = plt.subplots(num_plots, 2, figsize=(5 * 2, 5 * num_plots))

    for name, report, col in zip(
        ["Train", "Test"], [train_report, test_report], [0, 1]
    ):
        if spline_method:
            methods = sorted(list(set(report["full_spline_method"])))
            row = 0
            bar = axes[row, col].bar(
                methods,
                [
                    round(
                        report["full_spline_method"].count(m)
                        / len(report["full_spline_method"]),
                        3,
                    )
                    for m in methods
                ],
            )
            axes[row, col].set_xlabel("full_spline_method")
            axes[row, col].set_ylabel("precent")
            axes[row, col].set_title("Full Spline Methods".format(name))
            axes[row, col].bar_label(bar)

            methods = sorted(list(set(report["spline_method"])))
            row += 1
            bar = axes[row, col].bar(
                methods,
                [
                    round(
                        report["spline_method"].count(m) / len(report["spline_method"]),
                        3,
                    )
                    for m in methods
                ],
            )
            axes[row, col].set_xlabel("spline_method")
            axes[row, col].set_ylabel("percent")
            axes[row, col].set_title("Spline Methods".format(name))
            axes[row, col].bar_label(bar)

        else:
            row = 0
            axes[row, col].scatter(
                report["surface_area"],
                report["volume"],
            )
            axes[row, col].set_xlabel("surface area")
            axes[row, col].set_ylabel("volume")
            axes[row, col].set_title("{} set Surface Area to Volume".format(name))

            row += 1
            axes[row, col].hist(
                report["surface_area"],
                bins=100,
                weights=np.ones(len(report["surface_area"]))
                / len(report["surface_area"]),
            )
            axes[row, col].set_title("{} set Surface Area".format(name))
            axes[row, col].yaxis.set_major_formatter(PercentFormatter(1))

            row += 1
            axes[row, col].hist(
                report["volume"],
                bins=100,
                weights=np.ones(len(report["volume"])) / len(report["volume"]),
            )
            axes[row, col].set_title("{} set Volume".format(name))
            axes[row, col].yaxis.set_major_formatter(PercentFormatter(1))

            row += 1
            axes[row, col].hist(
                report["restoration_components"],
                bins=100,
                weights=np.ones(len(report["restoration_components"]))
                / len(report["restoration_components"]),
            )
            axes[row, col].set_title("{} set Restoration Components".format(name))
            axes[row, col].yaxis.set_major_formatter(PercentFormatter(1))

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tool for visualizing broken object occ/sdf samples. Pass "
        + "the full path to an occ/sdf sample file and an output file to dump "
        + "the visualization to. The sample points will be loaded automatically "
        + "by inferring the break index from the file path."
    )
    parser.add_argument(dest="root_dir", type=str, help="")
    parser.add_argument(dest="splits_file", type=str, help="")
    parser.add_argument(dest="output", type=str, help="Path to the output file.")
    parser.add_argument("--num_breaks", type=int, default=1, help="")
    parser.add_argument("--spline_method", default=False, action="store_true", help="")
    args = parser.parse_args()

    report(
        args.root_dir,
        args.splits_file,
        args.num_breaks,
        args.spline_method,
    ).savefig(args.output)
