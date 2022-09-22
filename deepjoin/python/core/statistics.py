import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import core.eval as evaluator
import core.errors as errors


def build_metric(reconstruct_list, reconstruction_handler, output_pairs, metric):
    """
    Gets a given metric
    """

    # Get the metric
    values = np.empty((len(reconstruct_list), len(output_pairs)))
    values[:] = np.nan

    for ii, idx in enumerate(reconstruct_list):
        for e, (gt_idx, pd_idx) in enumerate(output_pairs):
            path = reconstruction_handler.path_eval(
                idx,
                pd_idx,
                gt_shape_idx=gt_idx,
                metric=metric,
            )
            try:
                # Try to load the value from disk
                value = np.load(path)
            except (FileNotFoundError, ValueError):
                logging.debug(
                    "Metric: {} could not be loaded for pair {}, {}".format(
                        metric, (pd_idx, gt_idx), path
                    )
                )
                continue
            except errors.IsosurfaceExtractionError:
                logging.debug(
                    "Metric: {} isosurface extraction error {}, {}".format(
                        metric, (pd_idx, gt_idx), path
                    )
                )
                continue

            if np.isinf(value):
                logging.debug(
                    "Metric: {} isinf {}, {}".format(metric, (pd_idx, gt_idx), path)
                )
                continue
            values[ii, e] = value

    # Values will be size (num_objects, output_pars)
    logging.info(
        "Num generated {} {}".format(
            np.logical_not(np.isnan(values)).astype(int).sum(axis=0),
            metric,
        )
    )

    return values


def get_alias(name, aliases):
    """Given a metric name and a list of aliases, return the corresponding alias"""
    for (m_old, m_new) in aliases:
        if m_old == name:
            return m_new
    raise RuntimeError("Metric {} has no alias in {}".format(name, aliases))


def build_plot(
    f_out,
    experiment,
    metric_aliases,
    shape_idx_aliases,
    rounding=5,
    bins=40,
    has_one_component_list=None,
    include_generated=True,
    save=True,
    tight_layout=True,
    show_mean=True,
    show_med=True,
):

    # Apply human-readable name updates
    experiment_renamed = experiment.copy()
    metrics = []
    if include_generated:
        metrics = ["generated"]
    for (m_old, m_new) in metric_aliases:
        if m_old in experiment_renamed:
            experiment_renamed[m_new] = experiment_renamed.pop(m_old)
            metrics.append(m_new)

    # Plot the metrics
    n_rows = len(metrics)
    n_cols = len(shape_idx_aliases)
    plot_names = shape_idx_aliases
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 5 * n_rows)
    )

    for r in range(n_rows):
        current_metric_values = experiment_renamed[metrics[r]]

        for c in range(n_cols):

            # Get the column of data
            if isinstance(current_metric_values, list):

                # Lists dont have a second dimension
                if c >= 1:
                    break
                v = np.array(current_metric_values)

                # Prune NAN values
                if has_one_component_list is not None:
                    v = v[has_one_component_list]
                total_num_v = len(v)
                v = v[~np.isnan(v)]
                num_after = len(v)

                if "nested_binner" in metrics and metrics[r] == get_alias(
                    "nested_binner", metric_aliases
                ):
                    num_bins = 9
                elif "truth_tabler" in metrics and metrics[r] == get_alias(
                    "truth_tabler", metric_aliases
                ):
                    num_bins = 8
                else:
                    raise RuntimeError("Unhandled metric: {}".format(metrics[r]))

                # Bin the values
                v = v.astype(int)
                v = np.expand_dims(np.bincount(v, minlength=num_bins), axis=0)

                # Create colorized table
                plt.subplot(n_rows, 1, r + 1)
                plt.table(
                    ((v / v.sum()) * 100).round(rounding),
                    rowLabels=["Percent"],
                    colLabels=np.arange(0, num_bins),
                    cellColours=plt.cm.get_cmap("Greens")(
                        plt.Normalize(v.min(), v.max())(v)
                    ),
                    loc="center",
                )
                plt.axis("off")

                # Set the title, include how many nan and non-nan values
                plt.title("{} ({}/{})".format(metrics[r], num_after, total_num_v))
            else:
                v = current_metric_values[:, c].flatten()

                # Prune NAN values
                if has_one_component_list is not None:
                    v = v[has_one_component_list]
                total_num_v = len(v)
                v = v[~np.isnan(v)]
                num_after = len(v)

                # If any values remain, plot
                if len(v) != 0:
                    try:
                        ax = axes[r, c]
                    except IndexError:
                        ax = axes[c]

                    ax.hist(
                        v,
                        bins=bins,
                        weights=np.ones(len(v)) / len(v),
                    )
                    ax.yaxis.set_major_formatter(PercentFormatter(1))

                    # Add a mean line
                    if show_mean:
                        mean = np.mean(v)
                        _, max_ylim = ax.get_ylim()
                        ax.axvline(mean, color="r", linestyle="dashed", linewidth=1)
                        ax.text(
                            mean * 1.1,
                            max_ylim * 0.9,
                            "Mean: {}".format(round(mean, rounding)),
                        )
                    if show_med:
                        med = np.median(v)
                        _, max_ylim = ax.get_ylim()
                        ax.axvline(med, color="g", linestyle="dashed", linewidth=1)
                        ax.text(
                            med * 1.1,
                            max_ylim * 0.8,
                            "Median: {}".format(round(med, rounding)),
                        )

                    # Set the title, include how many nan and non-nan values
                    ax.set_title(
                        "{} of {} ({}/{})".format(
                            metrics[r], plot_names[c], num_after, total_num_v
                        )
                    )

    if tight_layout:
        fig.tight_layout()
    if save:
        fig.savefig(f_out)
    else:
        return fig


def export_report(
    out_metrics,
    reconstruction_handler,
    reconstruct_list,
    output_pairs,
    metrics=["chamfer"],
    shape_idx_aliases=["C", "B", "R"],
):
    """ """

    metric_aliases = [
        ["chamfer", "Chamfer (CH)"],
    ]

    # Extract the metrics data
    metrics_dict = {
        m: build_metric(reconstruct_list, reconstruction_handler, output_pairs, m)
        for m in metrics
    }
    for key in metrics_dict:
        assert len(metrics_dict[key]) == len(reconstruct_list)
    metrics_dict["reconstruct_list"] = reconstruct_list

    # Save the dictionary to disk
    np.save(out_metrics, metrics_dict)

    # Add generated
    evaluator.add_generated(metrics_dict)

    path = os.path.splitext(out_metrics)[0] + ".png"
    build_plot(
        f_out=path,
        experiment=metrics_dict,
        metric_aliases=metric_aliases,
        shape_idx_aliases=shape_idx_aliases,
    )
    logging.info("Saved plot to {}".format(path))
