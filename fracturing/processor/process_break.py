import argparse, os
import logging

import trimesh
import numpy as np
from scipy.spatial import cKDTree as KDTree

try:
    import pymesh
except ImportError:
    pass

import processor.errors as errors
import processor.logger as logger


def pymesh2trimesh(m):
    return trimesh.Trimesh(m.vertices, m.faces)


def trimesh2pymesh(m):
    return pymesh.form_mesh(m.vertices, m.faces)


def repair_self_intersection(m):
    m = trimesh2pymesh(m)

    m, _ = pymesh.remove_degenerated_triangles(m)
    mt = pymesh2trimesh(m)
    if mt.is_watertight:
        return mt

    m, _ = pymesh.remove_duplicated_vertices(m)
    mt = pymesh2trimesh(m)
    if mt.is_watertight:
        return mt

    m, _ = pymesh.remove_duplicated_faces(m)
    mt = pymesh2trimesh(m)
    if mt.is_watertight:
        return mt

    m = pymesh.resolve_self_intersection(m)
    return pymesh2trimesh(m)


def paint_mesh(mesh_from, mesh_to, vertex_inds):
    """Transfer vertex colors from one mesh to another"""
    # Vertices to transfer color to
    vertices = mesh_to.vertices[vertex_inds, :]

    # Find their nearest neighbor on source mesh
    _, v_idx = KDTree(mesh_from.vertices).query(vertices)

    # Standin color is white, opaque
    mesh_to.visual.vertex_colors = (
        np.ones((mesh_to.vertices.shape[0], 4)).astype(np.uint8) * 255
    )

    # Transfer the colors
    mesh_to.visual.vertex_colors[vertex_inds, :] = mesh_from.visual.vertex_colors[
        v_idx, :
    ]


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


def break_mesh(
    mesh,
    offset=0.0,
    rand_translation=0.1,
    noise=0.005,
    replicator=None,
    return_tool=False,
):
    """
    Break an object and return the broken and restoration objects.
    Returns a dictionary that can be used to replicate the break exactly.
    """

    if replicator is None:
        replicator = {}

    tool_type = replicator.setdefault("tool_type", np.random.randint(1, high=5))
    tool_type = 1
    if tool_type == 1:
        tool = pymesh.generate_box_mesh(
            box_min=[-0.5, -0.5, -0.5], box_max=[0.5, 0.5, 0.5], subdiv_order=6
        )
    else:
        if tool_type == 2:
            tool = pymesh.generate_icosphere(0.5, [0.0, 0.0, 0.0], refinement_order=0)
        elif tool_type == 3:
            tool = pymesh.generate_icosphere(0.5, [0.0, 0.0, 0.0], refinement_order=1)
        elif tool_type == 4:
            tool = pymesh.generate_icosphere(0.5, [0.0, 0.0, 0.0], refinement_order=2)

        # Disjoint the vertices so that the icosphere isn't regular
        random_disjoint = replicator.setdefault(
            "random_disjoint", np.random.random(tool.vertices.shape)
        )
        tool = pymesh.form_mesh(
            tool.vertices + (random_disjoint * (0.1) - (0.1 / 2)), tool.faces
        )

        # Subdivide the mesh
        tool, __ = pymesh.split_long_edges(tool, noise * 5)
    vertices = tool.vertices

    # Offset the tool so that the break is roughly in the center
    set_offset = replicator.setdefault("set_offset", np.array([0.5 + offset, 0, 0]))
    vertices = vertices + set_offset

    # Add random noise to simulate fracture geometry
    noise = np.asarray([noise, noise, noise])
    random_noise = replicator.setdefault(
        "random_noise", np.random.random(vertices.shape)
    )
    vertices = vertices + (random_noise * (noise) - (noise / 2))

    # Add a random rotation
    # http://planning.cs.uiuc.edu/node198.html
    u, v, w = replicator.setdefault("random_rotation", np.random.random(3))
    q = [
        np.sqrt(1 - u) * np.sin(2 * np.pi * v),
        np.sqrt(1 - u) * np.cos(2 * np.pi * v),
        np.sqrt(u) * np.sin(2 * np.pi * w),
        np.sqrt(u) * np.cos(2 * np.pi * v),
    ]
    # vertices = np.dot(pymesh.Quaternion(q).to_matrix(), vertices.T).T
    vertices[:, 1] -= 0.2

    # Add a small random translation
    random_translation = replicator.setdefault(
        "random_translation", np.random.random(3)
    )
    vertices += random_translation * (rand_translation) - (rand_translation / 2.0)

    # Add a warp
    warp = lambda vs: np.asarray([(v**3) for v in vs])
    vertices += np.apply_along_axis(warp, 1, vertices)

    # Break
    tool = pymesh.form_mesh(vertices, tool.faces)
    broken = pymesh.boolean(mesh, tool, "difference")
    restoration = pymesh.boolean(
        mesh, pymesh.form_mesh(vertices, tool.faces), "intersection"
    )
    if return_tool:
        return broken, restoration, replicator, tool
    return broken, restoration, replicator


def breaker(
    f_in,
    f_out,
    f_restoration=False,
    f_tool=False,
    export_color=False,
    export_normals=False,
    validate=True,
    save_meta=False,
    max_break=0.5,
    min_break=0.3,
    num_components=1,
    max_overall_retries=10,
    max_single_retries=5,
    break_method="surface-area",
    overwrite=False,
):

    assert max_break > min_break
    assert break_method in ["surface-area", "volume", "combined"]

    # Break parameters
    offset = 0.3
    refinement_offset = 0.1
    refinement_decay = 0.90

    # Load the mesh
    tri_mesh_in = trimesh.load(f_in)
    mesh_in = pymesh.form_mesh(tri_mesh_in.vertices, tri_mesh_in.faces)

    # Make sure mesh is closed
    if (not mesh_in.is_manifold()) or (not mesh_in.is_closed()):
        raise errors.MeshNotClosedError

    num_test_components = num_components
    if num_components is None:
        num_test_components = 0

    cur_retry = 0
    while cur_retry < max_overall_retries:
        logging.debug("== Restarting fracture ... ==")

        # Break for the first time
        mesh_out, rmesh_out, replicator, mesh_tool = break_mesh(
            mesh_in, replicator=None, offset=offset, return_tool=True
        )

        # Check to make sure enough of the object was removed
        for itr in range(max_single_retries):
            amount_removed_vol = compute_break_percent(
                rmesh_out, mesh_in, method="volume"
            )
            amount_removed_sa = compute_break_percent(
                rmesh_out, mesh_in, method="surface_area"
            )
            mesh_num_components = mesh_out.num_surface_components
            rmesh_num_components = rmesh_out.num_surface_components

            # trimesh.Trimesh(mesh_tool.vertices, mesh_tool.faces).export("tool_{}.obj".format(itr))

            # Print debug information
            logging.debug("Removed {}%% volume".format(round(amount_removed_vol, 3)))
            logging.debug(
                "Removed {}%% surface_area".format(round(amount_removed_sa, 3))
            )
            logging.debug("Broken has {} components".format(mesh_num_components))
            logging.debug("Restoration has {} components".format(rmesh_num_components))
            logging.debug("Replicator tool_type: {}".format(replicator["tool_type"]))
            logging.debug(
                "Replicator random_translation: {}".format(
                    replicator["random_translation"]
                )
            )
            logging.debug(
                "Replicator random_rotation: {}".format(replicator["random_rotation"])
            )
            logging.debug("Replicator set_offset: {}".format(replicator["set_offset"]))

            # Force the volume condition to be satisfied
            if break_method == "surface-area":
                amount_removed_vol = min_break

            # Force the surface area condition to be satisfied
            elif break_method == "volume":
                amount_removed_sa = min_break

            # Force the components condition to be satisfied
            if num_components is None:
                rmesh_num_components = num_test_components

            # Adjust the location of the tool
            if (
                (amount_removed_vol < min_break)
                or (amount_removed_sa < min_break)
                or (rmesh_num_components < num_test_components)
            ):
                logging.debug("Moving tool closer to center of object")
                replicator["set_offset"][0] -= refinement_offset * (
                    refinement_decay**itr
                )
            elif (
                (amount_removed_vol > max_break)
                or (amount_removed_sa > max_break)
                or (rmesh_num_components > num_test_components)
            ):
                logging.debug("Moving tool farther from center of object")
                replicator["set_offset"][0] += refinement_offset * (
                    refinement_decay**itr
                )
            else:
                break

            # Retry the break
            mesh_out, rmesh_out, replicator, mesh_tool = break_mesh(
                mesh_in, replicator=replicator, offset=offset, return_tool=True
            )
        else:
            logging.debug(
                "Failed {} times, re-randomizing tool".format(max_single_retries)
            )
            cur_retry += 1
            continue

        # Perform output validation
        if validate:
            # We removed all of the vertices, or no vertices
            if (len(mesh_out.vertices) == 0) or (
                len(mesh_out.vertices) == len(mesh_in.vertices)
            ):
                cur_retry += 1
                logging.debug("Mesh validation failed, all or no vertices removed")
                continue

            # This shouldn't happen
            elif (
                (not mesh_out.is_manifold())
                or (not mesh_out.is_closed())
                or (not rmesh_out.is_manifold())
                or (not rmesh_out.is_closed())
            ):
                cur_retry += 1
                logging.debug("Mesh validation failed, result is not waterproof")
                continue

        break

    # If we've completed the while loop then this mesh cant be broken
    else:
        raise errors.MeshBreakMaxRetriesError

    logging.debug(
        "Successfully removed {}%% volume".format(round(amount_removed_vol, 3))
    )
    logging.debug(
        "Successfully removed {}%% surface_area".format(round(amount_removed_sa, 3))
    )
    logging.debug("Broken has {} components".format(mesh_num_components))
    logging.debug("Restoration has {} components".format(rmesh_num_components))
    logging.debug("Broken has {} vertices".format(mesh_out.vertices.shape[0]))
    logging.debug("Restoration has {} vertices".format(rmesh_out.vertices.shape[0]))

    # Save metadata
    if save_meta:
        fracture_inds = np.logical_not(
            intersect_mesh(mesh_out.vertices, mesh_in.vertices)
        )
        f_meta = os.path.splitext(f_out)[0] + ".npz"
        if overwrite or not os.path.exists(f_meta):
            logging.debug("Saving metadata to: {}".format(f_meta))
            np.savez_compressed(
                f_meta,
                fracture_vertices=mesh_out.vertices[fracture_inds, :],
                **replicator,
            )
        else:
            return

    # Save the broken object
    tri_mesh_out_b = trimesh.Trimesh(vertices=mesh_out.vertices, faces=mesh_out.faces)
    if not tri_mesh_out_b.is_watertight:
        tri_mesh_out_b = repair_self_intersection(tri_mesh_out_b)
    if export_color:
        paint_mesh(
            tri_mesh_in,
            tri_mesh_out_b,
            intersect_mesh(tri_mesh_out_b.vertices, tri_mesh_in.vertices),
        )
    if export_normals:
        tri_mesh_out_b.vertex_normals
    if overwrite or not os.path.exists(f_out):
        logging.debug("Saving broken to: {}".format(f_out))
        tri_mesh_out_b.export(f_out)
    else:
        return

    # Save the restoration object
    if f_restoration:
        tri_mesh_out_r = trimesh.Trimesh(
            vertices=rmesh_out.vertices, faces=rmesh_out.faces
        )
        if not tri_mesh_out_r.is_watertight:
            tri_mesh_out_r = repair_self_intersection(tri_mesh_out_r)
        if export_color:
            paint_mesh(
                tri_mesh_in,
                tri_mesh_out_r,
                intersect_mesh(tri_mesh_out_r.vertices, tri_mesh_in.vertices),
            )
        if export_normals:
            tri_mesh_out_r.vertex_normals

        if overwrite or not os.path.exists(f_restoration):
            logging.debug("Saving restoration to: {}".format(f_restoration))
            tri_mesh_out_r.export(f_restoration)
        else:
            return

    # Save the restoration object
    if f_tool:
        tri_mesh_out_tool = trimesh.Trimesh(
            vertices=mesh_tool.vertices, faces=mesh_tool.faces
        )
        if not tri_mesh_out_tool.is_watertight:
            tri_mesh_out_tool = repair_self_intersection(tri_mesh_out_tool)
        if export_normals:
            tri_mesh_out_tool.vertex_normals

        if overwrite or not os.path.exists(f_tool):
            logging.debug("Saving tool to: {}".format(f_tool))
            tri_mesh_out_tool.export(f_tool)
        else:
            return


def process(
    obj,
    num_results,
    overwrite,
    executor,
    args,
):

    handler_breaker = None
    break_func = breaker
    num_components = None

    for idx in range(num_results):
        f_in = obj.path_c()
        f_out = obj.path_b(idx)
        f_res = obj.path_r(idx)

        f_tool = False
        if args.use_tool:
            f_tool = obj.path_tool(idx)

        if args.break_handle:
            if handler_breaker is None:
                from processor.process_break_handle import breaker as handler_breaker
            if np.random.random(1) > 0.8:
                num_components = 1
                break_func = handler_breaker
            else:
                num_components = None
                break_func = breaker

        if os.path.exists(f_in) and (not os.path.exists(f_out) or overwrite):
            executor.graceful_submit(
                break_func,
                f_in=f_in,
                f_out=f_out,
                f_restoration=f_res,
                f_tool=f_tool,
                break_method=args.break_method,
                validate=True,
                max_break=args.max_break,
                min_break=args.min_break,
                num_components=num_components,
                max_overall_retries=10,
                max_single_retries=5,
                overwrite=overwrite,
            )


def validate_outputs(
    obj,
    num_results,
    args,
):
    outputs = []
    for idx in range(num_results):
        if not os.path.exists(obj.path_b(idx)):
            outputs.append(False)
            continue
        if not os.path.exists(obj.path_r(idx)):
            outputs.append(False)
            continue
        if args.use_tool:
            if not os.path.exists(obj.path_tool(idx)):
                outputs.append(False)
                continue
        outputs.append(True)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Breaks an object")
    parser.add_argument(dest="input", type=str, help="Path to the input file.")
    parser.add_argument(dest="output", type=str, help="Path to the output file.")
    parser.add_argument(
        "--restoration",
        "-r",
        type=str,
        default=False,
        help="Optionally output restoration shape.",
    )
    parser.add_argument(
        "--tool",
        type=str,
        default=False,
        help="Optionally output tool shape.",
    )
    parser.add_argument(
        "--export_color",
        default=False,
        action="store_true",
        help="Optionally remap the vertex texture attributes of the input mesh "
        + "to the broken and restoration shapes.",
    )
    parser.add_argument(
        "--export_normals",
        default=False,
        action="store_true",
        help="Force the normals to be genrated for the output shapes.",
    )
    parser.add_argument(
        "--skip_validate",
        "-v",
        action="store_false",
        help="If passed will skip checking if object is watertight.",
    )
    parser.add_argument(
        "--meta",
        "-m",
        action="store_true",
        help="If passed will store the fracture metadata in an npz file.",
    )
    parser.add_argument(
        "--max_break",
        type=float,
        default=1.0,
        help="Max amount of the object to remove (by volume).",
    )
    parser.add_argument(
        "--min_break",
        type=float,
        default=0.0,
        help="Min amount of the object to remove (by volume).",
    )
    parser.add_argument(
        "--num_components",
        type=int,
        default=1,
        help="Number of desired restoration components.",
    )
    parser.add_argument(
        "--max_single_retries",
        type=int,
        default=5,
        help="Number of times to adjust a tool when breaking.",
    )
    parser.add_argument(
        "--max_overall_retries",
        type=int,
        default=5,
        help="Number of times to retry a break from scratch.",
    )
    parser.add_argument(
        "--break_method",
        type=str,
        default="combined",
        help="Method to evaluate break by: [surface-area, volume, combined].",
    )
    parser.add_argument(
        "--no_overwrite",
        action="store_true",
        default=False,
        help="If passed, will not overwrite output files on disk.",
    )
    logger.add_logger_args(parser)
    args = parser.parse_args()
    logger.configure_logging(args)

    breaker(
        f_in=args.input,
        f_out=args.output,
        f_restoration=args.restoration,
        f_tool=args.tool,
        export_color=args.export_color,
        export_normals=args.export_normals,
        validate=args.skip_validate,
        save_meta=args.meta,
        max_break=args.max_break,
        min_break=args.min_break,
        num_components=args.num_components,
        max_overall_retries=args.max_overall_retries,
        max_single_retries=args.max_single_retries,
        break_method=args.break_method,
        overwrite=not args.no_overwrite,
    )
