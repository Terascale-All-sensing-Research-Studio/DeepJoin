import os
import argparse
import logging
import subprocess

import trimesh
import pymeshfix

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


def repair_self_intersection(mt):
    if mt.is_watertight:
        return mt

    m = trimesh2pymesh(mt)
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


def repair_watertight(mesh):
    """Attempt to repair a mesh using the default pymeshfix procedure"""
    mesh = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
    mesh.repair(joincomp=True, remove_smallest_components=False)
    return trimesh.Trimesh(mesh.v, mesh.f)


def repair_watertight_handsoff(
    f_in,
    f_out,
    timeout=None,
    verbose=False,
):

    cmd = ["python" + " " + __file__ + " " + f_in + " " + f_out]

    if verbose:
        cmd[0] += " --debug"

    # Badness, but prevents segfault
    logging.debug("Executing command in the shell: \n{}".format(cmd))
    try:
        subprocess.call(cmd, shell=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logging.debug("Repair failed")

    if not os.path.exists(f_out):
        raise errors.MeshWaterproofError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(dest="input", type=str, help="Path to the input file.")
    parser.add_argument(dest="output", type=str, help="Path to the output file.")
    logger.add_logger_args(parser)
    args = parser.parse_args()
    logger.configure_logging(args)
    mesh = repair_watertight(
        trimesh.load(args.input),
    )
    mesh.export(args.output)
