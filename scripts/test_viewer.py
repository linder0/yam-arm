"""Check that what the viewer draws matches what the sim simulates.

The viewer never sees the MuJoCo model: it gets a manifest (geom -> body index,
local pose, mesh file) plus a pose stream (per-body world transform), and
multiplies them in the browser. Any drift between the model the manifest was
built from and the model the env streams shows up as geometry drawn at another
body's pose -- while body *counts* still agree, so a count check sails past it.

So reconstruct the browser's math here, from the exported files, and compare
vertex-for-vertex against MuJoCo's own world-space geometry.

    python scripts/test_viewer.py [--task pick_cube]
"""

import argparse
import os
import struct
import sys
import warnings

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from yam import embodiments as E  # noqa: E402
from yam import model as M  # noqa: E402
from yam import scene as S  # noqa: E402


def read_binary_stl(path):
    with open(path, "rb") as f:
        f.read(80)
        (n,) = struct.unpack("<I", f.read(4))
        rec = np.frombuffer(f.read(n * 50), dtype=[("d", "<f4", 12), ("a", "<u2")])
    return rec["d"][:, 3:12].reshape(-1, 3).astype(np.float64)


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=np.float64))
    return m.reshape(3, 3)


def check(key, task, out_dir):
    """Max distance between a browser-reconstructed vertex and MuJoCo's."""
    model, _ = S.build(E.REGISTRY[key], task=task)
    data = mujoco.MjData(model)
    # A non-home pose: at qpos0 a mis-ordered manifest can still look plausible.
    rng = np.random.default_rng(0)
    data.qpos[:] = model.qpos0
    nq = min(model.nq, 6)
    data.qpos[:nq] += rng.uniform(-0.3, 0.3, nq)
    mujoco.mj_forward(model, data)

    label = M.embodiment_task_id(task, key)
    import json
    with open(os.path.join(out_dir, label, "manifest.json")) as f:
        manifest = json.load(f)

    # What the browser has: the pose stream, indexed by body.
    xpos, xquat = data.xpos.copy(), data.xquat.copy()
    if manifest["nbody"] != model.nbody:
        return None, f"nbody {manifest['nbody']} != model {model.nbody}"

    # scene_manifest emits geoms in model order, skipping non-visual groups and
    # planes; replaying that filter pairs each entry back to its geom id.
    gids = [g for g in range(model.ngeom)
            if int(model.geom_group[g]) in (0, 1, 2)
            and M._GEOM_TYPE.get(model.geom_type[g]) not in (None, "plane")]
    if len(gids) != len(manifest["geoms"]):
        return None, f"{len(manifest['geoms'])} manifest geoms vs {len(gids)} in model"

    worst, worst_geom = 0.0, None
    for gid, g in zip(gids, manifest["geoms"]):
        if g["type"] != "mesh":
            continue
        b = g["body"]
        # Browser: body group transform, then the geom's local transform.
        R_b, R_g = quat_to_mat(xquat[b]), quat_to_mat(g["quat"])
        v = read_binary_stl(os.path.join(out_dir, "meshes", g["mesh"]))
        drawn = xpos[b] + (R_b @ (np.asarray(g["pos"]) + (R_g @ v.T).T).T).T

        # MuJoCo: the same mesh, placed by the compiled model.
        mid = int(model.geom_dataid[gid])
        va, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, nf = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        truth_v = model.mesh_vert[va:va + nv][model.mesh_face[fa:fa + nf]].reshape(-1, 3)
        truth = data.geom_xpos[gid] + (data.geom_xmat[gid].reshape(3, 3) @ truth_v.T).T

        err = np.abs(drawn - truth).max()
        if err > worst:
            worst, worst_geom = err, g["mesh"]
    return worst, worst_geom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pick_cube", choices=sorted(M.TASKS))
    args = ap.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "yam", "web",
                           "public", "model")
    print(f"task: {args.task}\n")
    print(f"{'arm':<12} {'max vertex error':<20} verdict")
    bad = 0
    for key in E.REGISTRY:
        M.export_embodiment_web(key, args.task)
        err, detail = check(key, args.task, out_dir)
        if err is None:
            print(f"{key:<12} {'-':<20} BROKEN ({detail})")
            bad += 1
            continue
        ok = err < 1e-5
        bad += not ok
        print(f"{key:<12} {err:<20.2e} {'ok' if ok else f'MISPLACED ({detail})'}")
    print(f"\n{bad} arm(s) would render wrong")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
