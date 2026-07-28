"""Load the vendored YAM MJCF + task scenes, render cameras, and export the
scene geometry the web viewer needs.

The model is the MuJoCo Menagerie ``i2rt_yam`` (MIT), vendored under
``yam/assets/yam/`` with two local additions:

  * a ``wrist`` camera on link_6 (for image observations), and
  * the upstream ``home`` keyframe removed (task scenes add free-joint bodies
    that change ``nq``); the home arm pose lives here as ``HOME_QPOS_ARM``.

Kinematics recap (checked against the compiled model):
  * 6 arm joints ``joint1..joint6`` + a coupled 2-finger gripper
    (``left_finger`` mirrored to ``right_finger`` by an equality).
  * 7 position actuators: ``joint1..joint6`` (rad) + ``gripper`` (metres,
    ctrlrange ``[0, 0.041]``; ``0`` = closed, ``0.041`` = open).
  * The point between the fingers is the ``grasp_site``.
"""

import os
import shutil

import numpy as np
import mujoco

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets", "yam")

# -- names (single source of truth for env / robot / ik) ---------------------
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
FINGER_JOINTS = ["left_finger", "right_finger"]
ARM_ACTUATORS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_ACTUATOR = "gripper"
GRASP_SITE = "grasp_site"
TCP_SITE = "tcp_site"

# Home arm pose (replaces the removed keyframe): joint2/joint3 lifted so the
# gripper hovers above the table in a natural, reachable posture.
HOME_QPOS_ARM = np.array([0.0, 1.047, 1.047, 0.0, 0.0, 0.0], dtype=np.float64)
GRIPPER_OPEN_CTRL = 0.041
GRIPPER_CLOSE_CTRL = 0.0

# Cameras defined in the task scenes (+ the wrist cam on the arm).
DEFAULT_CAMERAS = ["front", "wrist"]

# Task registry: scene file + task-specific body handles. Scene files live next
# to yam.xml so the arm's meshdir="assets" keeps resolving through <include>.
TASKS = {
    "pick_cube": {
        "xml": "pick_cube.xml",
        "has_object": True,
        "object_body": "cube",
        "object_joint": "cube_free",
        "object_geom": "cube_geom",
        "target_body": "target",   # mocap
        "target_radius": 0.06,
        "description": "pick up the cube and place it on the green target zone",
    },
    "reach": {
        "xml": "reach.xml",
        "has_object": False,
        "target_body": "target",   # mocap
        "target_radius": 0.05,
        "description": "move the gripper to the target sphere (smoke test)",
    },
}


def task_xml_path(task):
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}. Available: {sorted(TASKS)}")
    return os.path.join(ASSETS_DIR, TASKS[task]["xml"])


def load_model(task="pick_cube"):
    """Compile and return the MjModel for a task scene."""
    return mujoco.MjModel.from_xml_path(task_xml_path(task))


class Ids:
    """Resolved MuJoCo ids for the handles env/robot/ik need. Built once per
    model so hot loops index by int, never by name."""

    def __init__(self, model, task):
        self.task = task
        self.spec = TASKS[task]

        def jid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        def aid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        def sid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
        def bid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)

        self.arm_joints = [jid(n) for n in ARM_JOINTS]
        self.finger_joints = [jid(n) for n in FINGER_JOINTS]
        # qpos/qvel addresses for the arm joints (all hinge => 1 dof each).
        self.arm_qpos_adr = np.array([model.jnt_qposadr[j] for j in self.arm_joints])
        self.arm_dof_adr = np.array([model.jnt_dofadr[j] for j in self.arm_joints])
        self.left_finger_qpos_adr = model.jnt_qposadr[jid("left_finger")]

        self.arm_actuators = [aid(n) for n in ARM_ACTUATORS]
        self.gripper_actuator = aid(GRIPPER_ACTUATOR)

        self.grasp_site = sid(GRASP_SITE)
        self.tcp_site = sid(TCP_SITE)

        self.object_body = bid(self.spec["object_body"]) if self.spec["has_object"] else -1
        if self.spec["has_object"]:
            oj = jid(self.spec["object_joint"])
            self.object_qpos_adr = model.jnt_qposadr[oj]   # 7: xyz + wxyz
            self.object_dof_adr = model.jnt_dofadr[oj]     # 6: linear + angular
            self.object_geom = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, self.spec["object_geom"])
        self.target_body = bid(self.spec["target_body"])
        # mocap index for the target (used to move it at reset).
        self.target_mocap = int(model.body_mocapid[self.target_body])


def key_qpos_home(model, ids):
    """A full-length home qpos for a task model: arm at HOME_QPOS_ARM, gripper
    open, object (if any) at its scene default."""
    qpos = np.array(model.qpos0, dtype=np.float64)
    qpos[ids.arm_qpos_adr] = HOME_QPOS_ARM
    return qpos


# -- camera rendering --------------------------------------------------------

class CameraRig:
    """Offscreen RGB renderer for a set of named cameras. One MjvCamera-free
    ``mujoco.Renderer`` per resolution; kept modest for imitation-learning
    image obs (default 128x128 per view)."""

    def __init__(self, model, cameras=None, height=128, width=128):
        self.model = model
        self.cameras = list(cameras or DEFAULT_CAMERAS)
        self.height = height
        self.width = width
        self._renderer = mujoco.Renderer(model, height=height, width=width)

    def render(self, data, camera):
        self._renderer.update_scene(data, camera=camera)
        return self._renderer.render()  # (H, W, 3) uint8

    def render_all(self, data):
        return {c: self.render(data, c) for c in self.cameras}

    def close(self):
        try:
            self._renderer.close()
        except Exception:
            pass


# -- web viewer scene manifest ----------------------------------------------
#
# The live viewer (yam/web) rebuilds the scene like web-next's webgpuRenderer:
# one three.js group per MuJoCo body, child meshes per geom with the geom's
# *local* pos/quat, then each frame the body groups are moved to the streamed
# world xpos/xquat. This exports that static description + the STL meshes.

_GEOM_TYPE = {
    mujoco.mjtGeom.mjGEOM_PLANE: "plane",
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
    mujoco.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
    mujoco.mjtGeom.mjGEOM_BOX: "box",
    mujoco.mjtGeom.mjGEOM_MESH: "mesh",
}


def _geom_rgba(model, gid):
    matid = int(model.geom_matid[gid])
    if matid >= 0:
        return [float(x) for x in model.mat_rgba[matid]]
    return [float(x) for x in model.geom_rgba[gid]]


def scene_manifest(model, task, include_groups=(0, 1, 2)):
    """A JSON-able description of the renderable scene: bodies (in id order) and
    their geoms with local transforms. Visual geoms live in groups 0-2; the
    collision groups (3) are skipped by default."""
    bodies = []
    for b in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
        bodies.append({"id": b, "name": name})

    geoms = []
    meshes_used = set()
    for g in range(model.ngeom):
        group = int(model.geom_group[g])
        if group not in include_groups:
            continue
        gtype = _GEOM_TYPE.get(model.geom_type[g])
        if gtype is None or gtype == "plane":
            # Planes are drawn by the viewer as its own ground grid.
            continue
        entry = {
            "body": int(model.geom_bodyid[g]),
            "type": gtype,
            "size": [float(x) for x in model.geom_size[g]],
            "pos": [float(x) for x in model.geom_pos[g]],
            "quat": [float(x) for x in model.geom_quat[g]],  # wxyz
            "rgba": _geom_rgba(model, g),
        }
        if gtype == "mesh":
            mid = int(model.geom_dataid[g])
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid)
            entry["mesh"] = f"{mesh_name}.stl"
            entry["mesh_scale"] = [float(x) for x in model.mesh_scale[mid]]
            meshes_used.add(mesh_name)
        geoms.append(entry)

    return {
        "task": task,
        "description": TASKS[task]["description"],
        "nbody": int(model.nbody),
        "bodies": bodies,
        "geoms": geoms,
        "meshes": sorted(f"{m}.stl" for m in meshes_used),
    }


def export_web(task="pick_cube", out_dir=None):
    """Write ``<out>/<task>/manifest.json`` and copy the STL meshes it needs
    into ``<out>/meshes/`` for the three.js viewer."""
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "web", "public", "model")
    model = load_model(task)
    manifest = scene_manifest(model, task)

    task_dir = os.path.join(out_dir, task)
    mesh_dir = os.path.join(out_dir, "meshes")
    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(mesh_dir, exist_ok=True)

    import json
    with open(os.path.join(task_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    src_mesh_dir = os.path.join(ASSETS_DIR, "assets")
    for mesh in manifest["meshes"]:
        src = os.path.join(src_mesh_dir, mesh)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(mesh_dir, mesh))

    # A tiny index so the client can discover available tasks.
    index_path = os.path.join(out_dir, "index.json")
    index = {"tasks": []}
    if os.path.exists(index_path):
        try:
            index = json.load(open(index_path))
        except Exception:
            pass
    tasks = {t["task"]: t for t in index.get("tasks", [])}
    tasks[task] = {"task": task, "manifest": f"{task}/manifest.json",
                   "description": TASKS[task]["description"]}
    index["tasks"] = sorted(tasks.values(), key=lambda t: t["task"])
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"[yam.model] exported manifest + {len(manifest['meshes'])} meshes to {task_dir}")
    return task_dir


def _cli():
    import argparse
    p = argparse.ArgumentParser(description="YAM model utilities")
    p.add_argument("--task", default="pick_cube", choices=sorted(TASKS))
    p.add_argument("--export-web", action="store_true",
                   help="write the web viewer manifest + copy meshes")
    p.add_argument("--out", default=None, help="web export dir")
    p.add_argument("--check", action="store_true",
                   help="compile the model and print a summary")
    args = p.parse_args()

    if args.check or not args.export_web:
        model = load_model(args.task)
        ids = Ids(model, args.task)
        data = mujoco.MjData(model)
        data.qpos[:] = key_qpos_home(model, ids)
        mujoco.mj_forward(model, data)
        print(f"task={args.task} nq={model.nq} nv={model.nv} nu={model.nu} "
              f"ncam={model.ncam}")
        print("grasp_site (home):", np.round(data.site_xpos[ids.grasp_site], 3))
    if args.export_web:
        export_web(args.task, args.out)


if __name__ == "__main__":
    _cli()
