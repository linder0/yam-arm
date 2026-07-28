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


# -- dynamic scene composition (agent-spawned objects) ------------------------
# Extra free bodies injected into a task scene at runtime (the spawn_object
# skill). Each spec: {name, shape, size (per-shape MuJoCo halfsizes), pos
# (xyz), rgba, mass}. Free joints append to qpos in worldbody order, so a
# scene extended with more objects keeps the base layout as a prefix --
# that's what lets a rebuild preserve the poses of everything already there.
SPAWN_SHAPES = ("box", "sphere", "cylinder")


def _object_body_xml(o):
    size = " ".join(f"{float(s):g}" for s in o["size"])
    rgba = " ".join(f"{float(c):g}" for c in o["rgba"])
    x, y, z = (float(v) for v in o["pos"])
    return (
        f'    <body name="{o["name"]}" pos="{x:g} {y:g} {z:g}">\n'
        f'      <freejoint name="{o["name"]}_free"/>\n'
        f'      <geom name="{o["name"]}_geom" type="{o["shape"]}" '
        f'size="{size}" rgba="{rgba}" mass="{float(o["mass"]):g}" '
        f'condim="4" friction="1.0 0.05 0.01" contype="1" conaffinity="1"/>\n'
        f"    </body>\n")


def compose_scene_xml(task, extra_objects):
    """Write the task scene + spawned bodies to ``_live_<task>.xml`` (next to
    the base XML so the arm include/meshdir keep resolving); returns the path."""
    with open(task_xml_path(task)) as f:
        xml = f.read()
    bodies = "".join(_object_body_xml(o) for o in extra_objects)
    xml = xml.replace("</worldbody>", bodies + "  </worldbody>")
    path = os.path.join(ASSETS_DIR, f"_live_{task}.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path


def load_model(task="pick_cube", extra_objects=None):
    """Compile and return the MjModel for a task scene (plus any spawned
    objects)."""
    if extra_objects:
        return mujoco.MjModel.from_xml_path(
            compose_scene_xml(task, extra_objects))
    return mujoco.MjModel.from_xml_path(task_xml_path(task))


class Ids:
    """Resolved MuJoCo ids for the handles env/robot/ik need. Built once per
    model so hot loops index by int, never by name.

    ``emb`` supplies the per-robot naming (see ``yam.embodiments``); it defaults
    to YAM, for which these are the module constants above.
    """

    def __init__(self, model, task, emb=None):
        from . import embodiments as E
        self.task = task
        self.spec = TASKS[task]
        self.emb = emb = emb or E.REGISTRY["yam"]

        def jid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        def aid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        def sid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
        def bid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)

        self.arm_joints = [jid(n) for n in emb.arm_joints]
        if min(self.arm_joints) < 0:
            raise ValueError(f"{emb.key}: unresolved arm joints {emb.arm_joints}")
        # qpos/qvel addresses for the arm joints (all hinge => 1 dof each).
        self.arm_qpos_adr = np.array([model.jnt_qposadr[j] for j in self.arm_joints])
        self.arm_dof_adr = np.array([model.jnt_dofadr[j] for j in self.arm_joints])

        self.finger_joints = [jid(n) for n in emb.finger_bodies or ()]
        finger = jid(emb.finger_joint) if emb.finger_joint else -1
        self.left_finger_qpos_adr = model.jnt_qposadr[finger] if finger >= 0 else -1

        self.arm_actuators = [aid(n) for n in emb.arm_actuators]
        self.gripper_actuator = aid(emb.gripper_actuator)

        self.grasp_site = sid(E.grasp_site_name(emb))
        self.tcp_site = sid(TCP_SITE)  # YAM-only; -1 elsewhere

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


def key_qpos_home(model, ids, home_arm=None):
    """A full-length home qpos for a task model: arm at its home pose, gripper
    open, object (if any) at its scene default."""
    from . import embodiments as E
    qpos = np.array(model.qpos0, dtype=np.float64)
    if home_arm is None:
        home_arm = E.home_arm_qpos(getattr(ids, "emb", None) or E.REGISTRY["yam"],
                                   model)
    qpos[ids.arm_qpos_adr] = home_arm
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


def _mesh_triangles(model, mid):
    """Triangles of a compiled mesh, in the mesh's (recentered) frame."""
    va, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    fa, nf = model.mesh_faceadr[mid], model.mesh_facenum[mid]
    verts = model.mesh_vert[va:va + nv]
    faces = model.mesh_face[fa:fa + nf]
    return verts[faces]                      # (nfaces, 3, 3)


def write_binary_stl(path, tris):
    import struct
    n = len(tris)
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    nrm = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.where(ln == 0, 1.0, ln)
    rec = np.zeros(n, dtype=[("d", "<f4", 12), ("attr", "<u2")])
    rec["d"][:, 0:3] = nrm
    rec["d"][:, 3:6] = v0
    rec["d"][:, 6:9] = v1
    rec["d"][:, 9:12] = v2
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", n))
        f.write(rec.tobytes())


def export_model_meshes(model, out_dir, prefix=""):
    """Write every mesh in a compiled model as a binary STL.

    Emitting geometry from the compiled model rather than copying the vendor
    files is what makes the viewer embodiment-agnostic: a UR5e ships only
    ``.obj``, which the browser's STLLoader cannot read, and mesh names collide
    across arms. Compiled vertices are already scaled and in the recentered
    frame, so the manifest for these must omit ``mesh_pos``/``mesh_quat`` --
    the viewer's un-recentering correction would otherwise double-apply.
    """
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    for mid in range(model.nmesh):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid) or f"mesh{mid}"
        fname = f"{prefix}{name}.stl".replace("/", "_")
        write_binary_stl(os.path.join(out_dir, fname), _mesh_triangles(model, mid))
        files[mid] = fname
    return files


def _geom_rgba(model, gid):
    matid = int(model.geom_matid[gid])
    if matid >= 0:
        return [float(x) for x in model.mat_rgba[matid]]
    return [float(x) for x in model.geom_rgba[gid]]


def _strip_prefixes(name, prefixes):
    for p in prefixes:
        if name.startswith(p):
            return name[len(p):]
    return name


def scene_manifest(model, task, include_groups=(0, 1, 2), description=None,
                   mesh_prefixes=(), mesh_files=None):
    """A JSON-able description of the renderable scene: bodies (in id order) and
    their geoms with local transforms. Visual geoms live in groups 0-2; the
    collision groups (3) are skipped by default.

    ``mesh_prefixes`` lets attached models (e.g. the two-arm build, whose meshes
    are named ``left_model2``/``right_model2``) map back to the on-disk STL
    (``model2.stl``) so meshes are shared instead of duplicated per arm."""
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
            if mesh_files is not None:
                # Geometry emitted straight from the compiled model: already
                # scaled and recentered, so no correction fields.
                entry["mesh"] = mesh_files[mid]
                geoms.append(entry)
                meshes_used.add(mesh_files[mid][:-len(".stl")])
                continue
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid)
            mesh_file = _strip_prefixes(mesh_name, mesh_prefixes)
            entry["mesh"] = f"{mesh_file}.stl"
            entry["mesh_scale"] = [float(x) for x in model.mesh_scale[mid]]
            # MuJoCo recenters mesh vertices at compile time (to the mesh COM)
            # and stores the applied transform in mesh_pos/mesh_quat. geom_pos/
            # geom_quat are relative to that *recentered* frame, so a viewer that
            # loads the raw STL must undo this offset (see yam/web/app.js).
            entry["mesh_pos"] = [float(x) for x in model.mesh_pos[mid]]
            entry["mesh_quat"] = [float(x) for x in model.mesh_quat[mid]]  # wxyz
            meshes_used.add(mesh_file)
        geoms.append(entry)

    if description is None:
        description = TASKS[task]["description"]
    return {
        "task": task,
        "description": description,
        "nbody": int(model.nbody),
        "bodies": bodies,
        "geoms": geoms,
        "meshes": sorted(f"{m}.stl" for m in meshes_used),
    }


EMBODIMENT_SEP = "__"


def embodiment_task_id(task, key):
    """Viewer id for a (task, arm) pair, e.g. ``pick_cube__ur5e``. Plain
    ``pick_cube`` stays YAM so existing links and saved sessions keep working.

    The separator doubles as a URL path segment and a directory name, so it
    avoids characters the static file handler would have to percent-decode.
    Task names use single underscores only, so the split is unambiguous.
    """
    return task if key == "yam" else f"{task}{EMBODIMENT_SEP}{key}"


def export_embodiment_web(key, task="pick_cube", out_dir=None):
    """Build a task scene for one arm and write everything the viewer needs.

    The viewer is otherwise YAM-only: it reads a manifest plus STL meshes that
    were copied out of ``yam/assets/yam/assets``. Other arms live in Menagerie,
    ship ``.obj`` as often as ``.stl``, and reuse mesh names like ``base_link``
    across vendors -- so the meshes are regenerated from the compiled model and
    namespaced per arm.
    """
    from . import embodiments as E
    from . import scene as S

    emb = E.REGISTRY[key]
    model, info = S.build(emb, task=task)
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "web", "public", "model")

    files = export_model_meshes(model, os.path.join(out_dir, "meshes"),
                                prefix=f"{key}__")
    label = embodiment_task_id(task, key)
    desc = (f"{TASKS[task]['description']} -- {key} "
            f"({emb.dof} DOF, {info['reach']:.2f} m reach)")
    manifest = scene_manifest(model, task, description=desc, mesh_files=files)
    manifest["task"] = label
    manifest["embodiment"] = key
    export_manifest(manifest, out_dir, copy_meshes=False)
    return label


def export_manifest(manifest, out_dir=None, copy_meshes=True):
    """Write ``<out>/<task>/manifest.json``, copy the STL meshes it references
    into ``<out>/meshes/``, and register the task in ``<out>/index.json``. Works
    for any manifest built by ``scene_manifest`` (single-arm or bimanual)."""
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "web", "public", "model")
    task = manifest["task"]

    task_dir = os.path.join(out_dir, task)
    mesh_dir = os.path.join(out_dir, "meshes")
    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(mesh_dir, exist_ok=True)

    import json
    with open(os.path.join(task_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    if copy_meshes:
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
                   "description": manifest["description"]}
    index["tasks"] = sorted(tasks.values(), key=lambda t: t["task"])
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"[yam.model] exported manifest + {len(manifest['meshes'])} meshes to {task_dir}")
    return task_dir


def export_web(task="pick_cube", out_dir=None):
    """Single-arm web export: build the task manifest and write it."""
    model = load_model(task)
    manifest = scene_manifest(model, task)
    return export_manifest(manifest, out_dir)


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
