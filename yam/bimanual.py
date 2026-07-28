"""Two-arm (bimanual) YAM setup -- deliberately a SEPARATE code path.

The single-arm stack (``yam/model.py``, ``robot.py``, ``env.py``) is left
completely untouched: a bimanual policy differs from a single-arm one, so the
two-arm world gets its own model builder, id resolver and live-viewer session
here rather than trying to generalize the single-arm code.

The model is built by attaching the vendored one-arm MJCF (``yam.xml``) twice
into ``bimanual_scene.xml`` via MuJoCo's ``MjSpec`` attachment, with ``left_``/
``right_`` name prefixes so joints/actuators/sites never clash. The two arms sit
side by side at the near edge of the table, each with its own cube + target, so
each arm runs an independent pick-and-place.

The interactive controllers (``EEController``, ``scripted_expert``) and the IK
solver are reused unchanged: each arm gets a tiny ``_ArmView`` adapter that
exposes just the handful of attributes those classes read (model/data/ids +
ee_pose/arm_qpos + task ground truth).
"""

import os
import time

import numpy as np
import mujoco

from . import model as M
from .ik import DifferentialIK
from .teleop import EEController, scripted_expert

ARMS = ("left", "right")

# Base pose of each arm along the near edge of the table (world y offset).
ARM_BASE_Y = {"left": 0.22, "right": -0.22}

# Registry mirroring M.TASKS, but for the two-arm scenes. Each arm gets its own
# cube/target; success/description are per the whole scene.
BIMANUAL_TASKS = {
    "pick_cube_dual": {
        "scene": "bimanual_scene.xml",
        "description": "each arm picks its cube and places it on its target zone",
        "arms": {
            "left":  {"cube_body": "cube_l", "cube_joint": "cube_l_free",
                      "target_body": "target_l", "cube_y": (0.08, 0.30)},
            "right": {"cube_body": "cube_r", "cube_joint": "cube_r_free",
                      "target_body": "target_r", "cube_y": (-0.30, -0.08)},
        },
        "target_radius": 0.06,
    },
}


def build_model(task="pick_cube_dual"):
    """Compile and return the MjModel for a bimanual task: the scene with two
    prefixed YAM arms attached side by side."""
    spec = BIMANUAL_TASKS[task]
    scene_path = os.path.join(M.ASSETS_DIR, spec["scene"])
    parent = mujoco.MjSpec.from_file(scene_path)
    arm_xml = os.path.join(M.ASSETS_DIR, "yam.xml")
    for side in ARMS:
        # Load a fresh arm spec per side: attach_body mutates the source spec
        # (it would otherwise re-prefix the already-prefixed first arm).
        arm = mujoco.MjSpec.from_file(arm_xml)
        frame = parent.worldbody.add_frame(pos=[0.0, ARM_BASE_Y[side], 0.0])
        frame.attach_body(arm.body("arm"), f"{side}_", "")
    return parent.compile()


def export_web(task="pick_cube_dual", out_dir=None):
    """Build the two-arm model and export its web manifest (+ shared meshes).
    Mesh names are prefixed (``left_model2``); we strip the arm prefixes so the
    viewer reuses the same STL files as the single-arm scenes."""
    model = build_model(task)
    prefixes = tuple(f"{s}_" for s in ARMS)
    manifest = M.scene_manifest(model, task,
                                description=BIMANUAL_TASKS[task]["description"],
                                mesh_prefixes=prefixes)
    return M.export_manifest(manifest, out_dir)


class _ArmIds:
    """The subset of ``M.Ids`` that ``DifferentialIK`` / ``EEController`` /
    ``scripted_expert`` actually touch, resolved for one prefixed arm."""

    def __init__(self, model, side):
        p = f"{side}_"

        def jid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        def aid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        def sid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)

        self.arm_joints = [jid(f"{p}{n}") for n in M.ARM_JOINTS]
        self.arm_qpos_adr = np.array([model.jnt_qposadr[j] for j in self.arm_joints])
        self.arm_dof_adr = np.array([model.jnt_dofadr[j] for j in self.arm_joints])
        self.left_finger_qpos_adr = model.jnt_qposadr[jid(f"{p}left_finger")]

        self.arm_actuators = [aid(f"{p}{n}") for n in M.ARM_ACTUATORS]
        self.gripper_actuator = aid(f"{p}{M.GRIPPER_ACTUATOR}")

        self.grasp_site = sid(f"{p}{M.GRASP_SITE}")
        self.tcp_site = sid(f"{p}{M.TCP_SITE}")


class _ArmView:
    """Quacks like the single-arm ``YamEnv`` enough for the reused teleop /
    scripted-expert code: it carries the shared model+data, this arm's ids, and
    per-arm ground truth (cube/target). ``robot`` points back at itself so
    ``env.robot.ee_pose()`` / ``env.robot.arm_qpos()`` resolve."""

    def __init__(self, model, data, side, arm_spec):
        self.model = model
        self.data = data
        self.side = side
        self.task = "pick_cube"  # drives scripted_expert's pick-and-place branch
        self.ids = _ArmIds(model, side)
        self.robot = self
        self._gripper_open_qpos = 0.0376

        def bid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        def jid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.cube_body = bid(arm_spec["cube_body"])
        self.cube_qpos_adr = model.jnt_qposadr[jid(arm_spec["cube_joint"])]
        self.cube_dof_adr = model.jnt_dofadr[jid(arm_spec["cube_joint"])]
        self.target_body = bid(arm_spec["target_body"])
        self.target_mocap = int(model.body_mocapid[self.target_body])

    # -- interface the reused controllers read -------------------------------
    def arm_qpos(self):
        return self.data.qpos[self.ids.arm_qpos_adr].astype(np.float32)

    def gripper_pos(self):
        lf = float(self.data.qpos[self.ids.left_finger_qpos_adr])
        return float(np.clip(lf / self._gripper_open_qpos, 0.0, 1.0))

    def ee_pose(self):
        pos = self.data.site_xpos[self.ids.grasp_site].astype(np.float64).copy()
        xmat = self.data.site_xmat[self.ids.grasp_site].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, xmat.reshape(-1))
        return pos, quat

    def target_pos(self):
        return self.data.xpos[self.target_body].copy()

    def object_pos(self):
        adr = self.cube_qpos_adr
        return self.data.qpos[adr:adr + 3].copy()


class BimanualSession:
    """Live-viewer session for the two-arm world -- the bimanual counterpart of
    ``server.SimSession`` (same method surface so the server can drive either).
    Cameras/recording are intentionally omitted (the two-arm dataset/policy is
    future work); this class exists to visualize and hand-drive both arms."""

    MODES = ("idle", "scripted", "teleop", "policy")

    def __init__(self, task="pick_cube_dual", cam_size=0, action_fn=None, seed=None):
        self.task = task
        self.spec = BIMANUAL_TASKS[task]
        self.cam_size = 0            # cameras off for the bimanual viewer
        self.control_dt = 0.05
        self.mode = "idle"
        self.speed = 1.0
        self.active = "left"         # which arm teleop drives
        self.rng = np.random.default_rng(
            int(time.time()) % 100000 if seed is None else seed)

        self.model = build_model(task)
        self.data = mujoco.MjData(self.model)
        self._n_substeps = max(1, round(self.control_dt / self.model.opt.timestep))

        self._arm_low = {}
        self._arm_high = {}
        self.views, self.ctl, self.expert = {}, {}, {}
        for side in ARMS:
            v = _ArmView(self.model, self.data, side, self.spec["arms"][side])
            self.views[side] = v
            self._arm_low[side] = np.array([self.model.jnt_range[j][0]
                                            for j in v.ids.arm_joints])
            self._arm_high[side] = np.array([self.model.jnt_range[j][1]
                                             for j in v.ids.arm_joints])

        # A pseudo-env exposing .task/.control_dt so the server treats us like a
        # SimSession (which reads session.env.task / session.env.control_dt).
        self.env = self
        self.reset()

    # -- lifecycle -----------------------------------------------------------
    def set_task(self, task):
        # Only bimanual tasks are valid here; ignore anything else (the server
        # swaps in a single-arm SimSession for single-arm tasks).
        if task in BIMANUAL_TASKS and task != self.task:
            self.__init__(task=task)

    def _sample_cube_xy(self, side):
        ylo, yhi = self.spec["arms"][side]["cube_y"]
        return np.array([self.rng.uniform(0.34, 0.48), self.rng.uniform(ylo, yhi)])

    def _sample_target_xy(self, side):
        ylo, yhi = self.spec["arms"][side]["cube_y"]
        return np.array([self.rng.uniform(0.26, 0.42), self.rng.uniform(ylo, yhi)])

    def _reset_arm(self, side):
        v = self.views[side]
        noise = self.rng.uniform(-0.03, 0.03, size=6)
        q0 = np.clip(M.HOME_QPOS_ARM + noise, self._arm_low[side], self._arm_high[side])
        self.data.qpos[v.ids.arm_qpos_adr] = q0

        cube_xy = self._sample_cube_xy(side)
        target_xy = self._sample_target_xy(side)
        for _ in range(10):
            if np.linalg.norm(cube_xy - target_xy) > 0.14:
                break
            target_xy = self._sample_target_xy(side)
        adr = v.cube_qpos_adr
        self.data.qpos[adr:adr + 3] = [cube_xy[0], cube_xy[1], 0.025]
        self.data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[v.cube_dof_adr:v.cube_dof_adr + 6] = 0.0
        self.data.mocap_pos[v.target_mocap] = [target_xy[0], target_xy[1], 0.001]
        self.data.mocap_quat[v.target_mocap] = [1.0, 0.0, 0.0, 0.0]

    def reset(self):
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        for side in ARMS:
            self._reset_arm(side)
        mujoco.mj_forward(self.model, self.data)
        self._hold_all()
        for side in ARMS:
            self.ctl[side] = EEController(self.views[side])
            self.ctl[side].reset()
            self.expert[side] = None

    def _hold_all(self):
        for side in ARMS:
            v = self.views[side]
            self._command(side, v.arm_qpos(), v.gripper_pos())

    def set_mode(self, mode):
        if mode not in self.MODES:
            return
        self.mode = mode
        if mode == "scripted":
            for side in ARMS:
                self.expert[side] = scripted_expert(self.views[side])
        if mode == "teleop":
            for side in ARMS:
                self.ctl[side].reset()

    # -- control from the browser -------------------------------------------
    def teleop_delta(self, dx, dy, dz):
        self.ctl[self.active].move([dx, dy, dz])

    def _side(self, arm):
        # Commands may carry an explicit arm (two-hand MediaPipe teleop drives
        # both at once); otherwise they go to the keyboard-selected active arm.
        return arm if arm in ARMS else self.active

    def teleop_target(self, x, y, z, arm=None):
        self.ctl[self._side(arm)].set_target([x, y, z])

    def set_gripper(self, g, arm=None):
        self.ctl[self._side(arm)].set_gripper(g)

    def _command(self, side, arm_target, gripper):
        v = self.views[side]
        arm_target = np.asarray(arm_target, dtype=np.float64).reshape(-1)
        for a, val in zip(v.ids.arm_actuators, arm_target):
            self.data.ctrl[a] = val
        g = float(np.clip(gripper, 0.0, 1.0))
        self.data.ctrl[v.ids.gripper_actuator] = (
            M.GRIPPER_CLOSE_CTRL + g * (M.GRIPPER_OPEN_CTRL - M.GRIPPER_CLOSE_CTRL))

    # -- stepping ------------------------------------------------------------
    def step(self):
        if self.mode == "idle":
            return
        for side in ARMS:
            v = self.views[side]
            if self.mode == "scripted":
                if self.expert[side] is None:
                    self.expert[side] = scripted_expert(v)
                exp = self.expert[side]
                action = exp()
                if getattr(exp, "done", False):
                    # loop: fresh cube/target + expert for this arm
                    self._reset_arm(side)
                    self.expert[side] = scripted_expert(v)
            elif self.mode == "teleop":
                # Both arms track their controllers so two-hand teleop can
                # drive them simultaneously; an undriven arm simply holds the
                # target captured at mode entry (ctl.reset()).
                action = self.ctl[side].action()
            else:
                action = np.concatenate([v.arm_qpos(), [v.gripper_pos()]])
            self._command(side, action[:6], action[6])
        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)

    def state_message(self):
        return {
            "type": "state",
            "task": self.task,
            "mode": self.mode,
            "step": 0,
            "success": False,
            "recording": False,
            "active": self.active,
            "xpos": self.data.xpos.astype(np.float32).reshape(-1).round(5).tolist(),
            "xquat": self.data.xquat.astype(np.float32).reshape(-1).round(5).tolist(),
        }

    def on_control(self, msg):
        cmd = msg.get("cmd")
        if cmd == "reset":
            self.reset()
        elif cmd == "task":
            self.set_task(msg.get("task", self.task))
        elif cmd == "mode":
            self.set_mode(msg.get("mode", "idle"))
        elif cmd == "speed":
            self.speed = float(np.clip(msg.get("speed", 1.0), 0.1, 5.0))
        elif cmd == "active_arm":
            side = msg.get("arm", "left")
            if side in ARMS:
                self.active = side
        elif cmd == "teleop_delta":
            self.teleop_delta(msg.get("dx", 0), msg.get("dy", 0), msg.get("dz", 0))
        elif cmd == "teleop_target":
            self.teleop_target(msg["x"], msg["y"], msg["z"], msg.get("arm"))
        elif cmd == "gripper":
            self.set_gripper(msg.get("value", 1.0), msg.get("arm"))
        # record_start/record_stop intentionally unsupported for bimanual.

    def body_states(self):
        return (self.data.xpos.astype(np.float32).copy(),
                self.data.xquat.astype(np.float32).copy())

    def close(self):
        pass
