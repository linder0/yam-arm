"""The hardware-swap seam.

Everything above this layer (env, teleop, record, eval, server) talks to a
``YamRobot``: a small interface of "read proprioception + cameras, command joint
targets". Today the only implementation is ``SimYamRobot`` (MuJoCo). Swapping in
the physical arm means implementing ``HardwareYamRobot`` against the ``i2rt`` CAN
SDK and real cameras with the *same* interface -- nothing else changes.

Design choices that keep sim and hardware interchangeable:
  * Commands are **joint position targets** (rad) + a normalized gripper
    opening in ``[0, 1]`` (0 = closed, 1 = open). This is exactly YAM's native
    "Joint position PD" control mode, so it maps 1:1 to hardware.
  * Observations are proprioception (arm qpos/qvel, gripper) + camera images.
    Object ground-truth (cube pose) is *not* part of the robot interface -- it
    only exists in sim and is read straight off the env's MjData. On hardware
    the policy sees objects through the cameras, as intended.
"""

from __future__ import annotations

import abc
from typing import Dict, Tuple

import numpy as np

from . import model as M


class YamRobot(abc.ABC):
    """Minimal arm interface shared by sim and hardware."""

    num_arm_joints = 6
    control_dt: float

    @abc.abstractmethod
    def arm_qpos(self) -> np.ndarray: ...

    @abc.abstractmethod
    def arm_qvel(self) -> np.ndarray: ...

    @abc.abstractmethod
    def gripper_pos(self) -> float:
        """Normalized opening in [0, 1] (0 = closed, 1 = open)."""

    @abc.abstractmethod
    def ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """(pos[3], quat_wxyz[4]) of the grasp point in the world frame."""

    @abc.abstractmethod
    def cameras(self) -> Dict[str, np.ndarray]:
        """name -> HxWx3 uint8 RGB frame."""

    @abc.abstractmethod
    def command(self, arm_qpos_target: np.ndarray, gripper: float) -> None:
        """Set joint position targets (rad) and gripper opening in [0, 1]."""

    @abc.abstractmethod
    def step(self) -> None:
        """Advance one control period (``control_dt``)."""

    def proprio(self) -> np.ndarray:
        """Flat proprioceptive vector: arm qpos(6) + arm qvel(6) + gripper(1)."""
        return np.concatenate([self.arm_qpos(), self.arm_qvel(),
                               [self.gripper_pos()]]).astype(np.float32)


class SimYamRobot(YamRobot):
    """MuJoCo-backed YAM. Owns the sim (model + data); the env reads task/object
    state off ``self.data`` directly (sim-only ground truth)."""

    def __init__(self, task="pick_cube", control_dt=0.05,
                 camera_names=None, cam_height=128, cam_width=128,
                 render_cameras=True):
        import mujoco
        self.mujoco = mujoco
        self.task = task
        self.model = M.load_model(task)
        self.data = mujoco.MjData(self.model)
        self.ids = M.Ids(self.model, task)
        self.control_dt = control_dt
        self._n_substeps = max(1, round(control_dt / self.model.opt.timestep))

        self.camera_names = list(camera_names or M.DEFAULT_CAMERAS)
        self._render_cameras = render_cameras
        self._rig = None
        if render_cameras:
            self._rig = M.CameraRig(self.model, self.camera_names,
                                    height=cam_height, width=cam_width)

        # Gripper normalization: left_finger qpos at fully-open ctrl.
        self._gripper_open_qpos = 0.0376
        self.reset_home()

    # -- state ---------------------------------------------------------------
    def arm_qpos(self):
        return self.data.qpos[self.ids.arm_qpos_adr].astype(np.float32)

    def arm_qvel(self):
        return self.data.qvel[self.ids.arm_dof_adr].astype(np.float32)

    def gripper_pos(self):
        lf = float(self.data.qpos[self.ids.left_finger_qpos_adr])
        return float(np.clip(lf / self._gripper_open_qpos, 0.0, 1.0))

    def ee_pose(self):
        pos = self.data.site_xpos[self.ids.grasp_site].astype(np.float64).copy()
        xmat = self.data.site_xmat[self.ids.grasp_site].reshape(3, 3)
        quat = np.zeros(4)
        self.mujoco.mju_mat2Quat(quat, xmat.reshape(-1))
        return pos, quat  # quat is wxyz

    def cameras(self):
        if not self._render_cameras or self._rig is None:
            return {}
        return self._rig.render_all(self.data)

    # -- control -------------------------------------------------------------
    def command(self, arm_qpos_target, gripper):
        arm_qpos_target = np.asarray(arm_qpos_target, dtype=np.float64).reshape(-1)
        for a, adr in zip(self.ids.arm_actuators, range(6)):
            self.data.ctrl[a] = arm_qpos_target[adr]
        g = float(np.clip(gripper, 0.0, 1.0))
        self.data.ctrl[self.ids.gripper_actuator] = (
            M.GRIPPER_CLOSE_CTRL + g * (M.GRIPPER_OPEN_CTRL - M.GRIPPER_CLOSE_CTRL))

    def hold_current(self):
        """Set the ctrl targets to the current pose (used right after a reset so
        the PD controllers don't yank the arm from its teleport)."""
        self.command(self.arm_qpos(), self.gripper_pos())

    def step(self):
        for _ in range(self._n_substeps):
            self.mujoco.mj_step(self.model, self.data)

    # -- sim-only helpers ----------------------------------------------------
    def reset_home(self):
        self.data.qpos[:] = M.key_qpos_home(self.model, self.ids)
        self.data.qvel[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        self.hold_current()

    def close(self):
        if self._rig is not None:
            self._rig.close()


class HardwareYamRobot(YamRobot):
    """Stub for the physical arm. NOT wired -- documents exactly what to fill in
    so the rest of the stack runs unchanged on real hardware.

    Sketch::

        from i2rt.robots.get_robot import get_yam_robot   # i2rt CAN SDK
        self._arm = get_yam_robot(channel="can0")          # sim=False path
        self._cams = {name: RealSenseCamera(serial) for ...}

    Then:
      * ``arm_qpos``/``arm_qvel``  -> ``self._arm.get_joint_pos()/vel()``
      * ``gripper_pos``            -> normalize the last (gripper) joint
      * ``command``                -> ``self._arm.command_joint_pos([*arm, grip])``
        (YAM's native Joint-position-PD mode; add the gripper as the 7th target)
      * ``ee_pose``                -> forward kinematics (reuse the MuJoCo model,
        or the i2rt kinematics helper)
      * ``cameras``                -> grab synchronized RGB frames
      * ``step``                   -> sleep to hold ``control_dt`` (the arm runs
        its own realtime loop; here we just pace the control rate)

    Safety: YAM has a 400 ms motor timeout -- keep ``command`` calls flowing, and
    clamp targets to joint limits before sending.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "HardwareYamRobot is a documented stub. Implement it against the "
            "i2rt CAN SDK (get_yam_robot) + real cameras; the env/teleop/record/"
            "eval/server code above it needs no changes. See the class docstring "
            "and yam/README.md ('Swapping in real hardware').")


def make_robot(backend="sim", **kwargs) -> YamRobot:
    if backend == "sim":
        return SimYamRobot(**kwargs)
    if backend == "hardware":
        return HardwareYamRobot(**kwargs)
    raise ValueError(f"Unknown robot backend {backend!r} (use 'sim' or 'hardware').")
