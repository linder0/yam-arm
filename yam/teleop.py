"""End-effector teleoperation + scripted experts.

Two ways to produce actions for the env:

  * ``EEController`` -- Cartesian control of the grasp point via differential IK.
    A human drives it (from the browser: drag the target / keys) by nudging a
    target position and toggling the gripper; ``action()`` returns the env's
    7-vector (arm joint targets + gripper). This is the interactive teleop used
    for recording real demonstrations.

  * ``scripted_expert`` -- a privileged state-machine (it reads the sim's object
    pose) that solves the task with IK. It exists so the whole pipeline
    (record -> train -> eval -> viewer) runs end-to-end without a human in the
    loop, and as an autopilot demo in the live viewer.
"""

import numpy as np

from .ik import DifferentialIK
from . import model as M


# A grasp orientation with the gripper pointing straight down at the table.
# Measured as the grasp-site orientation at the home arm pose (the wrist already
# hangs vertically there); stored as wxyz.
DOWN_QUAT = np.array([0.6918, 0.0125, 0.7219, 0.0046])
DOWN_QUAT = DOWN_QUAT / np.linalg.norm(DOWN_QUAT)

# The fingertips sit this far below the grasp_site along the approach axis, so a
# cube whose centre is at height z is straddled when the grasp_site is commanded
# to z + FINGERTIP_OFFSET.
FINGERTIP_OFFSET = 0.044


class EEController:
    """Cartesian teleop: hold a target EE position (fixed downward orientation)
    and a gripper opening; convert to joint targets with IK each control step."""

    def __init__(self, env, use_orientation=True, workspace=None,
                 max_ee_speed=None):
        self.env = env
        self.ik = DifferentialIK(env.model, env.ids, max_step=0.12)
        info = getattr(env, "info", {})
        # A 5-DOF arm cannot achieve an arbitrary orientation, so it servos
        # position only.
        self.use_orientation = use_orientation and getattr(env, "dof", 6) >= 6
        # Workspace box and slew rate both scale with the arm: the YAM-tuned
        # box excludes an SO-ARM100's entire reachable set.
        if workspace is None:
            workspace = info.get("workspace", np.array([[0.20, 0.60],
                                                        [-0.35, 0.35],
                                                        [0.02, 0.55]]))
        self.workspace = np.asarray(workspace, dtype=np.float64)
        self.down_quat = np.asarray(info.get("down_quat", DOWN_QUAT))
        # Cartesian slew rate: the commanded EE point moves toward the goal by at
        # most this many metres per control step. Keeps motions smooth so the
        # position controllers never whip the arm (which flings grasped objects).
        self.max_ee_speed = max_ee_speed or 0.025 * info.get("reach", 0.79)
        self.reset()

    def reset(self):
        pos, _ = self.env.robot.ee_pose()
        self.target_pos = pos.copy()   # slewed commanded point
        self.goal_pos = pos.copy()     # where we ultimately want to go
        self.gripper = 1.0             # open

    def set_target(self, pos):
        pos = np.asarray(pos, dtype=np.float64)
        self.goal_pos = np.clip(pos, self.workspace[:, 0], self.workspace[:, 1])

    def move(self, delta):
        self.set_target(self.goal_pos + np.asarray(delta, dtype=np.float64))

    def set_gripper(self, g):
        self.gripper = float(np.clip(g, 0.0, 1.0))

    def _slew(self):
        step = self.goal_pos - self.target_pos
        dist = np.linalg.norm(step)
        if dist > self.max_ee_speed:
            step *= self.max_ee_speed / dist
        self.target_pos = self.target_pos + step

    def action(self):
        self._slew()
        quat = self.down_quat if self.use_orientation else None
        arm = self.ik.solve(self.env.robot.arm_qpos(), self.target_pos,
                            target_quat=quat, full_qpos=self.env.data.qpos.copy())
        return np.concatenate([arm, [self.gripper]]).astype(np.float32)


class scripted_expert:
    """Privileged pick-and-place / reach autopilot. Call ``__call__`` once per
    control step; it returns the next 7-vector action and advances its phase.
    Reads env ground truth (object/target pose), so it is a data *generator*,
    not a policy."""

    def __init__(self, env):
        self.env = env
        self.ctl = EEController(env)
        self.phase = 0
        self.timer = 0
        self.done = False
        self.pick_xy = None  # snapshotted at grasp so transport doesn't chase drift
        # Waypoint heights and tolerances are ratios of reach, calibrated so a
        # 0.79 m YAM keeps its original hand-tuned values.
        info = getattr(env, "info", {})
        self.s = info.get("reach", 0.79) / 0.79
        self.fo = info.get("fingertip_offset", FINGERTIP_OFFSET)
        self.cube_half = getattr(env, "cube_half", 0.025)

    def _goto(self, pos, gripper, tol=0.02, max_t=40):
        self.ctl.set_target(pos)
        self.ctl.set_gripper(gripper)
        self.timer += 1
        reached = np.linalg.norm(self.env.robot.ee_pose()[0] - np.asarray(pos)) < tol
        return reached or self.timer > max_t

    def _advance(self):
        self.phase += 1
        self.timer = 0

    def __call__(self):
        env = self.env
        if env.task == "reach":
            self.ctl.set_target(env.target_pos())
            self.ctl.set_gripper(0.5)
            return self.ctl.action()

        target = env.target_pos()
        cube = env.object_pos()  # only used pre-grasp (phases 0-1)
        pre = self.pick_xy if self.pick_xy is not None else cube[:2]

        s, fo = self.s, self.fo
        above_cube = np.array([cube[0], cube[1], cube[2] + 0.16 * s])
        grasp = np.array([cube[0], cube[1], cube[2] + fo])
        # Post-grasp waypoints are FIXED (snapshotted xy), so a little cube drift
        # can't turn into a runaway target-chasing feedback loop.
        lift = np.array([pre[0], pre[1], 0.18 * s])
        above_tgt = np.array([target[0], target[1], 0.18 * s])
        place = np.array([target[0], target[1],
                          self.cube_half + fo + 0.008 * s])

        if self.phase == 0 and self._goto(above_cube, 1.0, tol=0.02 * s):
            self._advance()
        elif self.phase == 1 and self._goto(grasp, 1.0, tol=0.012 * s, max_t=60):
            self.pick_xy = self.env.robot.ee_pose()[0][:2].copy()
            self._advance()
        elif self.phase == 2:  # close gripper (hold in place)
            self._goto(grasp, 0.0, tol=0.0, max_t=15)
            if self.timer > 15:
                self._advance()
        elif self.phase == 3 and self._goto(lift, 0.0, tol=0.02 * s, max_t=45):
            self._advance()
        elif self.phase == 4 and self._goto(above_tgt, 0.0, tol=0.02 * s, max_t=60):
            self._advance()
        elif self.phase == 5 and self._goto(place, 0.0, tol=0.025 * s, max_t=60):
            self._advance()
        elif self.phase == 6:  # open gripper
            self._goto(place, 1.0, tol=0.0, max_t=15)
            if self.timer > 15:
                self._advance()
        elif self.phase >= 7:
            self._goto(above_tgt, 1.0, max_t=30)
            self.done = True
        return self.ctl.action()
