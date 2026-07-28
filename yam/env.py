"""Tabletop manipulation tasks for the YAM arm.

The env composes a ``SimYamRobot`` (proprioception + cameras + joint-target
control) with task logic (objects, randomization, success, reward). It is
deliberately *not* a gym env: imitation learning consumes ``(observation,
action)`` demonstrations, and the same env is driven by teleop for recording, by
a trained policy for eval, and by the websocket server for the live viewer.

Canonical action (what gets recorded and what a policy predicts):
    ``[q1..q6 (rad), gripper]``  -- absolute joint position targets + gripper
    opening in [0, 1]. This matches YAM's native joint-position control, so it
    transfers to hardware unchanged.

Observation (a dict):
    ``state``   float32(13) = arm qpos(6) + arm qvel(6) + gripper(1)
    ``images``  {camera_name: HxWx3 uint8}
Object/target ground truth is privileged (sim only) and used for reward/success,
never placed in ``observation`` -- on hardware the policy sees objects via the
cameras.

Tasks:
    ``pick_cube``  pick the cube and place it on the green target zone.
    ``reach``      move the gripper to the target sphere (smoke test, no contact).
"""

import numpy as np
import mujoco

from . import model as M
from .robot import SimYamRobot


class YamEnv:
    ACTION_DIM = 7  # 6 arm joint targets + gripper

    def __init__(self, task="pick_cube", control_dt=0.05, seed=0,
                 camera_names=None, cam_height=128, cam_width=128,
                 render_cameras=True, max_steps=300):
        self.task = task
        self.spec = M.TASKS[task]
        self.control_dt = control_dt
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.robot = SimYamRobot(
            task=task, control_dt=control_dt, camera_names=camera_names,
            cam_height=cam_height, cam_width=cam_width,
            render_cameras=render_cameras)
        self.model = self.robot.model
        self.data = self.robot.data
        self.ids = self.robot.ids
        self.camera_names = self.robot.camera_names

        # Joint limits for clamping/denormalizing actions.
        self._arm_low = np.array([self.model.jnt_range[j][0] for j in self.ids.arm_joints])
        self._arm_high = np.array([self.model.jnt_range[j][1] for j in self.ids.arm_joints])

        self._steps = 0
        self._last_action = np.zeros(self.ACTION_DIM, dtype=np.float32)

    # -- randomization -------------------------------------------------------
    def _sample_cube_xy(self):
        return np.array([self.rng.uniform(0.30, 0.50),
                         self.rng.uniform(-0.20, 0.20)])

    def _sample_target(self):
        if self.task == "reach":
            return np.array([self.rng.uniform(0.32, 0.52),
                             self.rng.uniform(-0.24, 0.24),
                             self.rng.uniform(0.12, 0.42)])
        return np.array([self.rng.uniform(0.30, 0.50),
                         self.rng.uniform(-0.26, 0.26),
                         0.001])

    def reset(self):
        self.robot.reset_home()
        # small arm posture noise for demo/policy diversity
        noise = self.rng.uniform(-0.03, 0.03, size=6)
        q0 = M.HOME_QPOS_ARM + noise
        self.data.qpos[self.ids.arm_qpos_adr] = np.clip(q0, self._arm_low, self._arm_high)

        target = self._sample_target()
        if self.spec["has_object"]:
            cube_xy = self._sample_cube_xy()
            # keep target away from the cube so the task requires transport
            for _ in range(10):
                if np.linalg.norm(cube_xy - target[:2]) > 0.14:
                    break
                target[:2] = self._sample_target()[:2]
            adr = self.ids.object_qpos_adr
            self.data.qpos[adr:adr + 3] = [cube_xy[0], cube_xy[1], 0.025]
            self.data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qvel[self.ids.object_dof_adr:self.ids.object_dof_adr + 6] = 0.0

        self.data.mocap_pos[self.ids.target_mocap] = target
        self.data.mocap_quat[self.ids.target_mocap] = [1.0, 0.0, 0.0, 0.0]

        mujoco.mj_forward(self.model, self.data)
        self.robot.hold_current()
        self._steps = 0
        self._last_action = np.concatenate(
            [self.robot.arm_qpos(), [self.robot.gripper_pos()]]).astype(np.float32)
        return self.observation()

    # -- obs / action --------------------------------------------------------
    def observation(self):
        return {
            "state": self.robot.proprio(),
            "images": self.robot.cameras(),
        }

    def apply_action(self, action):
        """Command the robot from a 7-vector action (arm targets rad + gripper),
        step one control period. Returns nothing; call observation() after."""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        arm_target = np.clip(action[:6], self._arm_low, self._arm_high)
        gripper = float(np.clip(action[6], 0.0, 1.0))
        self.robot.command(arm_target, gripper)
        self.robot.step()
        self._last_action = np.concatenate([arm_target, [gripper]]).astype(np.float32)

    def step(self, action):
        self.apply_action(action)
        self._steps += 1
        reward = self.reward()
        success = self.success()
        done = success or self._steps >= self.max_steps
        return self.observation(), reward, done, {"success": success,
                                                  "steps": self._steps}

    # -- task ground truth (privileged) --------------------------------------
    def grasp_pos(self):
        return self.data.site_xpos[self.ids.grasp_site].copy()

    def target_pos(self):
        return self.data.xpos[self.ids.target_body].copy()

    def object_pos(self):
        if not self.spec["has_object"]:
            return None
        adr = self.ids.object_qpos_adr
        return self.data.qpos[adr:adr + 3].copy()

    def reward(self):
        if self.task == "reach":
            return float(-np.linalg.norm(self.grasp_pos() - self.target_pos()))
        cube = self.object_pos()
        d_reach = np.linalg.norm(self.grasp_pos() - cube)
        d_place = np.linalg.norm(cube[:2] - self.target_pos()[:2])
        lifted = cube[2] > 0.06
        r = -d_reach
        if lifted:
            r += 1.5 - d_place
        return float(r)

    def success(self):
        if self.task == "reach":
            d = np.linalg.norm(self.grasp_pos() - self.target_pos())
            return bool(d < self.spec["target_radius"])
        cube = self.object_pos()
        d_place = np.linalg.norm(cube[:2] - self.target_pos()[:2])
        resting = 0.01 < cube[2] < 0.06
        return bool(d_place < self.spec["target_radius"] and resting)

    # -- streaming for the viewer -------------------------------------------
    def body_states(self):
        """(xpos[nbody,3], xquat[nbody,4]) for the live 3-D render."""
        return (self.data.xpos.astype(np.float32).copy(),
                self.data.xquat.astype(np.float32).copy())

    def task_spec(self):
        return {
            "task": self.task,
            "description": self.spec["description"],
            "control_dt": self.control_dt,
            "control_hz": round(1.0 / self.control_dt, 2),
            "max_steps": self.max_steps,
            "action_dim": self.ACTION_DIM,
            "action": "abs joint targets (6 rad) + gripper [0,1]",
            "state_dim": 13,
            "state": "arm qpos(6) + arm qvel(6) + gripper(1)",
            "cameras": list(self.camera_names),
            "cam_shape": [self.robot._rig.height, self.robot._rig.width, 3]
            if self.robot._rig else None,
            "target_radius": self.spec["target_radius"],
            "randomization": {
                "cube_x": [0.30, 0.50], "cube_y": [-0.20, 0.20],
                "target": "reach: xyz box; pick_cube: xy on table",
                "arm_noise": 0.03,
            },
        }

    def close(self):
        self.robot.close()
