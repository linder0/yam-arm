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
                 render_cameras=True, max_steps=300, extra_objects=None,
                 embodiment="yam"):
        self.task = task
        self.spec = M.TASKS[task]
        self.control_dt = control_dt
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        # Agent-spawned free objects (see scene.world_xml).
        self.extra_objects = list(extra_objects or [])

        self.robot = SimYamRobot(
            task=task, control_dt=control_dt, camera_names=camera_names,
            cam_height=cam_height, cam_width=cam_width,
            render_cameras=render_cameras, extra_objects=self.extra_objects,
            embodiment=embodiment)
        self.model = self.robot.model
        self.data = self.robot.data
        self.ids = self.robot.ids
        self.emb = self.robot.emb
        self.info = self.robot.info
        self.camera_names = self.robot.camera_names

        # Everything below scales with the arm: a 0.47 m SO-ARM100 and a 1.55 m
        # UR10e cannot share the YAM-tuned constants this env used to hardcode.
        self.dof = self.emb.dof
        self.ACTION_DIM = self.dof + 1
        self.reach = self.info["reach"]
        self.cube_half = self.info["cube_half"]
        self.target_radius = self.info["target_radius"]

        # qpos/dof addresses of each spawned object's free joint.
        self._extra_adr = []
        for o in self.extra_objects:
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                  f"{o['name']}_free")
            self._extra_adr.append(
                (int(self.model.jnt_qposadr[j]), int(self.model.jnt_dofadr[j])))

        # Joint limits for clamping/denormalizing actions.
        self._arm_low = np.array([self.model.jnt_range[j][0] for j in self.ids.arm_joints])
        self._arm_high = np.array([self.model.jnt_range[j][1] for j in self.ids.arm_joints])

        self._steps = 0
        self._last_action = np.zeros(self.ACTION_DIM, dtype=np.float32)

    # -- randomization -------------------------------------------------------
    # Ratios of reach, calibrated so a 0.79 m YAM reproduces the original
    # hand-tuned ranges (x in [0.30, 0.50], y in [-0.20, 0.20], ...).
    def _sample_cube_xy(self):
        r = self.reach
        return np.array([self.rng.uniform(0.38 * r, 0.63 * r),
                         self.rng.uniform(-0.25 * r, 0.25 * r)])

    def _sample_target(self):
        r = self.reach
        if self.task == "reach":
            return np.array([self.rng.uniform(0.41 * r, 0.66 * r),
                             self.rng.uniform(-0.30 * r, 0.30 * r),
                             self.rng.uniform(0.15 * r, 0.53 * r)])
        return np.array([self.rng.uniform(0.38 * r, 0.63 * r),
                         self.rng.uniform(-0.33 * r, 0.33 * r),
                         0.001])

    def reset(self):
        self.robot.reset_home()
        # small arm posture noise for demo/policy diversity
        noise = self.rng.uniform(-0.03, 0.03, size=self.dof)
        q0 = self.robot._home_arm + noise
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
            self.data.qpos[adr:adr + 3] = [cube_xy[0], cube_xy[1], self.cube_half]
            self.data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qvel[self.ids.object_dof_adr:self.ids.object_dof_adr + 6] = 0.0

        self.data.mocap_pos[self.ids.target_mocap] = target
        self.data.mocap_quat[self.ids.target_mocap] = [1.0, 0.0, 0.0, 0.0]

        # Spawned objects go back to their declared spawn poses.
        for o, (qadr, dadr) in zip(self.extra_objects, self._extra_adr):
            self.data.qpos[qadr:qadr + 3] = o["pos"]
            self.data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qvel[dadr:dadr + 6] = 0.0

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
        arm_target = np.clip(action[:self.dof], self._arm_low, self._arm_high)
        gripper = float(np.clip(action[self.dof], 0.0, 1.0))
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

    def spawned_object_states(self):
        """Current pose + spec of each agent-spawned object."""
        out = {}
        for o, (qadr, _) in zip(self.extra_objects, self._extra_adr):
            out[o["name"]] = {
                "shape": o["shape"],
                "size": [float(s) for s in o["size"]],
                "pos": [float(v) for v in self.data.qpos[qadr:qadr + 3]],
            }
        return out

    def reward(self):
        if self.task == "reach":
            return float(-np.linalg.norm(self.grasp_pos() - self.target_pos()))
        cube = self.object_pos()
        d_reach = np.linalg.norm(self.grasp_pos() - cube)
        d_place = np.linalg.norm(cube[:2] - self.target_pos()[:2])
        lifted = cube[2] > 2.4 * self.cube_half
        r = -d_reach
        if lifted:
            r += 1.5 - d_place
        return float(r)

    def success(self):
        if self.task == "reach":
            d = np.linalg.norm(self.grasp_pos() - self.target_pos())
            return bool(d < self.target_radius)
        cube = self.object_pos()
        d_place = np.linalg.norm(cube[:2] - self.target_pos()[:2])
        resting = 0.4 * self.cube_half < cube[2] < 2.4 * self.cube_half
        return bool(d_place < self.target_radius and resting)

    # -- streaming for the viewer -------------------------------------------
    def body_states(self):
        """(xpos[nbody,3], xquat[nbody,4]) for the live 3-D render."""
        return (self.data.xpos.astype(np.float32).copy(),
                self.data.xquat.astype(np.float32).copy())

    def task_spec(self):
        n = self.dof
        r = self.reach
        return {
            "task": self.task,
            "embodiment": self.emb.key,
            "description": self.spec["description"],
            "control_dt": self.control_dt,
            "control_hz": round(1.0 / self.control_dt, 2),
            "max_steps": self.max_steps,
            "action_dim": self.ACTION_DIM,
            "action": f"abs joint targets ({n} rad) + gripper [0,1]",
            "state_dim": 2 * n + 1,
            "state": f"arm qpos({n}) + arm qvel({n}) + gripper(1)",
            "cameras": list(self.camera_names),
            "cam_shape": [self.robot._rig.height, self.robot._rig.width, 3]
            if self.robot._rig else None,
            "reach_m": round(r, 3),
            "cube_size_m": round(2 * self.cube_half, 4),
            "target_radius": self.target_radius,
            "randomization": {
                "cube_x": [round(0.38 * r, 3), round(0.63 * r, 3)],
                "cube_y": [round(-0.25 * r, 3), round(0.25 * r, 3)],
                "target": "reach: xyz box; pick_cube: xy on table",
                "arm_noise": 0.03,
            },
        }

    def close(self):
        self.robot.close()
