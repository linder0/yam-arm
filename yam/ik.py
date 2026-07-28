"""Damped-least-squares differential inverse kinematics for the YAM arm.

Turns a desired grasp-point pose into 6 arm-joint targets, so teleop can be done
in intuitive Cartesian space (move/drag the end-effector) instead of per-joint.

Implemented directly on the MuJoCo model (site Jacobian via ``mj_jacSite``) so
there is no extra IK dependency. It runs on a *scratch* MjData copy, leaving the
live sim untouched, and is seeded from the current arm pose each call for
smooth, continuous tracking. (The real ``yam_realtime`` stack uses ``pyroki``
for Cartesian control; this is the sim-side equivalent behind one interface.)
"""

import numpy as np
import mujoco

from . import model as M


def _quat_err(target_wxyz, current_wxyz):
    """Rotation-vector error taking current -> target (world frame), 3-vector."""
    neg_cur = np.array([current_wxyz[0], -current_wxyz[1],
                        -current_wxyz[2], -current_wxyz[3]])
    dq = np.zeros(4)
    mujoco.mju_mulQuat(dq, target_wxyz, neg_cur)
    if dq[0] < 0:  # shortest path
        dq = -dq
    axis_angle = np.zeros(3)
    mujoco.mju_quat2Vel(axis_angle, dq, 1.0)
    return axis_angle


class DifferentialIK:
    def __init__(self, model, ids, damping=0.08, pos_weight=1.0,
                 rot_weight=0.4, max_step=0.2):
        self.model = model
        self.ids = ids
        self.scratch = mujoco.MjData(model)
        self.damping = damping
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight
        self.max_step = max_step  # rad per iteration cap (per joint)

        self.arm_dof = np.asarray(ids.arm_dof_adr)
        self.arm_qadr = np.asarray(ids.arm_qpos_adr)
        arm_joint_ids = ids.arm_joints
        self.jnt_low = np.array([model.jnt_range[j][0] for j in arm_joint_ids])
        self.jnt_high = np.array([model.jnt_range[j][1] for j in arm_joint_ids])

    def solve(self, seed_arm_qpos, target_pos, target_quat=None,
              full_qpos=None, iters=8):
        """Return 6 arm-joint targets that put the grasp site at target_pos
        (and target_quat, if given). ``full_qpos`` seeds the non-arm dofs (e.g.
        the current object pose) so the Jacobian is evaluated in context."""
        d = self.scratch
        if full_qpos is not None:
            d.qpos[:] = full_qpos
        else:
            d.qpos[:] = self.model.qpos0
        q = np.asarray(seed_arm_qpos, dtype=np.float64).copy()
        d.qpos[self.arm_qadr] = q
        d.qvel[:] = 0.0

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        target_pos = np.asarray(target_pos, dtype=np.float64)

        for _ in range(iters):
            d.qpos[self.arm_qadr] = q
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)

            site = self.ids.grasp_site
            cur_pos = d.site_xpos[site]
            perr = (target_pos - cur_pos) * self.pos_weight

            mujoco.mj_jacSite(self.model, d, jacp, jacr, site)
            Jp = jacp[:, self.arm_dof]

            if target_quat is not None:
                cur_quat = np.zeros(4)
                mujoco.mju_mat2Quat(cur_quat, d.site_xmat[site].reshape(-1))
                rerr = _quat_err(np.asarray(target_quat), cur_quat) * self.rot_weight
                Jr = jacr[:, self.arm_dof]
                J = np.vstack([Jp, Jr])
                err = np.concatenate([perr, rerr])
            else:
                J = Jp
                err = perr

            # DLS: dq = J^T (J J^T + lambda^2 I)^-1 err
            lam2 = self.damping ** 2
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + lam2 * np.eye(JJt.shape[0]), err)
            dq = np.clip(dq, -self.max_step, self.max_step)
            q = np.clip(q + dq, self.jnt_low, self.jnt_high)

            if np.linalg.norm(err) < 1e-4:
                break
        return q.astype(np.float64)
