"""Does the YAM control stack transfer to other arms unchanged?

For each embodiment this builds the model (injecting a grasp site and wrist
camera where the vendor MJCF has none), auto-calibrates the gripper, measures
reach, and then runs the *existing, unmodified* ``DifferentialIK`` against
targets that are reachable by construction: sample a random in-limit joint
configuration, take the grasp site's pose under forward kinematics, then ask IK
to recover it starting from the home pose.

That isolates the control layer. If it passes everywhere, the parts of the stack
that do not generalize are model integration, not control.

Usage::

    python scripts/test_embodiments.py
    python scripts/test_embodiments.py --arms panda,so_arm100 --targets 100 --rot
"""

import argparse
import sys
import os

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yam import embodiments as E
from yam.ik import DifferentialIK

TOL = 0.005  # 5 mm counts as solved


class Handles:
    """The subset of ``yam.model.Ids`` that DifferentialIK actually reads."""

    def __init__(self, model, emb):
        jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.arm_joints = [jid(n) for n in emb.arm_joints]
        if min(self.arm_joints) < 0:
            raise ValueError(f"{emb.key}: unresolved arm joints {emb.arm_joints}")
        self.arm_qpos_adr = np.array([model.jnt_qposadr[j] for j in self.arm_joints])
        self.arm_dof_adr = np.array([model.jnt_dofadr[j] for j in self.arm_joints])
        self.grasp_site = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, E.grasp_site_name(emb))
        if self.grasp_site < 0:
            raise ValueError(f"{emb.key}: grasp site missing after build")


def home_qpos(model, emb):
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, emb.home_key)
    if key < 0:
        return np.array(model.qpos0, dtype=np.float64)
    return np.array(model.key_qpos[key], dtype=np.float64)


def ik_trial(emb, model, handles, targets=60, use_rot=False, seed=0, iters=60):
    """Return (success_rate, median_err_mm, p95_err_mm)."""
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    ik = DifferentialIK(model, handles, max_step=0.3)
    qhome = home_qpos(model, emb)
    seed_arm = qhome[handles.arm_qpos_adr].copy()

    errs = []
    for _ in range(targets):
        # Reachable-by-construction target via forward kinematics.
        q = qhome.copy()
        for j in handles.arm_joints:
            lo, hi = model.jnt_range[j]
            if not model.jnt_limited[j]:
                lo, hi = -np.pi, np.pi
            q[model.jnt_qposadr[j]] = rng.uniform(lo, hi)
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        tgt_pos = data.site_xpos[handles.grasp_site].copy()
        tgt_quat = None
        if use_rot:
            tgt_quat = np.zeros(4)
            mujoco.mju_mat2Quat(tgt_quat, data.site_xmat[handles.grasp_site].reshape(-1))

        sol = ik.solve(seed_arm, tgt_pos, tgt_quat, full_qpos=qhome, iters=iters)

        data.qpos[:] = qhome
        data.qpos[handles.arm_qpos_adr] = sol
        mujoco.mj_kinematics(model, data)
        errs.append(float(np.linalg.norm(
            data.site_xpos[handles.grasp_site] - tgt_pos)))

    errs = np.array(errs)
    return float((errs < TOL).mean()), float(np.median(errs) * 1e3), \
        float(np.percentile(errs, 95) * 1e3)


def run(keys, targets, use_rot, menagerie):
    rows = []
    for key in keys:
        emb = E.REGISTRY[key]
        try:
            model = E.build_model(emb, menagerie)
            handles = Handles(model, emb)
            reach = E.measure_reach(emb, model)
            closed, opened, gap_c, gap_o = E.calibrate_gripper(emb, model)
            rate, med, p95 = ik_trial(emb, model, handles, targets, use_rot)
            rows.append(dict(key=key, dof=emb.dof, reach=reach, ok=rate,
                             med=med, p95=p95, closed=closed, opened=opened,
                             gap_c=gap_c * 1e3, gap_o=gap_o * 1e3,
                             injected=emb.grasp_site is None, err=None))
        except Exception as e:
            rows.append(dict(key=key, err=f"{type(e).__name__}: {e}"))

    mode = "pos+rot" if use_rot else "pos-only"
    print(f"\nIK recovery over {targets} FK-generated targets ({mode}), "
          f"tolerance {TOL*1e3:.0f} mm\n")
    hdr = (f"{'arm':<12}{'dof':>4}{'reach':>7}{'site':>10}"
           f"{'solved':>8}{'med mm':>8}{'p95 mm':>8}   gripper ctrl closed->open")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("err"):
            print(f"{r['key']:<12}  FAILED  {r['err']}")
            continue
        site = "injected" if r["injected"] else "vendor"
        print(f"{r['key']:<12}{r['dof']:>4}{r['reach']:>7.2f}{site:>10}"
              f"{r['ok']*100:>7.0f}%{r['med']:>8.2f}{r['p95']:>8.2f}   "
              f"{r['closed']:>7.3f} -> {r['opened']:<7.3f} "
              f"(gap {r['gap_c']:.0f}->{r['gap_o']:.0f} mm)")
    return rows


def _cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", default=",".join(E.REGISTRY))
    p.add_argument("--targets", type=int, default=60)
    p.add_argument("--rot", action="store_true",
                   help="also require orientation (impossible for 5-DOF arms)")
    p.add_argument("--menagerie", default=E.MENAGERIE)
    args = p.parse_args()
    run([k.strip() for k in args.arms.split(",") if k.strip()],
        args.targets, args.rot, args.menagerie)


if __name__ == "__main__":
    _cli()
