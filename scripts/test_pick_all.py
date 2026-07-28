"""Run the same pick-and-place on every embodiment and report success rates.

This is the end-to-end version of ``test_embodiments.py``: not just whether IK
solves, but whether the unchanged ``scripted_expert`` -- approach, grasp, lift,
transport, place -- completes the task on an arm it was never tuned for. The
expert, the env and the controller are shared; only the embodiment changes.

Usage::

    python scripts/test_pick_all.py --episodes 10
    python scripts/test_pick_all.py --arms yam,so_arm100,ur5e --task reach
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from yam import embodiments as E
from yam.env import YamEnv
from yam.teleop import scripted_expert


def run_arm(key, task, episodes, max_steps):
    env = YamEnv(task=task, render_cameras=False, max_steps=max_steps,
                 embodiment=key)
    ok, steps, lifted = 0, [], 0
    lift_h = 2.4 * env.cube_half
    for s in range(episodes):
        env.rng = np.random.default_rng(s)
        env.reset()
        expert = scripted_expert(env)
        info, peak = {"success": False}, 0.0
        for _ in range(max_steps):
            _, _, done, info = env.step(expert())
            if task != "reach":
                peak = max(peak, float(env.object_pos()[2]))
            if done:
                break
        ok += int(info["success"])
        lifted += int(peak > lift_h)
        if info["success"]:
            steps.append(info["steps"])
    env.close()
    return {
        "key": key, "dof": env.dof, "reach": env.reach,
        "cube_cm": env.cube_half * 200, "success": ok / episodes,
        "lifted": lifted / episodes,
        "med_steps": float(np.median(steps)) if steps else float("nan"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", default=",".join(E.REGISTRY))
    p.add_argument("--task", default="pick_cube")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=400)
    args = p.parse_args()

    rows = []
    for key in [k.strip() for k in args.arms.split(",") if k.strip()]:
        t0 = time.time()
        try:
            r = run_arm(key, args.task, args.episodes, args.max_steps)
            r["secs"] = time.time() - t0
        except Exception as e:
            r = {"key": key, "err": f"{type(e).__name__}: {e}"[:90]}
        rows.append(r)

    print(f"\n{args.task}: {args.episodes} episodes per arm, shared scripted "
          f"expert, no per-robot tuning\n")
    hdr = (f"{'arm':<11}{'dof':>4}{'reach':>7}{'cube':>7}"
           f"{'lifted':>8}{'success':>9}{'steps':>7}{'sec':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("err"):
            print(f"{r['key']:<11}  FAILED  {r['err']}")
            continue
        print(f"{r['key']:<11}{r['dof']:>4}{r['reach']:>7.2f}"
              f"{r['cube_cm']:>6.1f}cm{r['lifted']*100:>7.0f}%"
              f"{r['success']*100:>8.0f}%{r['med_steps']:>7.0f}{r['secs']:>7.1f}")

    good = [r for r in rows if not r.get("err")]
    if good:
        print(f"\nmean success {np.mean([r['success'] for r in good])*100:.0f}% "
              f"across {len(good)} embodiments")


if __name__ == "__main__":
    main()
