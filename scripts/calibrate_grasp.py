"""Find each arm's grasp orientation by trying candidates, not by reasoning.

``embodiments.derive_down_quat`` defaults to the grasp-site orientation at the
home pose, which is right for YAM and most arms whose vendor keyframe is a
ready pose. It is wrong for the rest: an iiwa14 walks through every phase of
the pick with the cube untouched, and a Kinova's home pose puts its gripper
behind its own base.

Rather than reason about how each vendor oriented its wrist frame, run the
actual pick with each candidate orientation and keep whichever one lifts the
cube. Print the result as a registry field to paste into ``embodiments.py`` --
the search is the measurement, the registry stores the answer.

Usage::

    python scripts/calibrate_grasp.py --arms iiwa14,gen3,so_arm100
"""

import argparse
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from yam import embodiments as E
from yam.env import YamEnv
from yam.teleop import scripted_expert


def score(env, quat, episodes, max_steps):
    """(lifts, successes) for one candidate orientation."""
    lifts = successes = 0
    lift_h = 2.4 * env.cube_half
    for s in range(episodes):
        env.rng = np.random.default_rng(s)
        env.reset()
        expert = scripted_expert(env)
        expert.ctl.down_quat = quat
        expert.ctl.use_orientation = True
        peak, info = 0.0, {"success": False}
        for _ in range(max_steps):
            _, _, done, info = env.step(expert())
            peak = max(peak, float(env.object_pos()[2]))
            if done:
                break
        lifts += int(peak > lift_h)
        successes += int(info["success"])
    return lifts, successes


def calibrate(key, episodes, max_steps):
    env = YamEnv(task="pick_cube", render_cameras=False, max_steps=max_steps,
                 embodiment=key)
    cands = E.grasp_quat_candidates(env.emb, model=env.model)
    rows = []
    for name, q in cands:
        lifts, succ = score(env, q, episodes, max_steps)
        rows.append((name, q, lifts, succ))
        print(f"   {name:<5} lift {lifts}/{episodes}  success {succ}/{episodes}")
    env.close()
    return max(rows, key=lambda r: (r[3], r[2]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", default=",".join(E.REGISTRY))
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=400)
    args = p.parse_args()

    best = {}
    for key in [k.strip() for k in args.arms.split(",") if k.strip()]:
        print(f"\n== {key}")
        try:
            name, q, lifts, succ = calibrate(key, args.episodes, args.max_steps)
        except Exception as e:
            print(f"   FAILED {type(e).__name__}: {e}")
            continue
        best[key] = (name, q, lifts, succ)
        print(f"   -> best '{name}' lift {lifts}/{args.episodes} "
              f"success {succ}/{args.episodes}")

    print("\n\nregistry fields:\n")
    for key, (name, q, lifts, succ) in best.items():
        if name == "home":
            print(f"    # {key}: home pose is already grasp-ready")
            continue
        vals = ", ".join(f"{v:.6f}" for v in q)
        print(f"    # {key}: '{name}' axis down, "
              f"lift {lifts}/{args.episodes} success {succ}/{args.episodes}")
        print(f"    down_quat=[{vals}],")


if __name__ == "__main__":
    main()
