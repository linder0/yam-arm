"""Survey MuJoCo Menagerie manipulators to see what an embodiment-agnostic
layer would actually have to absorb.

For every candidate arm this compiles the MJCF and reports the handles the
YAM stack currently hardcodes in ``yam/model.py``: arm joints, actuators and
their ctrlranges, sites, cameras, keyframes, and whether a gripper is even
present. It also samples random configurations to estimate reach, since the
Cartesian workspace box in ``EEController`` is YAM-sized and has to be derived
per robot rather than copied.

Usage::

    python scripts/survey_arms.py --menagerie ~/mujoco_menagerie
    python scripts/survey_arms.py --all --json /tmp/arms.json
"""

import argparse
import json
import os
import glob

import numpy as np
import mujoco

# Substrings that mark a model as something other than a table-top manipulator.
_SKIP = ("go1", "go2", "go_bdx", "a1", "aliengo", "b1", "b2", "spot", "anymal",
         "barkour", "cassie", "h1", "g1", "t1", "op3", "humanoid", "talos",
         "skydio", "crazyflie", "bitcraze", "drone", "car", "hand", "allegro",
         "shadow", "leap", "ability", "robotiq", "adam", "booster", "berkeley")

_GRIPPER_WORDS = ("finger", "gripper", "grip", "claw", "jaw", "pinch", "hand")


def _names(model, objtype, count):
    out = []
    for i in range(count):
        out.append(mujoco.mj_id2name(model, objtype, i) or f"<{i}>")
    return out


def _is_gripper(name):
    low = name.lower()
    return any(w in low for w in _GRIPPER_WORDS)


def candidate_xmls(menagerie, include_all=False):
    """Yield (model_dir_name, xml_path) for each plausible manipulator.

    A directory often holds several MJCFs (``panda.xml``, ``panda_nohand.xml``,
    ``hand.xml``, ``mjx_*``). Picking by filename gets this wrong, so compile
    every candidate and keep the one with the most actuators -- the whole robot
    rather than a detached gripper.
    """
    for entry in sorted(os.listdir(menagerie)):
        path = os.path.join(menagerie, entry)
        if not os.path.isdir(path) or entry.startswith("."):
            continue
        if not include_all and any(s in entry.lower() for s in _SKIP):
            continue
        best, best_nu = None, -1
        for xml in sorted(glob.glob(os.path.join(path, "*.xml"))):
            base = os.path.basename(xml)
            if "scene" in base or base.startswith(("mjx_", "_")):
                continue
            try:
                nu = int(mujoco.MjModel.from_xml_path(xml).nu)
            except Exception:
                continue
            if nu > best_nu:
                best, best_nu = xml, nu
        if best is not None:
            yield entry, best


def estimate_reach(model, samples=400, seed=0):
    """Rough max distance (m) from the base body to any body, over random
    in-limit configurations. A proxy for how big the workspace box should be."""
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    lo = model.jnt_range[:, 0].copy()
    hi = model.jnt_range[:, 1].copy()
    limited = model.jnt_limited.astype(bool)
    best = 0.0
    for _ in range(samples):
        q = np.array(model.qpos0, dtype=np.float64)
        for j in range(model.njnt):
            if model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE,
                                         mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            adr = model.jnt_qposadr[j]
            if limited[j]:
                q[adr] = rng.uniform(lo[j], hi[j])
            else:
                q[adr] = rng.uniform(-np.pi, np.pi)
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        base = data.xpos[1] if model.nbody > 1 else np.zeros(3)
        d = np.linalg.norm(data.xpos[1:] - base, axis=1).max()
        best = max(best, float(d))
    return best


def survey(name, xml_path, reach_samples=400):
    model = mujoco.MjModel.from_xml_path(xml_path)

    joints = []
    for j in range(model.njnt):
        jtype = model.jnt_type[j]
        if jtype not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"<j{j}>"
        joints.append({
            "name": jname,
            "type": "hinge" if jtype == mujoco.mjtJoint.mjJNT_HINGE else "slide",
            "limited": bool(model.jnt_limited[j]),
            "range": [round(float(x), 4) for x in model.jnt_range[j]],
            "gripper": _is_gripper(jname),
        })

    actuators = []
    for a in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"<a{a}>"
        actuators.append({
            "name": aname,
            "ctrlrange": [round(float(x), 4) for x in model.actuator_ctrlrange[a]],
            "gripper": _is_gripper(aname),
        })

    arm_joints = [j for j in joints if not j["gripper"]]
    grip_joints = [j for j in joints if j["gripper"]]
    grip_acts = [a for a in actuators if a["gripper"]]

    return {
        "name": name,
        "xml": os.path.relpath(xml_path, os.path.dirname(os.path.dirname(xml_path))),
        "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
        "dof_arm": len(arm_joints),
        "has_gripper": bool(grip_joints or grip_acts),
        "arm_joints": [j["name"] for j in arm_joints],
        "gripper_joints": [j["name"] for j in grip_joints],
        "actuators": actuators,
        "sites": _names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite),
        "cameras": _names(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam),
        "keyframes": _names(model, mujoco.mjtObj.mjOBJ_KEY, model.nkey),
        "bodies": _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody),
        "reach_m": round(estimate_reach(model, reach_samples), 3),
        "joints": joints,
    }


def _cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--menagerie", default=os.path.expanduser("~/mujoco_menagerie"))
    p.add_argument("--all", action="store_true",
                   help="do not skip non-manipulator models")
    p.add_argument("--json", default=None, help="write full results here")
    p.add_argument("--reach-samples", type=int, default=400)
    args = p.parse_args()

    if not os.path.isdir(args.menagerie):
        raise SystemExit(f"menagerie not found at {args.menagerie}")

    results, failures = [], []
    for name, xml in candidate_xmls(args.menagerie, args.all):
        try:
            results.append(survey(name, xml, args.reach_samples))
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"[:120]))

    hdr = f"{'model':<28}{'dof':>4}{'nu':>4}{'grip':>6}{'reach':>7}  {'sites':<28}cameras"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(results, key=lambda r: (r["dof_arm"], r["name"])):
        sites = ",".join(r["sites"][:2]) or "-"
        cams = ",".join(r["cameras"][:3]) or "-"
        print(f"{r['name']:<28}{r['dof_arm']:>4}{r['nu']:>4}"
              f"{('yes' if r['has_gripper'] else 'NO'):>6}{r['reach_m']:>7.2f}  "
              f"{sites:<28}{cams}")

    print(f"\n{len(results)} compiled, {len(failures)} failed")
    for name, err in failures:
        print(f"  FAIL {name}: {err}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"models": results, "failures": failures}, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    _cli()
