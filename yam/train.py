"""Train an imitation-learning policy on recorded YAM demonstrations.

This is a thin wrapper around LeRobot's ``lerobot-train`` CLI (ACT by default,
Diffusion Policy with ``--policy diffusion``). Its job is to (a) build the right
command for our dataset and (b) write the same self-documenting run folder the
rest of this repo uses -- ``runs/<task>/<run>/config.json`` with the command,
git commit, package versions, the dataset reference, and the env ``task_spec``
copied from the dataset metadata.

Training needs the heavy stack (``pip install -r requirements.txt``:
lerobot + torch). Without it, this prints the exact command to run. The dataset
must be a native LeRobotDataset (record with lerobot installed so record.py's
backend is ``lerobot``); the numpy fallback is for pipeline smoke tests, not for
lerobot-train.
"""

import datetime as dt
import json
import os
import shutil
import subprocess
import sys

from . import runctx

POLICIES = {
    "act": ["--policy.type=act"],
    "diffusion": ["--policy.type=diffusion"],
}


def _load_dataset_meta(dataset_root):
    """Return (task_spec, fps, backend) from a dataset's metadata, tolerating
    both our npz layout and a native LeRobot layout."""
    for rel in ("meta/info.json", "meta/yam_provenance.json"):
        p = os.path.join(dataset_root, rel)
        if os.path.exists(p):
            try:
                info = json.load(open(p))
                return (info.get("task_spec"), info.get("fps"),
                        info.get("backend", "unknown"))
            except Exception:
                pass
    return None, None, "unknown"


def build_command(dataset_root, policy, steps, batch_size, output_dir,
                  extra_args=None):
    repo_id = f"yam/{os.path.basename(os.path.normpath(dataset_root))}"
    cmd = ["lerobot-train", *POLICIES[policy],
           f"--dataset.repo_id={repo_id}",
           f"--dataset.root={os.path.abspath(dataset_root)}",
           f"--output_dir={os.path.abspath(output_dir)}",
           f"--steps={steps}",
           f"--batch_size={batch_size}",
           "--policy.push_to_hub=false",
           "--wandb.enable=false"]
    if extra_args:
        cmd += list(extra_args)
    return cmd


def train(dataset, policy="act", steps=20000, batch_size=64, run_name=None,
          logdir=None, run=False, extra_args=None):
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {sorted(POLICIES)}")
    dataset = os.path.abspath(dataset)
    task = os.path.basename(os.path.normpath(dataset))
    logdir = logdir or os.path.join(os.path.dirname(__file__), "runs")
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"{policy}_{ts}"
    out_dir = os.path.join(logdir, task, run_name)
    os.makedirs(out_dir, exist_ok=True)

    task_spec, fps, backend = _load_dataset_meta(dataset)
    cmd = build_command(dataset, policy, steps, batch_size, out_dir, extra_args)

    config = runctx.base_config("lerobot_train", task_spec=task_spec, extra={
        "task": task,
        "policy": policy,
        "dataset": dataset,
        "dataset_backend": backend,
        "steps": steps,
        "batch_size": batch_size,
        "fps": fps,
        "lerobot_command": " ".join(cmd),
        "args": {"policy": policy, "steps": steps, "batch_size": batch_size,
                 "run_name": run_name},
    })
    runctx.write_config(out_dir, config)
    print(f"[yam.train] run folder: {out_dir}")
    print(f"[yam.train] command:\n  {' '.join(cmd)}")

    has_lerobot = shutil.which("lerobot-train") is not None
    if backend != "lerobot":
        print(f"[yam.train] NOTE: dataset backend is '{backend}'. lerobot-train "
              "needs a native LeRobotDataset -- record with lerobot installed "
              "(record.py auto-selects the lerobot backend then).")
    if run:
        if not has_lerobot:
            print("[yam.train] lerobot-train not found. Install the training "
                  "stack: pip install -r requirements.txt")
            return out_dir
        print("[yam.train] launching lerobot-train ...")
        subprocess.run(cmd, check=False)
    else:
        print("[yam.train] dry run (pass --run to launch). After training, "
              "evaluate with:\n"
              f"  python -m yam.eval --policy {out_dir} --task {task} --serve")
    return out_dir


def _cli():
    import argparse
    p = argparse.ArgumentParser(description="Train an IL policy on YAM demos")
    p.add_argument("--dataset", required=True, help="dataset root (yam/data/<task>)")
    p.add_argument("--policy", default="act", choices=sorted(POLICIES))
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--run-name", default=None)
    p.add_argument("--logdir", default=None)
    p.add_argument("--run", action="store_true",
                   help="actually launch lerobot-train (default: print command)")
    args, extra = p.parse_known_args()
    train(dataset=args.dataset, policy=args.policy, steps=args.steps,
          batch_size=args.batch_size, run_name=args.run_name, logdir=args.logdir,
          run=args.run, extra_args=extra)


if __name__ == "__main__":
    _cli()
