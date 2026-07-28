"""Run provenance for datasets and training/eval runs.

Every artifact-producing entry point (record.py, train.py, eval.py) writes a
``config.json`` capturing the command, git commit, package versions, and a
``task_spec`` describing exactly which task/reward constants were in effect, so
runs and datasets stay self-documenting and reproducible.
"""

import datetime as dt
import json
import os
import socket
import subprocess
import sys


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def pkg_version(name):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


def versions():
    return {
        "python": sys.version.split()[0],
        "mujoco": pkg_version("mujoco"),
        "numpy": pkg_version("numpy"),
        "lerobot": pkg_version("lerobot"),
        "torch": pkg_version("torch"),
        "imageio": pkg_version("imageio"),
        "websockets": pkg_version("websockets"),
    }


def base_config(kind, extra=None, task_spec=None):
    """Common provenance block shared by datasets and training/eval runs.

    kind: short tag, e.g. "dataset", "lerobot_train", "eval".
    """
    cfg = {
        "kind": kind,
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "command": " ".join(sys.argv),
        "git_commit": git_commit(),
        "versions": versions(),
        "task_spec": task_spec,
    }
    if extra:
        cfg.update(extra)
    return cfg


def write_config(out_dir, config, name="config.json"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    return path
