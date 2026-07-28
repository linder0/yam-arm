"""Record demonstrations to a dataset for imitation learning.

Frames use LeRobot's key convention so the data is recognizable across the
ecosystem:

    observation.state           float32(13)      arm qpos(6)+qvel(6)+gripper(1)
    observation.images.<cam>    uint8(H,W,3)     one per camera
    action                      float32(7)       abs joint targets + gripper

Two storage backends:

  * ``lerobot`` -- writes a real ``LeRobotDataset`` (parquet + mp4), directly
    consumable by ``lerobot-train``. Used automatically when ``lerobot`` is
    installed.
  * ``npz`` -- a dependency-light fallback (numpy ``.npz`` per episode + JSON
    metadata, same keys/shapes). Lets the collection loop and the rest of the
    pipeline run before the training stack is installed, and is trivially
    convertible to a LeRobotDataset later. Used automatically otherwise.

Actions come from any callable action-source; ``--source expert`` uses the
privileged scripted expert (teleop.scripted_expert) so a full dataset can be
generated headlessly.
"""

import json
import os

import numpy as np

from . import runctx
from .env import YamEnv


def _have_lerobot():
    try:
        import lerobot  # noqa: F401
        return True
    except Exception:
        return False


class NpzRecorder:
    """Numpy-only dataset writer (keys/shapes match the LeRobot convention)."""

    def __init__(self, root, env: YamEnv, fps, task_description):
        self.root = root
        self.env = env
        self.fps = fps
        self.task_description = task_description
        self.cameras = list(env.camera_names)
        os.makedirs(os.path.join(root, "data"), exist_ok=True)
        os.makedirs(os.path.join(root, "meta"), exist_ok=True)
        self._buf = None
        self.episodes = []          # meta rows
        self.total_frames = 0

    def start_episode(self):
        self._buf = {"observation.state": [], "action": [], "timestamp": [],
                     "next.reward": [], "next.done": [], "next.success": []}
        for c in self.cameras:
            self._buf[f"observation.images.{c}"] = []

    def add_frame(self, obs, action, reward, done, success):
        b = self._buf
        b["observation.state"].append(np.asarray(obs["state"], np.float32))
        b["action"].append(np.asarray(action, np.float32))
        b["timestamp"].append(len(b["action"]) / self.fps)
        b["next.reward"].append(np.float32(reward))
        b["next.done"].append(bool(done))
        b["next.success"].append(bool(success))
        for c in self.cameras:
            b[f"observation.images.{c}"].append(obs["images"][c].astype(np.uint8))

    def save_episode(self, success):
        idx = len(self.episodes)
        arrays = {}
        for k, v in self._buf.items():
            arrays[k] = np.asarray(v)
        length = len(self._buf["action"])
        path = os.path.join(self.root, "data", f"episode_{idx:06d}.npz")
        np.savez_compressed(path, **arrays)
        self.episodes.append({"episode_index": idx, "length": length,
                              "success": bool(success),
                              "task": self.task_description})
        self.total_frames += length
        self._buf = None
        return idx

    def finalize(self, task_spec):
        with open(os.path.join(self.root, "meta", "episodes.jsonl"), "w") as f:
            for row in self.episodes:
                f.write(json.dumps(row) + "\n")
        info = runctx.base_config("dataset", task_spec=task_spec, extra={
            "backend": "npz",
            "fps": self.fps,
            "num_episodes": len(self.episodes),
            "num_frames": self.total_frames,
            "cameras": self.cameras,
            "features": {
                "observation.state": {"dtype": "float32", "shape": [13]},
                "action": {"dtype": "float32", "shape": [7]},
                **{f"observation.images.{c}": {"dtype": "uint8",
                    "shape": list(self.env.observation()["images"][c].shape)}
                   for c in self.cameras},
            },
            "note": ("numpy fallback; install lerobot to record a native "
                     "LeRobotDataset, or convert these .npz episodes."),
        })
        runctx.write_config(self.root, info, name="meta/info.json")


class LeRobotRecorder:
    """Best-effort adapter to write a native LeRobotDataset. Falls back to npz
    (raising) if the installed lerobot API differs from what we expect."""

    def __init__(self, root, env: YamEnv, fps, task_description):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        self.env = env
        self.fps = fps
        self.task_description = task_description
        self.cameras = list(env.camera_names)
        img_shape = list(env.observation()["images"][self.cameras[0]].shape)
        features = {
            "observation.state": {"dtype": "float32", "shape": [13], "names": None},
            "action": {"dtype": "float32", "shape": [7], "names": None},
        }
        for c in self.cameras:
            features[f"observation.images.{c}"] = {
                "dtype": "video", "shape": img_shape,
                "names": ["height", "width", "channel"]}
        repo_id = f"yam/{env.task}"
        self.ds = LeRobotDataset.create(
            repo_id=repo_id, fps=fps, root=root, features=features,
            use_videos=True)
        self.episodes = []
        self.total_frames = 0

    def start_episode(self):
        self._len = 0

    def add_frame(self, obs, action, reward, done, success):
        frame = {
            "observation.state": np.asarray(obs["state"], np.float32),
            "action": np.asarray(action, np.float32),
        }
        for c in self.cameras:
            frame[f"observation.images.{c}"] = obs["images"][c].astype(np.uint8)
        self.ds.add_frame(frame, task=self.task_description)
        self._len += 1

    def save_episode(self, success):
        self.ds.save_episode()
        idx = len(self.episodes)
        self.episodes.append({"episode_index": idx, "length": self._len,
                              "success": bool(success)})
        self.total_frames += self._len
        return idx

    def finalize(self, task_spec):
        # Record our provenance alongside LeRobot's own metadata.
        info = runctx.base_config("dataset", task_spec=task_spec, extra={
            "backend": "lerobot",
            "fps": self.fps,
            "num_episodes": len(self.episodes),
            "num_frames": self.total_frames,
            "cameras": self.cameras,
        })
        runctx.write_config(os.path.join(self.ds.root, "meta"),
                            info, name="yam_provenance.json")


def make_recorder(root, env, fps, task_description, backend="auto"):
    if backend == "auto":
        backend = "lerobot" if _have_lerobot() else "npz"
    if backend == "lerobot":
        try:
            return LeRobotRecorder(root, env, fps, task_description)
        except Exception as e:
            print(f"[yam.record] lerobot backend unavailable ({e}); "
                  "falling back to npz.")
            backend = "npz"
    return NpzRecorder(root, env, fps, task_description)


def collect(task="pick_cube", out=None, episodes=50, source="expert",
            backend="auto", cam_size=128, max_steps=400, only_success=True,
            seed=0, cameras=None, verbose=True):
    """Run ``episodes`` rollouts and record them. Returns the output dir."""
    from .teleop import scripted_expert

    out = out or os.path.join(os.path.dirname(__file__), "data", task)
    env = YamEnv(task=task, seed=seed, cam_height=cam_size, cam_width=cam_size,
                 camera_names=cameras, max_steps=max_steps)
    fps = round(1.0 / env.control_dt)
    rec = make_recorder(out, env, fps, M_TASKS(task), backend=backend)

    kept = 0
    attempts = 0
    while kept < episodes:
        attempts += 1
        obs = env.reset()
        if source == "expert":
            src = scripted_expert(env)
            get_action = lambda: src()
        else:
            raise ValueError(
                f"source={source!r}: only 'expert' is supported from the CLI. "
                "For human teleop, record through the websocket server "
                "(python -m yam.server --record).")
        rec.start_episode()
        success = False
        for _ in range(env.max_steps):
            action = get_action()
            rec.add_frame(obs, action, 0.0, False, False)  # obs BEFORE action
            obs, reward, done, info = env.step(action)
            success = info["success"]
            if done:
                break
        if only_success and not success:
            # Discard: re-start the episode buffer without saving.
            rec._buf = None if isinstance(rec, NpzRecorder) else None
            if isinstance(rec, LeRobotRecorder):
                try:
                    rec.ds.clear_episode_buffer()
                except Exception:
                    pass
            if verbose:
                print(f"  [skip] attempt {attempts}: no success")
            continue
        idx = rec.save_episode(success)
        kept += 1
        if verbose:
            print(f"  episode {idx} saved (success={success}) "
                  f"[{kept}/{episodes}]")

    rec.finalize(env.task_spec())
    env.close()
    print(f"[yam.record] wrote {kept} episodes to {out} "
          f"(backend={'lerobot' if _have_lerobot() and backend!='npz' else 'npz'})")
    return out


def M_TASKS(task):
    from . import model as M
    return M.TASKS[task]["description"]


def _cli():
    import argparse
    p = argparse.ArgumentParser(description="Record YAM demonstrations")
    p.add_argument("--task", default="pick_cube", choices=["pick_cube", "reach"])
    p.add_argument("--out", default=None)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--source", default="expert", choices=["expert"])
    p.add_argument("--backend", default="auto", choices=["auto", "lerobot", "npz"])
    p.add_argument("--cam-size", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-failures", action="store_true",
                   help="also keep unsuccessful episodes (default: success-only)")
    args = p.parse_args()
    collect(task=args.task, out=args.out, episodes=args.episodes,
            source=args.source, backend=args.backend, cam_size=args.cam_size,
            max_steps=args.max_steps, only_success=not args.keep_failures,
            seed=args.seed)


if __name__ == "__main__":
    _cli()
