"""Evaluate a policy in sim: success rate, optional video, optional live view.

Two policy sources:
  * ``--policy scripted``     -- the privileged expert (runs with no extra deps;
    validates the eval harness + metric + video end to end).
  * ``--policy <run_dir>``    -- a trained LeRobot checkpoint (ACT / Diffusion);
    lazily imports torch + lerobot.

Modes:
  * headless (default)  -- roll out ``--episodes`` episodes, print success rate,
    and optionally save an MP4 (``--video``).
  * ``--serve``         -- stream the policy driving the arm to the live 3-D
    viewer (yam/web) via the websocket server, so you can watch it in a browser.
"""

import os

import numpy as np

from .env import YamEnv


# ---------------------------------------------------------------------------
# Policy adapters: obs dict -> 7-vector action.
# ---------------------------------------------------------------------------
def scripted_action_fn(env):
    """Stateful scripted expert bound to an env; auto-resets its state each
    time the env starts a new episode (detected via the step counter)."""
    from .teleop import scripted_expert
    state = {"exp": scripted_expert(env), "last_step": 0}

    def fn(obs):
        if env._steps == 0 and state["last_step"] != 0:
            state["exp"] = scripted_expert(env)
        state["last_step"] = env._steps
        return state["exp"]()
    return fn


def lerobot_action_fn(run_dir, env):
    """Load a trained LeRobot policy and return an obs->action function. Best
    effort against the installed lerobot API; raises with guidance on mismatch."""
    import torch  # noqa: F401

    # Resolve a policy loader across lerobot versions.
    policy = None
    ckpt = _find_checkpoint(run_dir)
    try:
        from lerobot.common.policies.act.modeling_act import ACTPolicy
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
        for cls in (ACTPolicy, DiffusionPolicy):
            try:
                policy = cls.from_pretrained(ckpt)
                break
            except Exception:
                continue
    except Exception as e:
        raise RuntimeError(
            f"Could not import a LeRobot policy class ({e}). Install/upgrade "
            "the training stack: pip install -r requirements.txt") from e
    if policy is None:
        raise RuntimeError(
            f"No loadable LeRobot policy found under {run_dir}. Point --policy "
            "at the training run's checkpoint dir.")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device).eval()
    if hasattr(policy, "reset"):
        policy.reset()

    def fn(obs):
        if env._steps == 0 and hasattr(policy, "reset"):
            policy.reset()
        batch = {
            "observation.state": torch.as_tensor(
                obs["state"], dtype=torch.float32, device=device)[None],
        }
        for cam, img in obs["images"].items():
            t = torch.as_tensor(img, dtype=torch.float32, device=device) / 255.0
            batch[f"observation.images.{cam}"] = t.permute(2, 0, 1)[None]
        with torch.no_grad():
            action = policy.select_action(batch)
        return action.squeeze(0).cpu().numpy()
    return fn


def _find_checkpoint(run_dir):
    """Return the pretrained-model dir inside a training run (or run_dir itself)."""
    for cand in (os.path.join(run_dir, "checkpoints", "last", "pretrained_model"),
                 os.path.join(run_dir, "pretrained_model"), run_dir):
        if os.path.isdir(cand):
            return cand
    return run_dir


# ---------------------------------------------------------------------------
# Headless evaluation.
# ---------------------------------------------------------------------------
def evaluate(task="pick_cube", policy="scripted", episodes=20, max_steps=400,
             seed=100, video=None, cam_size=128, video_cam="front"):
    need_cam = policy != "scripted"  # trained policies consume images
    env = YamEnv(task=task, seed=seed, render_cameras=need_cam or bool(video),
                 cam_height=cam_size, cam_width=cam_size, max_steps=max_steps)

    if policy == "scripted":
        action_fn = scripted_action_fn(env)
    else:
        action_fn = lerobot_action_fn(policy, env)

    # Dedicated high-res renderer for video (independent of obs cameras).
    writer = None
    rig = None
    if video:
        import mujoco
        rig = mujoco.Renderer(env.model, height=480, width=480)
        writer = _open_video(video, fps=round(1.0 / env.control_dt))

    successes = 0
    lengths = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action = action_fn(obs)
            obs, reward, done, info = env.step(action)
            if writer is not None:
                rig.update_scene(env.data, camera=video_cam)
                writer.append_data(rig.render())
        successes += int(info["success"])
        lengths.append(info["steps"])
        print(f"  episode {ep}: success={info['success']} steps={info['steps']}")

    rate = successes / episodes
    print(f"\n[yam.eval] {policy} on {task}: success {successes}/{episodes} "
          f"= {rate:.0%}, mean steps {np.mean(lengths):.0f}")
    if writer is not None:
        writer.close()
        print(f"[yam.eval] video saved to {video}")
    env.close()
    return rate


def _open_video(path, fps):
    import imageio
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    try:
        return imageio.get_writer(path, fps=fps, macro_block_size=None)
    except Exception as e:
        raise RuntimeError(
            f"Could not open video writer for {path} ({e}). MP4 needs "
            "imageio-ffmpeg (pip install imageio-ffmpeg).") from e


# ---------------------------------------------------------------------------
# Live streaming to the viewer.
# ---------------------------------------------------------------------------
def serve(task="pick_cube", policy="scripted", host="127.0.0.1", port=8080,
          cam_size=128):
    from . import server as S

    if policy == "scripted":
        print("[yam.eval] serving with autopilot available; click 'Autopilot' "
              "in the browser (or use --policy <run_dir> to stream a trained "
              "policy).")
        S.run(host=host, port=port, task=task, cam_size=0)
        return

    # Build the policy action fn against a throwaway env, then hand a closure to
    # the server (it drives its own env, so we bind lazily per session).
    def action_fn_factory(session_env):
        return lerobot_action_fn(policy, session_env)

    # The server instantiates one env per connection; wrap so 'policy' mode uses
    # a fresh policy bound to that env.
    _orig_set_task = S.SimSession.set_task

    def patched_set_task(self, t):
        _orig_set_task(self, t)
        try:
            self.action_fn = action_fn_factory(self.env)
        except Exception as e:
            print("[yam.eval] policy load failed:", e)
            self.action_fn = None
    S.SimSession.set_task = patched_set_task

    print("[yam.eval] serving trained policy; choose 'policy' mode is automatic "
          "-- click Autopilot is scripted, policy streams when loaded.")
    S.run(host=host, port=port, task=task, cam_size=cam_size)


def _cli():
    import argparse
    p = argparse.ArgumentParser(description="Evaluate a YAM policy")
    p.add_argument("--task", default="pick_cube", choices=["pick_cube", "reach"])
    p.add_argument("--policy", default="scripted",
                   help="'scripted' or a training run dir")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--video", default=None, help="save an MP4 to this path")
    p.add_argument("--cam-size", type=int, default=128)
    p.add_argument("--serve", action="store_true",
                   help="stream to the live viewer instead of headless eval")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    if args.serve:
        serve(task=args.task, policy=args.policy, host=args.host, port=args.port,
              cam_size=args.cam_size)
    else:
        evaluate(task=args.task, policy=args.policy, episodes=args.episodes,
                 max_steps=args.max_steps, seed=args.seed, video=args.video,
                 cam_size=args.cam_size)


if __name__ == "__main__":
    _cli()
