"""Task-level skills for LLM-agent control of the YAM arm.

An LLM agent decides in seconds, so it cannot close a 20 Hz control loop.
This layer gives it tools at the right abstraction instead: Cartesian
end-effector goals and gripper commands, each executed *closed-loop* by the
existing differential-IK controller (``EEController``) while the agent is away
thinking. ``yam.mcp_server`` exposes these skills over MCP so any tool-calling
model can drive the arm; ``python -m yam.agent --demo`` runs the same skill
sequence scripted, to validate the layer end-to-end without a model.

Privileged state: ``get_state`` reports sim ground-truth object/target poses
(the sim shortcut -- a real deployment grounds these from cameras). Camera
images are exposed too, so a vision-capable agent can operate from pixels and
ignore the privileged fields.
"""

import argparse
import json
import os
import re
import sys
import threading
import time

import mujoco
import numpy as np

from . import model as M
from .env import YamEnv
from .teleop import EEController, FINGERTIP_OFFSET, scripted_expert

# Normalized gripper opening above which fingers commanded shut are considered
# blocked by an object (a held 5 cm cube reads ~0.66; empty reads ~0).
HOLDING_THRESHOLD = 0.15

# spawn_object defaults/limits.
MAX_SPAWNED = 8
SPAWN_DEFAULT_SIZE = {"box": [0.025, 0.025, 0.025], "sphere": [0.025],
                      "cylinder": [0.025, 0.02]}
SPAWN_PALETTE = [  # cycled when no color is given (base cube is red)
    [0.2, 0.45, 0.9, 1.0],   # blue
    [0.95, 0.75, 0.1, 1.0],  # yellow
    [0.7, 0.3, 0.85, 1.0],   # purple
    [0.95, 0.5, 0.15, 1.0],  # orange
    [0.15, 0.75, 0.75, 1.0], # teal
]

# Saved sessions (replayable episodes): frames.npz (per-step body poses) +
# session.json (metadata + the agent activity events).
SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "sessions")


def _safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(name)).strip("._") or "session"


def list_sessions():
    """Summaries of all saved sessions, oldest first (timestamp-prefixed
    names sort chronologically)."""
    out = []
    if not os.path.isdir(SESSIONS_DIR):
        return out
    for n in sorted(os.listdir(SESSIONS_DIR)):
        p = os.path.join(SESSIONS_DIR, n, "session.json")
        if not os.path.isfile(p):
            continue
        try:
            with open(p) as f:
                meta = json.load(f)
        except Exception:
            continue
        out.append({k: meta.get(k) for k in
                    ("name", "task", "sim_steps", "frames", "success",
                     "saved_at", "control_dt")})
    return out


def load_session(name):
    """(meta, steps[T], xpos[T,nbody,3], xquat[T,nbody,4]) for replay."""
    d = os.path.join(SESSIONS_DIR, _safe_name(name))
    with open(os.path.join(d, "session.json")) as f:
        meta = json.load(f)
    npz = np.load(os.path.join(d, "frames.npz"))
    return meta, npz["steps"], npz["xpos"], npz["xquat"]


def _round(x, nd=4):
    return [round(float(v), nd) for v in np.asarray(x, dtype=float).reshape(-1)]


class AgentSession:
    """One env + IK controller; skills run synchronously and return JSON-able
    dicts. This is the layer the MCP tools call into."""

    MAX_EVENTS = 500
    MAX_FRAMES = 40000  # ~33 sim-minutes at 20 Hz (~25 MB in memory)
    MANUAL_MODES = ("idle", "scripted", "teleop")

    def __init__(self, task="pick_cube", cam_size=256, seed=None, pace=0.0,
                 autosave=True):
        self.cam_size = int(cam_size)
        # Real seconds slept per control step (0 = run flat out). Non-zero makes
        # motions watchable in the live viewer.
        self.pace = float(pace)
        self.autosave = bool(autosave)  # save the episode on reset
        self.env = None
        # Agent-spawned objects (spawn_object). Rebuilding the scene bumps
        # scene_version so the viewer knows to reload its 3-D geometry.
        self.extra_objects = []
        self.scene_version = 0
        # Agent activity log: thoughts, tool calls, and visual inputs, each
        # stamped with wall time + sim step. Streamed to the viewer's agent
        # pane. Written from the MCP thread, read from the viewer thread.
        self.events = []
        self._event_id = 0
        self._events_lock = threading.Lock()
        self.current_activity = None  # tool call in flight (live indicator)
        # Manual (human) control: browser autopilot/teleop, same semantics as
        # the modes yam.server had. The server's manual loop steps the sim
        # while mode != idle and no agent tool call is in flight.
        self.mode = "idle"
        self.speed = 1.0
        self.expert = None       # scripted autopilot state machine
        self._recorder = None    # teleop demo recorder (NpzRecorder)
        self._rec_success = False
        self._rec_obs = None     # obs preceding the next recorded action
        # Per-episode recording: (sim_step, xpos, xquat) every control step,
        # so any episode can be saved and replayed in the viewer.
        self._frames = []
        self._frames_lock = threading.Lock()
        self._episode_event_start = 0  # events before this id are a past episode
        self._build(task, seed)

    # -- lifecycle -------------------------------------------------------
    def _build(self, task, seed, preserve_qpos=None, preserve_mocap=None,
               preserve_gripper=None):
        """(Re)compile the scene. With ``preserve_qpos`` the old state is
        copied back in after the rebuild -- free joints append to qpos in
        worldbody order, so the old layout is a prefix of the new one and
        spawning an object keeps the arm and every existing object in place."""
        if self.env is not None:
            self.env.close()
        # Expose every camera the scene defines (front/top/side/wrist), not
        # just the IL default pair.
        m = M.load_model(task)
        cams = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i)
                for i in range(m.ncam)]
        self.env = YamEnv(
            task=task,
            seed=seed if seed is not None else int(time.time()) % 100000,
            camera_names=cams or None,
            cam_height=max(self.cam_size, 1), cam_width=max(self.cam_size, 1),
            render_cameras=self.cam_size > 0,
            max_steps=10 ** 9,  # the agent decides when an episode ends
            extra_objects=self.extra_objects)
        self.env.reset()
        if preserve_qpos is not None:
            n = min(len(preserve_qpos), self.env.model.nq)
            self.env.data.qpos[:n] = preserve_qpos[:n]
            self.env.data.qvel[:] = 0.0
            if preserve_mocap is not None:
                m2 = min(len(preserve_mocap), len(self.env.data.mocap_pos))
                self.env.data.mocap_pos[:m2] = preserve_mocap[:m2]
            mujoco.mj_forward(self.env.model, self.env.data)
            self.env.robot.hold_current()
        self.ctl = EEController(self.env)
        if preserve_gripper is not None:
            self.ctl.set_gripper(preserve_gripper)
        self._init_disturbance_monitor()
        self._begin_episode()

    def _begin_episode(self):
        self.steps = 0
        with self._frames_lock:
            self._frames = []
        with self._events_lock:
            self._episode_event_start = self._event_id
        self._record_frame()  # frame 0: the home pose

    def _scene_changed(self):
        """After a scene recompile: bump the version so viewer streams send
        the new manifest (see yam.mcp_server). Deliberately does NOT write
        the shared web export: manifests on disk are global, so an offline
        session (demo, tests, scripts) re-exporting would clobber the file a
        live viewer is using and desync its geometry from the stream."""
        self.scene_version += 1

    def manifest(self):
        """Scene manifest for THIS session's compiled model -- the viewer
        geometry that matches the body-pose stream exactly."""
        return M.scene_manifest(self.env.model, self.env.task)

    def reset(self, task=None, seed=None):
        task = task or self.env.task
        if task not in M.TASKS:
            raise ValueError(f"unknown task {task!r}; available: {sorted(M.TASKS)}")
        autosaved = None
        if self.autosave and self.steps > 0:
            try:
                autosaved = self.save_session()["name"]
            except Exception as e:  # a failed save must never block a reset
                print(f"[yam.agent] autosave failed: {e}", file=sys.stderr)
        if task != self.env.task or seed is not None:
            if task != self.env.task:  # spawned objects belong to their scene
                self.extra_objects = []
            self._build(task, seed)
            self._scene_changed()
        else:
            self.env.reset()  # in-place: same scene, spawned objects persist
            self.ctl.reset()
            self._begin_episode()
        # A fresh episode invalidates the autopilot's phase machine.
        self.expert = scripted_expert(self.env) if self.mode == "scripted" else None
        state = self.get_state()
        if autosaved:
            state["autosaved_session"] = autosaved
        return state

    def close(self):
        if self.env is not None:
            self.env.close()

    # -- agent activity log ----------------------------------------------
    def log_event(self, kind, **fields):
        """Append one activity event (kind: thought | tool | image)."""
        with self._events_lock:
            self._event_id += 1
            event = {"id": self._event_id, "kind": kind,
                     "wall_time": time.time(), "sim_step": self.steps,
                     **fields}
            self.events.append(event)
            if len(self.events) > self.MAX_EVENTS:
                del self.events[:len(self.events) - self.MAX_EVENTS]
        return event

    def events_since(self, last_id, limit=100):
        """Oldest-first events with id > last_id (at most ``limit``)."""
        with self._events_lock:
            return [e for e in self.events if e["id"] > last_id][:limit]

    # -- session save / replay ---------------------------------------------
    def save_session(self, name=None):
        """Persist the current episode -- per-step body poses + the activity
        events -- to ``yam/sessions/<name>/`` for later replay."""
        with self._frames_lock:
            frames = list(self._frames)
        if len(frames) < 2:
            raise RuntimeError("nothing recorded yet in this episode")
        with self._events_lock:
            events = [e for e in self.events
                      if e["id"] > self._episode_event_start]

        name = _safe_name(name) if name else \
            f"{time.strftime('%Y%m%d-%H%M%S')}_{self.env.task}"
        base, i = name, 2
        while os.path.exists(os.path.join(SESSIONS_DIR, name)):
            name = f"{base}-{i}"
            i += 1
        out_dir = os.path.join(SESSIONS_DIR, name)
        os.makedirs(out_dir, exist_ok=True)

        np.savez_compressed(
            os.path.join(out_dir, "frames.npz"),
            steps=np.array([f[0] for f in frames], dtype=np.int32),
            xpos=np.stack([f[1] for f in frames]),
            xquat=np.stack([f[2] for f in frames]))
        meta = {
            "name": name,
            "task": self.env.task,
            "saved_at": time.time(),
            "sim_steps": int(self.steps),
            "frames": len(frames),
            "success": bool(self.env.success()),
            "control_dt": self.env.control_dt,
            "events": events,
            # The scene as recorded (spawned objects change the body count),
            # so replay can rebuild exactly this geometry.
            "nbody": int(self.env.model.nbody),
            "manifest": M.scene_manifest(self.env.model, self.env.task),
        }
        with open(os.path.join(out_dir, "session.json"), "w") as f:
            json.dump(meta, f)
        return {"name": name, "task": self.env.task, "frames": len(frames),
                "events": len(events), "success": meta["success"],
                "path": out_dir}

    # -- physical disturbance monitoring -------------------------------------
    # The arm is an embodied object: any link can sweep into scene objects,
    # and the agent otherwise has no way to notice (motions servo on
    # proprioception; it only "sees" when it asks for a camera image). Every
    # skill therefore watches the objects while it runs and reports the ones
    # it displaced -- with the arm links that touched them -- so a bump shows
    # up in the tool result instead of going unnoticed until the next photo.
    def _init_disturbance_monitor(self):
        m = self.env.model
        # Robot side: every body in the arm's kinematic subtree. Finger
        # sub-bodies get the friendlier label "gripper".
        j1_body = int(m.jnt_bodyid[self.env.ids.arm_joints[0]])
        base = int(m.body_parentid[j1_body])
        self._robot_bodies = {}
        for b in range(m.nbody):
            cur = b
            while cur != 0:
                if cur == base:
                    name = mujoco.mj_id2name(
                        m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
                    if "finger" in name or name.startswith(("lf_", "rf_")):
                        name = "gripper"
                    self._robot_bodies[b] = name
                    break
                cur = int(m.body_parentid[cur])
        # Watched objects: the task object plus everything the agent spawned.
        self._object_bodies = {}
        if self.env.spec["has_object"]:
            self._object_bodies[int(self.env.ids.object_body)] = \
                self.env.spec["object_body"]
        for o in self.extra_objects:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["name"])
            if b >= 0:
                self._object_bodies[int(b)] = o["name"]
        self._watch_contacts = None  # active only while a skill runs
        self._motion_start_pos = {}

    def _object_positions(self):
        return {name: self.env.data.xpos[b].copy()
                for b, name in self._object_bodies.items()}

    def _begin_motion(self):
        self._motion_start_pos = self._object_positions()
        self._watch_contacts = {n: set() for n in self._object_bodies.values()}

    def _scan_contacts(self):
        d, m = self.env.data, self.env.model
        for i in range(d.ncon):
            b1 = int(m.geom_bodyid[d.contact.geom1[i]])
            b2 = int(m.geom_bodyid[d.contact.geom2[i]])
            for obj, other in ((b1, b2), (b2, b1)):
                name = self._object_bodies.get(obj)
                if name is not None and other in self._robot_bodies:
                    self._watch_contacts[name].add(self._robot_bodies[other])

    def _held_object(self):
        """Name of the object currently held between the fingers, if any."""
        if not (self.ctl.gripper < 0.2 and
                self.env.robot.gripper_pos() > HOLDING_THRESHOLD):
            return None
        ee = self.env.robot.ee_pose()[0]
        best, best_d = None, 0.08
        for b, name in self._object_bodies.items():
            dist = float(np.linalg.norm(self.env.data.xpos[b] - ee))
            if dist < best_d:
                best, best_d = name, dist
        return best

    def _end_motion(self, threshold=0.005):
        """{object: {moved_m, hit_by}} for objects this skill displaced.
        The held object is exempt (it is supposed to move with the arm);
        an empty hit_by means a chain reaction (knocked by another object)
        or the object toppled on its own."""
        contacts = self._watch_contacts or {}
        self._watch_contacts = None
        end = self._object_positions()
        held = self._held_object()
        out = {}
        for name, p0 in self._motion_start_pos.items():
            moved = float(np.linalg.norm(end[name] - p0))
            if name == held or moved < threshold:
                continue
            out[name] = {"moved_m": round(moved, 3),
                         "hit_by": sorted(contacts.get(name, ()))}
        return out

    # -- stepping ----------------------------------------------------------
    def _record_frame(self):
        with self._frames_lock:
            if len(self._frames) < self.MAX_FRAMES:
                xpos, xquat = self.env.body_states()
                self._frames.append((self.steps, xpos, xquat))

    def _step_once(self):
        self.env.apply_action(self.ctl.action())
        self.steps += 1
        if self._watch_contacts is not None:
            self._scan_contacts()
        self._record_frame()
        if self.pace > 0:
            time.sleep(self.pace)

    # -- skills --------------------------------------------------------------
    def move_to(self, x, y, z, tolerance=0.015, max_steps=200):
        """Servo the grasp point to a world position; blocks until reached or
        max_steps control steps have elapsed."""
        requested = np.array([float(x), float(y), float(z)])
        self.ctl.set_target(requested)
        goal = self.ctl.goal_pos.copy()
        err = float(np.linalg.norm(self.env.robot.ee_pose()[0] - goal))
        used = 0
        self._begin_motion()
        for used in range(1, int(max_steps) + 1):
            self._step_once()
            err = float(np.linalg.norm(self.env.robot.ee_pose()[0] - goal))
            if err < tolerance:
                break
        result = {
            "reached": err < tolerance,
            "requested": _round(requested),
            "goal": _round(goal),
            "clamped_to_workspace": bool(np.linalg.norm(goal - requested) > 1e-9),
            "ee_pos": _round(self.env.robot.ee_pose()[0]),
            "error_m": round(err, 4),
            "sim_steps": used,
        }
        disturbed = self._end_motion()
        if disturbed:
            result["disturbed_objects"] = disturbed
        return result

    def move_delta(self, dx=0.0, dy=0.0, dz=0.0, tolerance=0.015, max_steps=200):
        goal = self.ctl.goal_pos + np.array([dx, dy, dz], dtype=np.float64)
        return self.move_to(*goal, tolerance=tolerance, max_steps=max_steps)

    def set_gripper(self, opening, settle_steps=20):
        self._begin_motion()
        self.ctl.set_gripper(opening)
        for _ in range(int(settle_steps)):
            self._step_once()
        result = {
            "commanded": round(float(np.clip(opening, 0.0, 1.0)), 3),
            "opening": round(self.env.robot.gripper_pos(), 3),
        }
        disturbed = self._end_motion()
        if disturbed:
            result["disturbed_objects"] = disturbed
        return result

    def grasp(self):
        r = self.set_gripper(0.0)
        r["holding"] = r["opening"] > HOLDING_THRESHOLD
        return r

    def release(self):
        r = self.set_gripper(1.0)
        r["holding"] = False
        return r

    def wait(self, steps=20):
        self._begin_motion()
        for _ in range(int(steps)):
            self._step_once()
        result = {"waited_steps": int(steps),
                  "ee_pos": _round(self.env.robot.ee_pose()[0])}
        disturbed = self._end_motion()  # e.g. a stack settling or toppling
        if disturbed:
            result["disturbed_objects"] = disturbed
        return result

    # -- scene editing -------------------------------------------------------
    def spawn_object(self, shape="box", name=None, x=None, y=None, z=None,
                     size=None, rgba=None, mass=0.05):
        """Add a free object to the scene (recompiles it in place: the arm and
        every existing object keep their poses; a new recording segment starts
        because the body count changes)."""
        if len(self.extra_objects) >= MAX_SPAWNED:
            raise RuntimeError(f"at most {MAX_SPAWNED} spawned objects")
        if shape not in M.SPAWN_SHAPES:
            raise ValueError(f"shape must be one of {M.SPAWN_SHAPES}")

        size = [float(s) for s in (size or SPAWN_DEFAULT_SIZE[shape])]
        if len(size) != len(SPAWN_DEFAULT_SIZE[shape]):
            raise ValueError(
                f"{shape} needs {len(SPAWN_DEFAULT_SIZE[shape])} size value(s) "
                f"(MuJoCo halfsizes), e.g. {SPAWN_DEFAULT_SIZE[shape]}")
        if not all(0.005 <= s <= 0.08 for s in size):
            raise ValueError("size values must be within [0.005, 0.08] m")

        taken = ({o["name"] for o in self.extra_objects} |
                 {"cube", "target", "table", "floor"})
        name = _safe_name(name) if name else None
        if not name:
            i = 2
            while f"{shape}{i}" in taken:
                i += 1
            name = f"{shape}{i}"
        if name in taken:
            raise ValueError(f"name {name!r} is already in the scene")

        half_h = {"box": size[2] if len(size) > 2 else size[0],
                  "sphere": size[0], "cylinder": size[-1]}[shape]
        rng = self.env.rng
        pos = [float(x) if x is not None else float(rng.uniform(0.30, 0.50)),
               float(y) if y is not None else float(rng.uniform(-0.20, 0.20)),
               float(z) if z is not None else half_h + 0.001]
        pos[2] = max(pos[2], half_h)  # never inside the table

        rgba = ([float(c) for c in rgba] if rgba else
                SPAWN_PALETTE[len(self.extra_objects) % len(SPAWN_PALETTE)])
        if len(rgba) == 3:
            rgba.append(1.0)

        spec = {"name": name, "shape": shape, "size": size, "pos": pos,
                "rgba": rgba, "mass": float(mass)}
        return self._edit_scene(add=spec)

    def remove_objects(self):
        """Remove every spawned object (the base scene is untouched)."""
        if not self.extra_objects:
            return {"ok": True, "note": "no spawned objects to remove",
                    "objects": {}}
        return self._edit_scene(clear=True)

    def _edit_scene(self, add=None, clear=False):
        if self.autosave and self.steps > 0:
            try:
                self.save_session()
            except Exception as e:
                print(f"[yam.agent] autosave failed: {e}", file=sys.stderr)
        qpos = self.env.data.qpos.copy()
        mocap = self.env.data.mocap_pos.copy()
        gripper = self.ctl.gripper
        if clear:
            self.extra_objects = []
        if add is not None:
            self.extra_objects.append(add)
        self._build(self.env.task, seed=None, preserve_qpos=qpos,
                    preserve_mocap=mocap, preserve_gripper=gripper)
        self._scene_changed()
        out = {"ok": True, "scene_version": self.scene_version,
               "objects": self.env.spawned_object_states()}
        if add is not None:
            out["spawned"] = add
        return out

    def check_success(self):
        return {
            "success": bool(self.env.success()),
            "criteria": self.env.spec["description"],
            "sim_steps": self.steps,
        }

    def get_state(self):
        env = self.env
        ee, _ = env.robot.ee_pose()
        opening = env.robot.gripper_pos()
        state = {
            "task": env.task,
            "description": env.spec["description"],
            "ee_pos": _round(ee),
            "joint_angles_rad": _round(env.robot.arm_qpos()),
            "gripper_opening": round(opening, 3),
            "holding": bool(self.ctl.gripper < 0.2 and opening > HOLDING_THRESHOLD),
            "success": bool(env.success()),
            "sim_steps": self.steps,
            "workspace_min_max_xyz": [_round(self.ctl.workspace[:, 0]),
                                      _round(self.ctl.workspace[:, 1])],
            "cameras": list(env.camera_names) if self.cam_size > 0 else [],
            "privileged": {"target_pos": _round(env.target_pos())},
            "hints": {
                "fingertip_offset_m": FINGERTIP_OFFSET,
                "grasp_recipe": (
                    "release; move_to ~0.16 above the object; move_to object "
                    "z + fingertip_offset_m; grasp; lift to z>=0.15 before "
                    "transporting"),
                "body_awareness": (
                    "the WHOLE arm is solid and collides with the scene -- "
                    "every link and the gripper body, not just the "
                    "fingertips. Moving at low z sweeps the gripper through "
                    "anything along the path, and a held object hangs ~0.07 "
                    "m below the reported ee_pos. Travel above obstacles "
                    "(z >= tallest obstacle + 0.10), descend and retreat "
                    "vertically near other objects, and check "
                    "'disturbed_objects' in every motion result: it lists "
                    "what you bumped, how far it moved, and which links hit "
                    "it."),
            },
        }
        if env.spec["has_object"]:
            state["privileged"]["object_pos"] = _round(env.object_pos())
            state["privileged"]["object_halfsize_m"] = 0.025
        if self.extra_objects:
            state["privileged"]["spawned_objects"] = env.spawned_object_states()
        return state

    def camera_image(self, camera=None):
        """(camera_name, HxWx3 uint8) render, for the MCP image tool."""
        rig = self.env.robot._rig
        if rig is None:
            raise RuntimeError("cameras disabled (cam_size=0)")
        camera = camera or self.env.camera_names[0]
        if camera not in self.env.camera_names:
            raise ValueError(
                f"unknown camera {camera!r}; available: {self.env.camera_names}")
        return camera, rig.render(self.env.data, camera)

    # -- manual (human) control --------------------------------------------
    # The browser's sim controls -- mode buttons, keyboard/hand teleop,
    # demo recording -- drive the same sim the agent uses. Arbitration is
    # "last actor wins": an agent tool call drops the mode back to idle
    # (see yam.mcp_server); the human takes over again by picking a mode.
    def set_mode(self, mode):
        if mode not in self.MANUAL_MODES:
            return
        self.mode = mode
        if mode == "scripted":
            self.expert = scripted_expert(self.env)
        if mode == "teleop":
            self.ctl.reset()

    def manual_step(self):
        """One human-driven control step (autopilot or teleop). Called by the
        server's manual loop on the sim thread; a no-op in idle mode."""
        if self.mode == "scripted":
            if self.expert is None:
                self.expert = scripted_expert(self.env)
            action = self.expert()
        elif self.mode == "teleop":
            action = self.ctl.action()
        else:
            return
        if self._recorder is not None:
            self._recorder.add_frame(self._rec_obs, action, 0.0, False, False)
        self.env.apply_action(action)
        self.steps += 1
        self._record_frame()
        if self._recorder is not None:
            self._rec_obs = self.env.observation()
            if self.env.success():
                self._rec_success = True
        if self.mode == "scripted" and getattr(self.expert, "done", False):
            # Autopilot loops: restart the episode in place (no autosave --
            # a looping demo would flood yam/sessions).
            self.env.reset()
            self.ctl.reset()
            self._begin_episode()
            self.expert = scripted_expert(self.env)

    def record_start(self):
        """Start recording a teleop demonstration (IL dataset frames)."""
        from .record import NpzRecorder
        if self.cam_size <= 0:
            print("[yam.agent] recording needs cameras (cam_size > 0)",
                  file=sys.stderr)
            return False
        out = os.path.join(os.path.dirname(__file__), "data",
                           f"{self.env.task}_teleop")
        fps = round(1.0 / self.env.control_dt)
        self._recorder = NpzRecorder(out, self.env, fps,
                                     M.TASKS[self.env.task]["description"])
        self._recorder.start_episode()
        self._rec_success = False
        self._rec_obs = self.env.observation()
        return True

    def record_stop(self, keep=True):
        if self._recorder is None:
            return
        if keep:
            idx = self._recorder.save_episode(self._rec_success)
            self._recorder.finalize(self.env.task_spec())
            print(f"[yam.agent] saved teleop episode {idx} "
                  f"(success={self._rec_success})", file=sys.stderr)
        self._recorder = None
        self._rec_obs = None

    def on_control(self, msg):
        """One manual-control message from the browser (the same command
        vocabulary yam.server speaks). Runs on the sim thread."""
        cmd = msg.get("cmd")
        if cmd == "reset":
            self.reset()
        elif cmd == "task":
            task = msg.get("task")
            if task in M.TASKS and task != self.env.task:
                self.reset(task=task)
        elif cmd == "mode":
            self.set_mode(msg.get("mode", "idle"))
        elif cmd == "speed":
            self.speed = float(np.clip(msg.get("speed", 1.0), 0.1, 5.0))
        elif cmd == "teleop_delta":
            self.ctl.move([msg.get("dx", 0), msg.get("dy", 0), msg.get("dz", 0)])
        elif cmd == "teleop_target":
            self.ctl.set_target([msg["x"], msg["y"], msg["z"]])
        elif cmd == "gripper":
            self.ctl.set_gripper(msg.get("value", 1.0))
        elif cmd == "record_start":
            self.record_start()
        elif cmd == "record_stop":
            self.record_stop(keep=msg.get("keep", True))

    # -- live viewer bridge ----------------------------------------------
    def viewer_message(self):
        """State frame in the format yam/web/app.js expects (read-only)."""
        xpos, xquat = self.env.body_states()
        return {
            "type": "state",
            "task": self.env.task,
            "mode": "agent" if self.current_activity else self.mode,
            "step": int(self.steps),
            "success": bool(self.env.success()),
            "recording": self._recorder is not None,
            "activity": self.current_activity,
            "scene_version": self.scene_version,
            "xpos": xpos.reshape(-1).round(5).tolist(),
            "xquat": xquat.reshape(-1).round(5).tolist(),
        }


# ---------------------------------------------------------------------------
# Scripted demo: drives the same skills an agent would call, validating the
# whole layer (and documenting the expected tool-call sequence).
# ---------------------------------------------------------------------------
def demo(task="pick_cube", seed=None, pace=0.0):
    s = AgentSession(task=task, cam_size=0, seed=seed, pace=pace)

    def log(name, result):
        print(f"[demo] {name}: {json.dumps(result)}")

    try:
        st = s.get_state()
        log("get_state", st)
        target = st["privileged"]["target_pos"]

        if task == "reach":
            log("move_to target", s.move_to(*target, tolerance=0.04))
            result = s.check_success()
            log("check_success", result)
            return result["success"]

        cube = st["privileged"]["object_pos"]
        off = st["hints"]["fingertip_offset_m"]
        log("release", s.release())
        log("move above cube", s.move_to(cube[0], cube[1], cube[2] + 0.16))
        log("descend", s.move_to(cube[0], cube[1], cube[2] + off, tolerance=0.012))
        log("grasp", s.grasp())
        log("lift", s.move_to(cube[0], cube[1], 0.18))
        log("transport", s.move_to(target[0], target[1], 0.18))
        log("lower", s.move_to(target[0], target[1], 0.025 + off + 0.008,
                               tolerance=0.025))
        log("release", s.release())
        log("retreat", s.move_to(target[0], target[1], 0.18))
        result = s.check_success()
        log("check_success", result)
        return result["success"]
    finally:
        s.close()


def _cli():
    p = argparse.ArgumentParser(description="YAM agent skill layer")
    p.add_argument("--demo", action="store_true",
                   help="run a scripted pass through the skills; exit 0 on task success")
    p.add_argument("--task", default="pick_cube", choices=sorted(M.TASKS))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--pace", type=float, default=0.0,
                   help="real seconds slept per control step (0 = flat out)")
    args = p.parse_args()
    if not args.demo:
        p.print_help()
        return
    ok = demo(args.task, seed=args.seed, pace=args.pace)
    print(f"[demo] task success: {ok}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    _cli()
