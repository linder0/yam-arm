"""MCP (Model Context Protocol) server for agent control of the YAM arm.

Exposes the task-level skills in ``yam.agent`` as MCP tools over stdio, so any
tool-calling LLM (Cursor, Claude, etc.) can drive the simulated arm: read
state, look through cameras, servo the end-effector, grasp/release, and check
task success. Stdlib-only, like the rest of the stack -- the JSON-RPC framing
is a few dozen lines, no MCP SDK needed.

By default it also starts the live 3-D viewer (same three.js page as
``yam.server``) in a background thread, read-only, so you can watch the agent
work at http://localhost:8765. Register in Cursor via ``.cursor/mcp.json``.

Protocol notes: newline-delimited JSON-RPC 2.0 on stdin/stdout. Anything else
(mujoco chatter, asset export logs) is rerouted to stderr so it cannot corrupt
the stream.
"""

import argparse
import asyncio
import base64
import concurrent.futures
import hashlib
import json
import os
import sys
import threading
import time

from . import agent as A
from . import model as M
from .agent import AgentSession

PROTOCOL_VERSION = "2024-11-05"

# Bumped by the replay_session tool; viewer connections watch it so a replay
# can be triggered from chat ("replay the last session") as well as from the
# viewer's own Replay button.
_REPLAY_REQUEST = {"seq": 0, "name": None}

_OBJ = {"type": "object", "properties": {}, "additionalProperties": False}


def _schema(props, required=()):
    return {"type": "object", "properties": props,
            "required": list(required), "additionalProperties": False}


def _num(desc):
    return {"type": "number", "description": desc}


TOOLS = [
    {
        "name": "think",
        "description": (
            "Record one step of chain-of-thought reasoning: your plan, how "
            "you interpret an observation, or error analysis. Has no effect "
            "on the sim; it is timestamped and rendered in the live viewer's "
            "agent activity pane so a human can follow along. Call this "
            "before actions to narrate what you are about to do and why."),
        "inputSchema": _schema({"thought": {"type": "string",
                                            "description": "the reasoning step to record"}},
                               required=("thought",)),
    },
    {
        "name": "get_state",
        "description": (
            "Current world state: end-effector position, gripper opening, task "
            "success, workspace bounds, camera names, plus privileged sim "
            "ground truth (object/target positions) and grasp hints. Call this "
            "first each episode."),
        "inputSchema": _OBJ,
    },
    {
        "name": "reset",
        "description": (
            "Start a fresh episode: rehome the arm and randomize object/target "
            "placement. Optionally switch task or fix the randomization seed."),
        "inputSchema": _schema({
            "task": {"type": "string", "enum": sorted(M.TASKS),
                     "description": "task to load (default: keep current)"},
            "seed": {"type": "integer", "description": "randomization seed"},
        }),
    },
    {
        "name": "move_to",
        "description": (
            "Servo the end-effector (the grasp point between the fingertips; "
            "the gripper always points straight down) to world position "
            "[x, y, z] in metres, closed-loop via differential IK. Frame: "
            "origin at the arm base on the table, +x away from the base across "
            "the table, +y left, +z up; the table top is z=0. Targets are "
            "clamped to the reachable box x[0.20,0.60] y[-0.35,0.35] "
            "z[0.02,0.55]. To grasp an object whose centre is at height z, "
            "descend to z + 0.044 (fingertip offset). The rest of the arm "
            "follows the end-effector: links and the gripper body sweep "
            "through space and can topple objects along the path, so travel "
            "above obstacles and descend vertically. Returns whether the "
            "target was reached, the final position, and disturbed_objects "
            "-- anything the motion bumped (how far it moved and which arm "
            "links hit it; empty hit_by = a chain reaction)."),
        "inputSchema": _schema({
            "x": _num("world x (m)"), "y": _num("world y (m)"),
            "z": _num("world z (m), table top = 0"),
            "tolerance": _num("success radius in m (default 0.015)"),
            "max_steps": {"type": "integer",
                          "description": "control-step budget (default 200; 20 steps = 1 sim second)"},
        }, required=("x", "y", "z")),
    },
    {
        "name": "move_delta",
        "description": "Nudge the end-effector goal by [dx, dy, dz] metres (same servoing and clamping as move_to).",
        "inputSchema": _schema({
            "dx": _num("delta x (m)"), "dy": _num("delta y (m)"), "dz": _num("delta z (m)"),
        }),
    },
    {
        "name": "grasp",
        "description": (
            "Close the gripper and settle for 1 sim second. Reports 'holding': "
            "true when the fingers were blocked by an object before fully "
            "closing (the 5 cm cube reads opening ~0.6)."),
        "inputSchema": _OBJ,
    },
    {
        "name": "release",
        "description": "Open the gripper fully and settle for 1 sim second.",
        "inputSchema": _OBJ,
    },
    {
        "name": "set_gripper",
        "description": "Set gripper opening to a value in [0, 1] (0 = closed, 1 = open).",
        "inputSchema": _schema({"opening": _num("target opening in [0, 1]")},
                               required=("opening",)),
    },
    {
        "name": "wait",
        "description": ("Hold position and let physics settle for N control "
                        "steps (20 = 1 sim second). Reports disturbed_objects "
                        "if anything moved while settling (e.g. a stack "
                        "toppling)."),
        "inputSchema": _schema({"steps": {"type": "integer", "description": "control steps (default 20)"}}),
    },
    {
        "name": "spawn_object",
        "description": (
            "Add a new free object to the scene (box, sphere, or cylinder). "
            "The scene recompiles in place: the arm and every existing object "
            "keep their poses, the live viewer reloads automatically, and a "
            "new recording segment starts. Position defaults to a random "
            "spot on the table; sizes are MuJoCo halfsizes in metres "
            "(box [hx,hy,hz], sphere [r], cylinder [r,half-height]). "
            "Spawned objects persist across reset (which re-places them at "
            "their spawn pose) and are listed in get_state under "
            "privileged.spawned_objects."),
        "inputSchema": _schema({
            "shape": {"type": "string", "enum": sorted(M.SPAWN_SHAPES),
                      "description": "object shape (default box)"},
            "name": {"type": "string",
                     "description": "unique name (default e.g. box2)"},
            "x": _num("world x (m); default random on the table"),
            "y": _num("world y (m); default random on the table"),
            "z": _num("world z of the centre (m); default resting on the table"),
            "size": {"type": "array", "items": {"type": "number"},
                     "description": "MuJoCo halfsizes (m), each in [0.005, 0.08]; "
                                    "default 2.5 cm (matches the base cube)"},
            "rgba": {"type": "array", "items": {"type": "number"},
                     "description": "color [r,g,b] or [r,g,b,a] in 0..1 "
                                    "(default: cycles a palette)"},
            "mass": _num("mass in kg (default 0.05, like the base cube)"),
        }),
    },
    {
        "name": "remove_objects",
        "description": ("Remove every spawned object, restoring the base "
                        "scene (the task's own cube/target are untouched)."),
        "inputSchema": _OBJ,
    },
    {
        "name": "get_camera_image",
        "description": (
            "RGB render from a named camera. pick_cube has front/top/side "
            "(fixed) and wrist (on the arm); get_state lists what is "
            "available. Use for visual verification of the scene."),
        "inputSchema": _schema({"camera": {"type": "string", "description": "camera name (default: first available)"}}),
    },
    {
        "name": "check_success",
        "description": "Evaluate the task's success criteria (e.g. cube resting on the target zone).",
        "inputSchema": _OBJ,
    },
    {
        "name": "save_session",
        "description": (
            "Save the current episode as a replayable session: every control "
            "step's body poses plus the full activity feed (thoughts, tool "
            "calls, visual inputs). Episodes also autosave on reset. Saved "
            "under yam/sessions/."),
        "inputSchema": _schema({"name": {"type": "string",
                                         "description": "session name (default: timestamp_task)"}}),
    },
    {
        "name": "list_sessions",
        "description": "List saved sessions (name, task, frames, success), oldest first.",
        "inputSchema": _OBJ,
    },
    {
        "name": "replay_session",
        "description": (
            "Replay a saved session in the live viewer: the 3-D scene plays "
            "back the recorded motion while the agent pane replays the "
            "recorded thoughts, tool calls, and visual inputs in sync. "
            "Defaults to the most recently saved session."),
        "inputSchema": _schema({"name": {"type": "string",
                                         "description": "session to replay (default: latest)"}}),
    },
]


# Tools that command the sim. Calling one takes control back from any manual
# browser mode (autopilot/teleop): last actor wins, and the human can re-enable
# a mode in the viewer at any time.
SIM_TOOLS = {"move_to", "move_delta", "grasp", "release", "set_gripper",
             "wait", "reset", "spawn_object", "remove_objects"}

# Browser sim-control commands routed to AgentSession.on_control (the manual
# side of the unified server: modes, teleop, hand tracking, demo recording).
MANUAL_CMDS = {"reset", "task", "mode", "speed", "teleop_delta",
               "teleop_target", "gripper", "record_start", "record_stop"}


def _text(payload):
    return [{"type": "text", "text": json.dumps(payload)}]


def _dispatch(session, name, args):
    """Route one tool call to the skill layer; returns a JSON-able payload."""
    if name == "get_state":
        return session.get_state()
    if name == "reset":
        return session.reset(task=args.get("task"), seed=args.get("seed"))
    if name == "move_to":
        return session.move_to(
            args["x"], args["y"], args["z"],
            tolerance=args.get("tolerance", 0.015),
            max_steps=args.get("max_steps", 200))
    if name == "move_delta":
        return session.move_delta(
            args.get("dx", 0.0), args.get("dy", 0.0), args.get("dz", 0.0))
    if name == "grasp":
        return session.grasp()
    if name == "release":
        return session.release()
    if name == "set_gripper":
        return session.set_gripper(args["opening"])
    if name == "wait":
        return session.wait(args.get("steps", 20))
    if name == "spawn_object":
        return session.spawn_object(
            shape=args.get("shape", "box"), name=args.get("name"),
            x=args.get("x"), y=args.get("y"), z=args.get("z"),
            size=args.get("size"), rgba=args.get("rgba"),
            mass=args.get("mass", 0.05))
    if name == "remove_objects":
        return session.remove_objects()
    if name == "check_success":
        return session.check_success()
    if name == "save_session":
        return session.save_session(args.get("name"))
    if name == "list_sessions":
        return {"sessions": A.list_sessions()}
    if name == "replay_session":
        sessions = A.list_sessions()
        if not sessions:
            raise RuntimeError("no saved sessions yet (use save_session first)")
        wanted = args.get("name") or sessions[-1]["name"]
        if wanted not in {s["name"] for s in sessions}:
            raise ValueError(f"unknown session {wanted!r}; "
                             f"available: {[s['name'] for s in sessions]}")
        _REPLAY_REQUEST["seq"] += 1
        _REPLAY_REQUEST["name"] = wanted
        return {"ok": True, "replaying": wanted,
                "note": "playing in the live viewer (all connected browsers)"}
    raise ValueError(f"unknown tool {name!r}")


def call_tool(session, name, args):
    """Execute one tools/call, logging it to the agent activity feed. Every
    call becomes an event: 'thought' (think), 'image' (get_camera_image --
    the moment a visual input is taken, with the exact frame the agent saw),
    or 'tool' (everything else, with args, result, and sim-step span)."""
    if name == "think":
        session.log_event("thought", text=str(args.get("thought", "")))
        return _text({"ok": True})

    if name == "get_camera_image":
        import imageio.v3 as iio
        camera, img = session.camera_image(args.get("camera"))
        png = iio.imwrite("<bytes>", img, extension=".png")
        b64 = base64.b64encode(png).decode("ascii")
        session.log_event("image", camera=camera, shape=list(img.shape),
                          png_b64=b64)
        return [
            {"type": "image", "data": b64, "mimeType": "image/png"},
            {"type": "text", "text": json.dumps({"camera": camera,
                                                 "shape": list(img.shape)})},
        ]

    t0 = time.time()
    step0 = session.steps
    if name in SIM_TOOLS:
        session.mode = "idle"  # stop the manual loop; the agent has the arm
    session.current_activity = f"{name} {json.dumps(args)}" if args else name
    try:
        payload = _dispatch(session, name, args)
    except Exception as e:
        session.log_event("tool", tool=name, args=args, error=str(e),
                          sim_step_start=step0,
                          duration_s=round(time.time() - t0, 2))
        raise
    finally:
        session.current_activity = None
    session.log_event("tool", tool=name, args=args, result=payload,
                      sim_step_start=step0,
                      duration_s=round(time.time() - t0, 2))
    return _text(payload)


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 handling, transport-agnostic: the same handler serves stdio
# (Cursor spawns us) and the TCP endpoint (bridged clients like the chat
# agent), all against ONE shared sim.
#
# Every sim/camera operation -- including building the AgentSession -- runs on
# this ONE dedicated thread. MuJoCo's offscreen GL context is bound to the
# thread that created it, and rendering from any other thread deadlocks (seen
# on macOS: a seedless reset re-rendering cameras from a TCP executor thread
# hung forever). The single worker also serializes tool calls from every
# transport, so two fronts can't command the arm mid-motion.
# ---------------------------------------------------------------------------
_SIM_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="yam-sim")


class McpHandler:
    def __init__(self, session, viewer_url=None):
        self.session = session
        self.viewer_url = viewer_url

    def handle(self, msg):
        """Process one JSON-RPC message; returns the response dict or None
        (notifications get no response)."""
        mid = msg.get("id")
        try:
            return self._handle(msg)
        except Exception as e:
            print(f"[yam.mcp] error handling {msg.get('method')}: {e}",
                  file=sys.stderr)
            if mid is not None:
                return self._error(mid, -32603, f"internal error: {e}")
            return None

    @staticmethod
    def _reply(mid, result):
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    @staticmethod
    def _error(mid, code, message):
        return {"jsonrpc": "2.0", "id": mid,
                "error": {"code": code, "message": message}}

    def _handle(self, msg):
        method = msg.get("method", "")
        mid = msg.get("id")
        params = msg.get("params") or {}

        if mid is None:  # notification -- never respond
            return None

        if method == "initialize":
            instructions = (
                "Control a simulated I2RT YAM 6-DOF arm on a tabletop. "
                "Typical episode: reset -> get_state -> move_to / grasp / "
                "release (composing the grasp_recipe hint) -> check_success. "
                "Positions are metres in the world frame; the table top is "
                "z=0. Skills block until the motion completes, so each call "
                "returns the settled outcome. The arm is an embodied "
                "physical object: every link and the gripper body collide "
                "with the scene, not just the fingertips, and a held object "
                "extends ~0.07 m below the end-effector. Plan paths that "
                "clear obstacles -- travel high, descend and retreat "
                "vertically -- and treat 'disturbed_objects' in any motion "
                "result as a bump alarm: something you touched (or a chain "
                "reaction) moved, so re-check state before continuing your "
                "plan. Narrate your work: call think "
                "before each action to record your reasoning -- thoughts, "
                "tool calls, and camera captures are timestamped and shown "
                "live in the viewer's agent activity pane. Visual input is "
                "taken ONLY when you call get_camera_image; motions servo on "
                "proprioception alone, and get_state reads joint angles and "
                "the end-effector pose.")
            if self.viewer_url:
                instructions += f" Watch the arm live at {self.viewer_url}."
            return self._reply(mid, {
                "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "yam-arm", "version": "0.1.0"},
                "instructions": instructions,
            })
        if method == "tools/list":
            return self._reply(mid, {"tools": TOOLS})
        if method == "tools/call":
            name = params.get("name", "")
            args = params.get("arguments") or {}
            debug = bool(os.environ.get("YAM_MCP_DEBUG"))
            try:
                if debug:
                    print(f"[yam.mcp:call] {name}: queued on sim thread",
                          file=sys.stderr)
                content = _SIM_EXECUTOR.submit(
                    call_tool, self.session, name, args).result()
                if debug:
                    print(f"[yam.mcp:call] {name}: done", file=sys.stderr)
                return self._reply(mid, {"content": content, "isError": False})
            except Exception as e:
                return self._reply(mid, {"content": [{"type": "text",
                                                      "text": str(e)}],
                                         "isError": True})
        if method == "ping":
            return self._reply(mid, {})
        if method == "resources/list":
            return self._reply(mid, {"resources": []})
        if method == "prompts/list":
            return self._reply(mid, {"prompts": []})
        return self._error(mid, -32601, f"method not found: {method}")


def serve_stdio(handler, out):
    """Blocking stdio loop (the transport Cursor uses)."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[yam.mcp] bad JSON on stdin: {e}", file=sys.stderr)
            continue
        resp = handler.handle(msg)
        if resp is not None:
            out.write(json.dumps(resp, separators=(",", ":")) + "\n")
            out.flush()


# ---------------------------------------------------------------------------
# Web services (background thread or --console foreground):
#   * live viewer: serves yam/web, streams body poses + the activity feed
#   * chat: browser chat pane -> Cursor SDK agent (see yam/chat.py)
#   * TCP MCP endpoint: newline JSON-RPC against the SAME sim, so bridged
#     clients (the chat agent via yam.mcp_bridge) share the arm with stdio
#   * manual control: the browser's sim controls (autopilot/teleop/hand
#     tracking/recording) drive the same sim; a manual loop steps it while a
#     mode is active and no agent tool call is in flight
# ---------------------------------------------------------------------------
def _run_services(session, host, port, mcp_port=0, chat=None, viewer_url=None):
    from .server import WS_GUID, _serve_static, ws_read_frame, ws_send

    async def ws_stream(reader, writer):
        connected = {"ok": True}
        # reader -> streamer mailbox (both coroutines share the event loop)
        ctl = {"replay": None, "stop_replay": False, "sessions_dirty": False}

        async def read_loop():
            while connected["ok"]:
                opcode, payload = await ws_read_frame(reader)
                if opcode is None or opcode == 0x8:
                    connected["ok"] = False
                    break
                if opcode == 0x9:  # ping -> pong
                    await ws_send(writer, payload, opcode=0xA)
                    continue
                if opcode != 0x1 or not payload:
                    continue
                try:
                    msg = json.loads(payload.decode("utf-8"))
                except Exception:
                    continue
                cmd = msg.get("cmd")
                if cmd == "save_session":
                    try:
                        await asyncio.wrap_future(_SIM_EXECUTOR.submit(
                            session.save_session, msg.get("name")))
                        ctl["sessions_dirty"] = True
                    except Exception as e:
                        print(f"[yam.mcp] save_session: {e}", file=sys.stderr)
                elif cmd == "replay" and msg.get("name"):
                    ctl["replay"] = msg["name"]
                elif cmd == "replay_stop":
                    ctl["stop_replay"] = True
                elif cmd == "chat" and chat is not None and msg.get("text"):
                    chat.submit(str(msg["text"]))
                elif cmd in MANUAL_CMDS:
                    # Manual sim control (modes, teleop, hand tracking,
                    # recording): queued on the sim thread, fire-and-forget so
                    # 20 Hz hand-tracking targets never stall this reader.
                    def _ctl(m=msg):
                        try:
                            session.on_control(m)
                        except Exception as e:
                            print(f"[yam.mcp] control {m.get('cmd')}: {e}",
                                  file=sys.stderr)
                    _SIM_EXECUTOR.submit(_ctl)

        async def stream_loop():
            sessions = A.list_sessions()
            # The manifest travels over the socket (here and on every scene
            # recompile) so the viewer's geometry always matches THIS sim's
            # body-pose stream -- the manifest.json on disk is written by
            # whichever session exported last and cannot be trusted.
            await ws_send(writer, json.dumps({
                "type": "hello", "task": session.env.task,
                "tasks": sorted(M.TASKS), "bimanual_tasks": [],
                "sessions": sessions, "chat": chat is not None,
                "manifest": session.manifest(),
                "scene_version": session.scene_version}))
            last_scene_version = session.scene_version
            last_event_id = 0  # starts at 0 so a (re)connect replays history
            last_chat_id = 0
            last_seq = _REPLAY_REQUEST["seq"]
            known_sessions = [s["name"] for s in sessions]
            sessions_checked = time.time()
            replay = None  # {"meta", "steps", "xpos", "xquat", "i", "j"}

            def skip_live_backlog():
                """After a replay, resume the live feed from *now* rather than
                flooding the pane with everything missed during playback."""
                nonlocal last_event_id
                evs = session.events_since(last_event_id, limit=10 ** 6)
                if evs:
                    last_event_id = evs[-1]["id"]

            while connected["ok"]:
                try:
                    # replay requests: from this browser or the replay tool
                    if _REPLAY_REQUEST["seq"] != last_seq:
                        last_seq = _REPLAY_REQUEST["seq"]
                        ctl["replay"] = _REPLAY_REQUEST["name"]
                    if ctl["stop_replay"]:
                        ctl["stop_replay"] = False
                        if replay is not None:
                            replay = None
                            skip_live_backlog()
                            await ws_send(writer, json.dumps({"type": "replay_end"}))
                    if ctl["replay"]:
                        name, ctl["replay"] = ctl["replay"], None
                        try:
                            meta, steps, xpos, xquat = A.load_session(name)
                            replay = {"meta": meta, "steps": steps,
                                      "xpos": xpos, "xquat": xquat,
                                      "i": 0, "j": 0}
                            await ws_send(writer, json.dumps(
                                {"type": "replay_start", "name": meta["name"],
                                 "frames": int(len(steps)),
                                 # sessions record their scene (spawned objects
                                 # change the body count) for exact replay
                                 "manifest": meta.get("manifest")}))
                        except Exception as e:
                            print(f"[yam.mcp] replay: {e}", file=sys.stderr)

                    # chat flows regardless of live/replay mode
                    if chat is not None:
                        entries = chat.entries_since(last_chat_id)
                        if entries:
                            last_chat_id = entries[-1]["id"]
                            await ws_send(writer, json.dumps(
                                {"type": "chat", "messages": entries}))

                    if replay is not None:
                        i, meta = replay["i"], replay["meta"]
                        steps = replay["steps"]
                        last = i == len(steps) - 1
                        await ws_send(writer, json.dumps({
                            "type": "state", "task": meta["task"],
                            "mode": "replay", "step": int(steps[i]),
                            "success": bool(meta["success"]) and last,
                            "recording": False,
                            "chat_busy": bool(chat is not None and chat.busy),
                            "activity": f"replay {meta['name']} "
                                        f"({i + 1}/{len(steps)})",
                            "xpos": replay["xpos"][i].reshape(-1).round(5).tolist(),
                            "xquat": replay["xquat"][i].reshape(-1).round(5).tolist(),
                        }))
                        # replay the recorded feed in sync with the motion
                        due = []
                        events = meta.get("events", [])
                        while (replay["j"] < len(events) and
                               events[replay["j"]]["sim_step"] <= steps[i]):
                            due.append(events[replay["j"]])
                            replay["j"] += 1
                        if due:
                            await ws_send(writer, json.dumps(
                                {"type": "agent_events", "events": due,
                                 "replay": True}))
                        replay["i"] += 1
                        if replay["i"] >= len(steps):
                            replay = None
                            skip_live_backlog()
                            await ws_send(writer, json.dumps({"type": "replay_end"}))
                        await asyncio.sleep(meta.get("control_dt", 0.05))
                        continue

                    # live streaming: on a scene recompile, ship the new
                    # manifest before any state frame that references it
                    if session.scene_version != last_scene_version:
                        last_scene_version = session.scene_version
                        await ws_send(writer, json.dumps(
                            {"type": "scene", "task": session.env.task,
                             "manifest": session.manifest(),
                             "scene_version": last_scene_version}))
                    live = session.viewer_message()
                    live["chat_busy"] = bool(chat is not None and chat.busy)
                    await ws_send(writer, json.dumps(live))
                    events = session.events_since(last_event_id)
                    if events:
                        last_event_id = events[-1]["id"]
                        await ws_send(writer, json.dumps(
                            {"type": "agent_events", "events": events}))
                    # keep the sessions dropdown fresh (saves from any source)
                    if ctl["sessions_dirty"] or time.time() - sessions_checked > 2.0:
                        ctl["sessions_dirty"] = False
                        sessions_checked = time.time()
                        current = A.list_sessions()
                        if [s["name"] for s in current] != known_sessions:
                            known_sessions = [s["name"] for s in current]
                            await ws_send(writer, json.dumps(
                                {"type": "sessions", "sessions": current}))
                except (ConnectionError, RuntimeError):
                    connected["ok"] = False
                    break
                await asyncio.sleep(1 / 30)

        await asyncio.gather(read_loop(), stream_loop())

    async def handle(reader, writer):
        try:
            request_line = await reader.readline()
            if not request_line:
                writer.close()
                return
            parts = request_line.decode(errors="ignore").split()
            _, path = (parts + ["", ""])[:2]
            headers = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"", b"\n"):
                    break
                k, _, v = line.decode(errors="ignore").partition(":")
                headers[k.strip().lower()] = v.strip()

            if headers.get("upgrade", "").lower() == "websocket":
                key = headers.get("sec-websocket-key", "")
                accept = base64.b64encode(
                    hashlib.sha1((key + WS_GUID).encode()).digest()).decode()
                writer.write((
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode())
                await writer.drain()
                await ws_stream(reader, writer)
            else:
                await _serve_static(writer, path)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    debug = bool(os.environ.get("YAM_MCP_DEBUG"))

    async def handle_mcp_client(reader, writer):
        """One bridged MCP client (newline JSON-RPC over TCP). Tool calls can
        block for seconds (paced motion), so they run in the executor to keep
        the viewer stream flowing."""
        handler = McpHandler(session, viewer_url=viewer_url)
        loop = asyncio.get_running_loop()
        peer = writer.get_extra_info("peername")
        if debug:
            print(f"[yam.mcp:tcp] client connected {peer}", file=sys.stderr)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    if debug:
                        print(f"[yam.mcp:tcp] <- unparseable: {line[:200]}",
                              file=sys.stderr)
                    continue
                if debug:
                    print(f"[yam.mcp:tcp] <- {line[:300].decode(errors='replace')}",
                          file=sys.stderr)
                resp = await loop.run_in_executor(None, handler.handle, msg)
                if resp is not None:
                    out = json.dumps(resp, separators=(",", ":"))
                    if debug:
                        print(f"[yam.mcp:tcp] -> {out[:300]}", file=sys.stderr)
                    writer.write((out + "\n").encode())
                    await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            if debug:
                print(f"[yam.mcp:tcp] client disconnected {peer}",
                      file=sys.stderr)
            try:
                writer.close()
            except Exception:
                pass

    async def manual_loop():
        """Step the sim for the human's mode (autopilot/teleop) at control
        rate. Pauses while an agent tool call is in flight -- the single sim
        thread serializes everything, and agent calls drop the mode to idle
        anyway (last actor wins)."""
        loop = asyncio.get_running_loop()
        while True:
            t0 = loop.time()
            if session.mode != "idle" and session.current_activity is None:
                try:
                    await asyncio.wrap_future(
                        _SIM_EXECUTOR.submit(session.manual_step))
                except Exception as e:
                    print(f"[yam.mcp] manual step: {e}", file=sys.stderr)
            dt = session.env.control_dt / max(session.speed, 1e-3)
            await asyncio.sleep(max(0.005, dt - (loop.time() - t0)))

    async def main():
        servers = []
        if port:
            try:
                servers.append(await asyncio.start_server(handle, host, port))
            except OSError as e:
                print(f"[yam.mcp] viewer disabled ({e})", file=sys.stderr)
        if mcp_port:
            try:
                servers.append(await asyncio.start_server(
                    handle_mcp_client, host, mcp_port))
            except OSError as e:
                print(f"[yam.mcp] TCP MCP endpoint disabled ({e})",
                      file=sys.stderr)
        if not servers:
            return
        await asyncio.gather(manual_loop(),
                             *(s.serve_forever() for s in servers))

    asyncio.run(main())


def main():
    p = argparse.ArgumentParser(description="YAM arm MCP server + robot IDE")
    p.add_argument("--task", default="pick_cube", choices=sorted(M.TASKS))
    p.add_argument("--cam-size", type=int, default=256,
                   help="camera render size for get_camera_image (0 = off)")
    p.add_argument("--pace", type=float, default=None,
                   help="real seconds slept per control step; default 0.02 "
                        "with the viewer (watchable) else 0 (flat out)")
    p.add_argument("--console", action="store_true",
                   help="run without stdio MCP (viewer + chat + TCP endpoint "
                        "only) -- for `python -m yam.mcp_server --console`")
    p.add_argument("--viewer", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="serve the live 3-D viewer")
    p.add_argument("--viewer-host", default="127.0.0.1")
    p.add_argument("--viewer-port", type=int, default=8765)
    p.add_argument("--mcp-port", type=int, default=8766,
                   help="TCP MCP endpoint sharing this sim (0 = off); the "
                        "chat agent connects here via yam.mcp_bridge")
    p.add_argument("--chat", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="enable the browser chat pane (Cursor SDK agent)")
    p.add_argument("--chat-model", default="composer-2.5",
                   help="model for the chat agent")
    args = p.parse_args()

    # Protocol messages go to the real stdout; reroute everything else
    # (mujoco/export prints) to stderr so the JSON-RPC stream stays clean.
    real_out = sys.stdout
    sys.stdout = sys.stderr

    pace = args.pace if args.pace is not None else (0.02 if args.viewer else 0.0)
    # Built on the sim thread so the GL render context lives where all
    # subsequent tool calls run (see _SIM_EXECUTOR above).
    session = _SIM_EXECUTOR.submit(
        AgentSession, task=args.task, cam_size=args.cam_size, pace=pace,
    ).result()

    viewer_url = None
    if args.viewer:
        for t in M.TASKS:  # manifests + meshes for the browser scene
            M.export_web(t)
        viewer_url = f"http://{args.viewer_host}:{args.viewer_port}"
        print(f"[yam.mcp] live viewer at {viewer_url}", file=sys.stderr)

    chat = None
    if args.chat and args.viewer and args.mcp_port:
        from .chat import ChatManager
        chat = ChatManager(mcp_port=args.mcp_port, model=args.chat_model)
        print(f"[yam.mcp] chat pane enabled (model={args.chat_model}, "
              f"key={'set' if os.environ.get('CURSOR_API_KEY') else 'MISSING'})",
              file=sys.stderr)

    services = (session, args.viewer_host,
                args.viewer_port if args.viewer else 0,
                args.mcp_port, chat, viewer_url)
    if args.console:
        print("[yam.mcp] console mode (no stdio MCP); Ctrl-C to stop",
              file=sys.stderr)
        try:
            _run_services(*services)
        except KeyboardInterrupt:
            print("[yam.mcp] stopped", file=sys.stderr)
        return

    threading.Thread(target=_run_services, args=services, daemon=True).start()
    print(f"[yam.mcp] serving MCP on stdio (task={args.task}, "
          f"cam_size={args.cam_size}, pace={pace})", file=sys.stderr)
    serve_stdio(McpHandler(session, viewer_url=viewer_url), real_out)


if __name__ == "__main__":
    main()
