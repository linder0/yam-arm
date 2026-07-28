"""Live 3-D viewer bridge: a dependency-free HTTP + WebSocket server.

Runs the sim server-authoritatively (this is the design that also works for the
real arm later: Python owns the loop and streams state; the browser only
renders). Serves the static three.js viewer in ``yam/web`` and, over a
WebSocket, streams per-body world poses each control step. It also accepts
control messages from the browser: reset, switch task, pick a mode
(scripted autopilot / Cartesian teleop), nudge the teleop target, toggle the
gripper, and start/stop recording a demonstration.

No third-party packages required -- the WebSocket handshake and framing are
implemented against the standard library so ``python -m yam.server`` just works.
"""

import asyncio
import base64
import hashlib
import json
import mimetypes
import os
import struct
import time

import numpy as np

from . import model as M
from .env import YamEnv
from .teleop import EEController, scripted_expert

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


# ---------------------------------------------------------------------------
# Sim session: env stepping + mode logic (no networking -- unit-testable).
# ---------------------------------------------------------------------------
class SimSession:
    MODES = ("idle", "scripted", "teleop", "policy")

    def __init__(self, task="pick_cube", cam_size=0, action_fn=None):
        # cam_size=0 disables camera rendering (viewer renders 3-D from poses;
        # cameras aren't needed for the stream and skipping them is much faster).
        self.cam_size = cam_size
        self.action_fn = action_fn  # for policy mode: obs -> action
        self.env = None
        self.ctl = None
        self.expert = None
        self.mode = "idle"
        self.speed = 1.0
        self.last_obs = None
        self.last_info = {"success": False, "steps": 0}
        self._recorder = None
        self._rec_success = False
        self.set_task(task)

    # -- lifecycle -----------------------------------------------------------
    def set_task(self, task):
        if task not in M.TASKS:
            return
        if self.env is not None:
            self.env.close()
        render = self.cam_size > 0
        self.env = YamEnv(task=task, seed=int(time.time()) % 10000,
                          render_cameras=render,
                          cam_height=max(self.cam_size, 1),
                          cam_width=max(self.cam_size, 1))
        self.ctl = EEController(self.env)
        self.last_obs = self.env.reset()
        self.ctl.reset()
        self.expert = None
        self.mode = "idle"

    def reset(self):
        self.last_obs = self.env.reset()
        self.ctl.reset()
        self.expert = None
        self.last_info = {"success": False, "steps": 0}

    def set_mode(self, mode):
        if mode not in self.MODES:
            return
        self.mode = mode
        if mode == "scripted":
            self.expert = scripted_expert(self.env)
        if mode == "teleop":
            self.ctl.reset()

    # -- teleop input --------------------------------------------------------
    def teleop_delta(self, dx, dy, dz):
        self.ctl.move([dx, dy, dz])

    def teleop_target(self, x, y, z):
        self.ctl.set_target([x, y, z])

    def set_gripper(self, g):
        self.ctl.set_gripper(g)

    # -- recording -----------------------------------------------------------
    def record_start(self, out=None):
        from .record import NpzRecorder
        out = out or os.path.join(os.path.dirname(__file__), "data",
                                  f"{self.env.task}_teleop")
        fps = round(1.0 / self.env.control_dt)
        # camera frames are needed for a usable IL dataset
        if self.cam_size <= 0:
            print("[yam.server] recording needs cameras; restart with --cam-size")
            return False
        self._recorder = NpzRecorder(out, self.env, fps,
                                     M.TASKS[self.env.task]["description"])
        self._recorder.start_episode()
        self._rec_success = False
        return True

    def record_stop(self, keep=True):
        if self._recorder is None:
            return
        if keep:
            idx = self._recorder.save_episode(self._rec_success)
            self._recorder.finalize(self.env.task_spec())
            print(f"[yam.server] saved teleop episode {idx} "
                  f"(success={self._rec_success})")
        self._recorder = None

    # -- stepping ------------------------------------------------------------
    def step(self):
        if self.mode == "idle" or self.env is None:
            return
        if self.mode == "scripted":
            if self.expert is None:
                self.expert = scripted_expert(self.env)
            action = self.expert()
        elif self.mode == "teleop":
            action = self.ctl.action()
        elif self.mode == "policy" and self.action_fn is not None:
            action = self.action_fn(self.last_obs)
        else:
            return

        if self._recorder is not None:
            self._recorder.add_frame(self.last_obs, action, 0.0, False, False)

        self.last_obs, reward, done, info = self.env.step(action)
        self.last_info = info
        if info["success"]:
            self._rec_success = True
        if done and self.mode == "scripted":
            # autopilot loops: new episode
            self.reset()
            self.set_mode("scripted")

    def state_message(self):
        xpos, xquat = self.env.body_states()
        return {
            "type": "state",
            "task": self.env.task,
            "mode": self.mode,
            "step": int(self.last_info.get("steps", 0)),
            "success": bool(self.last_info.get("success", False)),
            "recording": self._recorder is not None,
            "xpos": xpos.reshape(-1).round(5).tolist(),
            "xquat": xquat.reshape(-1).round(5).tolist(),
        }

    def on_control(self, msg):
        cmd = msg.get("cmd")
        if cmd == "reset":
            self.reset()
        elif cmd == "task":
            self.set_task(msg.get("task", "pick_cube"))
        elif cmd == "mode":
            self.set_mode(msg.get("mode", "idle"))
        elif cmd == "speed":
            self.speed = float(np.clip(msg.get("speed", 1.0), 0.1, 5.0))
        elif cmd == "teleop_delta":
            self.teleop_delta(msg.get("dx", 0), msg.get("dy", 0), msg.get("dz", 0))
        elif cmd == "teleop_target":
            self.teleop_target(msg["x"], msg["y"], msg["z"])
        elif cmd == "gripper":
            self.set_gripper(msg.get("value", 1.0))
        elif cmd == "record_start":
            self.record_start()
        elif cmd == "record_stop":
            self.record_stop(keep=msg.get("keep", True))

    def close(self):
        if self.env is not None:
            self.env.close()


# ---------------------------------------------------------------------------
# Minimal WebSocket framing (stdlib only).
# ---------------------------------------------------------------------------
async def _read_exactly(reader, n):
    data = await reader.readexactly(n)
    return data


async def ws_read_frame(reader):
    """Return (opcode, payload_bytes) or (None, None) on close/EOF."""
    try:
        b0, b1 = await _read_exactly(reader, 2)
    except (asyncio.IncompleteReadError, ConnectionError):
        return None, None
    opcode = b0 & 0x0F
    masked = b1 & 0x80
    length = b1 & 0x7F
    if length == 126:
        (length,) = struct.unpack(">H", await _read_exactly(reader, 2))
    elif length == 127:
        (length,) = struct.unpack(">Q", await _read_exactly(reader, 8))
    mask = await _read_exactly(reader, 4) if masked else b"\x00\x00\x00\x00"
    payload = bytearray(await _read_exactly(reader, length))
    if masked:
        for i in range(length):
            payload[i] ^= mask[i % 4]
    return opcode, bytes(payload)


def ws_encode_frame(payload, opcode=0x1):
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    header = bytearray([0x80 | opcode])
    n = len(payload)
    if n < 126:
        header.append(n)
    elif n < 65536:
        header.append(126)
        header += struct.pack(">H", n)
    else:
        header.append(127)
        header += struct.pack(">Q", n)
    return bytes(header) + payload


async def ws_send(writer, payload, opcode=0x1):
    writer.write(ws_encode_frame(payload, opcode))
    await writer.drain()


# ---------------------------------------------------------------------------
# HTTP + WS connection handling.
# ---------------------------------------------------------------------------
def _content_type(path):
    ctype, _ = mimetypes.guess_type(path)
    if path.endswith(".stl"):
        return "model/stl"
    if path.endswith(".js"):
        return "text/javascript"
    return ctype or "application/octet-stream"


async def _serve_static(writer, path):
    rel = path.lstrip("/") or "index.html"
    full = os.path.normpath(os.path.join(WEB_DIR, rel))
    if not full.startswith(os.path.normpath(WEB_DIR)) or not os.path.isfile(full):
        writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        return
    with open(full, "rb") as f:
        body = f.read()
    header = (f"HTTP/1.1 200 OK\r\nContent-Type: {_content_type(full)}\r\n"
              f"Content-Length: {len(body)}\r\nAccess-Control-Allow-Origin: *\r\n"
              "Cache-Control: no-cache\r\n\r\n").encode()
    writer.write(header + body)
    await writer.drain()


async def _handle(reader, writer, make_session):
    try:
        request_line = await reader.readline()
        if not request_line:
            writer.close(); return
        parts = request_line.decode(errors="ignore").split()
        method, path = (parts + ["", ""])[:2]
        headers = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"", b"\n"):
                break
            k, _, v = line.decode(errors="ignore").partition(":")
            headers[k.strip().lower()] = v.strip()

        if headers.get("upgrade", "").lower() == "websocket":
            await _ws_session(reader, writer, headers, make_session)
        else:
            await _serve_static(writer, path)
            writer.close()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def _ws_session(reader, writer, headers, make_session):
    key = headers.get("sec-websocket-key", "")
    accept = base64.b64encode(
        hashlib.sha1((key + WS_GUID).encode()).digest()).decode()
    handshake = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n")
    writer.write(handshake.encode())
    await writer.drain()

    session = make_session()
    connected = {"ok": True}

    async def reader_loop():
        while connected["ok"]:
            opcode, payload = await ws_read_frame(reader)
            if opcode is None or opcode == 0x8:  # close/EOF
                connected["ok"] = False
                break
            if opcode == 0x9:  # ping -> pong
                await ws_send(writer, payload, opcode=0xA)
                continue
            if opcode == 0x1 and payload:
                try:
                    session.on_control(json.loads(payload.decode("utf-8")))
                except Exception as e:
                    print("[yam.server] bad control msg:", e)

    async def stream_loop():
        # Tell the client which task/manifest to load first.
        await ws_send(writer, json.dumps({"type": "hello",
                      "task": session.env.task, "tasks": sorted(M.TASKS)}))
        while connected["ok"]:
            t0 = time.perf_counter()
            if session.mode != "idle":
                session.step()
            try:
                await ws_send(writer, json.dumps(session.state_message()))
            except (ConnectionError, RuntimeError):
                connected["ok"] = False
                break
            dt = session.env.control_dt / max(session.speed, 1e-3)
            elapsed = time.perf_counter() - t0
            await asyncio.sleep(max(0.0, dt - elapsed))

    try:
        await asyncio.gather(reader_loop(), stream_loop())
    finally:
        connected["ok"] = False
        session.close()


async def _serve(host, port, task, cam_size, action_fn):
    def make_session():
        return SimSession(task=task, cam_size=cam_size, action_fn=action_fn)

    server = await asyncio.start_server(
        lambda r, w: _handle(r, w, make_session), host, port)
    url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"
    print(f"[yam.server] live viewer at {url}  (task={task}, "
          f"cameras={'on' if cam_size>0 else 'off'})", flush=True)
    print("[yam.server] Ctrl-C to stop.", flush=True)
    async with server:
        await server.serve_forever()


def run(host="127.0.0.1", port=8080, task="pick_cube", cam_size=0,
        action_fn=None, export_web=True):
    if export_web:
        print("[yam.server] exporting web assets (manifests + meshes) ...",
              flush=True)
        for t in M.TASKS:
            M.export_web(t)
    try:
        asyncio.run(_serve(host, port, task, cam_size, action_fn))
    except KeyboardInterrupt:
        print("\n[yam.server] stopped.")


def _cli():
    import argparse
    p = argparse.ArgumentParser(description="YAM live viewer server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--task", default="pick_cube", choices=sorted(M.TASKS))
    p.add_argument("--cam-size", type=int, default=0,
                   help="camera render size (0 = off; needed only to record "
                        "teleop demos through the browser)")
    args = p.parse_args()
    run(host=args.host, port=args.port, task=args.task, cam_size=args.cam_size)


if __name__ == "__main__":
    _cli()
