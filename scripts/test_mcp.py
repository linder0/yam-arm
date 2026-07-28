"""Smoke test for yam.mcp_server: spawns the server on stdio, speaks JSON-RPC,
and runs a full pick-and-place episode through the MCP tools (the same calls a
tool-calling LLM would make). Also connects to the live viewer WebSocket and
verifies the agent activity feed (thoughts, tool calls, visual inputs) streams
out. Exits non-zero on any failure.

Usage: .venv-yam/bin/python scripts/test_mcp.py
"""

import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time

VIEWER_PORT = 8767
MCP_PORT = 8768

env = dict(os.environ, CURSOR_API_KEY="")  # chat must fail politely, not spawn
proc = subprocess.Popen(
    [sys.executable, "-m", "yam.mcp_server", "--viewer",
     "--viewer-port", str(VIEWER_PORT), "--mcp-port", str(MCP_PORT),
     "--pace", "0"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)

_id = 0


def rpc(method, params=None):
    global _id
    _id += 1
    proc.stdin.write(json.dumps(
        {"jsonrpc": "2.0", "id": _id, "method": method,
         "params": params or {}}) + "\n")
    proc.stdin.flush()
    resp = json.loads(proc.stdout.readline())
    assert resp["id"] == _id, f"id mismatch: {resp}"
    assert "error" not in resp, f"{method} -> {resp['error']}"
    return resp["result"]


def tool(name, args=None):
    r = rpc("tools/call", {"name": name, "arguments": args or {}})
    assert not r["isError"], f"{name} failed: {r['content']}"
    texts = [c["text"] for c in r["content"] if c["type"] == "text"]
    images = [c for c in r["content"] if c["type"] == "image"]
    payload = json.loads(texts[0]) if texts else {}
    return payload, images


# ---------------------------------------------------------------------------
# Viewer WebSocket client: collects streamed messages in the background.
# ---------------------------------------------------------------------------
ws_messages = []


def ws_client():
    for _ in range(50):  # wait for the viewer thread to bind
        try:
            sock = socket.create_connection(("127.0.0.1", VIEWER_PORT), timeout=5)
            break
        except OSError:
            time.sleep(0.2)
    else:
        return
    sock.sendall((
        "GET /ws HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n"
        "Connection: Upgrade\r\nSec-WebSocket-Key: dGVzdCB0ZXN0IHRlc3Qh\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n").encode())
    buf = b""
    while b"\r\n\r\n" not in buf:  # swallow the 101 handshake
        buf += sock.recv(1024)
    buf = buf.split(b"\r\n\r\n", 1)[1]

    def read_exactly(n):
        nonlocal buf
        while len(buf) < n:
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError
            buf += chunk
        out, buf = buf[:n], buf[n:]
        return out

    try:
        while True:
            b0, b1 = read_exactly(2)
            length = b1 & 0x7F
            if length == 126:
                (length,) = struct.unpack(">H", read_exactly(2))
            elif length == 127:
                (length,) = struct.unpack(">Q", read_exactly(8))
            payload = read_exactly(length)
            if (b0 & 0x0F) == 0x1:
                ws_messages.append(json.loads(payload))
    except (ConnectionError, OSError):
        pass


ws_thread = threading.Thread(target=ws_client, daemon=True)
ws_thread.start()

# ---------------------------------------------------------------------------
# The episode, narrated the way an agent would do it.
# ---------------------------------------------------------------------------
init = rpc("initialize", {"protocolVersion": "2024-11-05",
                          "capabilities": {},
                          "clientInfo": {"name": "smoke", "version": "0"}})
assert init["serverInfo"]["name"] == "yam-arm"
proc.stdin.write(json.dumps({"jsonrpc": "2.0",
                             "method": "notifications/initialized"}) + "\n")
proc.stdin.flush()

tools = rpc("tools/list")["tools"]
names = {t["name"] for t in tools}
expected = {"think", "get_state", "reset", "move_to", "move_delta", "grasp",
            "release", "set_gripper", "wait", "get_camera_image",
            "check_success"}
assert expected <= names, f"missing tools: {expected - names}"
print(f"ok  initialize + tools/list ({len(tools)} tools)")

state, _ = tool("reset", {"task": "pick_cube", "seed": 7})
assert len(state["joint_angles_rad"]) == 6
cube = state["privileged"]["object_pos"]
target = state["privileged"]["target_pos"]
off = state["hints"]["fingertip_offset_m"]
print(f"ok  reset: cube={cube} target={target} joints={state['joint_angles_rad']}")

r, _ = tool("think", {"thought": "Cube located; approaching from above."})
assert r["ok"]
_, images = tool("get_camera_image", {"camera": "front"})
assert images and images[0]["mimeType"] == "image/png"
print(f"ok  think + get_camera_image ({len(images[0]['data'])} b64 chars)")

tool("release")
r, _ = tool("move_to", {"x": cube[0], "y": cube[1], "z": cube[2] + 0.16})
assert r["reached"], r
r, _ = tool("move_to", {"x": cube[0], "y": cube[1], "z": cube[2] + off,
                        "tolerance": 0.012})
assert r["reached"], r
r, _ = tool("grasp")
assert r["holding"], f"grasp missed the cube: {r}"
print("ok  approach + grasp (holding)")

tool("move_to", {"x": cube[0], "y": cube[1], "z": 0.18})
tool("move_to", {"x": target[0], "y": target[1], "z": 0.18})
tool("move_to", {"x": target[0], "y": target[1], "z": 0.025 + off + 0.008,
                 "tolerance": 0.025})
tool("release")
tool("move_to", {"x": target[0], "y": target[1], "z": 0.18})
r, _ = tool("check_success")
assert r["success"], f"episode did not succeed: {r}"
print(f"ok  full episode success in {r['sim_steps']} sim steps")

# error paths come back as tool errors, not crashes
bad = rpc("tools/call", {"name": "no_such_tool", "arguments": {}})
assert bad["isError"]
bad = rpc("tools/call", {"name": "get_camera_image",
                         "arguments": {"camera": "nope"}})
assert bad["isError"]
print("ok  error handling")

# ---------------------------------------------------------------------------
# Sessions: save, list, autosave on reset, replay into the viewer stream.
# ---------------------------------------------------------------------------
import shutil

for stale in os.listdir("yam/sessions") if os.path.isdir("yam/sessions") else []:
    if stale.startswith("smoke-episode"):
        shutil.rmtree(os.path.join("yam/sessions", stale), ignore_errors=True)

saved, _ = tool("save_session", {"name": "smoke-episode"})
assert saved["name"] == "smoke-episode" and saved["success"]
assert saved["frames"] > 100 and saved["events"] > 10
listed, _ = tool("list_sessions")
assert any(s["name"] == "smoke-episode" for s in listed["sessions"])
print(f"ok  save_session ({saved['frames']} frames, {saved['events']} events)")

state, _ = tool("reset", {"task": "pick_cube", "seed": 8})
assert "autosaved_session" in state, "reset should autosave the episode"
autosaved = state["autosaved_session"]
print(f"ok  autosave on reset ({autosaved})")

replaying, _ = tool("replay_session", {"name": "smoke-episode"})
assert replaying["ok"]
deadline = time.time() + 30
while time.time() < deadline:
    if any(m.get("type") == "replay_end" for m in ws_messages):
        break
    time.sleep(0.5)
assert any(m.get("type") == "replay_start" and m["name"] == "smoke-episode"
           for m in ws_messages), "viewer never started the replay"
assert any(m.get("type") == "replay_end" for m in ws_messages), \
    "replay never finished"
replay_states = [m for m in ws_messages
                 if m.get("type") == "state" and m.get("mode") == "replay"]
assert len(replay_states) > 50, f"too few replay frames: {len(replay_states)}"
replay_events = [e for m in ws_messages
                 if m.get("type") == "agent_events" and m.get("replay")
                 for e in m["events"]]
assert {"thought", "tool", "image"} <= {e["kind"] for e in replay_events}, \
    "replayed feed missing event kinds"
print(f"ok  replay: {len(replay_states)} frames + "
      f"{len(replay_events)} feed events streamed")

# ---------------------------------------------------------------------------
# TCP MCP endpoint + stdio bridge: a second client against the SAME sim.
# ---------------------------------------------------------------------------
bridge = subprocess.Popen(
    [sys.executable, "-m", "yam.mcp_bridge", "--port", str(MCP_PORT)],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)


def bridge_rpc(method, params=None, mid=99):
    bridge.stdin.write(json.dumps(
        {"jsonrpc": "2.0", "id": mid, "method": method,
         "params": params or {}}) + "\n")
    bridge.stdin.flush()
    resp = json.loads(bridge.stdout.readline())
    assert "error" not in resp, f"{method} -> {resp['error']}"
    return resp["result"]


init2 = bridge_rpc("initialize", {"protocolVersion": "2024-11-05",
                                  "capabilities": {},
                                  "clientInfo": {"name": "bridge", "version": "0"}})
assert init2["serverInfo"]["name"] == "yam-arm"
tools2 = bridge_rpc("tools/list")["tools"]
assert {t["name"] for t in tools2} == names
r2 = bridge_rpc("tools/call", {"name": "get_state", "arguments": {}})
state2 = json.loads(r2["content"][0]["text"])
# same sim: the bridged client sees the state the stdio client produced
state1, _ = tool("get_state")
assert state2["privileged"] == state1["privileged"], "bridge sees another sim!"

# Regression: a seedless reset takes the in-place branch, which re-renders
# cameras on the existing GL context. That used to deadlock when the call
# arrived on a different thread than the one that built the renderer (macOS
# GL thread affinity) -- all sim work must stay on the one sim thread.
def timed_bridge_call(name, arguments, timeout=30):
    box = {}

    def go():
        box["resp"] = bridge_rpc("tools/call",
                                 {"name": name, "arguments": arguments})

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), f"{name} via bridge hung >{timeout}s"
    return box["resp"]


r3 = timed_bridge_call("reset", {"task": "pick_cube"})
assert not r3["isError"], r3
r3 = timed_bridge_call("get_camera_image", {"camera": "front"})
assert not r3["isError"], r3
bridge.stdin.close()
bridge.wait(timeout=10)
print("ok  TCP endpoint + bridge share the sim (incl. seedless reset)")

# ---------------------------------------------------------------------------
# Scene editing: spawn a second cube, verify the rebuild preserved poses and
# the viewer was told to reload, then remove it.
# ---------------------------------------------------------------------------
state, _ = tool("get_state")
cube_before = state["privileged"]["object_pos"]

r, _ = tool("spawn_object", {"shape": "box", "name": "cube2",
                             "x": 0.36, "y": 0.12, "rgba": [0.2, 0.45, 0.9]})
assert r["ok"] and "cube2" in r["objects"], r
assert abs(r["objects"]["cube2"]["pos"][0] - 0.36) < 0.02, r

state, _ = tool("get_state")
assert "cube2" in state["privileged"]["spawned_objects"], state
# the rebuild must not have moved the base cube
drift = sum(abs(a - b) for a, b in zip(state["privileged"]["object_pos"],
                                       cube_before))
assert drift < 0.02, f"base cube moved during spawn: {drift}"

# the manifest travels over the viewer socket (hello + "scene" messages on
# recompile), never via the shared disk export -- an offline session writing
# manifest.json must not be able to desync a live viewer
assert any(m.get("type") == "hello" and m.get("manifest")
           for m in ws_messages), "hello carried no manifest"

# in-place (seedless) reset keeps spawned objects and re-places them
state, _ = tool("reset")
assert "cube2" in state["privileged"]["spawned_objects"], state


def scene_msgs():
    return [m for m in ws_messages if m.get("type") == "scene"]


deadline = time.time() + 5
while time.time() < deadline:
    if any(any(b["name"] == "cube2" for b in m["manifest"]["bodies"])
           for m in scene_msgs()):
        break
    time.sleep(0.2)
assert any(any(b["name"] == "cube2" for b in m["manifest"]["bodies"])
           for m in scene_msgs()), \
    "viewer stream never got the recompiled scene's manifest"
assert any(m.get("type") == "state" and m.get("scene_version", 0) >= 1
           for m in ws_messages), "viewer stream never saw the scene bump"

r, _ = tool("remove_objects")
assert r["ok"] and not r["objects"], r
deadline = time.time() + 5
while time.time() < deadline:
    if any(all(b["name"] != "cube2" for b in m["manifest"]["bodies"])
           for m in scene_msgs()):
        break
    time.sleep(0.2)
assert any(all(b["name"] != "cube2" for b in m["manifest"]["bodies"])
           for m in scene_msgs()), \
    "viewer stream never got the base-scene manifest after remove"
print("ok  spawn_object / remove_objects (recompile preserves poses, "
      "manifest streamed + scene_version updated)")

# ---------------------------------------------------------------------------
# Chat: with no API key the pane gets a polite system message, not a crash.
# ---------------------------------------------------------------------------
def ws_send_cmd(obj):
    """Send one masked client->server frame on a fresh WS connection."""
    s = socket.create_connection(("127.0.0.1", VIEWER_PORT), timeout=5)
    s.sendall((
        "GET /ws HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n"
        "Connection: Upgrade\r\nSec-WebSocket-Key: dGVzdCB0ZXN0IHRlc3Qh\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n").encode())
    b = b""
    while b"\r\n\r\n" not in b:
        b += s.recv(1024)
    payload = json.dumps(obj).encode()
    mask = b"\x01\x02\x03\x04"
    masked = bytes(c ^ mask[i % 4] for i, c in enumerate(payload))
    header = bytearray([0x81])
    if len(payload) < 126:
        header.append(0x80 | len(payload))
    else:
        header.append(0x80 | 126)
        header += struct.pack(">H", len(payload))
    s.sendall(bytes(header) + mask + masked)
    time.sleep(1.5)
    s.close()


ws_send_cmd({"cmd": "chat", "text": "pick up the cube"})
deadline = time.time() + 10
chat_msgs = []
while time.time() < deadline:
    chat_msgs = [e for m in ws_messages if m.get("type") == "chat"
                 for e in m["messages"]]
    if any(e["role"] == "system" for e in chat_msgs):
        break
    time.sleep(0.5)
assert any(e["role"] == "user" and "cube" in e["text"] for e in chat_msgs), \
    f"user chat message never echoed: {chat_msgs}"
assert any(e["role"] == "system" and "CURSOR_API_KEY" in e["text"]
           for e in chat_msgs), f"no polite key error: {chat_msgs}"
print("ok  chat pane: user echo + missing-key system message")

# ---------------------------------------------------------------------------
# The viewer stream must have carried the whole activity feed.
# ---------------------------------------------------------------------------
time.sleep(1.0)  # let the 30 Hz stream flush
events = [e for m in ws_messages if m.get("type") == "agent_events"
          for e in m["events"]]
kinds = {e["kind"] for e in events}
assert {"thought", "tool", "image"} <= kinds, f"feed missing kinds: {kinds}"
thought = next(e for e in events if e["kind"] == "thought")
assert "approaching" in thought["text"]
image = next(e for e in events if e["kind"] == "image")
assert image["camera"] == "front" and len(image["png_b64"]) > 1000
moves = [e for e in events if e["kind"] == "tool" and e["tool"] == "move_to"]
assert moves and all("result" in e and "sim_step_start" in e for e in moves)
assert any(m.get("type") == "state" and "activity" in m for m in ws_messages)
print(f"ok  viewer feed: {len(events)} events "
      f"({sum(e['kind'] == 'thought' for e in events)} thoughts, "
      f"{sum(e['kind'] == 'tool' for e in events)} tool calls, "
      f"{sum(e['kind'] == 'image' for e in events)} visual inputs)")

proc.stdin.close()
proc.wait(timeout=10)
print("PASS")
