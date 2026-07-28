"""Smoke test for the unified server's manual-control path: boots
``python -m yam`` (console mode), drives the sim from a browser-style
WebSocket client (autopilot mode, teleop nudges), and verifies that an agent
tool call over the TCP MCP endpoint preempts the manual mode (last actor
wins). Exits non-zero on any failure.

Usage: .venv-yam/bin/python scripts/test_manual.py
"""

import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time

VIEWER_PORT = 8771
MCP_PORT = 8772

env = dict(os.environ, CURSOR_API_KEY="")
proc = subprocess.Popen(
    [sys.executable, "-m", "yam", "--no-chat",
     "--viewer-port", str(VIEWER_PORT), "--mcp-port", str(MCP_PORT),
     "--pace", "0"],
    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env)

try:
    # -- browser-style WebSocket client ------------------------------------
    for _ in range(100):
        try:
            sock = socket.create_connection(("127.0.0.1", VIEWER_PORT),
                                            timeout=5)
            break
        except OSError:
            time.sleep(0.2)
    else:
        raise SystemExit("server never bound the viewer port")
    sock.sendall((
        "GET /ws HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n"
        "Connection: Upgrade\r\nSec-WebSocket-Key: dGVzdCB0ZXN0IHRlc3Qh\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n").encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        buf += sock.recv(1024)
    buf = buf.split(b"\r\n\r\n", 1)[1]

    messages = []

    def reader():
        global buf

        def read_exactly(n):
            global buf
            while len(buf) < n:
                chunk = sock.recv(65536)
                if not chunk:
                    raise ConnectionError
                buf += chunk
            out, rest = buf[:n], buf[n:]
            buf = rest
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
                    messages.append(json.loads(payload))
        except (ConnectionError, OSError):
            pass

    threading.Thread(target=reader, daemon=True).start()

    def send_cmd(obj):
        payload = json.dumps(obj).encode()
        mask = b"\x01\x02\x03\x04"
        masked = bytes(c ^ mask[i % 4] for i, c in enumerate(payload))
        header = bytearray([0x81])
        if len(payload) < 126:
            header.append(0x80 | len(payload))
        else:
            header.append(0x80 | 126)
            header += struct.pack(">H", len(payload))
        sock.sendall(bytes(header) + mask + masked)

    def latest_state():
        states = [m for m in messages if m.get("type") == "state"]
        return states[-1] if states else None

    def wait_for(pred, timeout=15, what="condition"):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if pred():
                return
            time.sleep(0.1)
        raise SystemExit(f"timed out waiting for {what} "
                         f"(last state: {latest_state()})")

    wait_for(lambda: any(m.get("type") == "hello" for m in messages),
             what="hello")
    print("ok  connected (hello received)")

    # -- autopilot: mode -> scripted must make sim steps advance -----------
    send_cmd({"cmd": "mode", "mode": "scripted"})
    wait_for(lambda: (s := latest_state()) and s["mode"] == "scripted",
             what="scripted mode in the stream")
    step0 = latest_state()["step"]
    wait_for(lambda: (s := latest_state()) and s["step"] > step0 + 20,
             what="autopilot stepping")
    print("ok  autopilot mode steps the sim")

    # -- agent preemption: a tool call over TCP flips the mode to idle -----
    mcp = socket.create_connection(("127.0.0.1", MCP_PORT), timeout=10)
    mcp_in = mcp.makefile("rb")

    def rpc(method, params, mid):
        mcp.sendall((json.dumps({"jsonrpc": "2.0", "id": mid,
                                 "method": method,
                                 "params": params}) + "\n").encode())
        resp = json.loads(mcp_in.readline())
        assert "error" not in resp, resp
        return resp["result"]

    rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "manual-smoke", "version": "0"}},
        1)
    r = rpc("tools/call", {"name": "move_to",
                           "arguments": {"x": 0.40, "y": 0.0, "z": 0.25}}, 2)
    assert not r["isError"], r
    wait_for(lambda: (s := latest_state()) and s["mode"] == "idle",
             what="agent preemption (mode back to idle)")
    print("ok  agent tool call preempts the manual mode")

    # -- teleop: nudges move the arm between agent calls --------------------
    send_cmd({"cmd": "mode", "mode": "teleop"})
    wait_for(lambda: (s := latest_state()) and s["mode"] == "teleop",
             what="teleop mode in the stream")
    step1 = latest_state()["step"]
    for _ in range(5):
        send_cmd({"cmd": "teleop_delta", "dz": 0.02})
        time.sleep(0.1)
    wait_for(lambda: (s := latest_state()) and s["step"] > step1 + 10,
             what="teleop stepping")
    r = json.loads(rpc("tools/call", {"name": "get_state",
                                      "arguments": {}}, 3)
                   ["content"][0]["text"])
    assert r["ee_pos"][2] > 0.26, f"teleop nudges had no effect: {r['ee_pos']}"
    print(f"ok  teleop nudges move the arm (ee z={r['ee_pos'][2]})")
    print("PASS")
finally:
    proc.terminate()
    proc.wait(timeout=10)
