"""End-to-end chat test: boots the robot IDE (console mode), sends one chat
message, and waits for the Cursor SDK agent to drive the arm to task success.

Requires CURSOR_API_KEY (from the environment or .cursor/mcp.json) and
network access, and consumes a small amount of Cursor usage -- so it is NOT
part of scripts/test_mcp.py. Run manually:

    .venv-yam/bin/python scripts/test_chat.py
"""

import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER_PORT = 8771
MCP_PORT = 8772
TIMEOUT_S = 360

env = os.environ.copy()
if not env.get("CURSOR_API_KEY"):
    try:
        with open(os.path.join(ROOT, ".cursor", "mcp.json")) as f:
            env["CURSOR_API_KEY"] = \
                json.load(f)["mcpServers"]["yam-arm"]["env"]["CURSOR_API_KEY"]
    except Exception:
        sys.exit("CURSOR_API_KEY not set and not found in .cursor/mcp.json")

# Refuse to run against a stale server -- we must test the current code.
for p in (VIEWER_PORT, MCP_PORT):
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", p))
    except OSError:
        sys.exit(f"port {p} is already in use (stale server? "
                 f"lsof -ti :{p} | xargs kill)")
    finally:
        probe.close()

proc = subprocess.Popen(
    [sys.executable, "-m", "yam.mcp_server", "--console",
     "--viewer-port", str(VIEWER_PORT), "--mcp-port", str(MCP_PORT),
     "--pace", "0.005"],
    env=env, cwd=ROOT)


class WsClient:
    """Minimal duplex WebSocket client (server frames are unmasked;
    client frames must be masked)."""

    def __init__(self, host, port, retries=50):
        for _ in range(retries):
            try:
                self.sock = socket.create_connection((host, port), timeout=5)
                break
            except OSError:
                time.sleep(0.2)
        else:
            raise ConnectionError("viewer never came up")
        self.sock.sendall((
            "GET /ws HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n"
            "Connection: Upgrade\r\nSec-WebSocket-Key: dGVzdCB0ZXN0IHRlc3Qh\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n").encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.sock.recv(1024)
        self._buf = buf.split(b"\r\n\r\n", 1)[1]
        self.messages = []
        self.sock.settimeout(None)
        threading.Thread(target=self._reader, daemon=True).start()

    def _read_exactly(self, n):
        while len(self._buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def _reader(self):
        try:
            while True:
                b0, b1 = self._read_exactly(2)
                length = b1 & 0x7F
                if length == 126:
                    (length,) = struct.unpack(">H", self._read_exactly(2))
                elif length == 127:
                    (length,) = struct.unpack(">Q", self._read_exactly(8))
                payload = self._read_exactly(length)
                if (b0 & 0x0F) == 0x1:
                    self.messages.append(json.loads(payload))
        except (ConnectionError, OSError):
            pass

    def send(self, obj):
        payload = json.dumps(obj).encode()
        mask = os.urandom(4)
        masked = bytes(c ^ mask[i % 4] for i, c in enumerate(payload))
        header = bytearray([0x81])
        if len(payload) < 126:
            header.append(0x80 | len(payload))
        else:
            header.append(0x80 | 126)
            header += struct.pack(">H", len(payload))
        self.sock.sendall(bytes(header) + mask + masked)


try:
    ws = WsClient("127.0.0.1", VIEWER_PORT)
    time.sleep(1)
    print("connected; sending chat message...")
    ws.send({"cmd": "chat",
             "text": "Pick up the cube and place it on the green target "
                     "zone, then confirm success."})

    t0 = time.time()
    done = False
    seen_chat = set()
    while time.time() - t0 < TIMEOUT_S:
        for m in ws.messages:
            if m.get("type") == "chat":
                for e in m["messages"]:
                    if e["id"] in seen_chat:
                        continue
                    seen_chat.add(e["id"])
                    print(f"  [chat/{e['role']}] {e['text'][:200]}")
        success = any(m.get("type") == "state" and m.get("success")
                      for m in ws.messages)
        chat_entries = [e for m in ws.messages if m.get("type") == "chat"
                        for e in m["messages"]]
        assistant_done = any(e["role"] == "assistant" for e in chat_entries)
        busy = any(m.get("type") == "state" for m in ws.messages[-5:]) and \
            ws.messages[-1].get("chat_busy", False)
        if success and assistant_done and not busy:
            done = True
            break
        if any(e["role"] == "system" and "error" in e["text"].lower()
               for e in chat_entries):
            break
        time.sleep(2)

    events = [e for m in ws.messages if m.get("type") == "agent_events"
              for e in m["events"]]
    thoughts = sum(e["kind"] == "thought" for e in events)
    tools = sum(e["kind"] == "tool" for e in events)
    images = sum(e["kind"] == "image" for e in events)
    print(f"\nfeed: {thoughts} thoughts, {tools} tool calls, "
          f"{images} visual inputs")
    print(f"elapsed: {time.time() - t0:.0f}s")
    if done:
        print("PASS: chat agent completed the task (sim success confirmed)")
    else:
        print("FAIL: task did not complete in time (see transcript above)")
        sys.exit(1)
finally:
    proc.terminate()
