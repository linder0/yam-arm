"""Stdio <-> TCP bridge for the YAM MCP endpoint.

MCP clients spawn stdio servers per client, which would give every client its
own simulator. This bridge lets any stdio MCP client (the chat pane's Cursor
SDK agent, or Cursor itself) attach to the ONE running sim instead: it pipes
newline-delimited JSON-RPC between stdin/stdout and the TCP endpoint that
``yam.mcp_server`` exposes (default port 8766).

Usage (as an MCP server command):
    python -m yam.mcp_bridge --port 8766
"""

import argparse
import os
import socket
import sys
import threading
import time

_DEBUG_PATH = os.environ.get("YAM_BRIDGE_DEBUG")


def _dbg(direction, data):
    if not _DEBUG_PATH:
        return
    try:
        with open(_DEBUG_PATH, "a") as f:
            f.write(f"{time.time():.3f} [{os.getpid()}] {direction} "
                    f"{data[:300]!r}\n")
    except OSError:
        pass


def main():
    p = argparse.ArgumentParser(description="stdio<->TCP MCP bridge")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    args = p.parse_args()
    _dbg("start", b"")

    try:
        sock = socket.create_connection((args.host, args.port), timeout=10)
        # create_connection's timeout would otherwise persist and kill the
        # read loop whenever the agent idles >10s between tool calls.
        sock.settimeout(None)
    except OSError as e:
        print(f"[yam.bridge] cannot reach yam-arm sim at "
              f"{args.host}:{args.port} ({e}); is yam.mcp_server running?",
              file=sys.stderr)
        raise SystemExit(1)

    def stdin_to_sock():
        try:
            for line in sys.stdin.buffer:
                _dbg("stdin->sock", line)
                sock.sendall(line)
        except (OSError, ValueError) as e:
            _dbg("stdin_err", str(e).encode())
        finally:
            _dbg("stdin_eof", b"")
            try:
                sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    threading.Thread(target=stdin_to_sock, daemon=True).start()

    try:
        for line in sock.makefile("rb"):
            _dbg("sock->stdout", line)
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
    except (OSError, ValueError) as e:
        _dbg("sock_err", str(e).encode())
    _dbg("sock_eof", b"")


if __name__ == "__main__":
    main()
