"""Browser chat -> Cursor SDK agent -> the arm.

The viewer's chat pane sends user messages over the existing WebSocket; this
module runs them through a Cursor agent (``cursor-sdk``) whose MCP config
points back at the running sim via ``yam.mcp_bridge``. The agent's tool calls
therefore land in the same simulator the browser is watching, and its
narration (``think``) and captures stream into the activity pane like any
other client's.

Chat messages are kept in an id-ordered log (same pattern as the activity
events) that the viewer stream polls, so history replays on reconnect and
multiple browsers stay in sync.

Requires ``pip install cursor-sdk`` and ``CURSOR_API_KEY`` in the server's
environment; both are checked lazily so the rest of the stack runs without
them.
"""

import os
import sys
import threading
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Sent once, ahead of the first user message, to shape the agent's behavior.
PREAMBLE = (
    "You are the assistant inside a friendly robot-arm IDE. A simulated I2RT "
    "YAM 6-DOF arm is available through the yam-arm MCP tools, and the human "
    "is watching a live 3-D viewer with an agent-activity pane beside this "
    "chat. Rules: (1) before each action, call the think tool to narrate "
    "your reasoning -- the human sees it in the activity pane; (2) keep chat "
    "replies to one or two short sentences, since your tool calls and "
    "thoughts are already displayed; (3) use get_camera_image when visual "
    "verification helps; (4) do not read or modify files, run shell "
    "commands, or start/stop/restart any servers or processes -- your only "
    "interface to the world is the yam-arm MCP tools; (5) if those tools "
    "error or disconnect, say so in one sentence and stop -- never try to "
    "work around it.\n\nUser message: ")

MAX_LOG = 300


class ChatManager:
    """One conversation with one SDK agent; thread-safe log for the viewer."""

    def __init__(self, mcp_port, model="composer-2.5", repo_root=REPO_ROOT):
        self.mcp_port = int(mcp_port)
        self.model = model
        self.repo_root = repo_root
        self.busy = False
        self._agent = None
        self._primed = False
        self._lock = threading.Lock()
        self._log = []
        self._next_id = 1

    # -- log (polled by the viewer stream) --------------------------------
    def _append(self, role, text):
        with self._lock:
            entry = {"id": self._next_id, "role": role, "text": text,
                     "wall_time": time.time()}
            self._next_id += 1
            self._log.append(entry)
            if len(self._log) > MAX_LOG:
                del self._log[:len(self._log) - MAX_LOG]
        return entry

    def entries_since(self, last_id, limit=50):
        with self._lock:
            return [e for e in self._log if e["id"] > last_id][:limit]

    # -- message handling --------------------------------------------------
    def submit(self, text):
        """Called from the viewer's WS reader; must not block."""
        with self._lock:
            if self.busy:
                reject = True
            else:
                self.busy = True
                reject = False
        if reject:
            self._append("system", "Still working on the previous message -- "
                                   "one command at a time.")
            return
        self._append("user", text)
        threading.Thread(target=self._run, args=(text,), daemon=True).start()

    def _run(self, text):
        try:
            try:
                import cursor_sdk  # noqa: F401
            except ImportError:
                self._append("system",
                             "cursor-sdk is not installed in the server's "
                             "venv (pip install cursor-sdk).")
                return
            if not os.environ.get("CURSOR_API_KEY"):
                self._append("system",
                             "CURSOR_API_KEY is not set for the yam-arm "
                             "server. Add it to the env block in "
                             ".cursor/mcp.json (key from "
                             "cursor.com/dashboard -> Integrations), then "
                             "restart the server.")
                return

            agent = self._ensure_agent()
            prompt = text if self._primed else PREAMBLE + text
            self._primed = True

            run = agent.send(prompt)
            # run.messages() yields streaming deltas: consecutive assistant
            # chunks are one utterance, so buffer them and flush a single
            # chat entry at each non-assistant boundary (tool call, etc.).
            buf = []
            for message in run.messages():
                if message.type == "assistant":
                    buf.extend(b.text for b in message.message.content
                               if getattr(b, "type", "") == "text" and b.text)
                elif buf:
                    joined = "".join(buf).strip()
                    if joined:
                        self._append("assistant", joined)
                    buf = []
            if buf and "".join(buf).strip():
                self._append("assistant", "".join(buf).strip())
            result = run.wait()
            if result.status != "finished":
                self._append("system", f"agent run ended: {result.status}")
        except Exception as e:
            print(f"[yam.chat] {e}", file=sys.stderr)
            self._append("system", f"agent error: {e}")
        finally:
            self.busy = False

    def _ensure_agent(self):
        if self._agent is not None:
            return self._agent
        from cursor_sdk import (Agent, AgentOptions, LocalAgentOptions,
                                StdioMcpServerConfig)
        bridge_env = {"PYTHONPATH": self.repo_root}
        if os.environ.get("YAM_BRIDGE_DEBUG"):
            bridge_env["YAM_BRIDGE_DEBUG"] = os.environ["YAM_BRIDGE_DEBUG"]
        self._agent = Agent.create(AgentOptions(
            model=self.model,
            api_key=os.environ["CURSOR_API_KEY"],
            name="yam-arm-ide",
            local=LocalAgentOptions(cwd=self.repo_root),
            mcp_servers={"yam-arm": StdioMcpServerConfig(
                command=sys.executable,
                args=["-m", "yam.mcp_bridge", "--port", str(self.mcp_port)],
                env=bridge_env,
                cwd=self.repo_root)},
        ))
        return self._agent

    def close(self):
        if self._agent is not None:
            try:
                self._agent.close()
            except Exception:
                pass
            self._agent = None
