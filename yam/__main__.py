"""``python -m yam`` -- start the whole robot IDE with one command.

Runs ``yam.mcp_server`` in console mode: live 3-D viewer, manual control
(autopilot / keyboard + hand teleop / demo recording), browser chat, and the
TCP MCP endpoint, all against one simulator. Any ``yam.mcp_server`` flag
works here too (e.g. ``python -m yam --task reach --no-chat``).
"""

import sys

from .mcp_server import main

if "--console" not in sys.argv:
    sys.argv.append("--console")
main()
