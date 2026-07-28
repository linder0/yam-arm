# YAM arm — sim-first imitation learning + live web render

A self-contained stack for teaching an [I2RT YAM](https://i2rt.com/products/yam-manipulator)
6-DOF arm to do tabletop tasks by **imitation learning**, running in MuJoCo
today and structured so a real YAM (i2rt CAN + cameras) drops in behind one
interface later. It ships a **live 3-D web viewer** that renders the arm in the
browser — no Node/npm build required.

Manipulation here is a different paradigm from locomotion RL: teleoperate →
record demos → train ACT/Diffusion (LeRobot) → deploy, rather than on-policy RL.
The project keeps the same good habits as its locomotion sibling
(self-documenting run folders, a browser render).

```
teleop / autopilot ──► record demos ──► lerobot-train ──► eval ──► live viewer
   (IK, browser)        LeRobot dataset   ACT / Diffusion   success   three.js
        │                                                      ▲          ▲
        └──────────────── MuJoCo sim (SimYamRobot) ────────────┴──────────┘
                          swap → HardwareYamRobot (i2rt CAN) for the real arm
```

## What's here

| File | Role |
|---|---|
| `model.py` | load the vendored YAM MJCF + task scenes; camera rendering; web scene manifest export |
| `robot.py` | `YamRobot` interface + `SimYamRobot` (MuJoCo) + `HardwareYamRobot` stub — the hardware seam |
| `env.py` | tabletop tasks: `pick_cube`, `reach` (obs/action/reward/success/reset) |
| `ik.py` | damped-least-squares differential IK (pure MuJoCo) |
| `teleop.py` | Cartesian EE teleop + a scripted expert (privileged demo generator) |
| `record.py` | write demos to a LeRobot-compatible dataset (numpy fallback if lerobot absent) |
| `train.py` | wrapper around `lerobot-train` (ACT/Diffusion) + run-folder provenance |
| `eval.py` | roll out scripted or trained policies; success rate, video, or live view |
| `agent.py` | task-level skills for LLM-agent control (Cartesian goals, grasp/release) |
| `mcp_server.py` | the unified server (`python -m yam`): viewer + manual control + chat + MCP, one sim |
| `server.py` | lightweight standalone viewer/teleop server; home of the bimanual scenes |
| `web/` | zero-build three.js viewer (CDN import map), served by `server.py` |
| `assets/yam/` | vendored MuJoCo Menagerie `i2rt_yam` (MIT) + task scenes |

## Setup

```bash
python3.12 -m venv .venv-yam && source .venv-yam/bin/activate
pip install -r requirements.txt
```

The sim, viewer, teleop, and demo collection run with just
`mujoco`/`numpy`/`imageio`. Training (`train.py`) and trained-policy eval add the
heavier `lerobot` + `torch`; those steps print install guidance if the stack
isn't present.

Quick sanity check of the model:

```bash
python -m yam.model --task pick_cube --check
```

## One server: the robot IDE

```bash
python -m yam                   # everything at http://localhost:8765
```

One command, one sim, every front: the live 3-D viewer, manual control
(autopilot / keyboard + hand teleop / demo recording), the browser chat pane,
and an MCP endpoint for external agents. Set `CURSOR_API_KEY` first if you
want the chat pane (see the IDE section below); everything else works without
it. Any `yam.mcp_server` flag works, e.g. `python -m yam --task reach`.

Humans and agents share the arm with "last actor wins" arbitration: an agent
tool call pauses manual stepping and flips the mode to idle; clicking a mode
(or using teleop) takes control back. Tool calls are serialized on one sim
thread, so nothing ever interleaves mid-motion.

`python -m yam.server` (port 8080) still exists as a lightweight,
dependency-free viewer — and it is the home of the **two-arm (bimanual)**
scenes, which have no agent story yet.

Open the URL and:
- **Autopilot** — the scripted expert solves pick-and-place on a loop.
- **Teleop** — drive the gripper with the keyboard: `W/S` = ±X, `A/D` = ±Y,
  `Q/E` = ±Z, `Space` toggles the gripper. Motions are slew-rate limited so the
  arm never whips.
- **Hand teleop (MediaPipe)** — click *Enable hand tracking* to drive the arm
  with your webcam: move your hand left/right/up/down to steer the gripper in
  Y/Z, move it toward/away from the camera for depth (X), and **pinch**
  thumb-to-index to close the gripper. Tracking runs entirely in the browser
  (MediaPipe `HandLandmarker` from CDN — no extra Python deps); it streams
  absolute EE targets over the same WebSocket as the keyboard, so the server's
  workspace clamp + slew limiting still apply. Losing the hand just holds the
  arm in place. In two-arm scenes both hands are tracked and each drives its
  own arm (left hand → left arm) with its own pinch-gripper.
- Switch tasks, reset, scrub speed, and **record** a teleop demonstration
  (cameras are on by default in `python -m yam`; on the lightweight
  `yam.server` start with `--cam-size 128` to capture frames).

The server is authoritative: Python owns the sim loop and streams every body's
world pose over a WebSocket; the browser only renders (three.js, MuJoCo Z-up →
Y-up, STL meshes for the arm links). This is the same shape that will drive the
real arm — swap the sim for hardware and the viewer is unchanged.

## Agent control (MCP)

Any tool-calling LLM can drive the arm through task-level skills — Cartesian
end-effector goals and gripper commands, each executed closed-loop by the
differential-IK controller while the model plans. The agent decides *what* to
do; the controller handles the 20 Hz *how*.

```bash
python -m yam.agent --demo      # scripted pass through the skills; validates the layer
python -m yam.mcp_server        # MCP server on stdio + live viewer at :8765
```

`.cursor/mcp.json` registers the server as `yam-arm` in Cursor: enable it in
Cursor Settings → MCP, then ask the agent to e.g. "pick up the cube and place
it on the target" and watch it work at http://localhost:8765.

Tools: `think` (chain of thought), `reset`, `get_state` (joint angles + EE
pose + privileged sim ground truth + grasp hints), `get_camera_image`
(front/top/side/wrist), `move_to` / `move_delta` (closed-loop servo,
workspace-clamped), `grasp` / `release` / `set_gripper`, `wait`,
`check_success`, and scene editing: `spawn_object` adds a box/sphere/cylinder
of any size/color to the table ("give me two cubes"), recompiling the scene
in place — the arm and existing objects keep their poses and the live viewer
reloads automatically; `remove_objects` restores the base scene. Every motion
result also reports `disturbed_objects` — anything the arm's body bumped out
of place (displacement + the arm links that touched it), so the agent notices
collisions with its own body without taking a photo. The server
is stdlib-only, like the rest of the stack;
`scripts/test_mcp.py` smoke-tests the protocol and the viewer feed end-to-end.

**Agent pane.** The viewer's right pane is a single Cursor-style
conversation, everything interleaved in the order it happened:

- **User messages** appear as cards; **assistant replies** as plain prose.
- **Chain of thought** — the agent's reasoning (recorded via the `think`
  tool; the server instructs models to narrate before acting) renders as
  dimmed indented text between actions.
- **Tool calls** are compact one-line rows (`● move_to {"x":0.42,...}` with
  duration); click a row to expand its arguments, result, and the sim-step
  span it executed over. State reads (`get_state`, `check_success`) are
  dimmed — that's the agent sampling proprioception.
- **Visual inputs** show the exact frame the agent saw, inline. Visual input
  is taken *only* when the agent calls `get_camera_image`; motions in between
  servo on proprioception alone, so the transcript shows precisely when the
  agent looked vs. moved blind.

The pane header shows the tool call currently in flight; reconnecting replays
recent history. The left header shows the connection status.

**Sessions: save + replay.** Every episode is recorded (per-step body poses +
the activity feed). Save via the `save_session` tool, the pane's **Save
session** button, or automatically on `reset`; sessions land in
`yam/sessions/<name>/` (`frames.npz` + `session.json`). Replay from the
pane's session dropdown, or by asking the agent (`replay_session`, defaults
to latest): the 3-D scene plays back the recorded motion in real time while
the right pane re-runs the recorded thoughts, tool calls, and camera captures
at the sim step they originally happened.

## Robot IDE: chat with the arm in the browser

The viewer doubles as a friendly IDE: the input box under the agent pane
sends your message to a **Cursor agent** (via the [Cursor SDK](https://cursor.com/docs/sdk/python)),
which drives the arm through the same MCP tools — its thoughts, tool calls,
and camera captures stream into the conversation while you watch the 3-D
scene.

```bash
export CURSOR_API_KEY="cursor_..."      # cursor.com/dashboard -> Integrations
python -m yam                           # IDE at http://localhost:8765, no Cursor app needed
```

Type "pick up the cube" in the chat pane and watch. Options:
`--chat-model composer-2.5` (default), `--no-chat` to disable.

One sim, many fronts: the server also exposes a TCP MCP endpoint
(`--mcp-port`, default 8766) against the *same* simulator, and
`yam.mcp_bridge` pipes any stdio MCP client into it — that's how the chat
agent connects, and how any other MCP client (including Cursor itself) can
share the arm instead of spawning its own sim. Tool calls across fronts are
serialized so two agents can't fight over the arm mid-motion.

When Cursor spawns the server via `.cursor/mcp.json`, the chat pane works
too — set `CURSOR_API_KEY` in that file's `env` block (and git-ignore the
file: it now holds a secret).

`scripts/test_chat.py` verifies the whole loop for real — it boots the IDE,
types a pick-and-place request into the chat, and waits for the SDK agent to
drive the arm to `check_success`. It needs the API key and spends a little
Cursor usage, so it's separate from the free `scripts/test_mcp.py` smoke test.

## Record demonstrations

Headless, using the scripted expert (great for a first dataset / baseline):

```bash
python -m yam.record --task pick_cube --episodes 100 --cam-size 128
# -> yam/data/pick_cube  (success-only by default; --keep-failures to keep all)
```

Human teleop demos: run `python -m yam`, switch to Teleop in the browser, hit
**Record**, do the task, hit **Save**.

Dataset frames use LeRobot keys: `observation.state` (13 = arm qpos(6)+qvel(6)+
gripper(1)), `observation.images.<cam>` (uint8 HxWx3), `action` (7 = abs joint
targets + gripper). With `lerobot` installed, `record.py` writes a native
`LeRobotDataset` (parquet + mp4). Without it, it writes an equivalent numpy
(`.npz`) dataset so the loop still runs — install `lerobot` for real training.

## Train

```bash
python -m yam.train --dataset yam/data/pick_cube --policy act --steps 20000 --run
# or --policy diffusion ; omit --run to just print the lerobot-train command
```

Writes `yam/runs/<task>/<run>/config.json` (command, git commit, versions,
dataset ref, and the env `task_spec`), then launches `lerobot-train`.

## Evaluate

```bash
# scripted expert (no extra deps) — validates the harness + records a video:
python -m yam.eval --task pick_cube --policy scripted --episodes 20 --video eval.mp4

# a trained checkpoint, headless success rate:
python -m yam.eval --task pick_cube --policy yam/runs/pick_cube/<run> --episodes 50

# watch a trained policy in the browser:
python -m yam.eval --task pick_cube --policy yam/runs/pick_cube/<run> --serve
```

## Tasks

- **`pick_cube`** — pick the cube and place it on the green target zone. Success
  = cube resting within the target radius. The scripted expert solves it ~100%.
- **`reach`** — move the gripper to a randomized target sphere (contact-free
  smoke test for the model / IK / recording / viewer).

Randomization (cube/target placement, arm posture noise) is recorded in each
run's `task_spec`.

## Swapping in real hardware

Only `robot.py` changes. Implement `HardwareYamRobot` (a documented stub today)
against the i2rt CAN SDK + real cameras, matching the `YamRobot` interface:

```python
from i2rt.robots.get_robot import get_yam_robot   # i2rt CAN SDK
arm = get_yam_robot(channel="can0")                # the sim=False path
# arm_qpos/qvel  <- arm.get_joint_pos()/vel()
# command(...)   -> arm.command_joint_pos([*arm_targets, gripper])  # native Joint-PD
# cameras()      -> grab synchronized RGB frames
# ee_pose()      -> forward kinematics (reuse the MuJoCo model)
```

Everything above it — `env`, `teleop`, `record`, `train`, `eval`, `server`, and
the viewer — is unchanged. Notes:
- Actions are already YAM's native joint-position targets + a normalized gripper,
  so they map 1:1 to hardware.
- Object/target ground truth is sim-only (used for reward/success); on hardware
  the policy sees objects through the cameras, which is why images are in the
  observation from day one.
- Safety: YAM has a 400 ms motor timeout — keep `command` flowing and clamp
  targets to joint limits before sending.

## Provenance / attribution

The arm model is MuJoCo Menagerie `i2rt_yam` (MIT, vendored in `assets/yam/`)
with two local edits noted in `assets/yam/yam.xml`: a `wrist` camera and the
removal of the fixed `home` keyframe (task scenes change `nq`; the home pose now
lives in `model.HOME_QPOS_ARM`).
