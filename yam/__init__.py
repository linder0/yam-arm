"""Sim-first imitation-learning stack for the I2RT YAM arm.

A self-contained project. The pieces:

    model.py   load the vendored YAM MJCF + task scenes; camera rendering; the
               web-viewer scene manifest.
    robot.py   the hardware-swap seam: a YamRobot protocol with a MuJoCo
               SimYamRobot now and a documented HardwareYamRobot stub for the
               real i2rt CAN arm later.
    env.py     tabletop tasks (pick-and-place a cube, and a reach smoke test):
               observations, actions, success, reset/randomization.
    ik.py      damped-least-squares differential IK (pure MuJoCo) for EE teleop.
    teleop.py  keyboard / websocket end-effector teleoperation.
    record.py  write demonstrations to a LeRobot-compatible dataset.
    train.py   thin wrapper around `lerobot-train` (ACT / Diffusion) + provenance.
    eval.py    roll out a trained policy in sim, report success, stream to viewer.
    server.py  websocket bridge: step the sim (teleop / policy) and stream poses
               + camera frames to yam/web (the live 3-D render).

Everything runs in sim today; swapping SimYamRobot -> HardwareYamRobot is the
only change needed to drive a physical YAM (see yam/README.md).
"""

__all__ = ["model", "robot", "env", "ik"]
