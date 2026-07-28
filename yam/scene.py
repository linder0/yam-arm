"""Compose a task scene around any embodiment.

The hand-written task scenes (``yam/assets/yam/pick_cube.xml``) ``<include>``
the vendored YAM arm and place the table, cube, target and cameras at
YAM-relative coordinates. That layout is unusable on an arm with 0.47 m of
reach or 1.22 m of it, and the ``<include>`` trick only works because the arm
MJCF sits in the same directory as the scene (so its ``meshdir`` resolves).

So the scene is generated instead: a world built from primitives only (no mesh
paths to resolve), with the robot attached into it via ``MjSpec``. Every
dimension is expressed as a multiple of the arm's measured reach, calibrated so
that a 0.79 m YAM reproduces the original hand-written layout.

Attaching also handles the arms that ship no gripper at all -- UR5e, iiwa, FR3,
Kinova and the rest expose only a flange ``attachment_site``, and get a Robotiq
2F-85 bolted on here.
"""

from __future__ import annotations

import numpy as np
import mujoco

from . import embodiments as E

# YAM's reach, used to convert the original hand-tuned scene into ratios.
_BASE_REACH = 0.79

def _adopt_physics(world, arm_model, emb):
    """Copy the arm's solver settings onto the composed world.

    MuJoCo resolves attach conflicts in favour of the *parent*, so a world that
    leaves these at their defaults silently downgrades the arm's physics -- YAM
    asks for ``implicitfast`` and would otherwise be integrated with Euler. The
    Robotiq gripper is the one case that overrides the arm, since it needs an
    elliptic friction cone and a high impedance ratio or its grasps slip.
    """
    opt = arm_model.opt
    world.option.timestep = float(opt.timestep)
    world.option.integrator = int(opt.integrator)
    world.option.cone = int(opt.cone)
    world.option.impratio = float(opt.impratio)
    if emb.attach_gripper == "robotiq_2f85":
        world.option.cone = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
        world.option.impratio = max(10.0, float(opt.impratio))


def _r(emb_reach, ratio):
    return emb_reach * ratio


def cube_half_size(open_gap):
    """Half-extent of the cube, sized to the gripper rather than the arm: a
    gripper that opens 70 mm cannot pick the 50 mm cube the YAM scene uses."""
    return float(np.clip(0.30 * open_gap, 0.012, 0.025))


def world_xml(reach, cube_half, target_radius, extra_objects=(), task="pick_cube"):
    """The task world, scaled to ``reach``. Table top sits at z = 0 so the arm
    base mounts at the origin, matching the original scene's convention."""
    tbl_x, tbl_sx, tbl_sy = _r(reach, 0.38), _r(reach, 0.57), _r(reach, 0.63)
    floor_z = -_r(reach, 0.51)
    cube_x = _r(reach, 0.51)
    tgt_x, tgt_y = _r(reach, 0.51), _r(reach, 0.35)

    has_object = task != "reach"
    obj = ""
    if has_object:
        obj = f"""
    <body name="cube" pos="{cube_x:.4f} 0 {cube_half:.4f}">
      <freejoint name="cube_free"/>
      <inertial pos="0 0 0" mass="0.05" diaginertia="2.1e-5 2.1e-5 2.1e-5"/>
      <geom name="cube_geom" type="box" size="{cube_half:.4f} {cube_half:.4f} {cube_half:.4f}"
        material="cube_mat" condim="4" friction="1.0 0.05 0.01" contype="1" conaffinity="1"/>
    </body>"""

    spawned = "".join(
        f"""
    <body name="{o['name']}" pos="{o['pos'][0]:g} {o['pos'][1]:g} {o['pos'][2]:g}">
      <freejoint name="{o['name']}_free"/>
      <geom name="{o['name']}_geom" type="{o['shape']}"
        size="{' '.join(f'{float(s):g}' for s in o['size'])}"
        rgba="{' '.join(f'{float(c):g}' for c in o['rgba'])}" mass="{float(o['mass']):g}"
        condim="4" friction="1.0 0.05 0.01" contype="1" conaffinity="1"/>
    </body>""" for o in extra_objects)

    return f"""<mujoco model="tabletop">
  <statistic center="{_r(reach, 0.38):.3f} 0 {_r(reach, 0.19):.3f}" extent="{_r(reach, 1.27):.3f}"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="140" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
      rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="tabletop" rgba="0.55 0.45 0.35 1"/>
    <material name="cube_mat" rgba="0.85 0.2 0.2 1"/>
    <material name="target_mat" rgba="0.2 0.8 0.3 0.35"/>
  </asset>
  <worldbody>
    <light pos="{_r(reach, 0.38):.3f} 0 {_r(reach, 1.9):.3f}" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"
      pos="0 0 {floor_z:.4f}" contype="1" conaffinity="1"/>
    <geom name="table" type="box" material="tabletop" pos="{tbl_x:.4f} 0 -0.02"
      size="{tbl_sx:.4f} {tbl_sy:.4f} 0.02" contype="1" conaffinity="1"/>{obj}
    <body name="target" pos="{tgt_x:.4f} {tgt_y:.4f} 0.001" mocap="true">
      <geom name="target_geom" type="cylinder" size="{target_radius:.4f} 0.001"
        material="target_mat" contype="0" conaffinity="0"/>
    </body>{spawned}
    <camera name="front" pos="{_r(reach, 1.2):.3f} 0 {_r(reach, 0.63):.3f}"
      xyaxes="0 1 0 -0.6 0 0.8" fovy="58"/>
    <camera name="top" pos="{_r(reach, 0.44):.3f} 0 {_r(reach, 1.08):.3f}"
      xyaxes="1 0 0 0 1 0" fovy="55"/>
    <camera name="side" pos="{_r(reach, 0.44):.3f} {_r(reach, 1.2):.3f} {_r(reach, 0.57):.3f}"
      xyaxes="-1 0 0 0 -0.5 0.85" fovy="58"/>
    <site name="mount" pos="0 0 0" size="0.001" group="4"/>
  </worldbody>
</mujoco>"""


def build(emb: E.Embodiment, task="pick_cube", menagerie: str = None,
          extra_objects=(), reach=None, add_wrist_cam=True):
    """Compile a full task scene for ``emb``.

    Returns ``(model, info)`` where ``info`` carries the measured quantities the
    env needs: reach, cube size, target radius, the gripper's calibrated ctrl
    endpoints, and the downward grasp orientation.
    """
    arm = E.build_spec(emb, menagerie, add_wrist_cam)
    arm_model = arm.compile()

    if reach is None:
        reach = E.measure_reach(emb, model=arm_model)
    closed, opened, gap_c, gap_o = E.calibrate_gripper(emb, model=arm_model)
    cube_half = cube_half_size(gap_o)
    target_radius = _r(reach, 0.076)

    world = mujoco.MjSpec.from_string(
        world_xml(reach, cube_half, target_radius, extra_objects, task))
    _adopt_physics(world, arm_model, emb)
    world.attach(arm, prefix="", site="mount")
    model = world.compile()

    info = {
        "reach": float(reach),
        "cube_half": float(cube_half),
        "target_radius": float(target_radius),
        "gripper_ctrl": (float(closed), float(opened)),
        "gripper_gap": (float(gap_c), float(gap_o)),
        "workspace": E.workspace_box(reach),
        "down_quat": E.derive_down_quat(emb, model=arm_model),
        "fingertip_offset": E.fingertip_offset(emb, model=arm_model),
        "home_arm_qpos": E.home_arm_qpos(emb, arm_model),
    }

    # Opt-in only. Applying this wherever the vendor home sits outside the
    # workspace regresses arms that work fine from it (UR5e drops from 100% to
    # 0%): the searched pose has an arbitrary wrist orientation, while
    # ``down_quat`` is still measured at the vendor home, so the expert servos
    # toward an orientation unrelated to where the arm actually starts.
    if emb.search_ready_pose:
        ready = E.find_ready_pose(emb, arm_model, info["workspace"])
        if ready is not None:
            info["home_arm_qpos"] = ready
            info["ready_pose_searched"] = True
    return model, info
