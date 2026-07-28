"""Per-robot descriptions, so the stack above ``robot.py`` stops being YAM-only.

``yam/model.py`` hardcodes one embodiment as module constants (``ARM_JOINTS``,
``GRIPPER_OPEN_CTRL``, ``GRASP_SITE``...). Surveying MuJoCo Menagerie shows why
that cannot be auto-detected: arm joints are ``joint1..6`` on YAM, ``waist/
shoulder/elbow/...`` on a Trossen, ``Rotation/Pitch/Elbow/...`` on an SO-ARM100;
Panda's actuators are ``actuator1..8`` while a Lite6's are unnamed. So the
naming is declared here, once per robot.

What is *measured* rather than declared:

  * **The grasp point.** Only YAM, Trossen (``pinch``) and Kinova (``pinch_site``)
    ship a usable site; Panda, SO-ARM100, ARX and Piper define none, and the
    industrial arms expose only a flange ``attachment_site``. So the pinch point
    is taken from the finger pads and injected as a site for IK to servo.
  * **Gripper polarity and travel.** Units are inconsistent to the point of
    being adversarial: metres on YAM ``[0, 0.041]``, metres with a nonzero floor
    on a Trossen ``[0.021, 0.057]``, radians on an SO-ARM100 jaw, and
    ``[0, 255]`` on Panda and Robotiq. Rather than trust a table, drive the
    actuator to each end of its ctrlrange and measure which way the jaws open.
  * **Reach**, which scales the workspace box and the whole task scene.
  * **The downward grasp orientation**, which ``teleop.DOWN_QUAT`` hardcodes
    from YAM's home pose and which differs on every wrist.

Two defects in the vendor models are repaired on load: arms that ship no gripper
get a Robotiq 2F-85 attached at their flange, and actuators with a degenerate
``[0, 0]`` ctrlrange (Kinova does this on its four continuous joints) are
widened, since MuJoCo would otherwise silently clamp those joints to zero.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import mujoco

MENAGERIE = os.environ.get("MENAGERIE_DIR",
                           os.path.expanduser("~/mujoco_menagerie"))

# The default arm stays vendored so the repo runs without a Menagerie clone.
REPO_ASSETS = os.path.join(os.path.dirname(__file__), "assets")

# Names for the handles we inject, so downstream code has one constant.
GRASP_SITE = "yam_grasp_site"
WRIST_CAM = "wrist"

GRIPPERS = {"robotiq_2f85": "robotiq_2f85/2f85.xml"}

_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class Embodiment:
    """What the stack must be told about a robot; everything else is measured.

    When ``attach_gripper`` is set, the gripper-side names below refer to the
    *prefixed* names produced by the attach (``rq_left_pad``, not ``left_pad``).
    """

    key: str
    xml: str                                  # relative to the menagerie root
    arm_joints: Sequence[str]
    arm_actuators: Sequence[str]
    gripper_actuator: Optional[str] = None
    finger_joint: Optional[str] = None        # normalized for gripper feedback
    finger_bodies: Optional[Tuple[str, str]] = None
    palm_body: Optional[str] = None           # parent for the injected site/camera
    grasp_site: Optional[str] = None          # existing site; else injected
    wrist_cam_parent: Optional[str] = None
    wrist_cam: Optional[str] = None           # existing camera, if any
    attach_gripper: Optional[str] = None      # key into GRIPPERS
    attach_site: Optional[str] = None         # site on the arm to bolt it to
    attach_prefix: str = "rq_"
    home_key: str = "home"
    home_qpos: Optional[Sequence[float]] = None   # if the MJCF ships no keyframe
    down_quat: Optional[Sequence[float]] = None   # grasp orientation, if the
                                                  # home pose is not grasp-ready
    vendored: bool = False                        # resolve under yam/assets
    search_ready_pose: bool = False               # vendor home is unusable
    notes: str = ""

    @property
    def dof(self) -> int:
        return len(self.arm_joints)

    def path(self, menagerie: str = None) -> str:
        if self.vendored:
            return os.path.join(REPO_ASSETS, self.xml)
        return os.path.join(menagerie or MENAGERIE, self.xml)


def gripper_path(name: str, menagerie: str = None) -> str:
    return os.path.join(menagerie or MENAGERIE, GRIPPERS[name])


def _robotiq(**kw):
    """Registry entry for an arm that ships without a gripper."""
    kw.setdefault("attach_gripper", "robotiq_2f85")
    kw.setdefault("attach_site", "attachment_site")
    p = kw.get("attach_prefix", "rq_")
    kw.setdefault("gripper_actuator", f"{p}fingers_actuator")
    kw.setdefault("finger_joint", f"{p}left_driver_joint")
    kw.setdefault("finger_bodies", (f"{p}left_pad", f"{p}right_pad"))
    kw.setdefault("palm_body", f"{p}base")
    return Embodiment(**kw)


REGISTRY = {e.key: e for e in [
    # -- arms with an integrated gripper -------------------------------------
    Embodiment(
        key="yam",
        xml="yam/yam.xml",
        vendored=True,
        arm_joints=[f"joint{i}" for i in range(1, 7)],
        arm_actuators=[f"joint{i}" for i in range(1, 7)],
        gripper_actuator="gripper",
        finger_joint="left_finger",
        finger_bodies=("link_left_finger", "link_right_finger"),
        palm_body="link_6",
        grasp_site="grasp_site",
        wrist_cam="wrist",
        # The vendored copy has its keyframe stripped (task scenes change nq),
        # so the home pose lives here, matching model.HOME_QPOS_ARM.
        home_qpos=[0.0, 1.047, 1.047, 0.0, 0.0, 0.0],
        notes="Baseline: the arm the stack was written against.",
    ),
    Embodiment(
        key="so_arm100",
        xml="trs_so_arm100/so_arm100.xml",
        arm_joints=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
        arm_actuators=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
        gripper_actuator="Jaw",
        finger_joint="Jaw",
        finger_bodies=("Fixed_Jaw", "Moving_Jaw"),
        palm_body="Wrist_Pitch_Roll",
        wrist_cam_parent="Wrist_Pitch_Roll",
        notes="5-DOF and 0.47 m reach: cannot hit arbitrary 6-DOF poses, so IK "
              "must run position-only. The cheap arm people actually own.",
    ),
    Embodiment(
        key="vx300s",
        xml="trossen_vx300s/vx300s.xml",
        arm_joints=["waist", "shoulder", "elbow", "forearm_roll",
                    "wrist_angle", "wrist_rotate"],
        arm_actuators=["waist", "shoulder", "elbow", "forearm_roll",
                       "wrist_angle", "wrist_rotate"],
        gripper_actuator="gripper",
        finger_joint="left_finger",
        finger_bodies=("left_finger_link", "right_finger_link"),
        palm_body="gripper_link",
        grasp_site="pinch",
        wrist_cam_parent="gripper_link",
        notes="Same DOF as YAM, entirely different names. The ALOHA arm.",
    ),
    Embodiment(
        key="panda",
        xml="franka_emika_panda/panda.xml",
        arm_joints=[f"joint{i}" for i in range(1, 8)],
        arm_actuators=[f"actuator{i}" for i in range(1, 8)],
        gripper_actuator="actuator8",
        finger_joint="finger_joint1",
        finger_bodies=("left_finger", "right_finger"),
        palm_body="hand",
        wrist_cam_parent="hand",
        notes="7-DOF and redundant. Gripper ctrl is [0, 255], not metres.",
    ),
    Embodiment(
        key="arx_l5",
        xml="arx_l5/arx_l5.xml",
        arm_joints=[f"joint{i}" for i in range(1, 7)],
        arm_actuators=[f"joint{i}" for i in range(1, 7)],
        gripper_actuator="gripper",
        finger_joint="joint7",
        finger_bodies=("link7", "link8"),
        palm_body="link6",
        wrist_cam="wrist_cam",
        notes="Fingers are joint7/joint8 and link7/link8 -- no keyword "
              "heuristic finds them.",
    ),
    Embodiment(
        key="piper",
        xml="agilex_piper/piper.xml",
        arm_joints=[f"joint{i}" for i in range(1, 7)],
        arm_actuators=[f"joint{i}" for i in range(1, 7)],
        gripper_actuator="gripper",
        finger_joint="joint7",
        finger_bodies=("link7", "link8"),
        palm_body="link6",
        wrist_cam_parent="link6",
        notes="Same shape as ARX L5 but opposite prismatic axes, so its gripper "
              "polarity has to be measured, not copied.",
    ),

    # -- arms with no gripper: a Robotiq 2F-85 is bolted on ------------------
    _robotiq(
        key="ur5e",
        xml="universal_robots_ur5e/ur5e.xml",
        arm_joints=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        arm_actuators=["shoulder_pan", "shoulder_lift", "elbow",
                       "wrist_1", "wrist_2", "wrist_3"],
        wrist_cam_parent="wrist_3_link",
        notes="Ships no gripper at all, only a flange site.",
    ),
    _robotiq(
        key="ur10e",
        xml="universal_robots_ur10e/ur10e.xml",
        arm_joints=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        arm_actuators=["shoulder_pan", "shoulder_lift", "elbow",
                       "wrist_1", "wrist_2", "wrist_3"],
        wrist_cam_parent="wrist_3_link",
        notes="1.49 m reach: the far end of the scale range.",
    ),
    _robotiq(
        key="iiwa14",
        xml="kuka_iiwa_14/iiwa14.xml",
        arm_joints=[f"joint{i}" for i in range(1, 8)],
        arm_actuators=[f"actuator{i}" for i in range(1, 8)],
        wrist_cam_parent="link7",
        # Its home pose holds the wrist frame such that the fingers never
        # straddle the cube; measured by scripts/calibrate_grasp.py.
        down_quat=[0.0, -0.707107, 0.707107, 0.0],
        notes="7-DOF, no gripper, and the one arm whose home orientation is "
              "not grasp-ready.",
    ),
    _robotiq(
        key="fr3",
        xml="franka_fr3/fr3.xml",
        arm_joints=[f"fr3_joint{i}" for i in range(1, 8)],
        arm_actuators=[f"fr3_joint{i}" for i in range(1, 8)],
        wrist_cam_parent="fr3_link7",
        notes="Every name carries an fr3_ prefix; the Panda entry does not.",
    ),
    _robotiq(
        key="gen3",
        xml="kinova_gen3/gen3.xml",
        arm_joints=[f"joint_{i}" for i in range(1, 8)],
        arm_actuators=[f"joint_{i}" for i in range(1, 8)],
        attach_site="pinch_site",
        wrist_cam="wrist",
        notes="Ships ctrlrange [0, 0] on its four continuous joints, which "
              "would clamp them to zero; repaired on load.",
    ),
]}


# -- model construction ------------------------------------------------------

def robot_spec(emb: Embodiment, menagerie: str = None):
    """The arm as an editable ``MjSpec``, with vendor defects repaired."""
    spec = mujoco.MjSpec.from_file(emb.path(menagerie))

    if emb.attach_gripper:
        gripper = mujoco.MjSpec.from_file(
            gripper_path(emb.attach_gripper, menagerie))
        spec.attach(gripper, prefix=emb.attach_prefix, site=emb.attach_site)

    # A degenerate ctrlrange is not "no limit" to MuJoCo -- it pins the actuator
    # at zero. Continuous joints get the full turn instead; IK still clamps to
    # the joint's own range.
    wanted = set(emb.arm_actuators)
    for act in spec.actuators:
        if act.name in wanted and float(act.ctrlrange[0]) == float(act.ctrlrange[1]):
            act.ctrlrange = [-_TWO_PI, _TWO_PI]

    return spec


def build_spec(emb: Embodiment, menagerie: str = None, add_wrist_cam=True):
    """Robot spec with a ``GRASP_SITE`` and wrist camera guaranteed to exist."""
    spec = robot_spec(emb, menagerie)
    if emb.grasp_site is None:
        local, _, _ = derive_grasp_point(emb, model=spec.compile())
        spec.body(emb.palm_body).add_site(
            name=GRASP_SITE, pos=local, size=[0.005, 0.005, 0.005],
            rgba=[1.0, 0.2, 0.2, 0.6])
    existing = {c.name for c in spec.cameras}
    if (add_wrist_cam and emb.wrist_cam is None and emb.wrist_cam_parent
            and WRIST_CAM not in existing):
        spec.body(emb.wrist_cam_parent).add_camera(
            name=WRIST_CAM, pos=[0.0, 0.0, 0.05], quat=[0.0, 0.0, 0.0, 1.0],
            fovy=58.0)
    return spec


def build_model(emb: Embodiment, menagerie: str = None, add_wrist_cam=True):
    return build_spec(emb, menagerie, add_wrist_cam).compile()


def grasp_site_name(emb: Embodiment) -> str:
    return emb.grasp_site or GRASP_SITE


def wrist_cam_name(emb: Embodiment) -> Optional[str]:
    return emb.wrist_cam or (WRIST_CAM if emb.wrist_cam_parent else None)


# -- measurement -------------------------------------------------------------

def _home_data(model, emb):
    """MjData posed at the arm's home configuration. Note that a model whose
    keyframe was stripped (the vendored YAM) would otherwise be measured at
    ``qpos0``, which is not a grasp-ready pose."""
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, emb.home_key)
    if emb.home_qpos is not None:
        for j, v in zip(emb.arm_joints, emb.home_qpos):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jid >= 0:
                data.qpos[model.jnt_qposadr[jid]] = v
    elif key >= 0:
        data.qpos[:] = model.key_qpos[key]
    mujoco.mj_forward(model, data)
    return data


def _bid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _pad_geom(model, data, body_id, palm_pos):
    """The geom of a finger farthest from the palm: its gripping pad.

    Restricting measurements to the pads matters for revolute jaws, where the
    finger bodies overlap near the hinge and a naive minimum distance reports
    that overlap rather than the opening between the jaws.
    """
    best, best_d = -1, -1.0
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body_id:
            continue
        d = float(np.linalg.norm(data.geom_xpos[g] - palm_pos))
        if d > best_d:
            best, best_d = g, d
    return best


def derive_grasp_point(emb: Embodiment, model=None, menagerie: str = None):
    """Pinch point in ``palm_body`` local coordinates, measured at the home pose
    as the midpoint of the two finger pads.

    Returns ``(local_pos, world_pos, pad_gap)``.
    """
    if model is None:
        model = robot_spec(emb, menagerie).compile()
    data = _home_data(model, emb)

    lid, rid = (_bid(model, b) for b in emb.finger_bodies)
    pid = _bid(model, emb.palm_body)
    if min(lid, rid, pid) < 0:
        raise ValueError(f"{emb.key}: finger/palm bodies not found "
                         f"({emb.finger_bodies}, {emb.palm_body})")

    palm = data.xpos[pid]
    ga, gb = _pad_geom(model, data, lid, palm), _pad_geom(model, data, rid, palm)
    world = 0.5 * (data.geom_xpos[ga] + data.geom_xpos[gb])
    gap = float(np.linalg.norm(data.geom_xpos[ga] - data.geom_xpos[gb]))

    local = data.xmat[pid].reshape(3, 3).T @ (world - palm)
    return local, world, gap


def calibrate_gripper(emb: Embodiment, model=None, menagerie: str = None,
                      settle=400):
    """Find which end of the gripper's ctrlrange opens it, by driving to each
    extreme and measuring the gap between the finger pads.

    Returns ``(closed_ctrl, open_ctrl, closed_gap, open_gap)``. This is what
    lets the stack keep a normalized ``gripper in [0, 1]`` interface across
    robots whose control units are metres, radians, or a 0-255 integer.
    """
    if model is None:
        model = build_model(emb, menagerie)
    act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                            emb.gripper_actuator)
    if act < 0:
        raise ValueError(f"{emb.key}: no gripper actuator {emb.gripper_actuator!r}")

    lid, rid = (_bid(model, b) for b in emb.finger_bodies)
    pid = _bid(model, emb.palm_body)
    lo, hi = (float(x) for x in model.actuator_ctrlrange[act])
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, emb.home_key)

    def gap_at(ctrl):
        data = mujoco.MjData(model)
        if key >= 0:
            data.qpos[:] = model.key_qpos[key]
        mujoco.mj_forward(model, data)
        data.ctrl[act] = ctrl
        for _ in range(settle):
            mujoco.mj_step(model, data)
        palm = data.xpos[pid]
        ga = _pad_geom(model, data, lid, palm)
        gb = _pad_geom(model, data, rid, palm)
        return float(mujoco.mj_geomDistance(model, data, ga, gb, 1.0, None))

    gap_lo, gap_hi = gap_at(lo), gap_at(hi)
    if gap_hi >= gap_lo:
        return lo, hi, gap_lo, gap_hi
    return hi, lo, gap_hi, gap_lo


def measure_reach(emb: Embodiment, model=None, menagerie: str = None,
                  samples=400, seed=0):
    """Max distance from the base to any body over random in-limit poses. Used
    to scale the workspace box and the task scene."""
    if model is None:
        model = build_model(emb, menagerie)
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in emb.arm_joints]
    best = 0.0
    for _ in range(samples):
        data.qpos[:] = model.qpos0
        for j in jids:
            lo, hi = model.jnt_range[j]
            if not model.jnt_limited[j]:
                lo, hi = -np.pi, np.pi
            data.qpos[model.jnt_qposadr[j]] = rng.uniform(lo, hi)
        mujoco.mj_kinematics(model, data)
        base = data.xpos[1]
        best = max(best, float(np.linalg.norm(data.xpos[1:] - base, axis=1).max()))
    return best


def derive_down_quat(emb: Embodiment, model=None, menagerie: str = None):
    """The grasp orientation IK holds during a pick: the grasp-site orientation
    at the arm's home pose.

    This is the generalization of ``teleop.DOWN_QUAT``, whose hardcoded value is
    exactly YAM's home-pose site orientation. Deriving it geometrically instead
    (rotate until palm->pinch points at -z) is tempting and wrong: on YAM that
    axis is horizontal at home while the fingers already hang correctly, so the
    "correction" tips the wrist over. Vendor home keyframes are ready poses, so
    reading the orientation off one is both simpler and more faithful.
    """
    if emb.down_quat is not None:
        q = np.asarray(emb.down_quat, dtype=np.float64)
        return q / np.linalg.norm(q)
    if model is None:
        model = build_model(emb, menagerie)
    data = _home_data(model, emb)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, grasp_site_name(emb))
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[sid].reshape(-1))
    return quat / np.linalg.norm(quat)


def grasp_quat_candidates(emb: Embodiment, model=None, menagerie: str = None):
    """Plausible grasp orientations to try when the home pose does not work.

    The home orientation, plus the minimal rotation of it that aims each of the
    grasp site's local axes at the table. One of these is almost always the
    pose that lets the fingers straddle an object; which one depends on how the
    vendor happened to orient the wrist frame, so it is cheaper to test them
    than to reason about it.
    """
    if model is None:
        model = build_model(emb, menagerie)
    data = _home_data(model, emb)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, grasp_site_name(emb))
    R = data.site_xmat[sid].reshape(3, 3)
    q_home = np.zeros(4)
    mujoco.mju_mat2Quat(q_home, R.reshape(-1))

    down = np.array([0.0, 0.0, -1.0])
    out = [("home", q_home / np.linalg.norm(q_home))]
    for name, axis in (("+x", R[:, 0]), ("-x", -R[:, 0]), ("+y", R[:, 1]),
                       ("-y", -R[:, 1]), ("+z", R[:, 2]), ("-z", -R[:, 2])):
        cross = np.cross(axis, down)
        s, c = np.linalg.norm(cross), float(np.dot(axis, down))
        if s < 1e-9:
            fix = np.array([1.0, 0.0, 0.0, 0.0]) if c > 0 else \
                np.array([0.0, 1.0, 0.0, 0.0])
        else:
            fix = np.zeros(4)
            mujoco.mju_axisAngle2Quat(fix, cross / s, float(np.arctan2(s, c)))
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, fix, q_home)
        out.append((name, q / np.linalg.norm(q)))
    return out


def fingertip_offset(emb: Embodiment, model=None, menagerie: str = None):
    """How far the pinch point sits below the grasp site along the approach
    axis. Zero for injected sites (they are placed at the pinch point); nonzero
    for vendor sites such as YAM's, whose ``FINGERTIP_OFFSET`` is 0.044 m.
    """
    if model is None:
        model = build_model(emb, menagerie)
    if emb.grasp_site is None:
        return 0.0
    data = _home_data(model, emb)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, emb.grasp_site)
    _, pinch_world, _ = derive_grasp_point(emb, model=model)
    return float(np.linalg.norm(pinch_world - data.site_xpos[sid]))


def home_arm_qpos(emb: Embodiment, model):
    """The arm's home joint values: an explicit override, else the ``home``
    keyframe, else ``qpos0`` for models that ship neither."""
    if emb.home_qpos is not None:
        return np.array(emb.home_qpos, dtype=np.float64)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, emb.home_key)
    src = model.key_qpos[key] if key >= 0 else model.qpos0
    adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
           for j in emb.arm_joints]
    return np.array([src[a] for a in adr], dtype=np.float64)


def find_ready_pose(emb: Embodiment, model, workspace, samples=4000, seed=0):
    """Search for a start pose that puts the grasp point inside the workspace.

    Some vendor ``home`` keyframes are storage poses, not ready poses: Kinova's
    leaves the gripper at x = -0.45, behind its own base and outside anything
    the task can ask for. IK then has to escape that pose before it can track
    anything, and the pick fails for reasons that look like control bugs.

    Returns arm joint values whose grasp point is nearest the centre of the
    workspace box, or ``None`` if the vendor pose is already inside it.
    """
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, grasp_site_name(emb))
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in emb.arm_joints]
    adr = [model.jnt_qposadr[j] for j in jids]
    lo = np.array([model.jnt_range[j][0] if model.jnt_limited[j] else -np.pi
                   for j in jids])
    hi = np.array([model.jnt_range[j][1] if model.jnt_limited[j] else np.pi
                   for j in jids])

    box = np.asarray(workspace, dtype=np.float64)
    centre = box.mean(axis=1)
    data = _home_data(model, emb)
    if np.all(data.site_xpos[sid] >= box[:, 0]) and \
       np.all(data.site_xpos[sid] <= box[:, 1]):
        return None

    rng = np.random.default_rng(seed)
    best, best_d = None, np.inf
    for _ in range(samples):
        q = rng.uniform(lo, hi)
        data.qpos[adr] = q
        mujoco.mj_kinematics(model, data)
        p = data.site_xpos[sid]
        inside = np.all(p >= box[:, 0]) and np.all(p <= box[:, 1])
        d = float(np.linalg.norm(p - centre)) + (0.0 if inside else 10.0)
        if d < best_d:
            best, best_d = q.copy(), d
    return best


def workspace_box(reach, floor=0.02):
    """A reachable Cartesian box scaled to the arm, replacing the hardcoded YAM
    box in ``EEController`` (x[0.20,0.60], y[+-0.35], z[0.02,0.55])."""
    return np.array([[0.25 * reach, 0.78 * reach],
                     [-0.45 * reach, 0.45 * reach],
                     [floor, 0.70 * reach]])
