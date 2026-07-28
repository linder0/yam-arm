// MediaPipe hand-tracking teleop.
//
// Runs Google's HandLandmarker (tasks-vision, loaded from CDN like three.js)
// on the user's webcam and turns tracked hands into the same WebSocket
// commands the keyboard teleop uses:
//
//   * hand left/right on screen  -> EE ±Y   (absolute `teleop_target`)
//   * hand up/down on screen     -> EE ±Z
//   * hand toward/away from cam  -> EE ±X   (apparent hand size as depth proxy)
//   * thumb-index pinch          -> gripper close/open (continuous `gripper`)
//
// Only the dashed inner box drawn on the preview maps onto the workspace
// (positions clamp at its edges): the detector needs the whole hand in frame,
// so the reliable control surface is the frame's centre, not its edges.
//
// Single-arm tasks use one hand. In two-arm tasks both hands are tracked and
// each drives its own arm (commands carry an `arm` field): in the raw camera
// frame the right half of the image is the user's LEFT hand, which is also the
// side of the left arm (world +Y) under the position mapping — so hands are
// assigned to arms by horizontal position and the mapping stays continuous.
//
// Targets are absolute and clamped/slew-limited server-side by EEController,
// so a jittery detection can never whip the arm. Everything is lazy: the
// MediaPipe wasm + model (~10 MB) only load the first time tracking is enabled.

const TASKS_VISION =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// Mirror of EEController's workspace box (metres, MuJoCo world frame).
const WS_X = [0.20, 0.60];
const WS_Y = [-0.35, 0.35];
const WS_Z = [0.02, 0.55];

// Only the central region of the frame is the control surface: MediaPipe
// needs the whole hand in view, so mapping the FULL frame onto the workspace
// meant reaching a workspace edge (e.g. table height, where every grasp
// happens) required pushing the hand to the frame edge -- exactly where
// detection cuts out. The inner box below spans the full workspace instead,
// and it is drawn on the preview so you can see the control region.
const ACTIVE_X = [0.18, 0.82];  // raw-image x -> full ±Y workspace
const ACTIVE_Y = [0.16, 0.80];  // raw-image y (top..bottom) -> full Z range

// Apparent hand size (wrist->middle-MCP, normalized image units) at the far
// and near ends of a comfortable seated range; linearly mapped onto WS_X.
const SIZE_FAR = 0.10;
const SIZE_NEAR = 0.28;

// Pinch ratio (thumb-tip<->index-tip distance / hand size): fully closed
// below GRIP_CLOSED, fully open above GRIP_OPEN, linear in between.
const GRIP_CLOSED = 0.45;
const GRIP_OPEN = 0.90;

const SMOOTH = 0.35;        // EMA weight for new samples (position + grip)
const SEND_INTERVAL = 33;   // ms between WS sends per arm (server steps at 20 Hz)
const MIN_MOVE = 0.002;     // m: don't resend an unchanged target
const MIN_GRIP = 0.03;      // don't resend an unchanged gripper
const LOST_AFTER = 500;     // ms without a hand before we report "no hand"

// Landmark indices + skeleton (standard 21-point MediaPipe hand).
const WRIST = 0, THUMB_TIP = 4, INDEX_MCP = 5, INDEX_TIP = 8, MIDDLE_MCP = 9,
      PINKY_MCP = 17;
const BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4],          // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],          // index
  [5, 9], [9, 10], [10, 11], [11, 12],     // middle
  [9, 13], [13, 14], [14, 15], [15, 16],   // ring
  [13, 17], [17, 18], [18, 19], [19, 20],  // pinky
  [0, 17],
];

const lerp = (lo, hi, t) => lo + (hi - lo) * t;
const clamp01 = (t) => Math.min(1, Math.max(0, t));

export function createHandTeleop({ send, canvas, onStatus, isBimanual }) {
  const ctx = canvas.getContext("2d");
  let landmarker = null;      // created once, reused across start/stop
  let video = null;
  let stream = null;
  let running = false;
  let rafId = 0;
  let lastSeen = 0;
  let singleArm = null;       // sticky arm choice when only one hand is up

  // Per-arm filter/send state, keyed "left" / "right" / "single".
  const armState = {};
  function stateFor(key) {
    if (!armState[key]) {
      armState[key] = { smoothed: null, grip: 1.0,
                        lastSent: null, lastSentGrip: null, lastSendTime: 0 };
    }
    return armState[key];
  }

  async function loadLandmarker() {
    onStatus("loading hand model…");
    const { FilesetResolver, HandLandmarker } =
      await import(`${TASKS_VISION}/vision_bundle.mjs`);
    const fileset = await FilesetResolver.forVisionTasks(`${TASKS_VISION}/wasm`);
    landmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: HAND_MODEL, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 2,
    });
  }

  // Hand size in normalized image units. The max of palm length and knuckle
  // span (scaled to comparable units): palm length alone foreshortens when
  // the palm pitches toward/away from the camera, which read as depth motion
  // and lurched the arm in X. Knuckle width is invariant to pitch, so the max
  // of the two stays stable under natural hand tilt.
  function handSize(lm, dist) {
    return Math.max(dist(lm[WRIST], lm[MIDDLE_MCP]),
                    1.5 * dist(lm[INDEX_MCP], lm[PINKY_MCP]));
  }

  // Raw landmarks -> workspace target + gripper opening.
  function mapHand(lm) {
    const aspect = video.videoWidth / video.videoHeight;
    const dist = (a, b) => Math.hypot((a.x - b.x) * aspect, a.y - b.y);

    const size = handSize(lm, dist);
    const depth = clamp01((size - SIZE_FAR) / (SIZE_NEAR - SIZE_FAR));
    const p = lm[MIDDLE_MCP];  // palm centre: steadier than fingertips

    // Raw image x runs left->right from the camera's point of view, which is
    // mirrored for the user — and that mirror is exactly what maps hand-right
    // to EE -Y (matching the D key). Vertical is inverted (image y grows down).
    // Positions are normalized within the ACTIVE box (clamped outside it) so
    // the full workspace is reachable without leaving the reliable region.
    const nx = clamp01((p.x - ACTIVE_X[0]) / (ACTIVE_X[1] - ACTIVE_X[0]));
    const ny = clamp01((p.y - ACTIVE_Y[0]) / (ACTIVE_Y[1] - ACTIVE_Y[0]));
    const target = [
      lerp(WS_X[0], WS_X[1], depth),
      lerp(WS_Y[0], WS_Y[1], nx),
      lerp(WS_Z[1], WS_Z[0], ny),
    ];

    const pinch = dist(lm[THUMB_TIP], lm[INDEX_TIP]) /
      Math.max(dist(lm[WRIST], lm[MIDDLE_MCP]), 1e-6);
    const grip = clamp01((pinch - GRIP_CLOSED) / (GRIP_OPEN - GRIP_CLOSED));
    return { target, grip };
  }

  // Decide which arm each detected hand drives: [armKey|null, landmarks][].
  function assignHands(hands) {
    if (hands.length === 0) return [];
    if (!isBimanual()) {
      // MediaPipe's detection order is arbitrary frame-to-frame, so taking
      // hands[0] let a stray second hand (or a false detection) yank the
      // target around. Drive from the largest hand -- the operator's, which
      // is closest to the camera.
      const flat = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
      const best = hands.reduce((a, b) =>
        handSize(a, flat) >= handSize(b, flat) ? a : b);
      return [[null, best]];
    }

    if (hands.length >= 2) {
      const [a, b] = hands;
      const swap = a[MIDDLE_MCP].x < b[MIDDLE_MCP].x;
      // larger raw x = user's left side = left arm (world +Y)
      return swap ? [["left", b], ["right", a]] : [["left", a], ["right", b]];
    }
    // One hand in a two-arm scene: pick by side, with hysteresis around the
    // midline so the assignment doesn't flicker.
    const x = hands[0][MIDDLE_MCP].x;
    if (singleArm === null) singleArm = x > 0.5 ? "left" : "right";
    else if (x > 0.55) singleArm = "left";
    else if (x < 0.45) singleArm = "right";
    return [[singleArm, hands[0]]];
  }

  function track(key, lm, now) {
    const st = stateFor(key ?? "single");
    const { target, grip } = mapHand(lm);
    if (st.smoothed === null) st.smoothed = target.slice();
    for (let i = 0; i < 3; i++)
      st.smoothed[i] += SMOOTH * (target[i] - st.smoothed[i]);
    st.grip += SMOOTH * (grip - st.grip);

    if (now - st.lastSendTime < SEND_INTERVAL) return st;
    const [x, y, z] = st.smoothed;
    const arm = key ? { arm: key } : {};
    if (st.lastSent === null ||
        Math.hypot(x - st.lastSent[0], y - st.lastSent[1], z - st.lastSent[2]) > MIN_MOVE) {
      send({ cmd: "teleop_target", x, y, z, ...arm });
      st.lastSent = [x, y, z];
      st.lastSendTime = now;
    }
    if (st.lastSentGrip === null || Math.abs(st.grip - st.lastSentGrip) > MIN_GRIP) {
      send({ cmd: "gripper", value: st.grip, ...arm });
      st.lastSentGrip = st.grip;
      st.lastSendTime = now;
    }
    return st;
  }

  function draw(tracked) {
    const w = canvas.width, h = canvas.height;
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);  // selfie view
    ctx.drawImage(video, 0, 0, w, h);
    // Control region: inside it the hand position maps onto the workspace;
    // outside it the target clamps to the nearest edge.
    ctx.strokeStyle = "rgba(255,255,255,0.45)";
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 4]);
    ctx.strokeRect(ACTIVE_X[0] * w, ACTIVE_Y[0] * h,
                   (ACTIVE_X[1] - ACTIVE_X[0]) * w,
                   (ACTIVE_Y[1] - ACTIVE_Y[0]) * h);
    ctx.setLineDash([]);
    for (const { lm, grip } of tracked) {
      ctx.strokeStyle = grip < 0.5 ? "#f85149" : "#3fb950";
      ctx.fillStyle = ctx.strokeStyle;
      ctx.lineWidth = 2;
      for (const [a, b] of BONES) {
        ctx.beginPath();
        ctx.moveTo(lm[a].x * w, lm[a].y * h);
        ctx.lineTo(lm[b].x * w, lm[b].y * h);
        ctx.stroke();
      }
      for (const pt of lm) {
        ctx.beginPath();
        ctx.arc(pt.x * w, pt.y * h, 3, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
    ctx.restore();
  }

  function loop() {
    if (!running) return;
    rafId = requestAnimationFrame(loop);
    if (video.readyState < 2) return;

    const now = performance.now();
    const result = landmarker.detectForVideo(video, now);
    const hands = result.landmarks || [];
    const assigned = assignHands(hands);

    const tracked = [];
    if (assigned.length) {
      lastSeen = now;
      const parts = [];
      for (const [key, lm] of assigned) {
        const st = track(key, lm, now);
        tracked.push({ lm, grip: st.grip });
        const label = key ? `${key[0].toUpperCase()} ` : "";
        parts.push(`${label}grip ${Math.round(st.grip * 100)}%`);
      }
      onStatus(`tracking ${assigned.length} hand${assigned.length > 1 ? "s" : ""}`
               + ` — ${parts.join(" · ")} open`);
    } else if (now - lastSeen > LOST_AFTER) {
      onStatus("no hand in view (arm holds position)");
    }
    draw(tracked);
  }

  async function start() {
    if (running) return;
    if (!landmarker) await loadLandmarker();
    onStatus("requesting camera…");
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
    });
    video = document.createElement("video");
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    await video.play();

    for (const key of Object.keys(armState)) delete armState[key];
    singleArm = null;
    lastSeen = performance.now();
    running = true;
    canvas.hidden = false;
    loop();
  }

  function stop() {
    running = false;
    cancelAnimationFrame(rafId);
    if (stream) {
      for (const track of stream.getTracks()) track.stop();
      stream = null;
    }
    video = null;
    canvas.hidden = true;
    onStatus("");
  }

  return { start, stop, get running() { return running; } };
}
