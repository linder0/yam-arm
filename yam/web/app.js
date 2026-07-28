// Live 3-D viewer for the YAM arm.
//
// Mirrors web-next's renderer approach: one three.js group per MuJoCo body, with
// child meshes per geom at the geom's local pose, all under a root group rotated
// -90 deg about X so MuJoCo's Z-up world maps to three.js Y-up. Each control
// step the server streams every body's world (xpos, xquat) and we move the body
// groups. STL meshes (the arm links) are loaded on demand.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { createHandTeleop } from "./hand.js";

const MODEL_BASE = "/public/model";

// ---- three.js scene --------------------------------------------------------
const container = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.shadowMap.enabled = true;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

const camera = new THREE.PerspectiveCamera(
  45, container.clientWidth / container.clientHeight, 0.01, 100);
camera.position.set(1.1, 0.9, 1.1);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0.35, 0.15, 0.0);
controls.update();

scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 1.1));
const dir = new THREE.DirectionalLight(0xffffff, 1.4);
dir.position.set(1.5, 3, 2); dir.castShadow = true; scene.add(dir);

// Root maps MuJoCo Z-up -> three.js Y-up (same trick as web-next).
const root = new THREE.Group();
root.rotation.x = -Math.PI / 2;
scene.add(root);

// Ground grid drawn in three.js space (already Y-up), so add outside root.
const grid = new THREE.GridHelper(3, 30, 0x30363d, 0x21262d);
grid.position.y = -0.001;
scene.add(grid);

const stlLoader = new STLLoader();
let bodyGroups = [];   // index -> THREE.Group (per MuJoCo body)

function wxyzToQuat(w, x, y, z) { return new THREE.Quaternion(x, y, z, w); }

function makeGeometry(g) {
  const s = g.size;
  switch (g.type) {
    case "box": return new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
    case "sphere": return new THREE.SphereGeometry(s[0], 24, 16);
    case "ellipsoid": {
      const geo = new THREE.SphereGeometry(1, 24, 16);
      geo.scale(s[0], s[1], s[2]); return geo;
    }
    case "capsule": {  // MuJoCo: [radius, half-length], axis local Z
      const geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 16);
      geo.rotateX(Math.PI / 2); return geo;  // three capsule is Y-aligned
    }
    case "cylinder": {  // MuJoCo: [radius, half-height], axis local Z
      const geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 24);
      geo.rotateX(Math.PI / 2); return geo;
    }
    default: return null;
  }
}

function addGeomMesh(group, g) {
  const color = new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]);
  const material = new THREE.MeshStandardMaterial({
    color, metalness: 0.15, roughness: 0.7,
    transparent: g.rgba[3] < 1.0, opacity: g.rgba[3] });

  const place = (mesh) => {
    mesh.position.set(g.pos[0], g.pos[1], g.pos[2]);
    mesh.quaternion.copy(wxyzToQuat(g.quat[0], g.quat[1], g.quat[2], g.quat[3]));
    mesh.castShadow = true; mesh.receiveShadow = true;
    group.add(mesh);
  };

  if (g.type === "mesh") {
    stlLoader.load(`${MODEL_BASE}/meshes/${g.mesh}`, (geo) => {
      if (g.mesh_scale) geo.scale(g.mesh_scale[0], g.mesh_scale[1], g.mesh_scale[2]);
      // Undo MuJoCo's compile-time mesh recentering: the raw STL vertices are in
      // the asset frame, but geom_pos/geom_quat reference the recentered mesh
      // frame (mesh_pos/mesh_quat). Bake the inverse into the geometry so the
      // links assemble correctly instead of scattering.
      if (g.mesh_pos && g.mesh_quat) {
        const meshFrame = new THREE.Matrix4().compose(
          new THREE.Vector3(g.mesh_pos[0], g.mesh_pos[1], g.mesh_pos[2]),
          wxyzToQuat(g.mesh_quat[0], g.mesh_quat[1], g.mesh_quat[2], g.mesh_quat[3]),
          new THREE.Vector3(1, 1, 1));
        geo.applyMatrix4(meshFrame.invert());
      }
      place(new THREE.Mesh(geo, material));
    });
  } else {
    const geo = makeGeometry(g);
    if (geo) place(new THREE.Mesh(geo, material));
  }
}

let currentTask = null;   // task whose scene is currently mounted
let pendingLoad = null;   // task currently being fetched/built
let sceneGen = 0;         // bumped per load so stale loads bail out
let sceneVersion = null;  // server bumps it when the scene is recompiled
let inReplayScene = false;  // replaying a session with its own manifest
// Live-scene manifest received over the socket (agent server). Authoritative
// over the manifest.json on disk, which any other session may have rewritten.
let liveManifest = null;
let liveTask = null;

function buildFromManifest(manifest, taskLabel, gen) {
  // Build into a local array, then swap it in atomically so bodyGroups is
  // never transiently empty.
  const groups = [];
  for (let i = 0; i < manifest.nbody; i++) groups.push(new THREE.Group());
  for (const g of manifest.geoms) {
    if (g.body >= 0 && g.body < groups.length) addGeomMesh(groups[g.body], g);
  }
  if (gen !== undefined && gen !== sceneGen) return;

  for (const grp of bodyGroups) root.remove(grp);
  bodyGroups = groups;
  for (const grp of bodyGroups) root.add(grp);
  document.getElementById("desc").textContent = manifest.description;
  currentTask = taskLabel;
}

async function loadScene(task) {
  // Guard against the state stream (which arrives ~20 Hz) kicking off a new
  // load every frame while one is already in flight — that used to clear the
  // scene repeatedly and made it flicker/blank.
  if (task === currentTask || task === pendingLoad) return;
  pendingLoad = task;
  const gen = ++sceneGen;
  try {
    // no-store: the server rewrites the manifest when objects are spawned
    const manifest = await fetch(`${MODEL_BASE}/${task}/manifest.json`,
      { cache: "no-store" }).then((r) => r.json());
    if (gen !== sceneGen) return;  // superseded by a newer load
    buildFromManifest(manifest, task, gen);
  } finally {
    if (pendingLoad === task) pendingLoad = null;
  }
}

function applyState(msg) {
  const { xpos, xquat } = msg;
  // Only apply when the stream matches the mounted scene (during a task switch
  // the counts differ for a few frames).
  if (xpos.length / 3 !== bodyGroups.length) return;
  for (let i = 0; i < bodyGroups.length; i++) {
    const grp = bodyGroups[i];
    grp.position.set(xpos[3 * i], xpos[3 * i + 1], xpos[3 * i + 2]);
    grp.quaternion.copy(wxyzToQuat(
      xquat[4 * i], xquat[4 * i + 1], xquat[4 * i + 2], xquat[4 * i + 3]));
  }
}

// ---- render loop -----------------------------------------------------------
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

window.addEventListener("resize", () => {
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
});

// ---- websocket -------------------------------------------------------------
let ws;
let gripOpen = true;

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => setStatus("connected");
  ws.onclose = () => { setStatus("disconnected — retrying…"); setTimeout(connect, 1500); };
  ws.onmessage = (ev) => onMessage(JSON.parse(ev.data));
}

// The server names scenes "<task>" (YAM) or "<task>__<arm>". Split that into
// two dropdowns: a task list that stays short however many arms exist, and an
// arm list beside it.
const ARM_SEP = "__";
let singleTasks = [];
let bimanualTasks = [];
let armList = [];

const baseOf = (id) => id.split(ARM_SEP)[0];
const armOf = (id) => (id.includes(ARM_SEP) ? id.split(ARM_SEP)[1] : "yam");
const taskId = (base, arm) => (arm === "yam" ? base : `${base}${ARM_SEP}${arm}`);

function fill(sel, list, select) {
  sel.innerHTML = "";
  for (const t of list) {
    const o = document.createElement("option");
    o.value = t; o.textContent = t; sel.appendChild(o);
  }
  if (select && list.includes(select)) sel.value = select;
  return sel.value;
}

function populateTasks(twoArms, select = null) {
  const list = twoArms ? bimanualTasks
                       : [...new Set(singleTasks.map(baseOf))];
  const value = fill(document.getElementById("task"), list,
                     twoArms ? select : select && baseOf(select));
  // Two-arm scenes are YAM-only, and a single-arm server has nothing to pick.
  const hide = twoArms || armList.length <= 1;
  document.getElementById("arm").hidden = hide;
  document.getElementById("armlabel").hidden = hide;
  return value;
}

function populateArms(select = null) {
  return fill(document.getElementById("arm"), armList, select);
}

/** The scene id the two dropdowns currently describe. */
function selectedTask() {
  if (document.getElementById("twoarms").checked) {
    return document.getElementById("task").value;
  }
  return taskId(document.getElementById("task").value,
                document.getElementById("arm").value);
}

async function onMessage(msg) {
  if (msg.type === "hello") {
    singleTasks = msg.tasks || [];
    bimanualTasks = msg.bimanual_tasks || [];
    armList = [...new Set(["yam", ...singleTasks.map(armOf)])];
    const twoArms = bimanualTasks.includes(msg.task);
    const twoArmsBox = document.getElementById("twoarms");
    twoArmsBox.checked = twoArms;
    // Servers without two-arm tasks ignore the toggle -- hide it rather than
    // let a click empty the task list and do nothing.
    twoArmsBox.closest("label").hidden = bimanualTasks.length === 0;
    populateArms(armOf(msg.task));
    populateTasks(twoArms, msg.task);
    if (msg.sessions !== undefined) {  // agent server: session save/replay UI
      document.getElementById("sessionsui").hidden = false;
      populateSessions(msg.sessions);
    }
    if (msg.chat) document.getElementById("chatform").hidden = false;
    if (msg.manifest) {  // agent server: geometry that matches its stream
      liveManifest = msg.manifest;
      liveTask = msg.task;
      sceneVersion = msg.scene_version ?? null;
      buildFromManifest(msg.manifest, msg.task);
    } else {
      await loadScene(msg.task);
    }
  } else if (msg.type === "scene") {  // live scene recompiled (e.g. spawn)
    liveManifest = msg.manifest;
    liveTask = msg.task;
    sceneVersion = msg.scene_version ?? sceneVersion;
    if (!inReplayScene) buildFromManifest(msg.manifest, msg.task);
  } else if (msg.type === "chat") {
    addChatMessages(msg.messages || []);
  } else if (msg.type === "sessions") {
    populateSessions(msg.sessions);
  } else if (msg.type === "replay_start") {
    replaying = true;
    if (msg.manifest) {  // sessions record their scene; mount it for playback
      inReplayScene = true;
      buildFromManifest(msg.manifest, currentTask);
    }
    clearConvo();
    addDivider(`replay: ${msg.name} (${msg.frames} frames)`);
  } else if (msg.type === "replay_end") {
    replaying = false;
    if (inReplayScene) {
      inReplayScene = false;
      if (liveManifest) buildFromManifest(liveManifest, liveTask);
      else currentTask = null;  // no socket manifest: reload from disk
    }
    renderLive();  // restore the live conversation
    addDivider("replay ended — back to live");
    setActivity(null);
  } else if (msg.type === "state") {
    if (msg.mode !== "replay") {
      if (msg.scene_version !== undefined && msg.scene_version !== sceneVersion) {
        if (sceneVersion !== null) currentTask = null;  // scene recompiled
        sceneVersion = msg.scene_version;
      }
      if (msg.task !== currentTask) {
        loadScene(msg.task);  // fire-and-forget; guarded against re-entry
      }
    }
    applyState(msg);
    // Keep the mode buttons honest: an agent tool call drops the server's
    // mode to idle (it takes the arm), and the buttons should show that.
    if (["idle", "scripted", "teleop"].includes(msg.mode)) {
      document.querySelectorAll("#modes button").forEach((x) =>
        x.classList.toggle("active", x.dataset.mode === msg.mode));
    }
    document.getElementById("s-mode").textContent =
      msg.active ? `${msg.mode} (arm: ${msg.active})` : msg.mode;
    document.getElementById("s-step").textContent = msg.step;
    const succ = document.getElementById("s-success");
    succ.textContent = msg.success ? "yes" : "no";
    succ.className = msg.success ? "ok" : "";
    document.getElementById("s-rec").textContent = msg.recording ? "REC" : "off";
    setActivity(msg.activity);
    setChatBusy(!!msg.chat_busy);
  } else if (msg.type === "agent_events") {
    addEvents(msg.events || [], !!msg.replay);
  }
}

function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }
function setStatus(s) {
  const el = document.getElementById("status");
  el.textContent = s;
  el.classList.toggle("ok", s === "connected");
}

// ---- agent conversation ------------------------------------------------
// One Cursor-style transcript: user messages as cards, the agent's chain of
// thought as dimmed text, tool calls as compact expandable rows, camera
// captures inline, and assistant replies as plain prose -- all interleaved
// in wall-clock order (chat and activity events arrive on separate streams).
const convo = document.getElementById("convo");
const chattyping = document.getElementById("chattyping");
const activityEl = document.getElementById("activity");
const READ_TOOLS = new Set(["get_state", "check_success", "list_sessions"]);
const MAX_ITEMS = 300;

let liveItems = [];      // {t, render} -- everything shown outside replay
let replaying = false;

function setActivity(activity) {
  activityEl.textContent = activity ? `running: ${activity}` : "idle";
  activityEl.classList.toggle("live", !!activity);
}

function fmtClock(t) {
  return new Date(t * 1000).toLocaleTimeString([], { hour12: false });
}

function makeEl(tag, cls, text) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text !== undefined) el.textContent = text;
  return el;
}

// Minimal markdown for assistant prose: `code`, **bold**, line breaks.
function renderProse(el, text) {
  const esc = text.replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  el.innerHTML = esc
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/\n/g, "<br>");
}

function renderChat(m) {
  if (m.role === "user") return makeEl("div", "msg-user", m.text);
  if (m.role === "system") return makeEl("div", "msg-system", m.text);
  const div = makeEl("div", "msg-assistant");
  renderProse(div, m.text);
  return div;
}

function renderEvent(e) {
  if (e.kind === "thought") return makeEl("div", "thought", e.text);

  if (e.kind === "image") {
    const div = makeEl("div", "capture");
    div.appendChild(makeEl("div", "cap",
      `${e.camera} camera · step ${e.sim_step} · ${fmtClock(e.wall_time)}`));
    const img = document.createElement("img");
    img.src = `data:image/png;base64,${e.png_b64}`;
    img.alt = `${e.camera} camera capture at sim step ${e.sim_step}`;
    div.appendChild(img);
    return div;
  }

  // tool call -> compact row, expandable for args/result
  const cls = e.error ? "tool err" : READ_TOOLS.has(e.tool) ? "tool read" : "tool";
  const det = makeEl("details", cls);
  const sum = makeEl("summary");
  sum.appendChild(makeEl("span", "glyph", "●"));
  sum.appendChild(makeEl("span", "tname", e.tool));
  const args = e.args && Object.keys(e.args).length
    ? JSON.stringify(e.args) : "";
  sum.appendChild(makeEl("span", "targs", args));
  const dur = e.duration_s !== undefined ? `${e.duration_s}s` : "";
  sum.appendChild(makeEl("span", "tmeta", dur));
  det.appendChild(sum);

  const span = e.sim_step_start !== undefined && e.sim_step_start !== e.sim_step
    ? `steps ${e.sim_step_start}→${e.sim_step}` : `step ${e.sim_step}`;
  det.appendChild(makeEl("pre", "",
    `${span} · ${fmtClock(e.wall_time)}\n` +
    (args ? `args: ${args}\n` : "") +
    (e.error ? `error: ${e.error}` : `result: ${JSON.stringify(e.result)}`)));
  return det;
}

function atBottom() {
  return convo.scrollTop + convo.clientHeight >= convo.scrollHeight - 80;
}

function trimConvo() {
  while (convo.children.length > MAX_ITEMS + 1) {  // +1: typing indicator
    const first = convo.firstChild;
    if (first === chattyping) break;
    convo.removeChild(first);
  }
}

// Insert keeping ascending wall-time order (the two streams can interleave
// slightly out of order); the typing indicator stays pinned at the bottom.
function insertOrdered(el, t) {
  const empty = document.getElementById("convo-empty");
  if (empty) empty.remove();
  el.dataset.t = t;
  let ref = chattyping;
  let sib = chattyping.previousElementSibling;
  while (sib && +sib.dataset.t > t) { ref = sib; sib = sib.previousElementSibling; }
  const stick = atBottom();
  convo.insertBefore(el, ref);
  trimConvo();
  if (stick) convo.scrollTop = convo.scrollHeight;
}

function pushLive(t, render) {
  liveItems.push({ t, render });
  if (liveItems.length > MAX_ITEMS) liveItems.splice(0, liveItems.length - MAX_ITEMS);
  insertOrdered(render(), t);  // during replay chat still flows; events don't arrive
}

function addEvents(events, isReplay) {
  for (const e of events) {
    if (isReplay) {  // replayed feed: append in playback order
      const stick = atBottom();
      convo.insertBefore(renderEvent(e), chattyping);
      if (stick) convo.scrollTop = convo.scrollHeight;
    } else {
      pushLive(e.wall_time, () => renderEvent(e));
    }
  }
}

function addChatMessages(messages) {
  for (const m of messages) {
    if (seenChatIds.has(m.id)) continue;
    seenChatIds.add(m.id);
    pushLive(m.wall_time, () => renderChat(m));
  }
}

function clearConvo() {
  for (const el of [...convo.children]) {
    if (el !== chattyping) el.remove();
  }
}

function renderLive() {
  clearConvo();
  for (const item of [...liveItems].sort((a, b) => a.t - b.t)) {
    const el = item.render();
    el.dataset.t = item.t;
    convo.insertBefore(el, chattyping);
  }
  convo.scrollTop = convo.scrollHeight;
}

function addDivider(text) {
  const stick = atBottom();
  const div = makeEl("div", "divider", text);
  div.dataset.t = Date.now() / 1000;
  convo.insertBefore(div, chattyping);
  if (stick) convo.scrollTop = convo.scrollHeight;
}

// ---- session save / replay ------------------------------------------------
function populateSessions(sessions) {
  const sel = document.getElementById("sessions");
  const prev = sel.value;
  sel.replaceChildren();
  for (const s of sessions || []) {
    const o = document.createElement("option");
    o.value = s.name;
    o.textContent = `${s.name} ${s.success ? "✓" : "✗"} (${s.frames}f)`;
    sel.appendChild(o);
  }
  // default to the newest session unless the user had one picked
  if (prev && [...sel.options].some((o) => o.value === prev)) sel.value = prev;
  else if (sel.options.length) sel.value = sel.options[sel.options.length - 1].value;
}

document.getElementById("replay").addEventListener("click", () => {
  const name = document.getElementById("sessions").value;
  if (name) send({ cmd: "replay", name });
});
document.getElementById("replaystop").addEventListener("click", () =>
  send({ cmd: "replay_stop" }));
document.getElementById("savesession").addEventListener("click", () =>
  send({ cmd: "save_session" }));

// ---- chat input --------------------------------------------------------
const seenChatIds = new Set();  // reconnects replay history; don't duplicate

function setChatBusy(busy) {
  const was = chattyping.classList.contains("on");
  chattyping.classList.toggle("on", busy);
  document.getElementById("chatsend").disabled = busy;
  if (busy && !was && atBottom()) convo.scrollTop = convo.scrollHeight;
}

document.getElementById("chatform").addEventListener("submit", (e) => {
  e.preventDefault();
  const input = document.getElementById("chatinput");
  const text = input.value.trim();
  if (!text) return;
  send({ cmd: "chat", text });
  input.value = "";
});
// keep keystrokes in the chat box from triggering W/S/A/D teleop nudges
document.getElementById("chatinput").addEventListener("keydown", (e) =>
  e.stopPropagation());

// ---- UI --------------------------------------------------------------------
// A task switch (dropdown or two-arm toggle) rebuilds the server session,
// which comes up in idle mode -- but the UI still shows the old mode and hand
// tracking keeps running, so the sim looks frozen. Re-send the selected mode
// so the new session resumes what the user was doing.
function currentMode() {
  const b = document.querySelector("#modes button.active");
  return b ? b.dataset.mode : "idle";
}
function switchTask(task) {
  if (!task) return;
  send({ cmd: "task", task });
  send({ cmd: "mode", mode: currentMode() });
  // The scene is (re)loaded by the state stream once the server switches task.
}
document.getElementById("task").addEventListener("change", () => {
  switchTask(selectedTask());
});
document.getElementById("arm").addEventListener("change", () => {
  switchTask(selectedTask());
});
document.getElementById("twoarms").addEventListener("change", (e) => {
  populateTasks(e.target.checked);
  switchTask(selectedTask());
});
document.querySelectorAll("#modes button").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll("#modes button").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    send({ cmd: "mode", mode: b.dataset.mode });
  });
});
document.getElementById("reset").addEventListener("click", () => send({ cmd: "reset" }));
document.getElementById("grip").addEventListener("click", () => {
  gripOpen = !gripOpen; send({ cmd: "gripper", value: gripOpen ? 1.0 : 0.0 });
});
const speed = document.getElementById("speed");
speed.addEventListener("input", () => {
  document.getElementById("speedval").textContent = `${(+speed.value).toFixed(1)}×`;
  send({ cmd: "speed", speed: +speed.value });
});
document.getElementById("recstart").addEventListener("click", (e) => {
  send({ cmd: "record_start" }); e.target.classList.add("rec");
});
document.getElementById("recstop").addEventListener("click", () => {
  send({ cmd: "record_stop", keep: true });
  document.getElementById("recstart").classList.remove("rec");
});

// ---- fullscreen toggle -------------------------------------------------------
const fsBtn = document.getElementById("fullscreen");

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen();
}

fsBtn.addEventListener("click", toggleFullscreen);
document.addEventListener("fullscreenchange", () => {
  const on = !!document.fullscreenElement;
  fsBtn.classList.toggle("active", on);
  fsBtn.title = on ? "Exit full screen (F)" : "Full screen (F)";
  document.body.classList.toggle("fs", on);  // panes hidden, scene only
});

// ---- MediaPipe hand teleop ---------------------------------------------------
const handBtn = document.getElementById("hand");
const handTeleop = createHandTeleop({
  send,
  canvas: document.getElementById("handview"),
  onStatus: (s) => { document.getElementById("handstatus").textContent = s; },
  isBimanual: () => document.getElementById("twoarms").checked,
});

function setModeUI(mode) {
  document.querySelectorAll("#modes button").forEach((x) =>
    x.classList.toggle("active", x.dataset.mode === mode));
  send({ cmd: "mode", mode });
}

handBtn.addEventListener("click", async () => {
  if (handTeleop.running) {
    handTeleop.stop();
    handBtn.textContent = "Enable hand tracking";
    handBtn.classList.remove("active");
    return;
  }
  try {
    await handTeleop.start();     // loads model + opens webcam on first use
    setModeUI("teleop");          // hand input only acts in teleop mode
    handBtn.textContent = "Disable hand tracking";
    handBtn.classList.add("active");
  } catch (err) {
    console.error("hand teleop:", err);
    document.getElementById("handstatus").textContent =
      err.name === "NotAllowedError"
        ? "camera permission denied"
        : `failed to start: ${err.message}`;
  }
});

// keyboard teleop -> incremental EE nudges (server slews smoothly)
const STEP = 0.02;
window.addEventListener("keydown", (e) => {
  const k = e.key.toLowerCase();
  if (k === "1") { send({ cmd: "active_arm", arm: "left" }); return; }
  if (k === "2") { send({ cmd: "active_arm", arm: "right" }); return; }
  if (k === "f") { toggleFullscreen(); return; }
  const d = { dx: 0, dy: 0, dz: 0 };
  if (k === "w") d.dx = STEP; else if (k === "s") d.dx = -STEP;
  else if (k === "a") d.dy = STEP; else if (k === "d") d.dy = -STEP;
  else if (k === "q") d.dz = STEP; else if (k === "e") d.dz = -STEP;
  else if (k === " ") { gripOpen = !gripOpen; send({ cmd: "gripper", value: gripOpen ? 1 : 0 }); e.preventDefault(); return; }
  else return;
  send({ cmd: "teleop_delta", ...d });
});

connect();
