// Shape editor for Mosaic Studio: draw a new shape or reshape a region by hand.
// Everything happens here in the browser, so drawing stays fluid; Python only
// sends a session and receives the finished outline.
//
// Gradio runs this once when the component appears, with:
//   element  the component (holding the toolbar and the canvas)
//   props    props.value = the session from Python (JSON), or the result we send
//   trigger  trigger('submit') makes Python read props.value
//   watch    watch('value', fn) runs fn when Python sends a new session
//
// A drawing is a list of paths. A path is a chain of segments (short
// polylines sharing their end points) and is open or closed. The two ends of
// an open path are big dots: a stroke started on a dot continues that line; a
// stroke ending on a dot joins the lines, or closes the shape. Coordinates are
// image pixels (x right, y down).

const stage = element.querySelector('.mz-stage');
const cv = element.querySelector('canvas');
const ctx = cv.getContext('2d');
const msg = element.querySelector('.mz-msg');
const zoomLabel = element.querySelector('.mz-zoom');
const btn = (a) => element.querySelector('[data-act="' + a + '"]');

const SNAP = 14;            // screen px: this close to a dot counts as on it
const HIT = 7;              // screen px: this close to a segment counts as on it

let S = null;               // the session from Python
let img = null;
let paths = [];             // [{segs: [[[x, y], ...], ...], closed}]
let tool = 'draw';
let view = {z: 1, x: 0, y: 0, fit: 1};   // screen = image * z + (x, y)
let undoStack = [], redoStack = [];
let stroke = null;          // {path, pts, fresh} while drawing
let erasing = null;         // {count} while erasing
let pan = null;             // {sx, sy, x, y} while panning
let hover = null;           // segment {path, k} (erase) or dot (draw) under the pointer
let space = false;
let raf = 0;
let dpr = window.devicePixelRatio || 1;

// ------------------------------------------------------------------ geometry
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const toImg = (s) => [(s[0] - view.x) / view.z, (s[1] - view.y) / view.z];
const toScr = (p) => [p[0] * view.z + view.x, p[1] * view.z + view.y];
const last = (a) => a[a.length - 1];

function evPos(e) {
  const r = cv.getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
}

function length(pts) {
  let L = 0;
  for (let i = 1; i < pts.length; i++) L += dist(pts[i - 1], pts[i]);
  return L;
}

function densify(pts, step) {
  const out = [pts[0]];
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1], b = pts[i], n = Math.ceil(dist(a, b) / step);
    for (let j = 1; j <= n; j++) out.push([a[0] + (b[0] - a[0]) * j / n, a[1] + (b[1] - a[1]) * j / n]);
  }
  return out;
}

function split(pts) {
  // Cut a polyline into segments about S.spacing long (what the eraser removes).
  pts = densify(pts, S.spacing / 3);
  const segs = [];
  let cur = [pts[0]], L = 0;
  for (let i = 1; i < pts.length; i++) {
    L += dist(pts[i - 1], pts[i]);
    cur.push(pts[i]);
    if (L >= S.spacing && i < pts.length - 1) { segs.push(cur); cur = [pts[i]]; L = 0; }
  }
  if (cur.length > 1) {
    if (segs.length && L < 0.4 * S.spacing) last(segs).push(...cur.slice(1));
    else segs.push(cur);
  }
  return segs;
}

function smooth(pts) {
  // Light smoothing of a hand-drawn stroke; both ends stay where they are.
  for (let pass = 0; pass < 2; pass++) {
    const q = pts.map((p) => p.slice());
    for (let i = 1; i < pts.length - 1; i++)
      for (let k = 0; k < 2; k++) q[i][k] = (pts[i - 1][k] + 2 * pts[i][k] + pts[i + 1][k]) / 4;
    pts = q;
  }
  return pts;
}

function reversePath(p) {
  p.segs.reverse();
  for (const s of p.segs) s.reverse();
}

function flat(p) {
  const out = [];
  p.segs.forEach((s, i) => out.push(...(i ? s.slice(1) : s)));
  if (p.closed) out.pop();
  return out.map((q) => [Math.round(q[0] * 100) / 100, Math.round(q[1] * 100) / 100]);
}

function ends() {
  const out = [];
  for (const p of paths) if (!p.closed && p.segs.length) {
    out.push({p, at: 'start', pt: p.segs[0][0]});
    out.push({p, at: 'end', pt: last(last(p.segs))});
  }
  return out;
}

function nearestEnd(sp, skip) {
  let best = null, bd = SNAP;
  for (const e of ends()) {
    if (skip && skip(e)) continue;
    const d = dist(toScr(e.pt), sp);
    if (d < bd) { bd = d; best = e; }
  }
  return best;
}

function segAt(sp) {
  const q = toImg(sp), tol = HIT / view.z;
  let best = null, bd = tol;
  for (const p of paths) p.segs.forEach((s, k) => {
    for (let i = 1; i < s.length; i++) {
      const a = s[i - 1], b = s[i], dx = b[0] - a[0], dy = b[1] - a[1];
      const t = Math.max(0, Math.min(1, ((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / (dx * dx + dy * dy || 1)));
      const d = dist(q, [a[0] + t * dx, a[1] + t * dy]);
      if (d < bd) { bd = d; best = {path: p, k}; }
    }
  });
  return best;
}

function removeSeg(p, k) {
  const i = paths.indexOf(p);
  if (p.closed) {                      // a closed outline opens where the segment was
    if (p.segs.length === 1) paths.splice(i, 1);
    else { p.segs = p.segs.slice(k + 1).concat(p.segs.slice(0, k)); p.closed = false; }
    return;
  }
  const parts = [p.segs.slice(0, k), p.segs.slice(k + 1)].filter((s) => s.length)
    .map((segs) => ({segs, closed: false}));
  paths.splice(i, 1, ...parts);
}

// ------------------------------------------------------------------ undo
function save() {
  undoStack.push(JSON.stringify(paths));
  if (undoStack.length > 200) undoStack.shift();
  redoStack = [];
}
function undo() {
  if (!undoStack.length) return;
  redoStack.push(JSON.stringify(paths));
  paths = JSON.parse(undoStack.pop());
  hover = null; update();
}
function redo() {
  if (!redoStack.length) return;
  undoStack.push(JSON.stringify(paths));
  paths = JSON.parse(redoStack.pop());
  hover = null; update();
}

// ------------------------------------------------------------------ view
function resize() {
  const w = stage.clientWidth, h = stage.clientHeight;
  if (!w || !h) return;
  dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  cv.style.width = w + 'px'; cv.style.height = h + 'px';
  if (img && img.complete && !view.fitted) fit();
  draw();
}

function fit() {
  const w = stage.clientWidth, h = stage.clientHeight;
  if (!img || !w || !h) return;
  const z = 0.97 * Math.min(w / S.w, h / S.h);
  view = {z, x: (w - S.w * z) / 2, y: (h - S.h * z) / 2, fit: z, fitted: true};
  draw();
}

function zoomAt(f, sp) {
  const z = Math.max(view.fit * 0.5, Math.min(view.z * f, 40));
  const k = z / view.z;
  view.x = sp[0] - (sp[0] - view.x) * k;
  view.y = sp[1] - (sp[1] - view.y) * k;
  view.z = z;
  draw();
}
const centre = () => [stage.clientWidth / 2, stage.clientHeight / 2];

function setTool(t) {
  tool = t; hover = null;
  btn('draw').classList.toggle('on', t === 'draw');
  btn('erase').classList.toggle('on', t === 'erase');
  cv.style.cursor = t === 'draw' ? 'crosshair' : 'pointer';
  draw();
}

// ------------------------------------------------------------------ drawing on screen
function draw() { if (!raf) raf = requestAnimationFrame(render); }

function tracePath(P, p) {
  p.segs.forEach((s, i) => {
    if (i === 0) P.moveTo(s[0][0], s[0][1]);
    for (let j = 1; j < s.length; j++) P.lineTo(s[j][0], s[j][1]);
  });
  if (p.closed) P.closePath();
}

function strokeLine(P, colour, width) {
  ctx.lineJoin = 'round'; ctx.lineCap = 'round';
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.6)'; ctx.lineWidth = (width + 2.5) / view.z; ctx.stroke(P);
  ctx.strokeStyle = colour; ctx.lineWidth = width / view.z; ctx.stroke(P);
}

function dot(p, r, fill) {
  ctx.beginPath(); ctx.arc(p[0], p[1], r / view.z, 0, 2 * Math.PI);
  ctx.fillStyle = fill; ctx.fill();
  ctx.lineWidth = 1.5 / view.z; ctx.strokeStyle = 'rgba(0, 0, 0, 0.75)'; ctx.stroke();
}

function render() {
  raf = 0;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#26262b'; ctx.fillRect(0, 0, cv.width, cv.height);
  if (!img || !S) return;
  ctx.setTransform(dpr * view.z, 0, 0, dpr * view.z, dpr * view.x, dpr * view.y);
  ctx.imageSmoothingEnabled = view.z < 3;            // see the pixels when zoomed far in
  ctx.drawImage(img, 0, 0);
  zoomLabel.textContent = Math.round(view.z * 100) + '%';

  const fill = new Path2D();
  for (const p of paths) if (p.closed) tracePath(fill, p);
  ctx.fillStyle = 'rgba(255, 214, 10, 0.25)'; ctx.fill(fill, 'evenodd');

  for (const p of paths) {
    const P = new Path2D(); tracePath(P, p);
    strokeLine(P, p.closed ? '#ffd60a' : '#ffe66d', 2);
  }
  if (hover && hover.path && paths.includes(hover.path)) {
    const s = hover.path.segs[hover.k];
    if (s) { const P = new Path2D(); tracePath(P, {segs: [s], closed: false}); strokeLine(P, '#ff453a', 4); }
  }
  if (stroke && stroke.pts.length > 1) {
    const P = new Path2D(); tracePath(P, {segs: [stroke.pts], closed: false});
    strokeLine(P, '#64d2ff', 2.5);
  }
  if (S.spacing * view.z >= 16 || tool === 'erase')       // joints between segments, once
    for (const p of paths) p.segs.forEach((s, i) => {       // they are far enough apart
      if (i > 0 || p.closed) dot(s[0], 2.5, '#ffffff');
    });
  for (const e of ends()) {                                // open ends: continue from here
    const on = hover && hover.dot && hover.dot.p === e.p && hover.dot.at === e.at;
    dot(e.pt, on ? 9 : 6.5, on ? '#64d2ff' : '#0a84ff');
  }
}

// ------------------------------------------------------------------ status & buttons
function update() {
  if (!S) return;
  const open = paths.filter((p) => !p.closed).length;
  const closed = paths.length - open;
  let text;
  if (!paths.length) text = S.mode === 'new'
    ? 'Draw the outline of the shape. You can stop any time and continue from the blue dots.'
    : 'Nothing left: draw the outline again, or Undo.';
  else if (open) text = (open === 1 ? 'The line is open' : open + ' lines are open') +
    ': drag from a blue dot to continue, and end on a dot to connect or close it.';
  else text = (S.mode === 'new' ? 'Shape closed ✓' : 'Outline closed ✓') +
    (closed > 1 ? ' (' + closed + ' outlines; one inside another makes a hole)' : '') +
    '. Confirm when it looks right' + (S.mode === 'edit' ? ', or ✂️ Erase a part to redraw it.' : '.');
  msg.textContent = text;
  btn('confirm').disabled = !paths.length || open > 0;
  btn('undo').disabled = !undoStack.length;
  btn('redo').disabled = !redoStack.length;
  draw();
}

function send(action) {
  props.value = JSON.stringify({
    result: true, token: S.token, mode: S.mode, region: S.region, action,
    loops: paths.filter((p) => p.closed).map(flat), before: S.loops || []});
  msg.textContent = action === 'confirm' ? 'Saving…' : 'Closing…';
  setTimeout(() => trigger('submit'), 30);
  setTimeout(update, 2500);          // if Python refuses (e.g. shape too small), carry on
}

// ------------------------------------------------------------------ pointer
cv.addEventListener('contextmenu', (e) => e.preventDefault());

cv.addEventListener('pointerdown', (e) => {
  if (!S || !img) return;
  cv.setPointerCapture(e.pointerId);
  const sp = evPos(e);
  if (e.button === 1 || e.button === 2 || space) {
    pan = {sx: sp[0], sy: sp[1], x: view.x, y: view.y};
    cv.style.cursor = 'grabbing';
    return;
  }
  if (e.button !== 0) return;
  save();
  if (tool === 'erase') {
    erasing = {count: 0};
    const h = segAt(sp);
    if (h) { removeSeg(h.path, h.k); erasing.count++; hover = null; update(); }
    return;
  }
  const hit = nearestEnd(sp);
  if (hit) {                                  // continue an open line from its dot
    if (hit.at === 'start') reversePath(hit.p);
    stroke = {path: hit.p, pts: [last(last(hit.p.segs))], fresh: false};
  } else {
    const p = {segs: [], closed: false};
    paths.push(p);
    stroke = {path: p, pts: [toImg(sp)], fresh: true};
  }
  hover = null; draw();
});

cv.addEventListener('pointermove', (e) => {
  if (!S) return;
  const sp = evPos(e);
  if (pan) {
    view.x = pan.x + sp[0] - pan.sx; view.y = pan.y + sp[1] - pan.sy;
    draw(); return;
  }
  if (stroke) {
    const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
    for (const ev of (evs.length ? evs : [e])) {
      const q = toImg(evPos(ev));
      if (dist(q, last(stroke.pts)) * view.z >= 1.5) stroke.pts.push(q);
    }
    draw(); return;
  }
  if (tool === 'erase') {
    const h = segAt(sp);
    if (erasing && h) { removeSeg(h.path, h.k); erasing.count++; hover = null; update(); return; }
    if ((h && h.path) !== (hover && hover.path) || (h && h.k) !== (hover && hover.k)) { hover = h; draw(); }
    return;
  }
  const d = nearestEnd(sp);
  const was = hover && hover.dot;
  if ((d && d.pt) !== (was && was.pt)) {
    hover = d ? {dot: d} : null;
    cv.style.cursor = d ? 'pointer' : 'crosshair';
    draw();
  }
});

function finishStroke() {
  const st = stroke; stroke = null;
  const p = st.path;
  if (st.pts.length < 2 || length(st.pts) * view.z < 4) {     // a click, not a line
    if (st.fresh) paths.splice(paths.indexOf(p), 1);
    undoStack.pop();
    update(); return;
  }
  const pts = smooth(st.pts);
  const endSp = toScr(last(pts));
  let close = false, join = null;
  if (st.fresh && dist(endSp, toScr(pts[0])) < SNAP && length(pts) * view.z > 4 * SNAP) {
    pts.push(pts[0].slice()); close = true;                     // drew a loop in one go
  } else {
    const t = nearestEnd(endSp, (e) => e.p === p && e.at === 'end');   // not the dot we left
    if (t) {
      pts.push(t.pt.slice());                                   // land exactly on the dot
      if (t.p === p) close = true; else join = t;
    }
  }
  p.segs.push(...split(pts));
  if (close) p.closed = true;
  if (join) {                                                   // two lines become one
    const q = join.p;
    if (join.at === 'end') reversePath(q);
    p.segs.push(...q.segs);
    paths.splice(paths.indexOf(q), 1);
  }
  update();
}

function pointerUp() {
  if (pan) { pan = null; cv.style.cursor = space ? 'grab' : (tool === 'draw' ? 'crosshair' : 'pointer'); return; }
  if (stroke) finishStroke();
  if (erasing) { if (!erasing.count) undoStack.pop(); erasing = null; update(); }
}
cv.addEventListener('pointerup', pointerUp);
cv.addEventListener('pointercancel', pointerUp);
cv.addEventListener('pointerleave', () => { if (hover && !stroke && !erasing) { hover = null; draw(); } });

cv.addEventListener('wheel', (e) => {
  if (!S) return;
  e.preventDefault();
  const sp = evPos(e);
  if (e.ctrlKey || e.metaKey) zoomAt(Math.exp(-e.deltaY * 0.01), sp);   // pinch, or ⌘/Ctrl + scroll
  else { view.x -= e.deltaX; view.y -= e.deltaY; draw(); }              // scroll moves the picture
}, {passive: false});

// ------------------------------------------------------------------ toolbar & keys
element.querySelector('.mz-bar').addEventListener('click', (e) => {
  const b = e.target.closest('[data-act]');
  if (!b || b.disabled || !S) return;
  const a = b.dataset.act;
  if (a === 'draw' || a === 'erase') setTool(a);
  else if (a === 'undo') undo();
  else if (a === 'redo') redo();
  else if (a === 'zoomin') zoomAt(1.4, centre());
  else if (a === 'zoomout') zoomAt(1 / 1.4, centre());
  else if (a === 'fit') fit();
  else if (a === 'confirm' || a === 'cancel') send(a);
});

function onKey(e) {
  if (!element.isConnected) { window.removeEventListener('keydown', onKey); window.removeEventListener('keyup', onKeyUp); return; }
  if (!S || cv.offsetParent === null) return;
  const t = e.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
  const mod = e.metaKey || e.ctrlKey, k = e.key.toLowerCase();
  if (mod && k === 'z') { e.preventDefault(); e.shiftKey ? redo() : undo(); }
  else if (mod && k === 'y') { e.preventDefault(); redo(); }
  else if (mod) return;
  else if (e.key === ' ') { e.preventDefault(); if (!space) { space = true; cv.style.cursor = 'grab'; } }
  else if (k === 'd') setTool('draw');
  else if (k === 'e') setTool('erase');
  else if (k === '+' || k === '=') zoomAt(1.4, centre());
  else if (k === '-') zoomAt(1 / 1.4, centre());
  else if (k === '0') fit();
}
function onKeyUp(e) {
  if (e.key === ' ' && space) { space = false; cv.style.cursor = tool === 'draw' ? 'crosshair' : 'pointer'; }
}
window.addEventListener('keydown', onKey);
window.addEventListener('keyup', onKeyUp);

// ------------------------------------------------------------------ sessions
function start(value) {
  let s = null;
  try { s = value ? JSON.parse(value) : null; } catch (err) { s = null; }
  if (!s || !s.token || s.result) return;          // empty, or our own result
  if (S && S.token === s.token) return;
  S = s;
  paths = (s.loops || []).filter((l) => l.length > 2)
    .map((l) => ({segs: split(l.concat([l[0]])), closed: true}));
  undoStack = []; redoStack = []; stroke = null; erasing = null; pan = null; hover = null;
  view.fitted = false;
  setTool('draw');
  img = new Image();
  img.onload = () => { resize(); fit(); update(); };
  img.src = s.img;
  update();
}

new ResizeObserver(resize).observe(stage);
watch('value', () => start(props.value));
start(props.value);
