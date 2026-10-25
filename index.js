// DEXO: GLB -> JSON konverter (DEXO GTA model formati).
// GLB dagi shakl (ichki va tashqi qismlar), rang (tekstura bo‘yicha), shaffoflik va aylanadigan parraklar JSON ga o‘tadi.
// JSON nomi GLB nomi bilan bir xil saqlanadi. Hamma narsa brauzerda ishlaydi, fayl serverga yuborilmaydi.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

// ===== CORE START =====
// Sof hisoblash qismi (brauzer/three.js ga bog‘liq emas): rang olish, bo‘lish, palitra, JSON yig‘ish
const S2L = new Float32Array(256);
for (let i = 0; i < 256; i++) { const c = i / 255; S2L[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
const L2S_N = 4096;
const L2S = new Uint8Array(L2S_N + 1);
for (let i = 0; i <= L2S_N; i++) { const l = i / L2S_N; const s = l <= 0.0031308 ? l * 12.92 : 1.055 * Math.pow(l, 1 / 2.4) - 0.055; L2S[i] = Math.round(Math.max(0, Math.min(1, s)) * 255); }
function l2s(l) { return L2S[l <= 0 ? 0 : l >= 1 ? L2S_N : (l * L2S_N + 0.5) | 0]; }

function wrapIdx(i, n, mode) {
  if (mode === 33071) return i < 0 ? 0 : i >= n ? n - 1 : i;                       // CLAMP_TO_EDGE
  if (mode === 33648) { const m = ((i % (2 * n)) + 2 * n) % (2 * n); return m < n ? m : 2 * n - 1 - m; }   // MIRRORED_REPEAT
  return ((i % n) + n) % n;                                                         // REPEAT
}

// Tekstura namuna oluvchi (bilinear). UV: glTF dagidek, (0,0) - yuqori chap burchak.
class Sampler {
  constructor(img, wrapS, wrapT, tr) { this.d = img.data; this.w = img.w; this.h = img.h; this.ws = wrapS; this.wt = wrapT; this.tr = tr; }
  map(u, v, o) {
    const t = this.tr;
    if (t) { const x = t.sx * u, y = t.sy * v; o[0] = t.cos * x + t.sin * y + t.ox; o[1] = -t.sin * x + t.cos * y + t.oy; }
    else { o[0] = u; o[1] = v; }
  }
  sample(u, v, out) {
    if (this.tr) { const t = this.tr, x = t.sx * u, y = t.sy * v; u = t.cos * x + t.sin * y + t.ox; v = -t.sin * x + t.cos * y + t.oy; }
    if (!(u === u) || !(v === v)) { u = 0; v = 0; }
    const w = this.w, h = this.h, d = this.d;
    const fx = u * w - 0.5, fy = v * h - 0.5;
    const x0 = Math.floor(fx), y0 = Math.floor(fy), tx = fx - x0, ty = fy - y0;
    const xa = wrapIdx(x0, w, this.ws), xb = wrapIdx(x0 + 1, w, this.ws), ya = wrapIdx(y0, h, this.wt), yb = wrapIdx(y0 + 1, h, this.wt);
    const i00 = (ya * w + xa) * 4, i10 = (ya * w + xb) * 4, i01 = (yb * w + xa) * 4, i11 = (yb * w + xb) * 4;
    const w00 = (1 - tx) * (1 - ty), w10 = tx * (1 - ty), w01 = (1 - tx) * ty, w11 = tx * ty;
    out[0] = d[i00] * w00 + d[i10] * w10 + d[i01] * w01 + d[i11] * w11;
    out[1] = d[i00 + 1] * w00 + d[i10 + 1] * w10 + d[i01 + 1] * w01 + d[i11 + 1] * w11;
    out[2] = d[i00 + 2] * w00 + d[i10 + 2] * w10 + d[i01 + 2] * w01 + d[i11 + 2] * w11;
    out[3] = d[i00 + 3] * w00 + d[i10 + 3] * w10 + d[i01 + 3] * w01 + d[i11 + 3] * w11;
  }
}

const _T = new Float64Array(4);
// Uchburchakning (w0,w1,w2) barisentrik nuqtasidagi rang: PBR bazaviy rang * tekstura * uch ranglari + emissive. out = [r,g,b (sRGB 0..255), alfa 0..1]
function evalColor(c, w0, w1, w2, out) {
  const m = c.mat;
  let r = m.base[0], g = m.base[1], b = m.base[2], a = m.base[3];
  if (c.uvA) {
    const u = w0 * c.uvA[0] + w1 * c.uvA[2] + w2 * c.uvA[4], v = w0 * c.uvA[1] + w1 * c.uvA[3] + w2 * c.uvA[5];
    m.texBase.sample(u, v, _T);
    r *= S2L[(_T[0] + 0.5) | 0]; g *= S2L[(_T[1] + 0.5) | 0]; b *= S2L[(_T[2] + 0.5) | 0]; a *= _T[3] / 255;
  }
  if (c.vc) {
    const q = c.vc;
    r *= w0 * q[0] + w1 * q[4] + w2 * q[8]; g *= w0 * q[1] + w1 * q[5] + w2 * q[9]; b *= w0 * q[2] + w1 * q[6] + w2 * q[10];
    a *= w0 * q[3] + w1 * q[7] + w2 * q[11];
  }
  if (m.emis) {
    let er = m.emis[0], eg = m.emis[1], eb = m.emis[2];
    if (c.uvE) {
      const u = w0 * c.uvE[0] + w1 * c.uvE[2] + w2 * c.uvE[4], v = w0 * c.uvE[1] + w1 * c.uvE[3] + w2 * c.uvE[5];
      m.texEmis.sample(u, v, _T);
      er *= S2L[(_T[0] + 0.5) | 0]; eg *= S2L[(_T[1] + 0.5) | 0]; eb *= S2L[(_T[2] + 0.5) | 0];
    }
    r += er; g += eg; b += eb;
  }
  out[0] = l2s(r); out[1] = l2s(g); out[2] = l2s(b); out[3] = a;
}

class Sink {
  constructor() { this.P = []; this.C = []; this.A = []; this.G = []; this.n = 0; }
  add(p0, p1, p2, p3, p4, p5, p6, p7, p8, r, g, b, bucket, gi) {
    this.P.push(p0, p1, p2, p3, p4, p5, p6, p7, p8);
    this.C.push(r, g, b); this.A.push(bucket); this.G.push(gi); this.n++;
  }
}

const _o = new Float64Array(4), _m = [0, 0], _m2 = [0, 0];
const _samp = new Float64Array(40 * 4);
function alphaBucket(mat, a) {
  if (mat.alphaMode === 'MASK') return a >= mat.cutoff ? 20 : 0;
  if (mat.alphaMode === 'BLEND') { const k = Math.round(a * 20); return k >= 20 ? 20 : k; }
  return 20;
}

// Bitta uchburchakni (dunyo koordinatasida) ranglarini aniq olish uchun kerak bo‘lsa mayda uchburchaklarga bo‘lib sink ga yozadi.
// P: 9 son. uvA/uvE: 6 son yoki null. vc: 12 son (uch uchun rgba) yoki null.
function processTriangle(P, uvA, uvE, vc, mat, opt, sink, gi) {
  const ctx = { mat, uvA: mat.texBase ? uvA : null, uvE: mat.emis && mat.texEmis ? uvE : null, vc };
  const hasTex = !!ctx.uvA;
  if (!hasTex && !ctx.uvE && !vc) {   // bir xil rang: bo‘lish shart emas
    evalColor(ctx, 1 / 3, 1 / 3, 1 / 3, _o);
    const bk = alphaBucket(mat, _o[3]);
    if (bk > 0) sink.add(P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], _o[0], _o[1], _o[2], bk, gi);
    return;
  }
  const stack = [[1, 0, 0, 0, 1, 0, 0, 0, 1, 0]];
  const maxDepth = sink.n > opt.maxFaces ? 0 : opt.depth;
  while (stack.length) {
    const s = stack.pop();
    const b0x = s[0], b0y = s[1], b0z = s[2], b1x = s[3], b1y = s[4], b1z = s[5], b2x = s[6], b2y = s[7], b2z = s[8], depth = s[9];
    // burchaklarning dunyodagi joyi
    const ax = b0x * P[0] + b0y * P[3] + b0z * P[6], ay = b0x * P[1] + b0y * P[4] + b0z * P[7], az = b0x * P[2] + b0y * P[5] + b0z * P[8];
    const bx = b1x * P[0] + b1y * P[3] + b1z * P[6], by = b1x * P[1] + b1y * P[4] + b1z * P[7], bz = b1x * P[2] + b1y * P[5] + b1z * P[8];
    const cx = b2x * P[0] + b2y * P[3] + b2z * P[6], cy = b2x * P[1] + b2y * P[4] + b2z * P[7], cz = b2x * P[2] + b2y * P[5] + b2z * P[8];
    // tekstura bo‘yicha o‘lcham (teksel)
    let texels = 0;
    if (hasTex) {
      const U = (b) => [b[0] * uvA[0] + b[1] * uvA[2] + b[2] * uvA[4], b[0] * uvA[1] + b[1] * uvA[3] + b[2] * uvA[5]];
      const ua = U([b0x, b0y, b0z]), ub = U([b1x, b1y, b1z]), uc = U([b2x, b2y, b2z]);
      const sm = ctx.mat.texBase;
      sm.map(ua[0], ua[1], _m); const a0 = _m[0], a1 = _m[1];
      sm.map(ub[0], ub[1], _m); const c0 = _m[0], c1 = _m[1];
      sm.map(uc[0], uc[1], _m2); const d0 = _m2[0], d1 = _m2[1];
      const W = sm.w, H = sm.h;
      texels = Math.max(Math.hypot((a0 - c0) * W, (a1 - c1) * H), Math.hypot((c0 - d0) * W, (c1 - d1) * H), Math.hypot((d0 - a0) * W, (d1 - a1) * H));
    }
    const n = hasTex ? Math.max(2, Math.min(opt.maxN, Math.ceil(texels / opt.spacing))) : 2;
    // namunalar
    let cnt = 0, sr = 0, sg = 0, sb = 0, sa = 0;
    for (let i = 0; i <= n; i++) for (let j = 0; j <= n - i; j++) {
      const q1 = i / n, q2 = j / n, q0 = 1 - q1 - q2;
      const w0 = q0 * b0x + q1 * b1x + q2 * b2x, w1 = q0 * b0y + q1 * b1y + q2 * b2y, w2 = q0 * b0z + q1 * b1z + q2 * b2z;
      evalColor(ctx, w0, w1, w2, _o);
      const k = cnt * 4; _samp[k] = _o[0]; _samp[k + 1] = _o[1]; _samp[k + 2] = _o[2]; _samp[k + 3] = _o[3];
      sr += _o[0]; sg += _o[1]; sb += _o[2]; sa += _o[3]; cnt++;
    }
    const mr = sr / cnt, mg = sg / cnt, mb = sb / cnt, ma = sa / cnt;
    let dev = 0;
    for (let k = 0; k < cnt; k++) {
      dev = Math.max(dev, Math.abs(_samp[k * 4] - mr), Math.abs(_samp[k * 4 + 1] - mg), Math.abs(_samp[k * 4 + 2] - mb), Math.abs(_samp[k * 4 + 3] - ma) * 255);
    }
    const edge = Math.max(Math.hypot(ax - bx, ay - by, az - bz), Math.hypot(bx - cx, by - cy, bz - cz), Math.hypot(cx - ax, cy - ay, cz - az));
    if (dev > opt.tol && depth < maxDepth && edge > opt.minEdge && (!hasTex || texels > 2)) {
      const m01 = [(b0x + b1x) / 2, (b0y + b1y) / 2, (b0z + b1z) / 2], m12 = [(b1x + b2x) / 2, (b1y + b2y) / 2, (b1z + b2z) / 2], m20 = [(b2x + b0x) / 2, (b2y + b0y) / 2, (b2z + b0z) / 2];
      const d = depth + 1;
      stack.push([b0x, b0y, b0z, ...m01, ...m20, d], [...m01, b1x, b1y, b1z, ...m12, d], [...m20, ...m12, b2x, b2y, b2z, d], [...m01, ...m12, ...m20, d]);
      continue;
    }
    const bk = alphaBucket(mat, ma);
    if (bk > 0) sink.add(ax, ay, az, bx, by, bz, cx, cy, cz, Math.round(mr), Math.round(mg), Math.round(mb), bk, gi);
  }
}

// ---------- Palitra (median cut). Ranglar soni palitradan kam bo‘lsa aniq, ko‘p bo‘lsa eng yaqin ranglarga jamlanadi ----------
function quantizeColors(C, K) {
  const counts = new Map();
  for (let i = 0; i < C.length; i += 3) { const key = (C[i] << 16) | (C[i + 1] << 8) | C[i + 2]; counts.set(key, (counts.get(key) || 0) + 1); }
  const keys = [...counts.keys()];
  const keyToPal = new Map();
  let pal = [];
  const mk = (items) => {
    let mn = [255, 255, 255], mx = [0, 0, 0], pop = 0;
    for (const it of items) { pop += it.n; for (let c = 0; c < 3; c++) { if (it.v[c] < mn[c]) mn[c] = it.v[c]; if (it.v[c] > mx[c]) mx[c] = it.v[c]; } }
    const rg = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]];
    const ch = rg[0] >= rg[1] && rg[0] >= rg[2] ? 0 : rg[1] >= rg[2] ? 1 : 2;
    return { items, ch, range: rg[ch], pop };
  };
  if (keys.length <= K) {
    keys.sort((a, b) => a - b);
    keys.forEach((k, i) => { keyToPal.set(k, i); pal.push([k >> 16, (k >> 8) & 255, k & 255]); });
  } else {
    let boxes = [mk(keys.map((k) => ({ k, n: counts.get(k), v: [k >> 16, (k >> 8) & 255, k & 255] })))];
    while (boxes.length < K) {
      let best = -1, bs = 0;
      for (let i = 0; i < boxes.length; i++) { const b = boxes[i]; if (b.items.length < 2 || b.range <= 0) continue; const sc = b.range * Math.sqrt(b.pop); if (sc > bs) { bs = sc; best = i; } }
      if (best < 0) break;
      const b = boxes[best], ch = b.ch;
      b.items.sort((x, y) => x.v[ch] - y.v[ch]);
      let acc = 0, cut = 1;
      for (let i = 0; i < b.items.length - 1; i++) { acc += b.items[i].n; cut = i + 1; if (acc >= b.pop / 2) break; }
      boxes.splice(best, 1, mk(b.items.slice(0, cut)), mk(b.items.slice(cut)));
    }
    boxes.forEach((b, i) => {
      let r = 0, g = 0, bl = 0;
      for (const it of b.items) { r += it.v[0] * it.n; g += it.v[1] * it.n; bl += it.v[2] * it.n; keyToPal.set(it.k, i); }
      pal.push([Math.round(r / b.pop), Math.round(g / b.pop), Math.round(bl / b.pop)]);
    });
  }
  const idx = new Int32Array(C.length / 3);
  let err = 0;
  for (let i = 0; i < idx.length; i++) {
    const p = keyToPal.get((C[i * 3] << 16) | (C[i * 3 + 1] << 8) | C[i * 3 + 2]);
    idx[i] = p; const c = pal[p];
    err += Math.abs(c[0] - C[i * 3]) + Math.abs(c[1] - C[i * 3 + 1]) + Math.abs(c[2] - C[i * 3 + 2]);
  }
  return { pal, idx, err: idx.length ? err / (idx.length * 3) : 0, unique: keys.length };
}
const hex2 = (v) => v.toString(16).padStart(2, '0');
const palHex = (c) => '#' + hex2(c[0]) + hex2(c[1]) + hex2(c[2]);

// ---------- Aylanish o‘qi: nuqtalar bulutining eng yupqa yo‘nalishi (rotor diski normali) ----------
function eigSym3(A) {
  const a = A.map((r) => r.slice()), V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 40; sweep++) {
    if (a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2] < 1e-24) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      if (Math.abs(a[p][q]) < 1e-30) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 3; k++) { const x = a[k][p], y = a[k][q]; a[k][p] = c * x - s * y; a[k][q] = s * x + c * y; }
      for (let k = 0; k < 3; k++) { const x = a[p][k], y = a[q][k]; a[p][k] = c * x - s * y; a[q][k] = s * x + c * y; }
      for (let k = 0; k < 3; k++) { const x = V[k][p], y = V[k][q]; V[k][p] = c * x - s * y; V[k][q] = s * x + c * y; }
    }
  }
  return { vals: [a[0][0], a[1][1], a[2][2]], vecs: [[V[0][0], V[1][0], V[2][0]], [V[0][1], V[1][1], V[2][1]], [V[0][2], V[1][2], V[2][2]]] };
}
function diskNormal(pts /* [x,y,z,...] */) {
  const n = pts.length / 3;
  if (n < 3) return [0, 1, 0];
  let mx = 0, my = 0, mz = 0;
  for (let i = 0; i < pts.length; i += 3) { mx += pts[i]; my += pts[i + 1]; mz += pts[i + 2]; }
  mx /= n; my /= n; mz /= n;
  let xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
  for (let i = 0; i < pts.length; i += 3) { const x = pts[i] - mx, y = pts[i + 1] - my, z = pts[i + 2] - mz; xx += x * x; xy += x * y; xz += x * z; yy += y * y; yz += y * z; zz += z * z; }
  const e = eigSym3([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]);
  let k = 0; for (let i = 1; i < 3; i++) if (e.vals[i] < e.vals[k]) k = i;
  return e.vecs[k];
}
function snapAxis(v) {
  const ax = Math.abs(v[0]), ay = Math.abs(v[1]), az = Math.abs(v[2]);
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  const m = Math.max(ax, ay, az);
  const axis = m === ax ? 'x' : m === ay ? 'y' : 'z';
  return { axis, dev: Math.acos(Math.min(1, m / len)) * 180 / Math.PI };
}

// ---------- JSON yig‘ish: sink -> o‘yin formatidagi matn ----------
// groups: [{ key, pivot:[x,y,z], axis:'x'|'y'|'z' }]; sink.G dagi qiymat groups indeksi yoki -1 (qotib turadigan kuzov)
function buildSpecText(sink, groups, opt, name) {
  const P = sink.P, nF = sink.n;
  let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < P.length; i += 3) for (let c = 0; c < 3; c++) { const v = P[i + c]; if (v < mn[c]) mn[c] = v; if (v > mx[c]) mx[c] = v; }
  const maxDim = Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2], 1e-9);
  const dec = opt.decimals > 0 ? opt.decimals : Math.min(6, Math.max(2, Math.ceil(Math.log10(80000 / maxDim))));
  const s10 = Math.pow(10, dec);
  const q = quantizeColors(sink.C, opt.palette);
  // paketlash: (guruh, shaffoflik) bo‘yicha
  const buckets = new Map();
  for (let f = 0; f < nF; f++) {
    const gi = sink.G[f];
    const bk = gi >= 0 ? 20 : sink.A[f];
    const key = gi * 100 + bk;
    let a = buckets.get(key); if (!a) { a = []; buckets.set(key, a); }
    a.push(f);
  }
  const order = [...buckets.keys()].sort((a, b) => {
    const ga = Math.floor(a / 100), gb = Math.floor(b / 100);
    const ba = a - ga * 100, bb = b - gb * 100;
    const ra = ga < 0 ? (ba === 20 ? 0 : 2) : 1, rb = gb < 0 ? (bb === 20 ? 0 : 2) : 1;
    return ra - rb || ga - gb || bb - ba;
  });
  const lines = [];
  let totalVerts = 0, totalFaces = 0;
  for (const key of order) {
    const faces = buckets.get(key);
    const gi = Math.floor(key / 100) < 0 ? -1 : Math.floor(key / 100);
    const bk = key - Math.floor(key / 100) * 100;
    const map = new Map(), verts = [], idx = [], fcs = [];
    for (const f of faces) {
      const ids = [0, 0, 0];
      for (let k = 0; k < 3; k++) {
        const ix = Math.round(P[f * 9 + k * 3] * s10), iy = Math.round(P[f * 9 + k * 3 + 1] * s10), iz = Math.round(P[f * 9 + k * 3 + 2] * s10);
        const kk = ix + ',' + iy + ',' + iz;
        let id = map.get(kk);
        if (id === undefined) { id = verts.length / 3; map.set(kk, id); verts.push(ix, iy, iz); }
        ids[k] = id;
      }
      if (ids[0] === ids[1] || ids[1] === ids[2] || ids[0] === ids[2]) continue;
      idx.push(ids[0], ids[1], ids[2]); fcs.push(q.idx[f]);
    }
    if (!idx.length) continue;
    const used = [...new Set(fcs)].sort((a, b) => a - b);
    const loc = new Map(used.map((g, i) => [g, i]));
    const vs = new Array(verts.length);
    for (let i = 0; i < verts.length; i++) vs[i] = verts[i] / s10;
    let s = '    {"t": "mesh"';
    if (gi >= 0) s += ', "g": ' + JSON.stringify(groups[gi].key);
    s += ', "ca": ' + opt.crease;
    if (gi < 0 && bk < 20) s += ', "o": ' + (bk / 20);
    if (used.length === 1) s += ', "c": "' + palHex(q.pal[used[0]]) + '"';
    else s += ', "pal": ' + JSON.stringify(used.map((g) => palHex(q.pal[g])));
    s += ', "v": [' + vs.join(',') + '], "f": [' + idx.join(',') + ']';
    if (used.length > 1) s += ', "fc": [' + fcs.map((g) => loc.get(g)).join(',') + ']';
    s += '}';
    lines.push(s);
    totalVerts += verts.length / 3; totalFaces += idx.length / 3;
  }
  const fmt = (v) => String(Math.round(v * s10) / s10);
  let head = '{\n  "name": ' + JSON.stringify(name) + ',\n';
  if (groups.length) {
    head += '  "groups": {\n' + groups.map((g) => '    ' + JSON.stringify(g.key) + ': {"p": [' + g.pivot.map(fmt).join(', ') + '], "axis": "' + g.axis + '"}').join(',\n') + '\n  },\n';
  }
  const text = head + '  "parts": [\n' + lines.join(',\n') + '\n  ]\n}\n';
  return { text, faces: totalFaces, verts: totalVerts, parts: lines.length, palette: q.pal.length, uniqueColors: q.unique, colorError: q.err, decimals: dec, size: [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]] };
}
// ===== CORE END =====

// ===================== BRAUZER QISMI (GLB o‘qish, interfeys, ko‘rish) =====================
const $ = (id) => document.getElementById(id);
const tick = () => new Promise((r) => setTimeout(r, 0));
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmtBytes = (n) => (n >= 1048576 ? (n / 1048576).toFixed(2) + ' MB' : Math.max(1, Math.round(n / 1024)) + ' KB');
const fmtNum = (n) => Number(n).toLocaleString('en-US').replace(/,/g, ' ');
const fmtDim = (v) => (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(2));

const PRESETS = {
  tez: { depth: 0, tol: 9999, maxN: 4, spacing: 8 },       // har bir uchburchak = bitta rang (eng yengil)
  yaxshi: { depth: 2, tol: 16, maxN: 4, spacing: 6 },      // rang o‘zgargan joylar mayda uchburchaklarga bo‘linadi
  aniq: { depth: 4, tol: 8, maxN: 6, spacing: 4 },         // juda aniq, lekin fayl katta bo‘ladi
};

const gltfLoader = new GLTFLoader();
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath('https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/libs/draco/gltf/');
gltfLoader.setDRACOLoader(dracoLoader);
gltfLoader.setMeshoptDecoder(MeshoptDecoder);

const jobs = [];
let jobSeq = 0;

function numVal(id, def, min, max) {
  const v = parseFloat($(id).value);
  if (!Number.isFinite(v)) return def;
  return Math.min(max, Math.max(min, v));
}
function readOptions() {
  const p = PRESETS[$('preset').value] || PRESETS.yaxshi;
  return {
    ...p,
    palette: parseInt($('palette').value, 10) || 256,
    crease: Math.round(numVal('crease', 40, 0, 180)),
    scale: numVal('scale', 1, 1e-6, 1e6),
    decimals: parseInt($('decimals').value, 10) || 0,
    maxFaces: Math.round(numVal('maxFaces', 1000000, 1000, 20000000)),
  };
}

// ---------- Fayllarni qo‘shish ----------
const drop = $('drop');
$('file').addEventListener('change', (e) => { const f = [...e.target.files]; e.target.value = ''; addFiles(f); });
['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => { addFiles([...(e.dataTransfer?.files || [])]); });

let addChain = Promise.resolve();
function addFiles(files) {
  for (const f of files) {
    if (!/\.(glb|gltf)$/i.test(f.name)) continue;
    addChain = addChain.then(() => addJob(f));   // xotira uchun bittadan tahlil qilinadi
  }
}

async function addJob(file) {
  const job = {
    id: ++jobSeq, file, base: file.name.replace(/\.(glb|gltf)$/i, ''),
    gltf: null, parser: null, json: null, items: [], groups: [], matCache: new Map(), imgCache: new Map(),
    status: 'GLB o‘qilmoqda…', statusClass: '', progress: -1, busy: false, result: null, notes: [], info: null, previewJson: null,
  };
  jobs.push(job);
  renderJobs();
  try {
    await analyze(job);
    job.status = 'Tayyor. Sozlamalarni tekshirib, “Aylantirish” ni bosing.';
  } catch (err) {
    console.error(err);
    job.status = 'Xatolik: ' + (err && err.message ? err.message : err);
    job.statusClass = 'bad';
  }
  job.progress = -1;
  renderJobs();
}

// ---------- Tahlil ----------
async function analyze(job) {
  const buf = await job.file.arrayBuffer();
  const gltf = await gltfLoader.parseAsync(buf, '');
  job.gltf = gltf; job.parser = gltf.parser; job.json = gltf.parser.json;
  gltf.scene.updateMatrixWorld(true);
  const items = [];
  let tris = 0;
  gltf.scene.traverseVisible((o) => {
    if (!o.isMesh || !o.geometry || !o.geometry.attributes.position) return;
    items.push({ mesh: o, gi: -1 });
    const g = o.geometry;
    tris += Math.floor((g.index ? g.index.count : g.attributes.position.count) / 3);
  });
  if (!items.length) throw new Error('Modelda mesh topilmadi');
  job.items = items;
  const box = new THREE.Box3().setFromObject(gltf.scene, true);
  const size = box.getSize(new THREE.Vector3());
  job.info = {
    tris, meshes: items.length, materials: (job.json.materials || []).length, textures: (job.json.textures || []).length,
    size: [size.x, size.y, size.z], box,
  };
  detectGroups(job);
}

function detectGroups(job) {
  if (!job.gltf) return;
  let re = null;
  try { const s = $('spinRe').value.trim(); re = s ? new RegExp(s, 'i') : null; } catch (e) { re = null; }
  const old = new Map(job.groups.map((g) => [g.node.uuid, g.axisMode]));
  const map = new Map();
  for (const it of job.items) {
    it.group = null;
    if (!re) continue;
    let top = null;
    for (let n = it.mesh; n && n !== job.gltf.scene; n = n.parent) if (n.name && re.test(n.name)) top = n;   // eng yuqori mos tugun
    if (!top) continue;
    if (!map.has(top.uuid)) map.set(top.uuid, { node: top, key: '', items: [], axisMode: old.get(top.uuid) || 'auto' });
    const g = map.get(top.uuid);
    g.items.push(it); it.group = g;
  }
  const used = new Set();
  job.groups = [...map.values()];
  job.groups.forEach((g, i) => {
    let k = g.node.name || ('guruh' + (i + 1)), n = 2;
    const base = k;
    while (used.has(k)) k = base + '_' + (n++);
    used.add(k); g.key = k;
  });
}
$('spinRe').addEventListener('change', () => { jobs.forEach((j) => { if (!j.busy) detectGroups(j); }); renderJobs(); });

// ---------- Tekstura / material ----------
async function getImage(job, srcIdx) {
  if (job.imgCache.has(srcIdx)) return job.imgCache.get(srcIdx);
  const p = (async () => {
    const def = job.json.images && job.json.images[srcIdx];
    if (!def) return null;
    let blob = null;
    if (def.bufferView !== undefined) {
      const ab = await job.parser.getDependency('bufferView', def.bufferView);
      blob = new Blob([ab], { type: def.mimeType || 'image/png' });
    } else if (def.uri) {
      try { blob = await (await fetch(def.uri)).blob(); } catch (e) { return null; }
    }
    if (!blob) return null;
    let bmp = null;
    try { bmp = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' }); }
    catch (e) {
      bmp = await new Promise((res, rej) => { const im = new Image(); const u = URL.createObjectURL(blob); im.onload = () => { URL.revokeObjectURL(u); res(im); }; im.onerror = () => rej(new Error('Tekstura o‘qilmadi')); im.src = u; });
    }
    let w = bmp.width, h = bmp.height;
    const MAXS = 4096;
    if (Math.max(w, h) > MAXS) { const k = MAXS / Math.max(w, h); w = Math.max(1, Math.round(w * k)); h = Math.max(1, Math.round(h * k)); job.notes.push('Katta tekstura ' + MAXS + ' pikselgacha kichraytirildi'); }
    const cv = document.createElement('canvas'); cv.width = w; cv.height = h;
    const cx = cv.getContext('2d', { willReadFrequently: true });
    cx.drawImage(bmp, 0, 0, w, h);
    if (bmp.close) bmp.close();
    const id = cx.getImageData(0, 0, w, h);
    cv.width = cv.height = 0;
    return { data: id.data, w, h };
  })();
  job.imgCache.set(srcIdx, p);
  return p;
}

async function makeSampler(job, texInfo) {
  if (!texInfo) return null;
  const tdef = job.json.textures && job.json.textures[texInfo.index];
  if (!tdef) return null;
  const ex = tdef.extensions || {};
  const src = tdef.source !== undefined ? tdef.source : (ex.EXT_texture_webp && ex.EXT_texture_webp.source) !== undefined ? ex.EXT_texture_webp.source : ex.EXT_texture_avif ? ex.EXT_texture_avif.source : undefined;
  if (src === undefined) { job.notes.push('KTX2/Basis tekstura o‘qib bo‘lmadi (faqat PNG/JPG/WebP)'); return null; }
  let img = null;
  try { img = await getImage(job, src); } catch (e) { job.notes.push('Tekstura o‘qilmadi: ' + e.message); }
  if (!img) return null;
  const sdef = tdef.sampler !== undefined && job.json.samplers ? job.json.samplers[tdef.sampler] : null;
  const tt = texInfo.extensions && texInfo.extensions.KHR_texture_transform;
  let tr = null;
  if (tt) { const rot = tt.rotation || 0; tr = { ox: (tt.offset || [0, 0])[0], oy: (tt.offset || [0, 0])[1], sx: (tt.scale || [1, 1])[0], sy: (tt.scale || [1, 1])[1], cos: Math.cos(rot), sin: Math.sin(rot) }; }
  return { sampler: new Sampler(img, (sdef && sdef.wrapS) || 10497, (sdef && sdef.wrapT) || 10497, tr), ch: (tt && tt.texCoord !== undefined ? tt.texCoord : texInfo.texCoord) || 0 };
}

async function buildMatInfo(job, idx, tm) {
  const info = { mat: { base: [1, 1, 1, 1], emis: null, texBase: null, texEmis: null, alphaMode: 'OPAQUE', cutoff: 0.5 }, chBase: 0, chEmis: 0 };
  const def = idx !== null ? job.json.materials[idx] : null;
  if (!def) {
    if (tm && tm.color) { info.mat.base = [tm.color.r, tm.color.g, tm.color.b, tm.opacity === undefined ? 1 : tm.opacity]; info.mat.alphaMode = tm.transparent ? 'BLEND' : 'OPAQUE'; }
    return info;
  }
  const pbr = def.pbrMetallicRoughness || {};
  const sg = def.extensions && def.extensions.KHR_materials_pbrSpecularGlossiness;
  const baseF = sg ? (sg.diffuseFactor || [1, 1, 1, 1]) : (pbr.baseColorFactor || [1, 1, 1, 1]);
  const baseT = sg ? sg.diffuseTexture : pbr.baseColorTexture;
  info.mat.base = [baseF[0], baseF[1], baseF[2], baseF[3] === undefined ? 1 : baseF[3]];
  info.mat.alphaMode = def.alphaMode || 'OPAQUE';
  info.mat.cutoff = def.alphaCutoff === undefined ? 0.5 : def.alphaCutoff;
  const bs = await makeSampler(job, baseT);
  if (bs) { info.mat.texBase = bs.sampler; info.chBase = bs.ch; }
  const ef = def.emissiveFactor;
  if (ef && (ef[0] || ef[1] || ef[2])) {
    const st = (def.extensions && def.extensions.KHR_materials_emissive_strength && def.extensions.KHR_materials_emissive_strength.emissiveStrength) || 1;
    info.mat.emis = [ef[0] * st, ef[1] * st, ef[2] * st];
    const es = await makeSampler(job, def.emissiveTexture);
    if (es) { info.mat.texEmis = es.sampler; info.chEmis = es.ch; }
  }
  return info;
}

async function getMatInfo(job, mesh) {
  const tm = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
  const assoc = tm && job.parser.associations ? job.parser.associations.get(tm) : null;
  const idx = assoc && assoc.materials !== undefined ? assoc.materials : null;
  const key = idx !== null ? idx : 'm:' + (tm ? tm.uuid : 'none');
  if (!job.matCache.has(key)) job.matCache.set(key, await buildMatInfo(job, idx, tm));
  return job.matCache.get(key);
}

// ---------- Aylantirish ----------
async function convertJob(job) {
  if (job.busy || !job.gltf) return;
  job.busy = true; job.result = null; job.previewJson = null; job.notes = job.notes.filter((n) => /kichraytirildi|KTX2|Tekstura/.test(n));
  const t0 = performance.now();
  const opt = readOptions();
  job.status = 'Boshlanmoqda…'; job.statusClass = ''; job.progress = 0;
  renderJobs();
  try {
    job.gltf.scene.updateMatrixWorld(true);
    const sc = opt.scale;
    const size = job.info.size;
    opt.minEdge = Math.max(size[0], size[1], size[2]) * sc * 2e-4;
    const sink = new Sink();
    let totalTris = 0;
    for (const it of job.items) { const g = it.mesh.geometry; totalTris += Math.floor((g.index ? g.index.count : g.attributes.position.count) / 3); }
    let done = 0, last = performance.now();
    const P = new Float64Array(9), uvA = new Float64Array(6), uvE = new Float64Array(6), vcA = new Float64Array(12);
    const groupIdx = new Map(job.groups.map((g, i) => [g, i]));
    const degEps = Math.pow(Math.max(size[0], size[1], size[2]) * sc * 1e-6, 4);   // |cross|² shu qiymatdan kichik bo‘lsa uchburchak yo‘q hisobi
    let mi = 0;
    for (const it of job.items) {
      mi++;
      job.status = 'Rang va shakl olinmoqda… (' + mi + '/' + job.items.length + ')';
      const mesh = it.mesh, geo = mesh.geometry, pos = geo.attributes.position, index = geo.index;
      const info = await getMatInfo(job, mesh);
      let mat = info.mat;
      const aB = mat.texBase ? geo.attributes[info.chBase === 0 ? 'uv' : 'uv' + info.chBase] : null;
      const aE = mat.emis && mat.texEmis ? geo.attributes[info.chEmis === 0 ? 'uv' : 'uv' + info.chEmis] : null;
      if ((mat.texBase && !aB) || (mat.texEmis && !aE)) mat = { ...mat, texBase: aB ? mat.texBase : null, texEmis: aE ? mat.texEmis : null };
      const tm = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
      const ca = tm && tm.vertexColors && geo.attributes.color ? geo.attributes.color : null;
      const e = mesh.matrixWorld.elements;
      const flip = mesh.matrixWorld.determinant() < 0;
      const gi = it.group ? groupIdx.get(it.group) : -1;
      const nT = Math.floor((index ? index.count : pos.count) / 3);
      for (let t = 0; t < nT; t++) {
        let ia = index ? index.getX(t * 3) : t * 3, ib = index ? index.getX(t * 3 + 1) : t * 3 + 1, ic = index ? index.getX(t * 3 + 2) : t * 3 + 2;
        if (flip) { const x = ib; ib = ic; ic = x; }
        const vs = [ia, ib, ic];
        for (let k = 0; k < 3; k++) {
          const x = pos.getX(vs[k]), y = pos.getY(vs[k]), z = pos.getZ(vs[k]);
          P[k * 3] = (e[0] * x + e[4] * y + e[8] * z + e[12]) * sc;
          P[k * 3 + 1] = (e[1] * x + e[5] * y + e[9] * z + e[13]) * sc;
          P[k * 3 + 2] = (e[2] * x + e[6] * y + e[10] * z + e[14]) * sc;
        }
        const ux = P[3] - P[0], uy = P[4] - P[1], uz = P[5] - P[2], wx = P[6] - P[0], wy = P[7] - P[1], wz = P[8] - P[2];
        const cx = uy * wz - uz * wy, cy = uz * wx - ux * wz, cz = ux * wy - uy * wx;
        if (cx * cx + cy * cy + cz * cz < degEps) { done++; continue; }   // yuzasi nolga teng uchburchak
        if (aB) for (let k = 0; k < 3; k++) { uvA[k * 2] = aB.getX(vs[k]); uvA[k * 2 + 1] = aB.getY(vs[k]); }
        if (aE) for (let k = 0; k < 3; k++) { uvE[k * 2] = aE.getX(vs[k]); uvE[k * 2 + 1] = aE.getY(vs[k]); }
        if (ca) for (let k = 0; k < 3; k++) { vcA[k * 4] = ca.getX(vs[k]); vcA[k * 4 + 1] = ca.getY(vs[k]); vcA[k * 4 + 2] = ca.getZ(vs[k]); vcA[k * 4 + 3] = ca.itemSize > 3 ? ca.getW(vs[k]) : 1; }
        processTriangle(P, aB ? uvA : null, aE ? uvE : null, ca ? vcA : null, mat, opt, sink, gi);
        done++;
        if ((t & 127) === 0 && performance.now() - last > 60) {
          job.progress = done / totalTris; updateJobDom(job); last = performance.now(); await tick();
        }
      }
    }
    if (!sink.n) throw new Error('Hech qanday yuza olinmadi');

    // Aylanadigan guruhlar: markaz va o‘q
    const gfaces = new Map();
    for (let f = 0; f < sink.n; f++) { const g = sink.G[f]; if (g >= 0) { if (!gfaces.has(g)) gfaces.set(g, []); gfaces.get(g).push(f); } }
    const gInfo = [], remap = new Map();
    for (const [oldI, faces] of gfaces) {
      const g = job.groups[oldI];
      const pts = [];
      const step = Math.max(1, Math.floor(faces.length / 20000));
      let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
      for (let q = 0; q < faces.length; q++) {
        const f = faces[q];
        for (let k = 0; k < 3; k++) for (let c = 0; c < 3; c++) { const v = sink.P[f * 9 + k * 3 + c]; if (v < mn[c]) mn[c] = v; if (v > mx[c]) mx[c] = v; }
        if (q % step === 0) for (let k = 0; k < 9; k++) pts.push(sink.P[f * 9 + k]);
      }
      let dir;
      if (g.axisMode === 'x') dir = [1, 0, 0]; else if (g.axisMode === 'y') dir = [0, 1, 0]; else if (g.axisMode === 'z') dir = [0, 0, 1];
      else if (g.axisMode === 'glb') { const v = new THREE.Vector3(0, 1, 0).transformDirection(g.node.matrixWorld); dir = [v.x, v.y, v.z]; }
      else dir = diskNormal(pts);
      const sn = snapAxis(dir);
      if (sn.dev > 10) job.notes.push('“' + g.key + '”: aylanish o‘qi x/y/z ga to‘g‘ri kelmaydi (' + sn.dev.toFixed(0) + '° farq). O‘qni qo‘lda tanlang.');
      const o = new THREE.Vector3().setFromMatrixPosition(g.node.matrixWorld).multiplyScalar(sc);
      const org = [o.x, o.y, o.z], ctr = [(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2];
      let inside = true;
      for (let c = 0; c < 3; c++) { const m = (mx[c] - mn[c]) * 0.15; if (org[c] < mn[c] - m || org[c] > mx[c] + m) inside = false; }
      remap.set(oldI, gInfo.length);
      gInfo.push({ key: g.key, pivot: inside ? org : ctr, axis: sn.axis });
    }
    for (let f = 0; f < sink.n; f++) if (sink.G[f] >= 0) sink.G[f] = remap.get(sink.G[f]);

    job.status = 'JSON yozilmoqda…'; job.progress = 1; updateJobDom(job); await tick();
    const res = buildSpecText(sink, gInfo, opt, job.base);
    res.faceSplit = sink.n;
    res.ms = performance.now() - t0;
    res.groups = gInfo.length;
    job.result = res;
    job.status = 'Tayyor: ' + job.base + '.json (' + fmtBytes(res.text.length) + ')';
    job.statusClass = 'ok';
    if (sink.n > opt.maxFaces) job.notes.push('Yuzalar soni chegaradan oshdi: keyingi uchburchaklar bo‘linmadi.');
    if (res.text.length > 12 * 1048576) job.notes.push('Fayl katta (' + fmtBytes(res.text.length) + '). O‘yinda yuklash sekin bo‘lsa “Yaxshi” yoki “Tez” ni tanlang.');
  } catch (err) {
    console.error(err);
    job.status = 'Xatolik: ' + (err && err.message ? err.message : err);
    job.statusClass = 'bad';
  }
  job.busy = false; job.progress = -1;
  renderJobs();
}

function downloadJob(job) {
  if (!job.result) return;
  const blob = new Blob([job.result.text], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = job.base + '.json';
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 20000);
}

function removeJob(job) {
  if (viewer.job === job) closeViewer();
  const i = jobs.indexOf(job); if (i >= 0) jobs.splice(i, 1);
  if (job.gltf) {
    job.gltf.scene.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      const ms = Array.isArray(o.material) ? o.material : o.material ? [o.material] : [];
      for (const m of ms) { for (const k of Object.keys(m)) { const v = m[k]; if (v && v.isTexture) { if (v.image && v.image.close) v.image.close(); v.dispose(); } } m.dispose(); }
    });
  }
  job.imgCache.clear(); job.matCache.clear(); job.gltf = null; job.result = null; job.previewJson = null;
  renderJobs();
}

// ---------- Interfeys ----------
function jobHtml(job) {
  const i = job.info;
  let h = '<div class="job-head"><div class="job-name">' + esc(job.file.name) + '</div><div class="job-size">' + fmtBytes(job.file.size) + '</div></div>';
  h += '<div class="job-status ' + job.statusClass + '" data-st>' + esc(job.status) + '</div>';
  h += '<div class="bar" data-bar' + (job.progress < 0 ? ' hidden' : '') + '><i style="width:' + Math.round(Math.max(0, job.progress) * 100) + '%"></i></div>';
  if (i) {
    h += '<div class="info"><div>Uchburchak: <b>' + fmtNum(i.tris) + '</b></div><div>Mesh: <b>' + i.meshes + '</b></div><div>Material: <b>' + i.materials + '</b></div><div>Tekstura: <b>' + i.textures + '</b></div><div>O‘lcham: <b>' + i.size.map(fmtDim).join(' × ') + '</b></div>';
    if (job.result) {
      const r = job.result;
      h += '<div>JSON yuza: <b>' + fmtNum(r.faces) + '</b></div><div>Uchlar: <b>' + fmtNum(r.verts) + '</b></div><div>Qism: <b>' + r.parts + '</b></div><div>Palitra: <b>' + r.palette + '</b> rang</div><div>Rang xatosi: <b>' + r.colorError.toFixed(2) + '</b>/255</div><div>Vaqt: <b>' + (r.ms / 1000).toFixed(1) + '</b> s</div><div>Koordinata: <b>' + r.decimals + '</b> raqam</div>';
    }
    h += '</div>';
  }
  if (job.groups.length) {
    h += '<div class="groups"><h3>Aylanadigan qismlar (o‘yinda parrak sifatida aylanadi)</h3>';
    for (const g of job.groups) {
      const idx = job.groups.indexOf(g);
      h += '<div class="grp"><div><div class="gname">' + esc(g.key) + '</div><div class="gmeta">' + g.items.length + ' ta mesh</div></div>' +
        '<select data-act="axis" data-gi="' + idx + '">' +
        [['auto', 'Avto (disk normali)'], ['glb', 'GLB (tugun Y o‘qi)'], ['x', 'X o‘qi'], ['y', 'Y o‘qi'], ['z', 'Z o‘qi']].map(([v, t]) => '<option value="' + v + '"' + (g.axisMode === v ? ' selected' : '') + '>' + t + '</option>').join('') + '</select></div>';
    }
    h += '</div>';
  }
  for (const n of job.notes) h += '<div class="warn">' + esc(n) + '</div>';
  h += '<div class="actions">' +
    '<button class="btn primary" data-act="convert"' + (job.busy || !job.gltf ? ' disabled' : '') + '>Aylantirish</button>' +
    '<button class="btn" data-act="download"' + (job.result ? '' : ' disabled') + '>Yuklab olish (' + esc(job.base) + '.json)</button>' +
    '<button class="btn" data-act="view"' + (job.gltf && !job.busy ? '' : ' disabled') + '>Ko‘rish</button>' +
    '<button class="btn ghost" data-act="remove"' + (job.busy ? ' disabled' : '') + '>Olib tashlash</button></div>';
  return h;
}
function renderJobs() {
  const box = $('jobs');
  const keep = new Map();
  for (const job of jobs) {
    let el = box.querySelector('[data-job="' + job.id + '"]');
    if (!el) { el = document.createElement('div'); el.className = 'job'; el.dataset.job = job.id; box.appendChild(el); }
    el.innerHTML = jobHtml(job);
    keep.set(String(job.id), el);
  }
  [...box.children].forEach((el) => { if (!keep.has(el.dataset.job)) el.remove(); });
}
function updateJobDom(job) {
  const el = $('jobs').querySelector('[data-job="' + job.id + '"]');
  if (!el) return;
  const st = el.querySelector('[data-st]'); if (st) st.textContent = job.status;
  const bar = el.querySelector('[data-bar]');
  if (bar) { bar.hidden = job.progress < 0; const i = bar.firstElementChild; if (i) i.style.width = Math.round(Math.max(0, job.progress) * 100) + '%'; }
}
$('jobs').addEventListener('click', (e) => {
  const b = e.target.closest('[data-act]'); if (!b || b.tagName !== 'BUTTON') return;
  const el = b.closest('[data-job]'); const job = jobs.find((j) => String(j.id) === el.dataset.job); if (!job) return;
  const act = b.dataset.act;
  if (act === 'convert') convertJob(job);
  else if (act === 'download') downloadJob(job);
  else if (act === 'remove') removeJob(job);
  else if (act === 'view') openViewer(job);
});
$('jobs').addEventListener('change', (e) => {
  const s = e.target.closest('select[data-act="axis"]'); if (!s) return;
  const el = s.closest('[data-job]'); const job = jobs.find((j) => String(j.id) === el.dataset.job); if (!job) return;
  job.groups[+s.dataset.gi].axisMode = s.value;
});

// ---------- Ko‘rish (GLB va hosil bo‘lgan JSON) ----------
const viewer = { inited: false, job: null, mode: 'json', spin: false, open: false, spinNodes: [], root: null, glbObj: null };

function initViewer() {
  if (viewer.inited) return;
  const canvas = $('viewCanvas');
  viewer.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  viewer.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  viewer.scene = new THREE.Scene();
  viewer.scene.background = new THREE.Color(0x151a24);
  viewer.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
  viewer.controls = new OrbitControls(viewer.camera, canvas);
  viewer.controls.enableDamping = true;
  const pm = new THREE.PMREMGenerator(viewer.renderer);
  viewer.scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture;
  viewer.scene.add(new THREE.AmbientLight(0xffffff, 0.9));
  const dl = new THREE.DirectionalLight(0xffffff, 1.6); dl.position.set(3, 6, 4); viewer.scene.add(dl);
  viewer.scene.add(new THREE.HemisphereLight(0xdfeaff, 0x303845, 0.7));
  viewer.root = new THREE.Group(); viewer.scene.add(viewer.root);
  viewer.clock = new THREE.Clock();
  window.addEventListener('resize', () => { if (viewer.open) resizeViewer(); });
  viewer.inited = true;
}
function resizeViewer() {
  const c = $('viewCanvas'); const w = c.clientWidth, h = c.clientHeight;
  if (!w || !h) return;
  viewer.renderer.setSize(w, h, false);
  viewer.camera.aspect = w / h; viewer.camera.updateProjectionMatrix();
}
function viewLoop() {
  if (!viewer.open) return;
  requestAnimationFrame(viewLoop);
  const dt = Math.min(0.05, viewer.clock.getDelta());
  if (viewer.spin) for (const n of viewer.spinNodes) n.rotation.y += dt * 10;
  viewer.controls.update();
  viewer.renderer.render(viewer.scene, viewer.camera);
}
function fitView(obj) {
  const box = new THREE.Box3().setFromObject(obj);
  if (box.isEmpty()) return;
  const c = box.getCenter(new THREE.Vector3()), s = box.getSize(new THREE.Vector3()), r = Math.max(s.x, s.y, s.z) || 1;
  viewer.camera.near = r / 500; viewer.camera.far = r * 50; viewer.camera.updateProjectionMatrix();
  viewer.camera.position.copy(c).add(new THREE.Vector3(0.9, 0.55, 1.1).multiplyScalar(r * 1.15));
  viewer.controls.target.copy(c); viewer.controls.update();
}

// Preview: o‘yindagi loader bilan bir xil mantiq (silliqlash, palitra, shisha, aylanish guruhlari)
function previewMeshGeometry(P) {
  const v = P.v || [], f = P.f || [];
  const col = new THREE.Color();
  const pal = Array.isArray(P.pal) && Array.isArray(P.fc) && P.pal.length ? P.pal.map((h) => { col.set(h); return [col.r, col.g, col.b]; }) : null;
  col.set(typeof P.c === 'string' ? P.c : '#c8c8c8');
  const single = [col.r, col.g, col.b];
  const nt = Math.floor(f.length / 3);
  const fn = new Float32Array(nt * 3), adj = new Array(v.length / 3);
  for (let t = 0; t < nt; t++) {
    const a = f[t * 3] * 3, b = f[t * 3 + 1] * 3, c = f[t * 3 + 2] * 3;
    const ux = v[b] - v[a], uy = v[b + 1] - v[a + 1], uz = v[b + 2] - v[a + 2], wx = v[c] - v[a], wy = v[c + 1] - v[a + 1], wz = v[c + 2] - v[a + 2];
    fn[t * 3] = uy * wz - uz * wy; fn[t * 3 + 1] = uz * wx - ux * wz; fn[t * 3 + 2] = ux * wy - uy * wx;
    for (let k = 0; k < 3; k++) { const vi = f[t * 3 + k]; (adj[vi] || (adj[vi] = [])).push(t); }
  }
  const ca = P.ca === undefined ? 40 : P.ca, cosT = Math.cos(Math.max(0, Math.min(180, ca)) * Math.PI / 180), smooth = ca > 0.5;
  const pos = new Float32Array(nt * 9), nor = new Float32Array(nt * 9), cl = new Float32Array(nt * 9);
  for (let t = 0; t < nt; t++) {
    const tx = fn[t * 3], ty = fn[t * 3 + 1], tz = fn[t * 3 + 2], tl0 = Math.hypot(tx, ty, tz), tl = tl0 || 1;
    const c = pal ? pal[P.fc[t] % pal.length] : single;
    for (let k = 0; k < 3; k++) {
      const vi = f[t * 3 + k], o = t * 9 + k * 3;
      pos[o] = v[vi * 3]; pos[o + 1] = v[vi * 3 + 1]; pos[o + 2] = v[vi * 3 + 2];
      let nx = tx / tl, ny = ty / tl, nz = tz / tl;
      if (tl0 < 1e-12) { nx = 0; ny = 1; nz = 0; }
      else if (smooth) {
        let sx = 0, sy = 0, sz = 0;
        for (const u of adj[vi]) {
          const ux = fn[u * 3], uy = fn[u * 3 + 1], uz = fn[u * 3 + 2], ul = Math.hypot(ux, uy, uz);
          if (ul < 1e-12) continue;
          if ((ux * tx + uy * ty + uz * tz) / (ul * tl) >= cosT) { sx += ux; sy += uy; sz += uz; }
        }
        const sl = Math.hypot(sx, sy, sz);
        if (sl > 1e-12) { nx = sx / sl; ny = sy / sl; nz = sz / sl; }
      }
      nor[o] = nx; nor[o + 1] = ny; nor[o + 2] = nz;
      cl[o] = c[0]; cl[o + 1] = c[1]; cl[o + 2] = c[2];
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  g.setAttribute('color', new THREE.BufferAttribute(cl, 3));
  return g;
}
function buildJsonPreview(spec) {
  const root = new THREE.Group();
  const solid = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
  const glassMats = new Map();
  const defs = spec.groups || {};
  const holders = new Map();
  const spins = [];
  for (const P of spec.parts || []) {
    const geo = previewMeshGeometry(P);
    const key = typeof P.g === 'string' ? P.g : '';
    if (!key) {
      let mat = solid;
      if (P.o !== undefined && P.o < 1) {
        const o = Math.max(0.02, Math.min(0.98, Math.round(P.o * 100) / 100));
        if (!glassMats.has(o)) glassMats.set(o, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, transparent: true, opacity: o, depthWrite: false }));
        mat = glassMats.get(o);
      }
      const m = new THREE.Mesh(geo, mat); if (mat !== solid) m.renderOrder = 2;
      root.add(m);
      continue;
    }
    if (!holders.has(key)) {
      const d = defs[key] || {}, p = d.p || [0, 0, 0], axis = String(d.axis || 'y').toLowerCase();
      const holder = new THREE.Group(); holder.position.set(p[0], p[1], p[2]);
      holder.rotation.set(axis === 'z' ? Math.PI / 2 : 0, 0, axis === 'x' ? -Math.PI / 2 : 0);
      holder.updateMatrix();
      const spin = new THREE.Group(); holder.add(spin); root.add(holder);
      spins.push(spin);
      holders.set(key, { holder, spin, p, inv: new THREE.Matrix4().copy(holder.matrix).setPosition(0, 0, 0).invert() });
    }
    const h = holders.get(key);
    geo.translate(-h.p[0], -h.p[1], -h.p[2]); geo.applyMatrix4(h.inv);
    h.spin.add(new THREE.Mesh(geo, solid));
  }
  root.userData.spinNodes = spins;
  return root;
}

function setViewMode(mode) {
  const job = viewer.job; if (!job) return;
  if (mode === 'json' && !job.result) mode = 'glb';
  viewer.mode = mode;
  while (viewer.root.children.length) viewer.root.remove(viewer.root.children[0]);
  viewer.spinNodes = [];
  let obj;
  if (mode === 'json') {
    if (!job.previewJson) job.previewJson = buildJsonPreview(JSON.parse(job.result.text));
    obj = job.previewJson;
    viewer.spinNodes = obj.userData.spinNodes || [];
  } else {
    obj = job.gltf.scene;
  }
  viewer.root.add(obj);
  fitView(obj);
  $('vGlb').classList.toggle('active', mode === 'glb');
  $('vJson').classList.toggle('active', mode === 'json');
  $('vJson').disabled = !job.result;
  $('vSpin').disabled = mode !== 'json' || !viewer.spinNodes.length;
  $('vSpin').classList.toggle('active', viewer.spin);
}
function openViewer(job) {
  initViewer();
  viewer.job = job; viewer.open = true; viewer.spin = false;
  $('viewer').hidden = false;
  $('viewerTitle').textContent = job.file.name;
  resizeViewer();
  setViewMode(job.result ? 'json' : 'glb');
  viewer.clock.getDelta();
  viewLoop();
}
function closeViewer() {
  viewer.open = false; viewer.job = null;
  $('viewer').hidden = true;
  if (viewer.root) while (viewer.root.children.length) viewer.root.remove(viewer.root.children[0]);
}
$('vGlb').addEventListener('click', () => setViewMode('glb'));
$('vJson').addEventListener('click', () => setViewMode('json'));
$('vSpin').addEventListener('click', () => { viewer.spin = !viewer.spin; $('vSpin').classList.toggle('active', viewer.spin); });
$('vClose').addEventListener('click', closeViewer);
