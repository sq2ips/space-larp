import { useEffect, useMemo, useRef, useState } from "react";
import { API } from "./utils";

const WAVELENGTHS = [
  0.75, 1.0, 1.3, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 4.4, 4.8, 5.0,
];
const DEFAULT_WL_IDX = WAVELENGTHS.indexOf(1.7);
const FOV_MIN = 0.1;
const FOV_MAX = 10;
const DEBOUNCE_MS = 700;
const MAX_PARALLEL = 4;
const CACHE_NAME = "skybox-v1";

const DS_LEFT = { id: "dr1", label: "February 2026", color: "#7dd3fc" };
const DS_RIGHT = { id: "qr3", label: "QR3", color: "#fbbf24" };

type CMap = "grey" | "inferno" | "spectral";

interface PostcardOptions {
  title: string;
  subtitle: string;
  showCoords: boolean;
  showWavelength: boolean;
  showScale: boolean;
  showTimestamp: boolean;
  showNasaLogo: boolean;
  showFavicon: boolean;
  renderMode: "composite" | "active" | "left" | "right";
  resolution: "1200x900" | "1600x1200" | "2400x1800";
}

const clamp = (v: number, lo: number, hi: number) =>
  Math.min(hi, Math.max(lo, v));
const wrap360 = (a: number) => ((a % 360) + 360) % 360;

const parseNum = (t: string) => {
  const v = parseFloat(t);
  return Number.isFinite(v) ? v : 0;
};
const isNumberish = (t: string) => /^-?\d*\.?\d*$/.test(t);
const RAD = Math.PI / 180;

function unproject(ra0: number, dec0: number, xi: number, eta: number) {
  const d0 = dec0 * RAD;
  const rho = Math.hypot(xi, eta);
  if (rho < 1e-12) return { ra: ra0, dec: dec0 };
  const c = Math.atan(rho);
  const dec = Math.asin(
    Math.cos(c) * Math.sin(d0) + (eta * Math.sin(c) * Math.cos(d0)) / rho,
  );
  const ra =
    ra0 * RAD +
    Math.atan2(
      xi * Math.sin(c),
      rho * Math.cos(d0) * Math.cos(c) - eta * Math.sin(d0) * Math.sin(c),
    );
  return { ra: wrap360(ra / RAD), dec: dec / RAD };
}

interface Meta {
  px: number;
  py: number;
  w: number;
  h: number;
  side: number;
  rot: number;
  ra: number;
  dec: number;
  fromServer: boolean;
}

function parseMeta(
  h: Headers,
  img: HTMLImageElement,
  req: { ra: number; dec: number; fov: number },
): Meta {
  const num = (k: string) => {
    const v = h.get(k);
    const n = v === null || v === "" ? NaN : parseFloat(v);
    return Number.isFinite(n) ? n : NaN;
  };
  const w = img.naturalWidth;
  const ht = img.naturalHeight;
  const px = num("x-point-x");
  const py = num("x-point-y");
  const side = num("x-tile-size-deg");
  const rot = num("x-rotation-deg");
  return {
    px: Number.isFinite(px) ? px : w / 2,
    py: Number.isFinite(py) ? py : ht / 2,
    side: Number.isFinite(side) ? side : req.fov,
    rot: Number.isFinite(rot) ? rot : 0,
    ra: req.ra,
    dec: req.dec,
    fromServer: Number.isFinite(px) && Number.isFinite(py),
    w,
    h: ht,
  };
}

const snapFov = (raw: number) =>
  clamp(Math.round(raw * 10) / 10, FOV_MIN, FOV_MAX);

const buildUrl = (
  ds: string,
  ra: number,
  dec: number,
  fov: number,
  wl: number,
) =>
  `${API}/cutout?ra=${ra.toFixed(5)}&dec=${dec.toFixed(5)}&size=${fov.toFixed(1)}` +
  `&wavelength=${wl}&dataset=${ds}`;

function useDebounced<T>(value: T, ms: number) {
  const [d, setD] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setD(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return d;
}

interface Transform {
  fx: boolean;
  fy: boolean;
  angle: number;
  x: number;
  y: number;
  s: number;
}

const IDENTITY: Transform = {
  fx: false,
  fy: false,
  angle: 0,
  x: 0,
  y: 0,
  s: 1,
};

const wrapAngle = (a: number) => {
  const w = ((((a + 180) % 360) + 360) % 360) - 180;
  return w === -180 ? 180 : w;
};

function invTransform(t: Transform, fx: number, fy: number) {
  const th = (t.angle * Math.PI) / 180;
  let x = fx - 0.5 - t.x / 100;
  let y = fy - 0.5 - t.y / 100;
  [x, y] = [
    x * Math.cos(th) + y * Math.sin(th),
    -x * Math.sin(th) + y * Math.cos(th),
  ];
  x /= t.fx ? -t.s : t.s;
  y /= t.fy ? -t.s : t.s;
  return { u: clamp(x + 0.5, 0, 1), v: clamp(y + 0.5, 0, 1) };
}

function forwardTransform(t: Transform, u: number, v: number) {
  let x = (u - 0.5) * (t.fx ? -t.s : t.s);
  let y = (v - 0.5) * (t.fy ? -t.s : t.s);
  const th = (t.angle * Math.PI) / 180;
  const rx = x * Math.cos(th) - y * Math.sin(th);
  const ry = x * Math.sin(th) + y * Math.cos(th);
  return {
    fx: rx + 0.5 + t.x / 100,
    fy: ry + 0.5 + t.y / 100,
  };
}

const transformCss = (t: Transform, animate = true): React.CSSProperties => ({
  transform: `translate(${t.x}%, ${t.y}%) rotate(${t.angle}deg) scale(${t.fx ? -t.s : t.s}, ${t.fy ? -t.s : t.s})`,
  transformOrigin: "center",
  transition: animate ? "transform 120ms ease-out" : "none",
});

const DB_NAME = "skybox";
const DB_STORE = "transforms";
let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = () => req.result.createObjectStore(DB_STORE);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  return dbPromise;
}

async function dbGet(key: string): Promise<Transform | null> {
  try {
    const db = await openDb();
    return await new Promise((resolve, reject) => {
      const r = db.transaction(DB_STORE).objectStore(DB_STORE).get(key);
      r.onsuccess = () => {
        const v = r.result as
          (Partial<Transform> & { rot?: number }) | undefined;
        resolve(
          v ? { ...IDENTITY, ...v, angle: v.angle ?? (v.rot ?? 0) * 90 } : null,
        );
      };
      r.onerror = () => reject(r.error);
    });
  } catch {
    return null;
  }
}

async function dbPut(key: string, value: Transform) {
  try {
    const db = await openDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(DB_STORE, "readwrite");
      tx.objectStore(DB_STORE).put(value, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch {
    // fallback
  }
}

async function dbDelete(key: string) {
  try {
    const db = await openDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(DB_STORE, "readwrite");
      tx.objectStore(DB_STORE).delete(key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch {
    // fallback
  }
}

const memory = new Map<string, { obj: string; meta: Meta }>();
const inflight = new Map<string, Promise<{ obj: string; meta: Meta }>>();
let active = 0;
const waiting: (() => void)[] = [];

async function limited<T>(fn: () => Promise<T>): Promise<T> {
  if (active >= MAX_PARALLEL) await new Promise<void>((r) => waiting.push(r));
  active++;
  try {
    return await fn();
  } finally {
    active--;
    waiting.shift()?.();
  }
}

function loadImage(
  url: string,
  req: { ra: number; dec: number; fov: number },
): Promise<{ obj: string; meta: Meta }> {
  const hit = memory.get(url);
  if (hit) return Promise.resolve(hit);
  const running = inflight.get(url);
  if (running) return running;
  const p = (async () => {
    let cache: Cache | null = null;
    try {
      if (typeof caches !== "undefined") cache = await caches.open(CACHE_NAME);
    } catch {
      cache = null;
    }
    let res = cache ? await cache.match(url) : undefined;
    if (!res) {
      res = await limited(() => fetch(url, { mode: "cors" }));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      try {
        await cache?.put(url, res.clone());
      } catch {
        // quota
      }
    }
    const headers = res.headers;
    const obj = URL.createObjectURL(await res.blob());
    const img = new Image();
    img.src = obj;
    await img.decode();
    const out = { obj, meta: parseMeta(headers, img, req) };
    memory.set(url, out);
    return out;
  })().finally(() => inflight.delete(url));
  inflight.set(url, p);
  return p;
}

async function decodeUrl(obj: string): Promise<HTMLImageElement> {
  const img = new Image();
  img.src = obj;
  await img.decode();
  return img;
}

const INFERNO = [
  [0, 0, 4],
  [87, 16, 110],
  [188, 55, 84],
  [249, 142, 9],
  [252, 255, 164],
];

function infernoAt(t: number): [number, number, number] {
  const x = clamp(t, 0, 1) * (INFERNO.length - 1);
  const i = Math.min(Math.floor(x), INFERNO.length - 2);
  const f = x - i;
  return [0, 1, 2].map(
    (k) => INFERNO[i][k] + (INFERNO[i + 1][k] - INFERNO[i][k]) * f,
  ) as [number, number, number];
}

function wavelengthRgb(wl: number): [number, number, number] {
  const t = clamp(
    (wl - WAVELENGTHS[0]) /
      (WAVELENGTHS[WAVELENGTHS.length - 1] - WAVELENGTHS[0]),
    0,
    1,
  );
  const h = (1 - t) * 250;
  const s = 0.9,
    l = 0.55;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => {
    const k = (n + h / 30) % 12;
    return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))));
  };
  return [f(0), f(8), f(4)];
}

const colourCache = new Map<string, string>();
async function colourise(raw: string, cmap: CMap, wl: number): Promise<string> {
  if (cmap === "grey") return raw;
  const key = `${raw}|${cmap}|${cmap === "spectral" ? wl : ""}`;
  const hit = colourCache.get(key);
  if (hit) return hit;
  const img = await decodeUrl(raw);
  const c = document.createElement("canvas");
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const ctx = c.getContext("2d")!;
  ctx.drawImage(img, 0, 0);
  const d = ctx.getImageData(0, 0, c.width, c.height);
  const lut = new Uint8ClampedArray(256 * 3);
  const tint = wavelengthRgb(wl);
  for (let v = 0; v < 256; v++) {
    const rgb =
      cmap === "inferno" ? infernoAt(v / 255) : tint.map((x) => (x * v) / 255);
    lut[v * 3] = rgb[0];
    lut[v * 3 + 1] = rgb[1];
    lut[v * 3 + 2] = rgb[2];
  }
  for (let i = 0; i < d.data.length; i += 4) {
    const v = d.data[i];
    d.data[i] = lut[v * 3];
    d.data[i + 1] = lut[v * 3 + 1];
    d.data[i + 2] = lut[v * 3 + 2];
  }
  ctx.putImageData(d, 0, 0);
  const blob = await new Promise<Blob>((res, rej) =>
    c.toBlob(
      (b) => (b ? res(b) : rej(new Error("toBlob failed"))),
      "image/png",
    ),
  );
  const out = URL.createObjectURL(blob);
  colourCache.set(key, out);
  return out;
}

const SLICE_N = 128;
interface Analysis {
  grid: Uint8Array;
}

function analyse(img: HTMLImageElement): Analysis {
  const N = SLICE_N;
  const c = document.createElement("canvas");
  c.width = c.height = N;
  const ctx = c.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0, N, N);
  const d = ctx.getImageData(0, 0, N, N).data;
  const grid = new Uint8Array(N * N);
  for (let i = 0; i < N * N; i++) grid[i] = d[i * 4];
  return { grid };
}

function sampleMean(a: Analysis, u: number, v: number): number {
  const N = SLICE_N;
  const cx = u * N,
    cy = v * N,
    r = N * 0.06;
  let sum = 0,
    n = 0;
  for (
    let y = Math.max(0, Math.floor(cy - r));
    y <= Math.min(N - 1, Math.ceil(cy + r));
    y++
  )
    for (
      let x = Math.max(0, Math.floor(cx - r));
      x <= Math.min(N - 1, Math.ceil(cx + r));
      x++
    )
      if (Math.hypot(x + 0.5 - cx, y + 0.5 - cy) <= r) {
        sum += a.grid[y * N + x];
        n++;
      }
  return n
    ? sum / n
    : a.grid[
        Math.min(N - 1, Math.floor(cy)) * N + Math.min(N - 1, Math.floor(cx))
      ];
}

function sliceRow(a: Analysis, v: number): Float32Array {
  const N = SLICE_N;
  const row = clamp(Math.round(v * N - 0.5), 0, N - 1);
  const y0 = Math.max(0, row - 2),
    y1 = Math.min(N - 1, row + 2);
  const out = new Float32Array(N);
  for (let x = 0; x < N; x++) {
    let s = 0;
    for (let y = y0; y <= y1; y++) s += a.grid[y * N + x];
    out[x] = s / (y1 - y0 + 1);
  }
  return out;
}

const PW = 560,
  PH = 260;
const M = { l: 58, r: 62, t: 16, b: 42 };
const W_MIN = WAVELENGTHS[0],
  W_MAX = WAVELENGTHS[WAVELENGTHS.length - 1];

function setupCanvas(c: HTMLCanvasElement) {
  const dpr = window.devicePixelRatio || 1;
  c.width = PW * dpr;
  c.height = PH * dpr;
  const ctx = c.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#0a0e17";
  ctx.fillRect(0, 0, PW, PH);
  ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
  return ctx;
}

function drawXAxis(ctx: CanvasRenderingContext2D) {
  const w = PW - M.l - M.r;
  ctx.fillStyle = "#94a3b8";
  ctx.strokeStyle = "#475569";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let v = 1; v <= 5; v++) {
    const x = M.l + ((v - W_MIN) / (W_MAX - W_MIN)) * w;
    ctx.beginPath();
    ctx.moveTo(x, PH - M.b);
    ctx.lineTo(x, PH - M.b + 4);
    ctx.stroke();
    ctx.fillText(String(v), x, PH - M.b + 6);
  }
  ctx.fillText("Wavelength (µm)", M.l + w / 2, PH - 16);
}

function drawEmpty(ctx: CanvasRenderingContext2D, text: string) {
  ctx.fillStyle = "#64748b";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, PW / 2, PH / 2);
}

function drawSpectrogram(
  c: HTMLCanvasElement,
  cols: (Float32Array | null)[],
  fovArcsec: number,
  curWl: number,
  mark: number | null,
) {
  const ctx = setupCanvas(c);
  const w = PW - M.l - M.r,
    h = PH - M.t - M.b;
  const have = cols.filter(Boolean) as Float32Array[];
  if (!have.length) return drawEmpty(ctx, "Waiting for layers…");
  let lo = Infinity,
    hi = -Infinity;
  for (const a of have)
    for (const v of a) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
  if (hi - lo < 1) hi = lo + 1;
  const img = ctx.createImageData(w, h);
  for (let px = 0; px < w; px++) {
    const wl = W_MIN + (px / (w - 1)) * (W_MAX - W_MIN);
    let i = WAVELENGTHS.findIndex(
      (_, k) => k < WAVELENGTHS.length - 1 && wl <= WAVELENGTHS[k + 1],
    );
    if (i < 0) i = WAVELENGTHS.length - 2;
    const f = (wl - WAVELENGTHS[i]) / (WAVELENGTHS[i + 1] - WAVELENGTHS[i]);
    const A = cols[i],
      B = cols[i + 1];
    for (let py = 0; py < h; py++) {
      const si = Math.min(SLICE_N - 1, Math.floor((py / h) * SLICE_N));
      let v: number | null = null;
      if (A && B) v = A[si] * (1 - f) + B[si] * f;
      else if (A && f < 0.5) v = A[si];
      else if (B && f >= 0.5) v = B[si];
      const o = (py * w + px) * 4;
      if (v === null) img.data[o] = img.data[o + 1] = img.data[o + 2] = 22;
      else {
        const [r, g, b] = infernoAt((v - lo) / (hi - lo));
        img.data[o] = r;
        img.data[o + 1] = g;
        img.data[o + 2] = b;
      }
      img.data[o + 3] = 255;
    }
  }
  const tmp = document.createElement("canvas");
  tmp.width = w;
  tmp.height = h;
  tmp.getContext("2d")!.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(tmp, M.l, M.t, w, h);
  ctx.strokeStyle = "#475569";
  ctx.strokeRect(M.l + 0.5, M.t + 0.5, w, h);
  ctx.fillStyle = "#94a3b8";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let k = 0; k <= 4; k++) {
    const y = M.t + (h * k) / 4;
    const off = fovArcsec / 2 - (fovArcsec * k) / 4;
    ctx.beginPath();
    ctx.moveTo(M.l - 4, y);
    ctx.lineTo(M.l, y);
    ctx.stroke();
    ctx.fillText(off.toFixed(0), M.l - 7, y);
  }
  ctx.save();
  ctx.translate(14, M.t + h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("Slice offset (arcsec)", 0, 0);
  ctx.restore();
  drawXAxis(ctx);
  ctx.strokeStyle = "rgba(255,255,255,.18)";
  for (const wv of WAVELENGTHS) {
    const x = M.l + ((wv - W_MIN) / (W_MAX - W_MIN)) * w;
    ctx.beginPath();
    ctx.moveTo(x, M.t + h - 5);
    ctx.lineTo(x, M.t + h);
    ctx.stroke();
  }
  const cx = M.l + ((curWl - W_MIN) / (W_MAX - W_MIN)) * w;
  ctx.strokeStyle = "rgba(125,211,252,.9)";
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(cx, M.t);
  ctx.lineTo(cx, M.t + h);
  ctx.stroke();
  ctx.setLineDash([]);
  if (mark !== null) {
    const my = M.t + clamp(mark, 0, 1) * h;
    ctx.strokeStyle = "rgba(255,255,255,.85)";
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(M.l, my);
    ctx.lineTo(M.l + w, my);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.moveTo(M.l - 1, my);
    ctx.lineTo(M.l - 7, my - 4);
    ctx.lineTo(M.l - 7, my + 4);
    ctx.fill();
  }
  const bx = PW - M.r + 14,
    bw = 10;
  for (let y = 0; y < h; y++) {
    const [r, g, b] = infernoAt(1 - y / h);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(bx, M.t + y, bw, 1);
  }
  ctx.strokeStyle = "#475569";
  ctx.strokeRect(bx + 0.5, M.t + 0.5, bw, h);
  ctx.fillStyle = "#94a3b8";
  ctx.textAlign = "left";
  ctx.fillText(hi.toFixed(0), bx + bw + 4, M.t + 4);
  ctx.fillText(lo.toFixed(0), bx + bw + 4, M.t + h - 4);
}

function drawSpectrum(
  c: HTMLCanvasElement,
  series: { name: string; color: string; vals: (number | null)[] }[],
  curWl: number,
) {
  const ctx = setupCanvas(c);
  const w = PW - M.l - M.r,
    h = PH - M.t - M.b;
  const all = series.flatMap((s) =>
    s.vals.filter((v): v is number => v !== null),
  );
  if (!all.length) return drawEmpty(ctx, "Waiting for layers…");
  let lo = Math.min(...all),
    hi = Math.max(...all);
  const pad = Math.max((hi - lo) * 0.12, 1);
  lo = Math.max(0, lo - pad);
  hi = Math.min(255, hi + pad);
  const X = (wl: number) => M.l + ((wl - W_MIN) / (W_MAX - W_MIN)) * w;
  const Y = (v: number) => M.t + h - ((v - lo) / (hi - lo)) * h;
  ctx.strokeStyle = "#1e293b";
  ctx.fillStyle = "#94a3b8";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let k = 0; k <= 4; k++) {
    const v = lo + ((hi - lo) * k) / 4;
    ctx.beginPath();
    ctx.moveTo(M.l, Y(v));
    ctx.lineTo(M.l + w, Y(v));
    ctx.stroke();
    ctx.fillText(v.toFixed(1), M.l - 7, Y(v));
  }
  for (let v = 1; v <= 5; v++) {
    ctx.beginPath();
    ctx.moveTo(X(v), M.t);
    ctx.lineTo(X(v), M.t + h);
    ctx.stroke();
  }
  ctx.strokeStyle = "#475569";
  ctx.strokeRect(M.l + 0.5, M.t + 0.5, w, h);
  ctx.save();
  ctx.translate(14, M.t + h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("Mean intensity (0–255)", 0, 0);
  ctx.restore();
  drawXAxis(ctx);
  ctx.strokeStyle = "rgba(125,211,252,.9)";
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(X(curWl), M.t);
  ctx.lineTo(X(curWl), M.t + h);
  ctx.stroke();
  ctx.setLineDash([]);
  series.forEach((s, si) => {
    ctx.strokeStyle = s.color;
    ctx.fillStyle = s.color;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    let started = false;
    s.vals.forEach((v, i) => {
      if (v === null) return;
      if (!started) ctx.moveTo(X(WAVELENGTHS[i]), Y(v));
      else ctx.lineTo(X(WAVELENGTHS[i]), Y(v));
      started = true;
    });
    ctx.stroke();
    s.vals.forEach((v, i) => {
      if (v === null) return;
      ctx.beginPath();
      ctx.arc(X(WAVELENGTHS[i]), Y(v), 3, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.lineWidth = 1;
    ctx.fillRect(M.l + 8 + si * 120, M.t + 8, 10, 3);
    ctx.fillStyle = "#cbd5e1";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(s.name, M.l + 22 + si * 120, M.t + 10);
  });
}

interface Region {
  ra: number;
  dec: number;
  fov: number;
}

export default function SkyBox({
  initialRa = 269.452077,
  initialDec = 4.69336497,
  initialFov = 0.5,
}: {
  initialRa?: number;
  initialDec?: number;
  initialFov?: number;
}) {
  const qp = useMemo(() => new URLSearchParams(window.location.search), []);
  const [raText, setRaText] = useState(qp.get("ra") ?? String(initialRa));
  const [decText, setDecText] = useState(qp.get("dec") ?? String(initialDec));
  const [fovText, setFovText] = useState(qp.get("fov") ?? String(initialFov));
  const [wlIdx, setWlIdx] = useState(() => {
    const w = parseFloat(qp.get("wl") ?? "");
    if (!Number.isFinite(w)) return DEFAULT_WL_IDX;
    return WAVELENGTHS.reduce(
      (best, v, i) =>
        Math.abs(v - w) < Math.abs(WAVELENGTHS[best] - w) ? i : best,
      0,
    );
  });
  const [time, setTime] = useState(() =>
    clamp(parseFloat(qp.get("time") ?? "0") || 0, 0, 1),
  );
  const [cmap, setCmap] = useState<CMap>(() => {
    const c = qp.get("cmap");
    return c === "grey" || c === "spectral" ? c : "inferno";
  });

  // Tutorial State
  const [showTutorial, setShowTutorial] = useState(() => {
    if (typeof window !== "undefined") {
      return !window.localStorage.getItem("skybox-tutorial");
    }
    return false;
  });
  const [tutorialTab, setTutorialTab] = useState(0);

  const dismissTutorial = () => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem("skybox-tutorial", "true");
    }
    setShowTutorial(false);
  };

  useEffect(() => {
    const p = new URLSearchParams(window.location.search);
    p.set("ra", raText);
    p.set("dec", decText);
    p.set("fov", fovText);
    p.set("wl", String(WAVELENGTHS[wlIdx]));
    p.set("time", time.toFixed(3));
    p.set("cmap", cmap);
    p.delete("objs");
    window.history.replaceState(
      null,
      "",
      `${window.location.pathname}?${p.toString()}`,
    );
  }, [raText, decText, fovText, wlIdx, time, cmap]);

  const ra = wrap360(parseNum(raText));
  const dec = clamp(parseNum(decText), -90, 90);
  const fovRaw = parseNum(fovText);
  const fov = snapFov(fovRaw);
  const wl = WAVELENGTHS[wlIdx];

  const paramKey = `${ra.toFixed(5)}|${dec.toFixed(5)}|${fov.toFixed(1)}|${wlIdx}`;
  const settled = useDebounced(paramKey, DEBOUNCE_MS);
  const settling = settled !== paramKey;
  const [raw, setRaw] = useState<{ left: string | null; right: string | null }>(
    {
      left: null,
      right: null,
    },
  );
  const [shown, setShown] = useState<{
    left: string | null;
    right: string | null;
  }>({
    left: null,
    right: null,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const reqId = useRef(0);
  const [cursor, setCursor] = useState<{ ra: number; dec: number } | null>(
    null,
  );
  const [meta, setMeta] = useState<Meta | null>(null);
  const metaRef = useRef<Meta | null>(null);
  metaRef.current = meta;

  const imageContainerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const [dragging, setDragging] = useState(false);
  const [cursorFrac, setCursorFrac] = useState<{
    fx: number;
    fy: number;
  } | null>(null);
  const cursorFracRef = useRef<{ fx: number; fy: number } | null>(null);
  cursorFracRef.current = cursorFrac;

  const tileKey = settled.split("|").slice(0, 3).join("|");
  const [tf, setTf] = useState<{ left: Transform; right: Transform }>({
    left: IDENTITY,
    right: IDENTITY,
  });
  const tfRef = useRef(tf);
  tfRef.current = tf;

  const activeSide: "left" | "right" = time >= 0.5 ? "right" : "left";
  const activeSideRef = useRef(activeSide);
  activeSideRef.current = activeSide;

  const activeDs = activeSide === "left" ? DS_LEFT : DS_RIGHT;
  const saveTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>({});

  const [postcardModalOpen, setPostcardModalOpen] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [postcardOptions, setPostcardOptions] = useState<PostcardOptions>({
    title: "SPHEREXIUM",
    subtitle: "Explorer",
    showCoords: true,
    showWavelength: true,
    showScale: true,
    showTimestamp: true,
    showNasaLogo: true,
    showFavicon: true,
    renderMode: "composite",
    resolution: "1600x1200",
  });

  useEffect(() => {
    let live = true;
    setTf({ left: IDENTITY, right: IDENTITY });
    Promise.all([
      dbGet(`${DS_LEFT.id}|${tileKey}`),
      dbGet(`${DS_RIGHT.id}|${tileKey}`),
    ]).then(([l, r]) => {
      if (live) setTf({ left: l ?? IDENTITY, right: r ?? IDENTITY });
    });
    return () => {
      live = false;
    };
  }, [tileKey]);

  const editTransform = (
    patch: Partial<Transform> | ((t: Transform) => Partial<Transform>),
  ) => {
    const side = activeSide;
    const key = `${(side === "left" ? DS_LEFT : DS_RIGHT).id}|${tileKey}`;
    setTf((prev) => {
      const cur = prev[side];
      const next = {
        ...cur,
        ...(typeof patch === "function" ? patch(cur) : patch),
      };
      clearTimeout(saveTimers.current[key]);
      saveTimers.current[key] = setTimeout(() => dbPut(key, next), 300);
      return { ...prev, [side]: next };
    });
  };

  const resetCurrentTransform = () => {
    const side = activeSide;
    const key = `${(side === "left" ? DS_LEFT : DS_RIGHT).id}|${tileKey}`;
    clearTimeout(saveTimers.current[key]);
    dbDelete(key);
    setTf((prev) => ({ ...prev, [side]: IDENTITY }));
  };

  const resetAllTransforms = () => {
    const lKey = `${DS_LEFT.id}|${tileKey}`;
    const rKey = `${DS_RIGHT.id}|${tileKey}`;
    clearTimeout(saveTimers.current[lKey]);
    clearTimeout(saveTimers.current[rKey]);
    dbDelete(lKey);
    dbDelete(rKey);
    setTf({ left: IDENTITY, right: IDENTITY });
  };

  const updateCursorSkyPosition = (
    fx: number,
    fy: number,
    currTf: Transform,
    m: Meta | null,
  ) => {
    if (!m) return;
    const src = invTransform(currTf, fx, fy);
    const x = src.u * m.w;
    const y = src.v * m.h;
    const dx = x - m.px;
    const dy = y - m.py;
    const sPx = (m.side / m.w) * RAD;
    const th = m.rot * RAD;
    const north = dx * Math.sin(th) - dy * Math.cos(th);
    const east = -dx * Math.cos(th) - dy * Math.sin(th);
    setCursor(unproject(m.ra, m.dec, east * sPx, north * sPx));
  };

  const onImageMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const r0 = e.currentTarget.getBoundingClientRect();
    if (dragRef.current) {
      const dxp = ((e.clientX - dragRef.current.x) / r0.width) * 100;
      const dyp = ((e.clientY - dragRef.current.y) / r0.height) * 100;
      dragRef.current = { x: e.clientX, y: e.clientY };
      editTransform((t) => ({
        x: clamp(t.x + dxp, -200, 200),
        y: clamp(t.y + dyp, -200, 200),
      }));
    }
    const fx = clamp((e.clientX - r0.left) / r0.width, 0, 1);
    const fy = clamp((e.clientY - r0.top) / r0.height, 0, 1);
    setCursorFrac({ fx, fy });
    updateCursorSkyPosition(fx, fy, tf[activeSide], meta);
  };

  useEffect(() => {
    const el = imageContainerRef.current;
    if (!el) return;

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();

      const r = el.getBoundingClientRect();
      const fx = clamp((e.clientX - r.left) / r.width, 0, 1);
      const fy = clamp((e.clientY - r.top) / r.height, 0, 1);

      setCursorFrac({ fx, fy });

      const zoomFactor = Math.exp(-e.deltaY * 0.002);
      const prevTf = tfRef.current;

      const calcZoomed = (cur: Transform): Transform => {
        const targetS = clamp(cur.s * zoomFactor, 0.5, 10.0);
        if (Math.abs(targetS - cur.s) < 1e-4) return cur;

        const pX = (fx - 0.5) * 100;
        const pY = (fy - 0.5) * 100;
        const ratio = targetS / cur.s;

        const nextX = clamp(pX - (pX - cur.x) * ratio, -300, 300);
        const nextY = clamp(pY - (pY - cur.y) * ratio, -300, 300);

        return {
          ...cur,
          s: targetS,
          x: nextX,
          y: nextY,
        };
      };

      const nextLeft = calcZoomed(prevTf.left);
      const nextRight = calcZoomed(prevTf.right);

      const lKey = `${DS_LEFT.id}|${tileKey}`;
      const rKey = `${DS_RIGHT.id}|${tileKey}`;

      clearTimeout(saveTimers.current[lKey]);
      clearTimeout(saveTimers.current[rKey]);
      saveTimers.current[lKey] = setTimeout(() => dbPut(lKey, nextLeft), 300);
      saveTimers.current[rKey] = setTimeout(() => dbPut(rKey, nextRight), 300);

      const updated = { left: nextLeft, right: nextRight };
      setTf(updated);

      const side = activeSideRef.current;
      updateCursorSkyPosition(fx, fy, updated[side], metaRef.current);
    };

    el.addEventListener("wheel", handleWheel, { passive: false });
    return () => {
      el.removeEventListener("wheel", handleWheel);
    };
  }, [tileKey]);

  useEffect(() => {
    const [sra, sdec, sfov, sidx] = settled.split("|");
    const id = ++reqId.current;
    const w = WAVELENGTHS[parseInt(sidx, 10)];
    setLoading(true);
    setError(null);
    const one = (ds: string, key: "left" | "right") =>
      loadImage(buildUrl(ds, +sra, +sdec, +sfov, w), {
        ra: +sra,
        dec: +sdec,
        fov: +sfov,
      }).then(({ obj, meta: m }) => {
        if (id !== reqId.current) return;
        setRaw((p) => ({ ...p, [key]: obj }));
        if (key === "left") setMeta(m);
      });
    Promise.all([one(DS_LEFT.id, "left"), one(DS_RIGHT.id, "right")])
      .catch(
        () =>
          id === reqId.current &&
          setError("Could not load the image for these parameters."),
      )
      .finally(() => id === reqId.current && setLoading(false));
  }, [settled]);

  useEffect(() => {
    let live = true;
    const wlNow = WAVELENGTHS[parseInt(settled.split("|")[3], 10)];
    Promise.all([
      raw.left ? colourise(raw.left, cmap, wlNow) : null,
      raw.right ? colourise(raw.right, cmap, wlNow) : null,
    ])
      .then(([left, right]) => live && setShown({ left, right }))
      .catch(() => live && setShown(raw));
    return () => {
      live = false;
    };
  }, [raw, cmap, settled]);

  const [region, setRegion] = useState<Region | null>(null);
  const [analyses, setAnalyses] = useState<Record<string, Analysis>>({});
  const [failed, setFailed] = useState(0);
  const inspectId = useRef(0);
  const specCanvas = useRef<HTMLCanvasElement>(null);
  const lineCanvas = useRef<HTMLCanvasElement>(null);

  const inspect = () => {
    const r: Region = { ra, dec, fov };
    const id = ++inspectId.current;
    setRegion(r);
    setAnalyses({});
    setFailed(0);
    for (const ds of [DS_LEFT.id, DS_RIGHT.id])
      for (const w of WAVELENGTHS)
        loadImage(buildUrl(ds, r.ra, r.dec, r.fov, w), r)
          .then((x) => decodeUrl(x.obj))
          .then((img) => {
            if (id === inspectId.current)
              setAnalyses((p) => ({ ...p, [`${ds}|${w}`]: analyse(img) }));
          })
          .catch(() => id === inspectId.current && setFailed((n) => n + 1));
  };

  const total = WAVELENGTHS.length * 2;
  const loadedCount = Object.keys(analyses).length;
  const regionMatches =
    !!region &&
    `${region.ra.toFixed(5)}|${region.dec.toFixed(5)}|${region.fov.toFixed(1)}` ===
      tileKey;
  const live = regionMatches && cursorFrac !== null;

  const toSource = (side: "left" | "right"): { u: number; v: number } => {
    if (!live || !cursorFrac) {
      return regionMatches && meta
        ? { u: meta.px / meta.w, v: meta.py / meta.h }
        : { u: 0.5, v: 0.5 };
    }
    return invTransform(tf[side], cursorFrac.fx, cursorFrac.fy);
  };

  useEffect(() => {
    if (!region || !specCanvas.current || !lineCanvas.current) return;
    const id = requestAnimationFrame(() => {
      if (!specCanvas.current || !lineCanvas.current) return;
      const pos = { left: toSource("left"), right: toSource("right") };
      const mainSide = time >= 0.5 ? "right" : "left";
      const main = mainSide === "right" ? DS_RIGHT : DS_LEFT;
      drawSpectrogram(
        specCanvas.current,
        WAVELENGTHS.map((w) => {
          const a = analyses[`${main.id}|${w}`];
          return a ? sliceRow(a, pos[mainSide].v) : null;
        }),
        region.fov * 3600,
        wl,
        pos[mainSide].v,
      );
      drawSpectrum(
        lineCanvas.current,
        (["left", "right"] as const).map((side) => {
          const d = side === "left" ? DS_LEFT : DS_RIGHT;
          return {
            name: d.label,
            color: d.color,
            vals: WAVELENGTHS.map((w) => {
              const a = analyses[`${d.id}|${w}`];
              return a ? sampleMean(a, pos[side].u, pos[side].v) : null;
            }),
          };
        }),
        wl,
      );
    });
    return () => cancelAnimationFrame(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [region, analyses, time, wl, cursorFrac, tf, meta, tileKey]);

  const crosshairStyle = useMemo(() => {
    if (!meta) return null;
    const pt = forwardTransform(
      tf[activeSide],
      meta.px / meta.w,
      meta.py / meta.h,
    );
    return {
      left: `${pt.fx * 100}%`,
      top: `${pt.fy * 100}%`,
    };
  }, [meta, tf, activeSide]);

  const generatePostcard = async () => {
    if (!shown.left && !shown.right) return;
    setIsExporting(true);

    try {
      const [widthStr, heightStr] = postcardOptions.resolution.split("x");
      const W = parseInt(widthStr, 10);
      const H = parseInt(heightStr, 10);

      const canvas = document.createElement("canvas");
      canvas.width = W;
      canvas.height = H;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.fillStyle = "#020408";
      ctx.fillRect(0, 0, W, H);

      const drawLayer = async (src: string, t: Transform, opacity: number) => {
        if (opacity <= 0) return;
        const img = await decodeUrl(src);
        ctx.save();
        ctx.globalAlpha = opacity;
        ctx.translate(W / 2 + (t.x / 100) * W, H / 2 + (t.y / 100) * H);
        ctx.rotate((t.angle * Math.PI) / 180);
        ctx.scale(t.fx ? -t.s : t.s, t.fy ? -t.s : t.s);
        ctx.drawImage(img, -W / 2, -H / 2, W, H);
        ctx.restore();
      };

      if (postcardOptions.renderMode === "composite") {
        if (shown.left) await drawLayer(shown.left, tf.left, 1);
        if (shown.right) await drawLayer(shown.right, tf.right, time);
      } else if (postcardOptions.renderMode === "left" && shown.left) {
        await drawLayer(shown.left, tf.left, 1);
      } else if (postcardOptions.renderMode === "right" && shown.right) {
        await drawLayer(shown.right, tf.right, 1);
      } else if (postcardOptions.renderMode === "active") {
        const activeImg = activeSide === "left" ? shown.left : shown.right;
        if (activeImg) await drawLayer(activeImg, tf[activeSide], 1);
      }

      const grad = ctx.createRadialGradient(
        W / 2,
        H / 2,
        Math.min(W, H) * 0.45,
        W / 2,
        H / 2,
        Math.hypot(W, H) / 2,
      );
      grad.addColorStop(0, "rgba(0,0,0,0)");
      grad.addColorStop(1, "rgba(2, 6, 14, 0.75)");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, W, H);

      ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(32, 32, W - 64, H - 64);

      ctx.save();
      ctx.shadowColor = "rgba(0, 0, 0, 0.9)";
      ctx.shadowBlur = 10;

      ctx.fillStyle = "#38bdf8";
      ctx.font = "bold 20px ui-sans-serif, system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(postcardOptions.title, 56, 54);

      ctx.fillStyle = "#ffffff";
      ctx.font = "600 38px ui-sans-serif, system-ui, sans-serif";
      ctx.fillText(postcardOptions.subtitle, 56, 80);

      let textY = 132;
      ctx.font = "14px ui-monospace, monospace";
      if (postcardOptions.showCoords) {
        ctx.fillStyle = "#cbd5e1";
        ctx.fillText(
          `RA ${ra.toFixed(5)}°   DEC ${dec.toFixed(5)}°   FOV ${fov.toFixed(1)}°`,
          56,
          textY,
        );
        textY += 22;
      }
      if (postcardOptions.showWavelength) {
        ctx.fillStyle = "#94a3b8";
        ctx.fillText(
          `BAND: ${wl.toFixed(2)} µm   ·   CMAP: ${cmap.toUpperCase()}`,
          56,
          textY,
        );
      }
      ctx.restore();

      if (postcardOptions.showFavicon) {
        ctx.save();
        ctx.shadowColor = "rgba(0, 0, 0, 0.9)";
        ctx.shadowBlur = 10;
        const favicon = new Image();
        favicon.src = "/favicon.png";
        const faviconLoaded = await new Promise<boolean>((resolve) => {
          favicon.onload = () => resolve(true);
          favicon.onerror = () => resolve(false);
        });

        const trRight = W - 56;
        if (faviconLoaded) {
          ctx.drawImage(favicon, trRight - 36, 56, 36, 36);
          ctx.fillStyle = "#f8fafc";
          ctx.font = "bold 22px ui-sans-serif, system-ui, sans-serif";
          ctx.textAlign = "right";
          ctx.textBaseline = "middle";
          ctx.fillText("Spherexium", trRight - 48, 74);
        } else {
          ctx.fillStyle = "#f8fafc";
          ctx.font = "bold 24px ui-sans-serif, system-ui, sans-serif";
          ctx.textAlign = "right";
          ctx.textBaseline = "top";
          ctx.fillText("Spherexium", trRight, 58);
        }
        ctx.restore();
      }

      if (postcardOptions.showNasaLogo) {
        ctx.save();
        const drawNasaLogo = (cx: number, cy: number, r: number) => {
          ctx.shadowColor = "rgba(0, 0, 0, 0.75)";
          ctx.shadowBlur = 10;

          ctx.beginPath();
          ctx.arc(cx, cy, r, 0, Math.PI * 2);
          ctx.fillStyle = "#0b3d91";
          ctx.fill();

          ctx.fillStyle = "#ffffff";
          const stars = [
            [-0.5, -0.4, 1.5],
            [0.6, -0.5, 1.2],
            [-0.3, 0.5, 1.8],
            [0.4, 0.4, 1.4],
            [-0.7, 0.1, 1.2],
            [0.2, -0.2, 1.5],
          ];
          stars.forEach(([sx, sy, sr]) => {
            ctx.beginPath();
            ctx.arc(cx + sx * r, cy + sy * r, sr, 0, Math.PI * 2);
            ctx.fill();
          });

          ctx.strokeStyle = "#fc3d21";
          ctx.lineWidth = Math.max(3, r * 0.09);
          ctx.lineCap = "round";
          ctx.beginPath();
          ctx.moveTo(cx - r * 0.85, cy + r * 0.4);
          ctx.lineTo(cx, cy - r * 0.45);
          ctx.lineTo(cx + r * 0.8, cy + r * 0.6);
          ctx.stroke();

          ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.ellipse(cx, cy, r * 0.9, r * 0.35, -Math.PI / 6, 0, Math.PI * 2);
          ctx.stroke();

          ctx.fillStyle = "#ffffff";
          ctx.font = `900 ${Math.round(r * 0.5)}px ui-sans-serif, system-ui, sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText("NASA", cx, cy + 1);
        };

        const nasaImg = new Image();
        nasaImg.src = "/nasa-logo.png";
        const nasaLoaded = await new Promise<boolean>((resolve) => {
          nasaImg.onload = () => resolve(true);
          nasaImg.onerror = () => resolve(false);
        });

        const nX = W - 110;
        const nY = H - 105;
        if (nasaLoaded) {
          ctx.drawImage(nasaImg, nX - 45, nY - 45, 90, 90);
        } else {
          drawNasaLogo(nX, nY, 44);
        }
        ctx.restore();
      }

      if (postcardOptions.showScale || postcardOptions.showTimestamp) {
        ctx.save();
        let bottomY = H - 46;

        if (postcardOptions.showTimestamp) {
          ctx.fillStyle = "rgba(148, 163, 184, 0.75)";
          ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
          ctx.textAlign = "left";
          ctx.textBaseline = "bottom";
          ctx.fillText(
            `Generated ${new Date().toISOString().slice(0, 10)}`,
            56,
            bottomY,
          );
          bottomY -= 18;
        }

        if (postcardOptions.showScale) {
          ctx.strokeStyle = "rgba(255, 255, 255, 0.7)";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(56, bottomY);
          ctx.lineTo(176, bottomY);
          ctx.moveTo(56, bottomY - 5);
          ctx.lineTo(56, bottomY + 5);
          ctx.moveTo(176, bottomY - 5);
          ctx.lineTo(176, bottomY + 5);
          ctx.stroke();

          ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
          ctx.font = "12px ui-monospace, monospace";
          ctx.textBaseline = "bottom";
          ctx.fillText(`FOV: ${fov.toFixed(1)}°`, 56, bottomY - 8);
        }
        ctx.restore();
      }

      const blob = await new Promise<Blob | null>((res) =>
        canvas.toBlob((b) => res(b), "image/png"),
      );
      if (!blob) return;

      const link = document.createElement("a");
      link.download = `postcard-${ra.toFixed(3)}_${dec.toFixed(3)}.png`;
      link.href = URL.createObjectURL(blob);
      link.click();
      URL.revokeObjectURL(link.href);
      setPostcardModalOpen(false);
    } catch (err) {
      console.error("Export failed", err);
    } finally {
      setIsExporting(false);
    }
  };

  const fovSlider = useMemo(() => fov, [fov]);
  const numberField = (
    label: string,
    value: string,
    set: (v: string) => void,
  ) => (
    <label className="flex flex-col gap-1">
      <span className="text-xs text-slate-400">{label}</span>
      <input
        value={value}
        inputMode="decimal"
        placeholder="0"
        onChange={(e) => isNumberish(e.target.value) && set(e.target.value)}
        className="w-36 rounded bg-slate-900 px-2 py-1.5 tabular-nums outline-none ring-1 ring-white/10 focus:ring-sky-300"
      />
    </label>
  );

  return (
    <div className="flex min-h-full w-full flex-col items-center gap-4 bg-[#04060b] p-4 text-slate-200">
      {showTutorial && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 p-4 backdrop-blur-sm">
          <div className="flex w-full max-w-md flex-col overflow-hidden rounded-lg border border-slate-700 bg-slate-900 shadow-2xl">
            <div className="p-6">
              <div className="mb-4 flex items-center justify-between">
                <h3 className="text-xl font-semibold text-slate-100">
                  Welcome to Spherexium Explorer
                </h3>
                <button
                  type="button"
                  onClick={dismissTutorial}
                  className="text-slate-400 hover:text-white"
                >
                  ✕
                </button>
              </div>

              <div className="min-h-[190px] text-sm text-slate-300">
                {tutorialTab === 0 && (
                  <div className="flex animate-in flex-col gap-3 duration-200 fade-in zoom-in-95">
                    <p className="font-medium text-sky-300">
                      Section 1: Main Controls
                    </p>
                    <ul className="flex list-disc flex-col gap-2 pl-5">
                      <li>
                        <strong>RA / Dec / FOV:</strong> Dial in celestial
                        coordinates. You can also click and scroll over the
                        image viewport to pan and zoom.
                      </li>
                      <li>
                        <strong>Wavelength:</strong> Slide left/right to view
                        imagery in different infrared bands.
                      </li>
                      <li>
                        <strong>Colour:</strong> Apply standard or false-colour
                        palettes for better feature contrast.
                      </li>
                      <li>
                        <strong>Time:</strong> Slide to smoothly crossfade
                        between the older and newer datasets.
                      </li>
                    </ul>
                  </div>
                )}
                {tutorialTab === 1 && (
                  <div className="flex animate-in flex-col gap-3 duration-200 fade-in zoom-in-95">
                    <p className="font-medium text-amber-300">
                      Section 2: Transforms
                    </p>
                    <ul className="flex list-disc flex-col gap-2 pl-5">
                      <li>
                        <strong>Orientation:</strong> Use Flip H/V and the 90°
                        rotation buttons to orient the active layer.
                      </li>
                      <li>
                        <strong>Precision Sliders:</strong> Fine-tune rotation
                        angle, scale, and offset for the current image.
                      </li>
                      <li>
                        <strong>Reset:</strong> Immediately undo your tweaks or
                        clear all adjustments across both datasets.
                      </li>
                    </ul>
                  </div>
                )}
                {tutorialTab === 2 && (
                  <div className="flex animate-in flex-col gap-3 duration-200 fade-in zoom-in-95">
                    <p className="font-medium text-emerald-300">
                      Section 3: Analysis & Export
                    </p>
                    <ul className="flex list-disc flex-col gap-2 pl-5">
                      <li>
                        <strong>Inspect region:</strong> Fetches all wavelength
                        layers to render live spectrogram and spectrum line
                        charts right below the viewer.
                      </li>
                      <li>
                        <strong>Create Postcard:</strong> Opens a menu to
                        assemble a customized, high-resolution snapshot (with
                        coordinates and NASA branding) for sharing.
                      </li>
                    </ul>
                  </div>
                )}
              </div>
            </div>

            <div className="flex border-t border-slate-800 bg-slate-950/50">
              <button
                className={`flex-1 py-3 text-xs font-medium transition-colors ${
                  tutorialTab === 0
                    ? "border-t-2 border-sky-400 bg-sky-900/40 text-sky-300"
                    : "border-t-2 border-transparent text-slate-400 hover:bg-slate-800 hover:text-slate-200"
                }`}
                onClick={() => setTutorialTab(0)}
              >
                1. Controls
              </button>
              <button
                className={`flex-1 py-3 text-xs font-medium transition-colors ${
                  tutorialTab === 1
                    ? "border-t-2 border-amber-400 bg-amber-900/40 text-amber-300"
                    : "border-t-2 border-transparent text-slate-400 hover:bg-slate-800 hover:text-slate-200"
                }`}
                onClick={() => setTutorialTab(1)}
              >
                2. Transforms
              </button>
              <button
                className={`flex-1 py-3 text-xs font-medium transition-colors ${
                  tutorialTab === 2
                    ? "border-t-2 border-emerald-400 bg-emerald-900/40 text-emerald-300"
                    : "border-t-2 border-transparent text-slate-400 hover:bg-slate-800 hover:text-slate-200"
                }`}
                onClick={() => setTutorialTab(2)}
              >
                3. Analysis & Export
              </button>
            </div>

            <div className="bg-slate-950/80 p-4 text-right">
              <button
                onClick={dismissTutorial}
                className="rounded bg-sky-300 px-4 py-1.5 text-sm font-medium text-slate-900 hover:bg-sky-200"
              >
                Get Started
              </button>
            </div>
          </div>
        </div>
      )}

      <h1 className="flex w-full max-w-7xl items-center gap-4 text-2xl">
        Spherexium <img src="/favicon.png" className="h-12" alt="logo" />
        <button
          onClick={() => setShowTutorial(true)}
          className="ml-auto flex items-center gap-1 text-sm text-slate-400 transition-colors hover:text-sky-300"
          title="Show how to use the UI"
        >
          <span className="flex h-5 w-5 items-center justify-center rounded-full border border-current text-xs">
            ?
          </span>
          Guide
        </button>
      </h1>

      {/* Widescreen: image left, settings right. Narrow: stacked. */}
      <div className="flex w-full max-w-7xl flex-col items-center gap-4 lg:flex-row lg:items-start">
        {/* ---------------- Image column ---------------- */}
        <div className="flex w-full flex-col items-center gap-2 lg:sticky lg:top-4 lg:min-w-0 lg:flex-1">
          <div
            ref={imageContainerRef}
            onPointerMove={onImageMove}
            onPointerDown={(e) => {
              e.currentTarget.setPointerCapture(e.pointerId);
              dragRef.current = { x: e.clientX, y: e.clientY };
              setDragging(true);
            }}
            onPointerUp={() => {
              dragRef.current = null;
              setDragging(false);
            }}
            onPointerCancel={() => {
              dragRef.current = null;
              setDragging(false);
            }}
            onMouseLeave={() => {
              setCursor(null);
              setCursorFrac(null);
            }}
            style={{ aspectRatio: meta ? `${meta.w} / ${meta.h}` : "1 / 1" }}
            className="relative w-full max-w-[min(100%,70vh)] cursor-crosshair touch-none select-none overflow-hidden rounded bg-black ring-1 ring-white/10 lg:max-w-[min(100%,88vh)]"
          >
            {shown.left && (
              <img
                src={shown.left}
                alt={`${DS_LEFT.label} cutout`}
                draggable={false}
                style={transformCss(tf.left, !dragging)}
                className="absolute inset-0 h-full w-full object-fill"
              />
            )}
            {shown.right && (
              <img
                src={shown.right}
                alt={`${DS_RIGHT.label} cutout`}
                draggable={false}
                style={{ opacity: time, ...transformCss(tf.right, !dragging) }}
                className="absolute inset-0 h-full w-full object-fill"
              />
            )}

            {(loading || settling) && (
              <div className="absolute right-2 top-2 rounded bg-black/60 px-2 py-1 text-xs text-sky-200">
                {settling
                  ? "Setting up..."
                  : "Loading... (warning database is slow!)"}
              </div>
            )}

            {crosshairStyle && (
              <div
                className="pointer-events-none absolute h-6 w-6 -translate-x-1/2 -translate-y-1/2"
                style={crosshairStyle}
              >
                <div className="absolute left-1/2 top-0 h-2 w-px bg-sky-300" />
                <div className="absolute bottom-0 left-1/2 h-2 w-px bg-sky-300" />
                <div className="absolute left-0 top-1/2 h-px w-2 bg-sky-300" />
                <div className="absolute right-0 top-1/2 h-px w-2 bg-sky-300" />
                <div className="absolute left-1/2 top-1/2 h-1 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-sky-300" />
              </div>
            )}

            {meta && !meta.fromServer && (
              <div className="pointer-events-none absolute left-2 top-2 rounded bg-black/65 px-2 py-1 text-[11px] text-amber-300">
                No point metadata in response; assuming image centre
              </div>
            )}

            <div className="pointer-events-none absolute bottom-2 right-2 rounded bg-black/75 px-2 py-0.5 font-mono text-xs tabular-nums text-sky-300 ring-1 ring-white/10">
              x{tf[activeSide].s.toFixed(1)}
            </div>

            {cursor && (
              <div className="pointer-events-none absolute bottom-2 left-2 rounded bg-black/65 px-2 py-1 text-xs tabular-nums text-slate-100">
                RA {cursor.ra.toFixed(5)}° Dec {cursor.dec.toFixed(5)}°
              </div>
            )}
            {error && (
              <div className="absolute inset-x-0 bottom-3 text-center text-sm text-red-300">
                {error} (possibly no data in QR3 for given region?)
              </div>
            )}
            {!shown.left && !shown.right && !loading && !error && (
              <div className="absolute inset-0 flex items-center justify-center text-sm text-slate-500">
                No image yet
              </div>
            )}
          </div>
        </div>

        {/* ---------------- Settings column ---------------- */}
        <div className="flex w-full flex-wrap items-start gap-x-6 gap-y-4 text-sm lg:w-[30rem] lg:shrink-0">
          {numberField("RA (°)", raText, setRaText)}
          {numberField("Dec (°)", decText, setDecText)}
          <div className="flex min-w-[260px] flex-1 flex-col gap-1">
            <span className="text-xs text-slate-400">
              FOV (°), steps of 0.1
            </span>
            <div className="flex items-center gap-3">
              <input
                value={fovText}
                inputMode="decimal"
                placeholder="0"
                onChange={(e) =>
                  isNumberish(e.target.value) && setFovText(e.target.value)
                }
                className="w-28 rounded bg-slate-900 px-2 py-1.5 tabular-nums outline-none ring-1 ring-white/10 focus:ring-sky-300"
              />
              <input
                type="range"
                min={FOV_MIN}
                max={FOV_MAX}
                step={0.1}
                value={fovSlider}
                onChange={(e) =>
                  setFovText(parseFloat(e.target.value).toFixed(1))
                }
                aria-label="Field of view"
                className="flex-1 accent-sky-300"
              />
            </div>
            {fovRaw !== fov && (
              <span className="text-[11px] text-amber-300/80">
                Using {fov.toFixed(1)}° (multiples of 0.1, range {FOV_MIN}–
                {FOV_MAX}°)
              </span>
            )}
          </div>
          <label className="flex min-w-[260px] flex-1 flex-col gap-1">
            <span className="flex justify-between text-xs text-slate-400">
              <span>Wavelength</span>
              <span className="tabular-nums text-slate-200">
                {wl.toFixed(2)} µm
              </span>
            </span>
            <input
              type="range"
              min={0}
              max={WAVELENGTHS.length - 1}
              step={1}
              value={wlIdx}
              onChange={(e) => setWlIdx(parseInt(e.target.value, 10))}
              className="accent-sky-300"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-slate-400">Colour</span>
            <select
              value={cmap}
              onChange={(e) => setCmap(e.target.value as CMap)}
              className="rounded bg-slate-900 px-2 py-1.5 ring-1 ring-white/10"
            >
              <option value="grey">Greyscale</option>
              <option value="inferno">Inferno</option>
              <option value="spectral">Wavelength tint</option>
            </select>
          </label>
          <label className="flex w-full flex-col gap-1">
            <span className="text-xs text-slate-400">Time</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.001}
              value={time}
              onChange={(e) => setTime(parseFloat(e.target.value))}
              className="accent-sky-300"
            />
            <span className="flex justify-between text-xs text-slate-400">
              <span>{DS_LEFT.label}</span>
              <span>{DS_RIGHT.label}</span>
            </span>
          </label>

          <fieldset className="w-full rounded bg-[#080b13] p-3 ring-1 ring-white/10">
            <legend className="px-1 text-xs text-slate-400">
              Transform ·{" "}
              <span style={{ color: activeDs.color }}>{activeDs.label}</span>
            </legend>
            <div className="flex flex-wrap items-center gap-2">
              {(
                [
                  [
                    "Flip H",
                    tf[activeSide].fx,
                    () => editTransform((t) => ({ fx: !t.fx })),
                  ],
                  [
                    "Flip V",
                    tf[activeSide].fy,
                    () => editTransform((t) => ({ fy: !t.fy })),
                  ],
                ] as const
              ).map(([label, on, fn]) => (
                <button
                  key={label}
                  type="button"
                  aria-pressed={on}
                  onClick={fn}
                  className={`rounded px-3 py-1.5 ring-1 ring-white/15 hover:bg-white/10 ${
                    on ? "bg-sky-300 text-slate-900 hover:bg-sky-200" : ""
                  }`}
                >
                  {label}
                </button>
              ))}
              <button
                type="button"
                aria-label="Rotate 90 degrees counter-clockwise"
                onClick={() =>
                  editTransform((t) => ({ angle: wrapAngle(t.angle - 90) }))
                }
                className="rounded px-3 py-1.5 ring-1 ring-white/15 hover:bg-white/10"
              >
                ↺ 90°
              </button>
              <button
                type="button"
                aria-label="Rotate 90 degrees clockwise"
                onClick={() =>
                  editTransform((t) => ({ angle: wrapAngle(t.angle + 90) }))
                }
                className="rounded px-3 py-1.5 ring-1 ring-white/15 hover:bg-white/10"
              >
                ↻ 90°
              </button>
              <span className="text-xs tabular-nums text-slate-400">
                {tf[activeSide].angle.toFixed(1)}°
              </span>

              <div className="flex w-full flex-wrap items-center gap-2 sm:ml-auto sm:w-auto">
                <button
                  type="button"
                  onClick={resetCurrentTransform}
                  className="rounded px-3 py-1.5 text-slate-300 ring-1 ring-white/15 hover:bg-white/10 hover:text-white"
                >
                  Reset {activeDs.label}
                </button>
                <button
                  type="button"
                  onClick={resetAllTransforms}
                  className="rounded px-3 py-1.5 text-rose-300 ring-1 ring-rose-500/30 hover:bg-rose-950/40"
                >
                  Reset All Transforms
                </button>
              </div>
            </div>

            <div className="mt-3 flex flex-col gap-3">
              {(
                [
                  ["Rotation", "angle", -180, 180, 0.1, "°"],
                  ["Position X", "x", -200, 200, 0.1, "%"],
                  ["Position Y", "y", -200, 200, 0.1, "%"],
                  ["Scale", "s", 0.5, 10, 0.01, "×"],
                ] as const
              ).map(([label, k, min, max, step, unit]) => (
                <label key={k} className="flex flex-col gap-1">
                  <span className="flex justify-between text-xs text-slate-400">
                    <span>{label}</span>
                    <span className="tabular-nums text-slate-200">
                      {tf[activeSide][k].toFixed(k === "s" ? 2 : 1)}
                      {unit}
                    </span>
                  </span>
                  <input
                    type="range"
                    min={min}
                    max={max}
                    step={step}
                    value={tf[activeSide][k]}
                    onChange={(e) =>
                      editTransform({
                        [k]: parseFloat(e.target.value),
                      } as Partial<Transform>)
                    }
                    onDoubleClick={() =>
                      editTransform({ [k]: IDENTITY[k] } as Partial<Transform>)
                    }
                    className="accent-sky-300"
                  />
                </label>
              ))}
            </div>
          </fieldset>

          <div className="flex w-full flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={inspect}
                disabled={settling}
                className="rounded bg-sky-300 px-4 py-1.5 font-medium text-slate-900 hover:bg-sky-200 disabled:opacity-50"
              >
                Inspect region
              </button>
              <span className="text-xs text-slate-400">
                Loads all {WAVELENGTHS.length} bands for both datasets
              </span>
            </div>

            <button
              type="button"
              onClick={() => setPostcardModalOpen(true)}
              disabled={!shown.left && !shown.right}
              className="flex items-center gap-2 rounded bg-amber-400 px-4 py-1.5 font-medium text-slate-950 shadow hover:bg-amber-300 disabled:opacity-50"
            >
              Create Postcard
            </button>
          </div>
        </div>
      </div>

      {region && (
        <div className="w-full max-w-7xl rounded bg-[#080b13] p-3 ring-1 ring-white/10">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
            <span>
              RA {region.ra.toFixed(4)}° Dec {region.dec.toFixed(4)}° · FOV{" "}
              {region.fov.toFixed(1)}° · {loadedCount}/{total} layers
              {failed > 0 && (
                <span className="ml-2 text-red-300">{failed} failed</span>
              )}
            </span>
            <span className="inline-block h-1.5 w-28 overflow-hidden rounded bg-slate-800">
              <span
                className="block h-full bg-sky-300 transition-[width]"
                style={{ width: `${(loadedCount / total) * 100}%` }}
              />
            </span>
          </div>
          {!regionMatches && (
            <p className="mb-2 text-xs text-amber-300/80">
              The view has moved away from the inspected region. Click Inspect
              region to analyse the current one.
            </p>
          )}
          <div className="grid gap-3 md:grid-cols-2">
            <figure>
              <figcaption className="mb-1 text-xs text-slate-400">
                Spectrogram, slice through {live ? "cursor" : "requested point"}{" "}
                ({time >= 0.5 ? DS_RIGHT.label : DS_LEFT.label})
              </figcaption>
              <canvas
                ref={specCanvas}
                style={{ width: "100%", aspectRatio: `${PW}/${PH}` }}
              />
            </figure>
            <figure>
              <figcaption className="mb-1 text-xs text-slate-400">
                Spectrum at{" "}
                {live && cursor
                  ? `RA ${cursor.ra.toFixed(5)}° Dec ${cursor.dec.toFixed(5)}°`
                  : "requested point"}
              </figcaption>
              <canvas
                ref={lineCanvas}
                style={{ width: "100%", aspectRatio: `${PW}/${PH}` }}
              />
            </figure>
          </div>
        </div>
      )}

      {postcardModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-sm">
          <div className="flex max-h-[90vh] w-full max-w-lg flex-col overflow-y-auto rounded-lg border border-slate-700 bg-slate-900 p-6 shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="text-lg font-semibold text-slate-100">
                Postcard Generator
              </h3>
              <button
                type="button"
                onClick={() => setPostcardModalOpen(false)}
                className="text-slate-400 hover:text-white"
              >
                ✕
              </button>
            </div>

            <div className="mt-4 flex flex-col gap-4 text-sm">
              <div className="grid grid-cols-2 gap-3">
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-slate-400">Main Title</span>
                  <input
                    type="text"
                    value={postcardOptions.title}
                    onChange={(e) =>
                      setPostcardOptions((p) => ({
                        ...p,
                        title: e.target.value,
                      }))
                    }
                    className="rounded bg-slate-800 px-3 py-1.5 text-white ring-1 ring-white/10 focus:ring-sky-300"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-slate-400">Subtitle</span>
                  <input
                    type="text"
                    value={postcardOptions.subtitle}
                    onChange={(e) =>
                      setPostcardOptions((p) => ({
                        ...p,
                        subtitle: e.target.value,
                      }))
                    }
                    className="rounded bg-slate-800 px-3 py-1.5 text-white ring-1 ring-white/10 focus:ring-sky-300"
                  />
                </label>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-slate-400">Render Source</span>
                  <select
                    value={postcardOptions.renderMode}
                    onChange={(e) =>
                      setPostcardOptions((p) => ({
                        ...p,
                        renderMode: e.target
                          .value as PostcardOptions["renderMode"],
                      }))
                    }
                    className="rounded bg-slate-800 px-3 py-1.5 text-white ring-1 ring-white/10"
                  >
                    <option value="composite">
                      Composite (Blend {DS_LEFT.id} &amp; {DS_RIGHT.id})
                    </option>
                    <option value="active">Active View</option>
                    <option value="left">{DS_LEFT.label}</option>
                    <option value="right">{DS_RIGHT.label}</option>
                  </select>
                </label>

                <label className="flex flex-col gap-1">
                  <span className="text-xs text-slate-400">Output Size</span>
                  <select
                    value={postcardOptions.resolution}
                    onChange={(e) =>
                      setPostcardOptions((p) => ({
                        ...p,
                        resolution: e.target
                          .value as PostcardOptions["resolution"],
                      }))
                    }
                    className="rounded bg-slate-800 px-3 py-1.5 text-white ring-1 ring-white/10"
                  >
                    <option value="1200x900">1200 × 900</option>
                    <option value="1600x1200">1600 × 1200</option>
                    <option value="2400x1800">2400 × 1800 (Hi-Res)</option>
                  </select>
                </label>
              </div>

              <div className="rounded border border-slate-800 bg-slate-950/40 p-3">
                <span className="mb-2 block text-xs font-semibold text-slate-400">
                  Visible Elements
                </span>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={postcardOptions.showCoords}
                      onChange={(e) =>
                        setPostcardOptions((p) => ({
                          ...p,
                          showCoords: e.target.checked,
                        }))
                      }
                      className="accent-sky-400"
                    />
                    <span>RA / Dec / FOV</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={postcardOptions.showWavelength}
                      onChange={(e) =>
                        setPostcardOptions((p) => ({
                          ...p,
                          showWavelength: e.target.checked,
                        }))
                      }
                      className="accent-sky-400"
                    />
                    <span>Wavelength &amp; Colormap</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={postcardOptions.showScale}
                      onChange={(e) =>
                        setPostcardOptions((p) => ({
                          ...p,
                          showScale: e.target.checked,
                        }))
                      }
                      className="accent-sky-400"
                    />
                    <span>Scale Bar</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={postcardOptions.showTimestamp}
                      onChange={(e) =>
                        setPostcardOptions((p) => ({
                          ...p,
                          showTimestamp: e.target.checked,
                        }))
                      }
                      className="accent-sky-400"
                    />
                    <span>Timestamp</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={postcardOptions.showFavicon}
                      onChange={(e) =>
                        setPostcardOptions((p) => ({
                          ...p,
                          showFavicon: e.target.checked,
                        }))
                      }
                      className="accent-sky-400"
                    />
                    <span>Top-Right Spherexium Logo</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={postcardOptions.showNasaLogo}
                      onChange={(e) =>
                        setPostcardOptions((p) => ({
                          ...p,
                          showNasaLogo: e.target.checked,
                        }))
                      }
                      className="accent-sky-400"
                    />
                    <span>NASA Badge</span>
                  </label>
                </div>
              </div>

              <div className="rounded border border-slate-800 bg-slate-950/60 p-2.5 text-xs text-slate-400">
                <p>
                  Current tile: RA {ra.toFixed(4)}°, Dec {dec.toFixed(4)}°, FOV{" "}
                  {fov.toFixed(1)}° at {wl.toFixed(2)} µm. Active
                  transformations will be rendered directly to the final image.
                </p>
              </div>
            </div>

            <div className="mt-6 flex items-center justify-end gap-3 border-t border-slate-800 pt-3">
              <button
                type="button"
                onClick={() => setPostcardModalOpen(false)}
                className="rounded px-4 py-1.5 text-slate-300 hover:text-white"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={generatePostcard}
                disabled={isExporting}
                className="rounded bg-sky-300 px-4 py-1.5 font-medium text-slate-900 hover:bg-sky-200 disabled:opacity-50"
              >
                {isExporting ? "Rendering..." : "Export PNG"}
              </button>
            </div>
          </div>
        </div>
      )}
      <p className="mt-4 text-sm text-white/60">
        Created by: Paweł Skrzypkowski, Adam Ryan, Malwina Lewara, Maria
        Mikulik, Grzegorz Wiktor, Leonard Piotorwski
      </p>
    </div>
  );
}
