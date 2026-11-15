export interface SatelliteConfig {
  name: string;
  hipsUrl: string;
  minFov: number;
  maxFov: number;
  maxOrder: number;
  maxLayer: number;
  blankLayers: number[];
  wavelengthProperty: string;
  preloadConcurrency?: number;
}

export interface TileRequest {
  ra: number;
  dec: number;
  fov: number;
  layer: number;
}

export interface PreloadProgress {
  completed: number;
  total: number;
}

export interface PreloadResult {
  completed: number;
  failedLayers: number[];
  total: number;
}

export interface TilePosition {
  x: number;
  y: number;
  normalizedX: number;
  normalizedY: number;
}

export interface TileMetadata {
  url: string;
  order: number;
  nside: number;
  pixel: number;
  directory: number;
  imageWidth: number;
  imageHeight: number;
  position: TilePosition;
}

export interface TileWithMetadata {
  image: HTMLImageElement;
  metadata: TileMetadata;
}

type Vec3 = [number, number, number];

const imageCache = new Map<string, Promise<HTMLImageElement>>();
const loadedImages = new Map<string, HTMLImageElement>();

function loadImage(url: string): Promise<HTMLImageElement> {
  const existing = loadedImages.get(url);
  if (existing) return Promise.resolve(existing);

  const cached = imageCache.get(url);
  if (cached) return cached;

  const promise = new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = "anonymous";

    image.onload = () => {
      loadedImages.set(url, image);
      resolve(image);
    };

    image.onerror = () => {
      imageCache.delete(url);
      loadedImages.delete(url);
      reject(new Error(`Could not load image: ${url}`));
    };

    image.src = url;
  });

  imageCache.set(url, promise);
  return promise;
}

// ============================================================
// HiPS TILE PIXEL MAPPING
//
// HEALPix NESTED numbering is NOT the row-major pixel order inside
// a HiPS tile image. Each tile is a Z-order (Morton) curve.
//
// ORIENTATION (this was the bug):
//   The reference gist (Thomas Boch / CDS, hips2healpix.py) reads FITS
//   tiles, where array row 0 is the BOTTOM of the image.
//   PNG/JPEG tiles are vertically flipped relative to FITS: row 0 is the
//   TOP (HiPS spec: "Contrary to FITS convention, in JPG and PNG the
//   lines of the pixel are written from top to bottom").
//
//   FITS (gist):   first half of rows  -> low bit 1
//   PNG  (here):   first half of rows  -> low bit 0   <-- required
//
//   Result for PNG, display orientation:
//     top-left = 0, bottom-left = 1, top-right = 2, bottom-right = 3
//   at every level of the recursion (most significant digit first).
//
//   Verified against cds-astro/hips2fits-cutout (png branch:
//   tile[yy][xx] with [xx, yy] = hpx2xy[nested]) for orders 1, 2, 3, 5.
//
// xy2hpx[imageY * TILE_SIZE + imageX] = nested child pixel
// hpx2xy[nested child pixel]          = imageY * TILE_SIZE + imageX
// ============================================================

const TILE_SIZE = 512;
const TILE_ORDER_BITS = 9;
const PIXELS_PER_TILE = TILE_SIZE * TILE_SIZE;
const EXTRAPOLATION_STEP = 4;

interface HiPSPixelMaps {
  xy2hpx: Int32Array;
  hpx2xy: Int32Array;
}

let hipsPixelMaps: HiPSPixelMaps | null = null;

function createHiPSPixelMaps(): HiPSPixelMaps {
  const count = PIXELS_PER_TILE;

  const xy2hpx = new Int32Array(count);
  const hpx2xy = new Int32Array(count);

  function fillUp(
    npix: Int32Array,
    nsize: number,
    pos: Int32Array | null,
  ): void {
    const size = nsize * nsize;
    const quarterSize = size / 4;

    const children = [
      new Int32Array(quarterSize),
      new Int32Array(quarterSize),
      new Int32Array(quarterSize),
      new Int32Array(quarterSize),
    ];

    const childCounts = [0, 0, 0, 0];

    for (let i = 0; i < size; i++) {
      // Left/right half of the current image region.
      const dg = i % nsize < nsize / 2 ? 0 : 1;

      // PNG: row 0 is the TOP. Top half -> 0, bottom half -> 1.
      // (The FITS gist uses the opposite because FITS row 0 is the bottom.)
      const bh = i < size / 2 ? 0 : 1;

      const quadrant = (dg << 1) | bh;

      const originalIndex = pos === null ? i : pos[i];

      npix[originalIndex] = npix[originalIndex] * 4 + quadrant;

      children[quadrant][childCounts[quadrant]++] = originalIndex;
    }

    if (size > 4) {
      for (let quadrant = 0; quadrant < 4; quadrant++) {
        fillUp(npix, nsize / 2, children[quadrant]);
      }
    }
  }

  fillUp(xy2hpx, TILE_SIZE, null);

  // Invert: nested HEALPix child pixel -> row-major image index.
  for (let imageIndex = 0; imageIndex < count; imageIndex++) {
    hpx2xy[xy2hpx[imageIndex]] = imageIndex;
  }

  return { xy2hpx, hpx2xy };
}

function getHiPSPixelMaps(): HiPSPixelMaps {
  if (!hipsPixelMaps) {
    hipsPixelMaps = createHiPSPixelMaps();
  }

  return hipsPixelMaps;
}

/** Nested child pixel -> image pixel coordinates. */
function nestedChildToImageXY(childPixel: number): { x: number; y: number } {
  if (
    !Number.isInteger(childPixel) ||
    childPixel < 0 ||
    childPixel >= PIXELS_PER_TILE
  ) {
    throw new Error(`Invalid HiPS child pixel: ${childPixel}`);
  }

  const { hpx2xy } = getHiPSPixelMaps();
  const imageIndex = hpx2xy[childPixel];

  return {
    x: imageIndex % TILE_SIZE,
    y: Math.floor(imageIndex / TILE_SIZE),
  };
}

/** Image pixel coordinates -> nested child pixel. */
function imageXYToNestedChild(x: number, y: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);

  if (ix < 0 || ix >= TILE_SIZE || iy < 0 || iy >= TILE_SIZE) {
    throw new Error(`Invalid HiPS image pixel: ${ix}, ${iy}`);
  }

  const { xy2hpx } = getHiPSPixelMaps();

  return xy2hpx[iy * TILE_SIZE + ix];
}

// ============================================================
// GENERAL HELPERS
// ============================================================

function clampInt(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function normalizeRA(value: number) {
  return ((value % 360) + 360) % 360;
}

function vecToSky(v: Vec3): { ra: number; dec: number } {
  const length = Math.hypot(v[0], v[1], v[2]) || 1;

  const x = v[0] / length;
  const y = v[1] / length;
  const z = clampInt(v[2] / length, -1, 1);

  const theta = Math.acos(z);
  const phi = Math.atan2(y, x);

  return {
    ra: normalizeRA((phi * 180) / Math.PI),
    dec: 90 - (theta * 180) / Math.PI,
  };
}

export function skyToVec(ra: number, dec: number): Vec3 {
  const theta = ((90 - dec) * Math.PI) / 180;
  const phi = (normalizeRA(ra) * Math.PI) / 180;

  const sinTheta = Math.sin(theta);

  return [sinTheta * Math.cos(phi), sinTheta * Math.sin(phi), Math.cos(theta)];
}

function normalizeVec(v: Vec3): Vec3 {
  const length = Math.hypot(v[0], v[1], v[2]) || 1;

  return [v[0] / length, v[1] / length, v[2] / length];
}

// ============================================================
// HEALPIX (NESTED) - own implementation using doubles only.
//
// healpixjs uses 32-bit integer ops. At order 14 (nside 16384, 3.2e9
// pixels > 2^31) its indices wrap negative ("calculated=-6171").
// Everything below stays in float64 (exact up to 2^53) and avoids
// bitwise operators.
// ============================================================

const JRLL = [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4];
const JPLL = [1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7];

/** Spread bits of v so bit k moves to bit 2k. */
function spreadBits(v: number): number {
  let result = 0;
  let mask = 1;
  while (v > 0) {
    if (v % 2 === 1) result += mask;
    v = Math.floor(v / 2);
    mask *= 4;
  }
  return result;
}

/** Inverse of spreadBits: gather even bits of v (bit 2k -> bit k). */
function compactBits(v: number): number {
  let result = 0;
  let mask = 1;
  while (v > 0) {
    if (v % 2 === 1) result += mask;
    v = Math.floor(v / 4);
    mask *= 2;
  }
  return result;
}

function getHealpixPixel(nside: number, ra: number, dec: number): number {
  const z = Math.sin((dec * Math.PI) / 180);
  const za = Math.abs(z);
  const phi = normalizeRA(ra) * (Math.PI / 180);
  const tt = (phi * 2) / Math.PI; // [0, 4)

  let face: number;
  let ix: number;
  let iy: number;

  if (za <= 2 / 3) {
    const temp1 = nside * (0.5 + tt);
    const temp2 = nside * z * 0.75;
    const jp = Math.floor(temp1 - temp2);
    const jm = Math.floor(temp1 + temp2);
    const ifp = Math.floor(jp / nside);
    const ifm = Math.floor(jm / nside);

    if (ifp === ifm) face = ifp === 4 ? 4 : ifp + 4;
    else if (ifp < ifm) face = ifp;
    else face = ifm + 8;

    ix = ((jm % nside) + nside) % nside;
    iy = nside - (((jp % nside) + nside) % nside) - 1;
  } else {
    const ntt = Math.min(3, Math.floor(tt));
    const tp = tt - ntt;
    const tmp = nside * Math.sqrt(3 * (1 - za));

    const jp = Math.min(nside - 1, Math.floor(tp * tmp));
    const jm = Math.min(nside - 1, Math.floor((1 - tp) * tmp));

    if (z >= 0) {
      face = ntt;
      ix = nside - jm - 1;
      iy = nside - jp - 1;
    } else {
      face = ntt + 8;
      ix = jp;
      iy = jm;
    }
  }

  ix = clampInt(ix, 0, nside - 1);
  iy = clampInt(iy, 0, nside - 1);

  return face * nside * nside + spreadBits(ix) + 2 * spreadBits(iy);
}

/** Center of a NESTED pixel -> colatitude / longitude (radians). */
function healpixPixelToAng(
  nside: number,
  pix: number,
): { theta: number; phi: number } {
  const npface = nside * nside;
  const face = Math.floor(pix / npface);
  const rem = pix - face * npface;

  const ix = compactBits(rem);
  const iy = compactBits(Math.floor(rem / 2));

  const nl4 = 4 * nside;
  const fact2 = 4 / (12 * npface);
  const jr = JRLL[face] * nside - ix - iy - 1;

  let nr: number;
  let z: number;
  let kshift = 0;

  if (jr < nside) {
    nr = jr;
    z = 1 - nr * nr * fact2;
  } else if (jr > 3 * nside) {
    nr = nl4 - jr;
    z = nr * nr * fact2 - 1;
  } else {
    nr = nside;
    z = (2 * nside - jr) * (2 * nside * fact2);
    kshift = (jr - nside) % 2;
  }

  let jp = Math.trunc((JPLL[face] * nr + ix - iy + 1 + kshift) / 2);
  if (jp > nl4) jp -= nl4;
  if (jp < 1) jp += nl4;

  const phi = (jp - (kshift + 1) * 0.5) * (Math.PI / 2 / nr);

  return { theta: Math.acos(clampInt(z, -1, 1)), phi };
}

// ============================================================
// DATA SOURCE
// ============================================================

export class SatelliteDataSource {
  readonly config: SatelliteConfig;
  readonly validLayers: number[];

  private wavelengthPromise: Promise<number[]> | null = null;

  constructor(config: SatelliteConfig) {
    this.config = config;

    this.validLayers = Array.from(
      { length: config.maxLayer + 1 },
      (_, index) => index,
    ).filter((layer) => !config.blankLayers.includes(layer));
  }

  get name(): string {
    return this.config.name;
  }

  get cachedImageCount(): number {
    return imageCache.size;
  }

  get availableLayers(): number[] {
    return this.validLayers;
  }

  validateRequest(request: TileRequest): void {
    const { ra, dec, fov, layer } = request;

    if (!Number.isFinite(ra) || ra < 0 || ra >= 360) {
      throw new Error("Right Ascension must be between 0 and 360 degrees.");
    }

    if (!Number.isFinite(dec) || dec < -90 || dec > 90) {
      throw new Error("Declination must be between -90 and +90 degrees.");
    }

    if (
      !Number.isFinite(fov) ||
      fov < this.config.minFov ||
      fov > this.config.maxFov
    ) {
      throw new Error("Invalid field of view.");
    }

    if (!this.validLayers.includes(layer)) {
      throw new Error("This spectral layer is unavailable.");
    }
  }

  getOrder(fov: number): number {
    if (!Number.isFinite(fov) || fov <= 0) {
      throw new Error("Field of view must be greater than zero.");
    }

    const calculatedOrder = Math.round(Math.log2(this.config.maxFov / fov));

    return Math.max(0, Math.min(this.config.maxOrder, calculatedOrder));
  }

  // ==========================================================
  // SKY -> TILE METADATA AND IMAGE POSITION
  // ==========================================================

  getTileMetadata(request: TileRequest): TileMetadata {
    this.validateRequest(request);

    const { ra, dec, fov, layer } = request;

    const order = this.getOrder(fov);
    const nside = 2 ** order;

    const normalizedRA = normalizeRA(ra);

    const pixel = getHealpixPixel(nside, normalizedRA, dec);

    const directory = Math.floor(pixel / 10000) * 10000;

    const url =
      `${this.config.hipsUrl}/Norder${order}` +
      `/Dir${directory}` +
      `/Npix${pixel}_${layer}.png`;

    // A HiPS tile contains 512 x 512 pixels.
    // Each image pixel corresponds to one nested child pixel
    // at order + 9.
    const fineOrder = order + TILE_ORDER_BITS;
    const fineNside = 2 ** fineOrder;

    const finePixel = getHealpixPixel(fineNside, normalizedRA, dec);

    const parentPixel = Math.floor(finePixel / PIXELS_PER_TILE);

    if (parentPixel !== pixel) {
      throw new Error(
        `HEALPix parent/child mismatch: ` +
          `parent=${pixel}, calculated=${parentPixel}.`,
      );
    }

    const childPixel = finePixel % PIXELS_PER_TILE;

    const { x, y } = nestedChildToImageXY(childPixel);

    return {
      url,
      order,
      nside,
      pixel,
      directory,
      imageWidth: TILE_SIZE,
      imageHeight: TILE_SIZE,
      position: {
        x: x + 0.5,
        y: y + 0.5,
        normalizedX: (x + 0.5) / TILE_SIZE,
        normalizedY: (y + 0.5) / TILE_SIZE,
      },
    };
  }

  getTileUrl(request: TileRequest): string {
    return this.getTileMetadata(request).url;
  }

  /** Tile identity only (no layer). Used for cache keys. */
  getTileAddress(
    ra: number,
    dec: number,
    fov: number,
  ): { order: number; pixel: number } {
    if (
      !Number.isFinite(ra) ||
      !Number.isFinite(dec) ||
      dec < -90 ||
      dec > 90
    ) {
      throw new Error("Invalid coordinates.");
    }

    const order = this.getOrder(fov);

    return {
      order,
      pixel: getHealpixPixel(2 ** order, normalizeRA(ra), dec),
    };
  }

  // ==========================================================
  // SKY -> IMAGE POSITION
  // ==========================================================

  skyToTilePosition(
    tile: Pick<TileMetadata, "order" | "pixel">,
    ra: number,
    dec: number,
  ): { normalizedX: number; normalizedY: number } | null {
    if (
      !Number.isFinite(ra) ||
      !Number.isFinite(dec) ||
      dec < -90 ||
      dec > 90
    ) {
      return null;
    }

    const fineNside = 2 ** (tile.order + TILE_ORDER_BITS);

    const finePixel = getHealpixPixel(fineNside, normalizeRA(ra), dec);

    const parentPixel = Math.floor(finePixel / PIXELS_PER_TILE);

    if (parentPixel !== tile.pixel) {
      return null;
    }

    const childPixel = finePixel % PIXELS_PER_TILE;

    const { x, y } = nestedChildToImageXY(childPixel);

    return {
      normalizedX: (x + 0.5) / TILE_SIZE,
      normalizedY: (y + 0.5) / TILE_SIZE,
    };
  }

  // ==========================================================
  // IMAGE PIXEL -> SKY VECTOR
  // ==========================================================

  private imagePixelToVec(
    tile: Pick<TileMetadata, "order" | "pixel">,
    x: number,
    y: number,
  ): Vec3 {
    const ix = clampInt(Math.floor(x), 0, TILE_SIZE - 1);
    const iy = clampInt(Math.floor(y), 0, TILE_SIZE - 1);

    const childPixel = imageXYToNestedChild(ix, iy);

    const finePixel = tile.pixel * PIXELS_PER_TILE + childPixel;

    const pointing = healpixPixelToAng(
      2 ** (tile.order + TILE_ORDER_BITS),
      finePixel,
    );

    const sinTheta = Math.sin(pointing.theta);

    return [
      sinTheta * Math.cos(pointing.phi),
      sinTheta * Math.sin(pointing.phi),
      Math.cos(pointing.theta),
    ];
  }

  // ==========================================================
  // IMAGE POSITION -> SKY
  // ==========================================================

  /**
   * Converts a position in the displayed image to sky coordinates.
   * Sub-pixel positions interpolate local 3D vectors and normalize
   * (safe across the RA 0/360 boundary). Positions outside the tile
   * extrapolate from the nearest edge pixel's local basis.
   */
  tilePositionToSky(
    tile: Pick<TileMetadata, "order" | "pixel">,
    normalizedX: number,
    normalizedY: number,
  ): { ra: number; dec: number } {
    if (!Number.isFinite(normalizedX) || !Number.isFinite(normalizedY)) {
      throw new Error("Invalid tile position.");
    }

    const px = normalizedX * TILE_SIZE;
    const py = normalizedY * TILE_SIZE;

    const cx = clampInt(Math.floor(px), 0, TILE_SIZE - 1);
    const cy = clampInt(Math.floor(py), 0, TILE_SIZE - 1);

    const p0 = this.imagePixelToVec(tile, cx, cy);

    // Position relative to the selected pixel center.
    const ox = px - (cx + 0.5);
    const oy = py - (cy + 0.5);

    // Find a valid neighboring sample in each image direction.
    const sx =
      cx + EXTRAPOLATION_STEP <= TILE_SIZE - 1
        ? EXTRAPOLATION_STEP
        : -EXTRAPOLATION_STEP;

    const sy =
      cy + EXTRAPOLATION_STEP <= TILE_SIZE - 1
        ? EXTRAPOLATION_STEP
        : -EXTRAPOLATION_STEP;

    const pX = this.imagePixelToVec(tile, cx + sx, cy);
    const pY = this.imagePixelToVec(tile, cx, cy + sy);

    const vx: Vec3 = [
      (pX[0] - p0[0]) / sx,
      (pX[1] - p0[1]) / sx,
      (pX[2] - p0[2]) / sx,
    ];

    const vy: Vec3 = [
      (pY[0] - p0[0]) / sy,
      (pY[1] - p0[1]) / sy,
      (pY[2] - p0[2]) / sy,
    ];

    const result: Vec3 = normalizeVec([
      p0[0] + ox * vx[0] + oy * vy[0],
      p0[1] + ox * vx[1] + oy * vy[1],
      p0[2] + ox * vx[2] + oy * vy[2],
    ]);

    return vecToSky(result);
  }

  // ==========================================================
  // IMAGE LOADING
  // ==========================================================

  async getTile(request: TileRequest): Promise<HTMLImageElement> {
    const metadata = this.getTileMetadata(request);
    return loadImage(metadata.url);
  }

  async getTileWithMetadata(request: TileRequest): Promise<TileWithMetadata> {
    const metadata = this.getTileMetadata(request);
    const image = await loadImage(metadata.url);

    return {
      image,
      metadata: {
        ...metadata,
        imageWidth: image.naturalWidth,
        imageHeight: image.naturalHeight,
        position: {
          x: metadata.position.normalizedX * image.naturalWidth,
          y: metadata.position.normalizedY * image.naturalHeight,
          normalizedX: metadata.position.normalizedX,
          normalizedY: metadata.position.normalizedY,
        },
      },
    };
  }

  // ==========================================================
  // WAVELENGTH METADATA
  // ==========================================================

  async getWavelengths(): Promise<number[]> {
    if (!this.wavelengthPromise) {
      this.wavelengthPromise = fetch(`${this.config.hipsUrl}/properties`)
        .then((response) => {
          if (!response.ok) {
            throw new Error("Could not load wavelength metadata.");
          }

          return response.text();
        })
        .then((text) => {
          const properties: Record<string, string> = {};

          for (const line of text.split(/\r?\n/)) {
            const trimmed = line.trim();

            if (!trimmed || trimmed.startsWith("#")) {
              continue;
            }

            const separator = trimmed.indexOf("=");

            if (separator < 0) continue;

            const key = trimmed.slice(0, separator).trim();
            const value = trimmed.slice(separator + 1).trim();

            properties[key] = value;
          }

          const values = properties[this.config.wavelengthProperty];

          if (!values) {
            throw new Error("Wavelength metadata not found.");
          }

          const result = values.trim().split(/\s+/).map(Number);

          if (result.some((value) => !Number.isFinite(value))) {
            throw new Error("Invalid wavelength metadata.");
          }

          return result;
        })
        .catch((error) => {
          this.wavelengthPromise = null;
          throw error;
        });
    }

    return this.wavelengthPromise;
  }

  // ==========================================================
  // PRELOAD
  // ==========================================================

  async preloadAllLayers(
    request: Omit<TileRequest, "layer">,
    onProgress: (progress: PreloadProgress) => void,
    shouldCancel: () => boolean,
  ): Promise<PreloadResult> {
    let nextIndex = 0;
    let completed = 0;

    const failedLayers: number[] = [];
    const total = this.validLayers.length;

    const concurrency = Math.max(1, this.config.preloadConcurrency ?? 4);

    const worker = async () => {
      while (!shouldCancel()) {
        const index = nextIndex++;

        if (index >= total) return;

        const layer = this.validLayers[index];

        try {
          await this.getTile({
            ...request,
            layer,
          });
        } catch (error) {
          failedLayers.push(layer);

          console.warn(`Could not load ${this.name} layer ${layer}`, error);
        }

        completed++;

        onProgress({
          completed,
          total,
        });
      }
    };

    await Promise.all(Array.from({ length: concurrency }, () => worker()));

    return {
      completed,
      failedLayers,
      total,
    };
  }

  // ==========================================================
  // CACHE CONTROL
  // ==========================================================

  getCachedImage(request: TileRequest): HTMLImageElement | null {
    const metadata = this.getTileMetadata(request);
    return loadedImages.get(metadata.url) ?? null;
  }

  clearImageCache(): void {
    imageCache.clear();
    loadedImages.clear();
  }
}

// ============================================================
// SPHEREX
// ============================================================

export class SpherexDataSource extends SatelliteDataSource {
  constructor() {
    super({
      name: "SPHEREx",

      hipsUrl:
        "https://irsa.ipac.caltech.edu/data/hips/SPHEREx/spherex_allsky_cube",

      minFov: 1.8,
      maxFov: 58.6,

      maxOrder: 5,
      maxLayer: 101,

      blankLayers: [0, 50, 51],

      wavelengthProperty: "data_cube_crval3_arr",

      preloadConcurrency: 4,
    });
  }
}
