#!/usr/bin/env python3
"""
SPHEREx cutout backend (FastAPI), tile-based

/cutout returns a grayscale PNG of the fixed sky TILE that contains the
requested RA/Dec (not a cutout centred on it), plus metadata saying where the
requested point is inside that image. Nearby RA/Dec values hit the same cached
tile, so nothing is re-requested.

Run:
    pip install fastapi uvicorn astroquery astropy numpy matplotlib
    uvicorn spherex_api:app --host 0.0.0.0 --port 8000

Examples:
    /cutout?ra=210.80227&dec=54.34895&size=0.1&wavelength=2.5
    /cutout?ra=210.80227&dec=54.34895&size=0.1&wavelength=2.5&dataset=dr1
    /wavelengths?ra=...&dec=...&size=...    (what is already cached for the tile)
    /datasets   /collections   /docs

The tile grid (deterministic, depends only on `size`)
-----------------------------------------------------
`size` is the grid step in degrees.
  * Declination is cut into bands of height `size`, starting at
    dec = -90 + size:  band j spans [-90+size + j*size, +size], and the last
    band is clipped at 90 - size.
  * Two polar caps (dec >= 90-size and dec <= -90+size) are one tile each,
    centred on the pole.
  * Band j is cut into n = ceil(360 * cos(d) / size) equal RA cells, where d is
    the edge of the band closest to the equator (0 if the band straddles it).
    So a cell is never wider than `size` on the sky, and RA cells shrink
    toward the poles just like the sphere does.
  * Cell (j, i) is the region lo <= dec < hi, i*360/n <= ra < (i+1)*360/n.
    Every RA/Dec belongs to exactly one cell, and the cell centre is a pure
    function of (size, j, i).

The image fetched for a tile is centred on the cell centre. Its side is
2 * (largest great-circle distance from the centre to the cell boundary),
plus TILE_MARGIN. That is a square that contains the cell's circumscribed
circle, so the cell is covered whatever the detector's orientation on the sky
(SPHEREx images are not north-up).

Where the point is in the image
-------------------------------
Every pixel position is computed from the real WCS of the downloaded cutout
(header saved next to the PNG, SIP included), never from a flat-sky
approximation. The saved wavelength map is the real spectral WCS evaluated per
pixel, so the wavelength at the point is exact for any RA/Dec in the tile.
"""

import concurrent.futures
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Literal, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.utils.data import conf
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CACHE_DIR = Path(os.environ.get("SPHEREX_CACHE_DIR", "spherex_cache"))
WORKERS = int(os.environ.get("SPHEREX_WORKERS", "8"))

# "Close enough": |found - wanted| <= TOLERANCE_FACTOR * wanted
# (SPHEREx resolving power is R ~ 35-130, i.e. 1-3 %; 0.02 = 2 % is a sane default)
TOLERANCE_FACTOR = float(os.environ.get("SPHEREX_TOLERANCE_FACTOR", "0.02"))

# Max number of new exposures probed per request before giving up and
# returning the closest one found so far.
MAX_PROBES = int(os.environ.get("SPHEREX_MAX_PROBES", "120"))

# Size (deg) of the small cutouts used just to read the wavelength.
# 0 disables probing (always download the full tile).
PROBE_SIZE_DEG = float(os.environ.get("SPHEREX_PROBE_SIZE_DEG", "0.01"))

# Extra fraction added to the tile's circumscribed square (safety margin).
TILE_MARGIN = float(os.environ.get("SPHEREX_TILE_MARGIN", "0.02"))

SEARCH_RADIUS_ARCSEC = 10
RETRIES = 3

# Wavelength coverage (micron) of each SPHEREx detector.
DETECTOR_BANDS = {
    1: (0.75, 1.09), 2: (1.10, 1.62), 3: (1.63, 2.41),
    4: (2.42, 3.82), 5: (3.83, 4.41), 6: (4.42, 5.00),
}

# Public dataset names -> IRSA SIA collection names.
# The DR1 collection name is an assumption: confirm it with GET /datasets and
# override without editing code:  export SPHEREX_COLLECTION_DR1=<name>
DATASETS = {
    "dr1": os.environ.get("SPHEREX_COLLECTION_DR1", "spherex_dr1"),
    "qr3": os.environ.get("SPHEREX_COLLECTION_QR3", "spherex_qr3"),
}
Dataset = Literal["dr1", "qr3"]
DEFAULT_DATASET = "qr3"

logging.getLogger("astropy").setLevel(logging.ERROR)
log = logging.getLogger("spherex_api")
conf.remote_timeout = 120

CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SPHEREx cutout service", version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Cache", "X-Match", "X-Probes", "X-Tolerance-um",
                    "X-Wavelength-um", "X-Bandwidth-um",
                    "X-Wavelength-Offset-um", "X-Source-Image",
                    "X-Tile-Id", "X-Point-X", "X-Point-Y",
                    "X-Point-Fraction-X", "X-Point-Fraction-Y",
                    "X-Point-In-Image", "X-Point-Wavelength-um",
                    "X-Image-Width", "X-Image-Height", "X-Metadata"],
)

# One lock per cache key: identical concurrent requests share the search.
_locks: dict = {}
_locks_guard = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


# ---------------------------------------------------------
# Tile grid (pure functions of size, ra, dec)
# ---------------------------------------------------------

def _sep_deg(ra1, dec1, ra2, dec2):
    """Great-circle separation in degrees (Vincenty, stable at all scales)."""
    l1, p1, l2, p2 = map(np.radians, (ra1, dec1, ra2, dec2))
    dl = l2 - l1
    num = np.hypot(np.cos(p2) * np.sin(dl),
                   np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))
    den = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(num, den))


def tile_for(ra: float, dec: float, size: float) -> dict:
    """The unique grid cell containing (ra, dec), and how to fetch it."""
    ra = ra % 360.0
    if ra >= 360.0:          # guards ra % 360 rounding to 360.0
        ra = 0.0

    if dec >= 90.0 - size or dec <= -90.0 + size:
        north = dec > 0
        pole = 90.0 if north else -90.0
        return _finish_tile(
            tile_id="capN" if north else "capS",
            band=None, index=None, n_ra=1,
            ra_lo=0.0, ra_hi=360.0,
            dec_lo=(90.0 - size) if north else -90.0,
            dec_hi=90.0 if north else (-90.0 + size),
            center_ra=0.0, center_dec=pole,
            radius=size, size=size, polar_cap=True,
        )

    start = -90.0 + size
    top = 90.0 - size
    nrows = max(1, math.ceil((top - start) / size - 1e-9))
    j = min(int((dec - start) // size), nrows - 1)
    lo = start + j * size
    hi = min(lo + size, top)

    d_min = 0.0 if lo <= 0.0 <= hi else min(abs(lo), abs(hi))
    n_ra = max(1, math.ceil(360.0 * math.cos(math.radians(d_min)) / size - 1e-9))
    d_ra = 360.0 / n_ra
    i = min(int(ra // d_ra), n_ra - 1)

    ra_lo, ra_hi = i * d_ra, (i + 1) * d_ra
    c_ra, c_dec = (ra_lo + ra_hi) / 2.0, (lo + hi) / 2.0

    # Radius of the circle around the centre that contains the whole cell.
    t = np.linspace(0.0, 1.0, 33)
    ras = np.concatenate([ra_lo + t * d_ra, ra_lo + t * d_ra,
                          np.full_like(t, ra_lo), np.full_like(t, ra_hi)])
    decs = np.concatenate([np.full_like(t, lo), np.full_like(t, hi),
                           lo + t * (hi - lo), lo + t * (hi - lo)])
    radius = float(np.max(_sep_deg(c_ra, c_dec, ras, decs)))

    return _finish_tile(
        tile_id=f"r{j}_c{i}", band=j, index=i, n_ra=n_ra,
        ra_lo=ra_lo, ra_hi=ra_hi, dec_lo=lo, dec_hi=hi,
        center_ra=c_ra, center_dec=c_dec,
        radius=radius, size=size, polar_cap=False,
    )


def _finish_tile(tile_id, band, index, n_ra, ra_lo, ra_hi, dec_lo, dec_hi,
                 center_ra, center_dec, radius, size, polar_cap):
    fetch = math.ceil(2.0 * radius * (1.0 + TILE_MARGIN) * 1e6) / 1e6
    return {
        "id": tile_id, "band": band, "index": index, "n_ra_in_band": n_ra,
        "grid_size_deg": size, "polar_cap": polar_cap,
        "ra_range_deg": [ra_lo, ra_hi], "dec_range_deg": [dec_lo, dec_hi],
        "center_ra": center_ra, "center_dec": center_dec,
        "fetch_size_deg": fetch,
    }


# ---------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------

def resolve_collection(dataset: str) -> str:
    return DATASETS[dataset]


def cache_key(tile, collection, detector) -> str:
    raw = (f"tile|{tile['grid_size_deg']:.6f}|{tile['id']}|"
           f"{tile['fetch_size_deg']:.6f}|{collection}|{detector}")
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def key_dir(key: str) -> Path:
    return CACHE_DIR / key


def read_state(key: str) -> Optional[dict]:
    path = key_dir(key) / "state.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def write_state(key: str, state: dict) -> None:
    d = key_dir(key)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "state.json.tmp"
    tmp.write_text(json.dumps(state))
    os.replace(tmp, d / "state.json")  # atomic


# ---------------------------------------------------------
# IRSA query (SIA) -> candidate exposures
# ---------------------------------------------------------

def _to_um(value):
    """ObsCore em_min/em_max are in metres; tolerate values already in um."""
    try:
        v = float(value)
    except Exception:
        return None
    if not math.isfinite(v) or v <= 0:
        return None
    return v * 1e6 if v < 1e-3 else v


def query_candidates(coord, collection, detector):
    res = Irsa.query_sia(
        pos=(coord, SEARCH_RADIUS_ARCSEC * u.arcsec),
        collection=collection,
    )

    if "access_url" not in res.colnames:
        raise RuntimeError(
            f"No access_url column. Available columns: {res.colnames}"
        )

    has_em = "em_min" in res.colnames and "em_max" in res.colnames
    has_t = "t_min" in res.colnames

    rows = []
    for r in res:
        url = str(r["access_url"])
        m = re.search(r"D(\d)_spx", url)
        det = int(m.group(1)) if m else None

        if detector is not None and det != detector:
            continue

        t = float(r["t_min"]) if has_t else 0.0
        rows.append({
            "url": url,
            "det": det,
            "em_min": _to_um(r["em_min"]) if has_em else None,
            "em_max": _to_um(r["em_max"]) if has_em else None,
            "t": t if math.isfinite(t) else 0.0,
        })

    rows.sort(key=lambda c: c["t"])
    return rows


def select_candidates(cands, wavelength, tol):
    """Keep only exposures whose band could contain the wanted wavelength."""
    out = []
    for c in cands:
        lo, hi = c.get("em_min"), c.get("em_max")
        if lo is None or hi is None:
            band = DETECTOR_BANDS.get(c.get("det"))
            if band:
                lo, hi = band
        if lo is not None and hi is not None:
            if not (lo - tol <= wavelength <= hi + tol):
                continue
        out.append(c)
    return out


# ---------------------------------------------------------
# Wavelength / geometry of a cutout
# ---------------------------------------------------------

def get_wavelength(hdul, coord):
    image_hdu = hdul["IMAGE"]

    spatial_wcs = WCS(image_hdu.header)
    x, y = spatial_wcs.world_to_pixel(coord)

    ny, nx = image_hdu.data.shape
    if not (0 <= x < nx and 0 <= y < ny):
        raise ValueError(
            f"Requested position falls outside cutout: x={x:.2f}, y={y:.2f}"
        )

    spectral_wcs = WCS(header=image_hdu.header, fobj=hdul, key="W")
    spectral_wcs.sip = None  # spectral WCS does not use spatial SIP

    wavelength, bandwidth = spectral_wcs.pixel_to_world(x, y)
    wl = wavelength.to_value(u.um)
    bw = bandwidth.to_value(u.um)

    if not np.isfinite(wl) or wl <= 0:
        raise ValueError("Invalid wavelength returned by spectral WCS")

    return float(wl), float(bw)


def build_geometry(hdul) -> dict:
    """
    Everything needed to locate any sky position in this cutout later without
    downloading it again: the image header (spatial WCS, SIP included) and the
    per-pixel wavelength / bandwidth maps from the spectral WCS.
    """
    image_hdu = hdul["IMAGE"]
    geom = {"header": image_hdu.header.tostring(sep="\n"),
            "wl_map": None, "bw_map": None}
    try:
        spectral_wcs = WCS(header=image_hdu.header, fobj=hdul, key="W")
        spectral_wcs.sip = None
        ny, nx = image_hdu.data.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        wl, bw = spectral_wcs.pixel_to_world(xx, yy)
        geom["wl_map"] = np.asarray(wl.to_value(u.um), dtype=np.float32)
        geom["bw_map"] = np.asarray(bw.to_value(u.um), dtype=np.float32)
    except Exception as exc:
        log.warning("Could not build wavelength map: %s", exc)
    return geom


def source_name(url: str) -> str:
    return os.path.splitext(os.path.basename(url.split("?")[0]))[0]


def read_cutout(url, coord, size_deg):
    """Download one cutout. Returns (data, wavelength_um, bandwidth_um, geometry)."""
    err = None
    for attempt in range(RETRIES):
        try:
            cutout_url = (
                f"{url}?center={coord.ra.deg},{coord.dec.deg}&size={size_deg}"
            )
            with fits.open(cutout_url, cache=False) as hdul:
                data = np.asarray(hdul["IMAGE"].data, dtype=float)
                wl, bw = get_wavelength(hdul, coord)
                geom = build_geometry(hdul)

            if data.ndim != 2:
                raise ValueError(f"Expected a 2D image, got shape {data.shape}")
            return data, wl, bw, geom

        except ValueError as exc:      # bad data: retrying will not help
            raise exc
        except Exception as exc:       # network problems: retry
            err = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(str(err))


def render_png(data, out: Path) -> None:
    finite = np.isfinite(data)
    if not np.any(finite):
        raise ValueError("Image contains no valid pixels")

    data = np.nan_to_num(
        data,
        nan=np.nanmedian(data[finite]),
        posinf=np.nanmax(data[finite]),
        neginf=np.nanmin(data[finite]),
    )
    vmin, vmax = np.percentile(data, [1, 99])
    if vmax <= vmin:
        vmax = vmin + 1e-10

    tmp = out.with_name(out.stem + ".tmp.png")
    plt.imsave(tmp, data, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    os.replace(tmp, out)


def save_geometry(png_path: Path, geom: dict) -> None:
    """Sidecar files next to the PNG: <stem>.hdr and <stem>.npz."""
    hdr = png_path.with_suffix(".hdr")
    tmp = hdr.with_name(hdr.name + ".tmp")
    tmp.write_text(geom["header"])
    os.replace(tmp, hdr)

    if geom.get("wl_map") is not None:
        npz = png_path.with_suffix(".npz")
        tmp_npz = png_path.with_name(png_path.stem + ".tmp.npz")
        np.savez_compressed(tmp_npz, wl=geom["wl_map"], bw=geom["bw_map"])
        os.replace(tmp_npz, npz)


def probe(url, coord, fetch_size):
    """Read just enough to learn the wavelength. Returns (meta, (data, geom)|None)."""
    probe_size = fetch_size
    if PROBE_SIZE_DEG > 0:
        probe_size = min(fetch_size, PROBE_SIZE_DEG)

    try:
        data, wl, bw, geom = read_cutout(url, coord, probe_size)
    except Exception as exc:
        return {"error": str(exc)}, None

    full = probe_size >= fetch_size
    return {"wavelength_um": wl, "bandwidth_um": bw}, ((data, geom) if full else None)


# ---------------------------------------------------------
# Where is the requested point inside the cached tile image?
# ---------------------------------------------------------

def _bilinear(arr, x, y):
    ny, nx = arr.shape
    if not (-0.5 <= x <= nx - 0.5 and -0.5 <= y <= ny - 0.5):
        return None
    x = min(max(x, 0.0), nx - 1.0)
    y = min(max(y, 0.0), ny - 1.0)
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1)
    fx, fy = x - x0, y - y0
    v = (arr[y0, x0] * (1 - fx) * (1 - fy) + arr[y0, x1] * fx * (1 - fy)
         + arr[y1, x0] * (1 - fx) * fy + arr[y1, x1] * fx * fy)
    return float(v) if np.isfinite(v) else None


def locate_point(png_path: Path, ra, dec, wavelength, tol) -> dict:
    hdr_path = png_path.with_suffix(".hdr")
    header = fits.Header.fromstring(hdr_path.read_text(), sep="\n")
    wcs = WCS(header)
    nx, ny = int(header["NAXIS1"]), int(header["NAXIS2"])

    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    x, y = (float(v) for v in wcs.world_to_pixel(coord))

    # Image directions of north and east (1 arcmin steps along the sphere).
    def direction(pa_deg):
        p = coord.directional_offset_by(pa_deg * u.deg, 1 * u.arcmin)
        x2, y2 = wcs.world_to_pixel(p)
        dx, dy = float(x2) - x, -(float(y2) - y)       # dy in PNG orientation
        n = math.hypot(dx, dy)
        return [dx / n, dy / n], n

    north, px_per_arcmin = direction(0.0)
    east, _ = direction(90.0)

    # PNG coordinates: origin top-left, in pixel-edge units (pixel k spans k..k+1).
    png_x = x + 0.5
    png_y = ny - (y + 0.5)
    in_image = (-0.5 <= x <= nx - 0.5) and (-0.5 <= y <= ny - 0.5)

    wl = bw = None
    npz = png_path.with_suffix(".npz")
    if npz.exists() and in_image:
        with np.load(npz) as z:
            wl = _bilinear(z["wl"], x, y)
            bw = _bilinear(z["bw"], x, y)

    return {
        "image": {
            "width": nx, "height": ny,
            "note": ("Detector frame, not north-up. PNG origin is top-left; "
                     "x_px/y_px are 0-based array indices with y measured "
                     "from the bottom; png_x/png_y are pixel-edge coordinates "
                     "from the top-left corner."),
            "pixel_scale_arcsec": 60.0 / px_per_arcmin,
            "north_unit_vector_png": north,
            "east_unit_vector_png": east,
        },
        "point": {
            "x_px": x, "y_px": y,
            "png_x": png_x, "png_y": png_y,
            "fraction_x": png_x / nx, "fraction_y": png_y / ny,
            "in_image": bool(in_image),
            "wavelength_um": wl, "bandwidth_um": bw,
            "wavelength_offset_um": None if wl is None else wl - wavelength,
            "within_tolerance_at_point":
                None if wl is None else abs(wl - wavelength) <= tol,
        },
    }


# ---------------------------------------------------------
# The search
# ---------------------------------------------------------

def best_probed(state, wavelength):
    best_url, best_meta = None, None
    for url, meta in state["probes"].items():
        if "wavelength_um" not in meta:
            continue
        if best_meta is None or (
            abs(meta["wavelength_um"] - wavelength)
            < abs(best_meta["wavelength_um"] - wavelength)
        ):
            best_url, best_meta = url, meta
    return best_url, best_meta


def find_cutout(ra, dec, size, collection, detector, wavelength,
                factor, max_probes, refresh):
    tile = tile_for(ra, dec, size)
    key = cache_key(tile, collection, detector)
    coord = SkyCoord(tile["center_ra"] * u.deg, tile["center_dec"] * u.deg,
                     frame="icrs")                       # tile centre, not the request
    fetch_size = tile["fetch_size_deg"]
    tol = factor * wavelength

    with _lock_for(key):
        state = None if refresh else read_state(key)

        # ---- candidate list (one IRSA query per tile, then cached) ----
        if state is None:
            try:
                cands = query_candidates(coord, collection, detector)
            except Exception as exc:
                raise HTTPException(
                    502,
                    f"IRSA query failed for collection '{collection}': {exc}. "
                    "Check GET /datasets to see which collections IRSA offers.",
                )
            if not cands:
                raise HTTPException(
                    404, "No SPHEREx observations found for this tile."
                )
            if refresh:
                import shutil
                shutil.rmtree(key_dir(key), ignore_errors=True)
            state = {
                "tile": tile,
                "collection": collection, "detector": detector,
                "candidates": cands, "probes": {}, "files": {},
            }
            write_state(key, state)

        # ---- already have something close enough? ----
        best_url, best_meta = best_probed(state, wavelength)
        probes_done = 0
        kept_data = {}

        def is_match(meta):
            return meta is not None and \
                abs(meta["wavelength_um"] - wavelength) <= tol

        if not is_match(best_meta):
            todo = [
                c for c in select_candidates(state["candidates"], wavelength, tol)
                if c["url"] not in state["probes"]
            ]
            todo.sort(key=lambda c: (
                abs(wavelength - (c["em_min"] + c["em_max"]) / 2)
                if c.get("em_min") and c.get("em_max") else 0.0
            ))

            ex = concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS)
            pending: dict = {}
            it = iter(todo)
            submitted = 0
            try:
                while True:
                    while len(pending) < WORKERS and submitted < max_probes:
                        c = next(it, None)
                        if c is None:
                            break
                        pending[ex.submit(probe, c["url"], coord, fetch_size)] = c["url"]
                        submitted += 1

                    if not pending:
                        break

                    done, _ = concurrent.futures.wait(
                        pending, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    found = False
                    for f in done:
                        url = pending.pop(f)
                        meta, payload = f.result()
                        state["probes"][url] = meta
                        probes_done += 1
                        if payload is not None:
                            kept_data[url] = payload
                        if is_match(meta):
                            found = True
                    if found:
                        break
            finally:
                # Don't wait for downloads still in flight; their results are dropped.
                ex.shutdown(wait=False, cancel_futures=True)
                write_state(key, state)

            best_url, best_meta = best_probed(state, wavelength)

        if best_meta is None:
            errors = [m["error"] for m in state["probes"].values() if "error" in m]
            raise HTTPException(
                404,
                "No usable exposure found for this wavelength "
                f"(probed {len(state['probes'])}, failures: {len(errors)}"
                + (f", e.g. {errors[0]}" if errors else "") + ").",
            )

        # ---- make sure the PNG + geometry for the winner exist ----
        png_downloaded = False
        fname = state["files"].get(best_url)
        have_files = bool(
            fname
            and (key_dir(key) / fname).exists()
            and (key_dir(key) / fname).with_suffix(".hdr").exists()
        )
        if not have_files:
            payload = kept_data.get(best_url)
            if payload is None:
                try:
                    data, _, _, geom = read_cutout(best_url, coord, fetch_size)
                except Exception as exc:
                    raise HTTPException(502, f"Cutout download failed: {exc}")
            else:
                data, geom = payload
            wl_str = f"{best_meta['wavelength_um']:.4f}".replace(".", "p")
            fname = f"{source_name(best_url)}_{wl_str}um.png"
            key_dir(key).mkdir(parents=True, exist_ok=True)
            try:
                render_png(data, key_dir(key) / fname)
                save_geometry(key_dir(key) / fname, geom)
            except Exception as exc:
                raise HTTPException(502, f"Could not render image: {exc}")
            state["files"][best_url] = fname
            write_state(key, state)
            png_downloaded = True

        png_path = key_dir(key) / fname
        located = locate_point(png_path, ra, dec, wavelength, tol)

        return {
            "path": png_path,
            "meta": best_meta,
            "source": source_name(best_url),
            "match": is_match(best_meta),
            "tolerance_um": tol,
            "probes": probes_done,
            "cached": probes_done == 0 and not png_downloaded,
            "tile": tile,
            "located": located,
        }


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

RA_Q = Query(..., ge=0, le=360, description="Right ascension, degrees")
DEC_Q = Query(..., ge=-90, le=90, description="Declination, degrees")
SIZE_Q = Query(0.1, gt=0, le=10.0,
               description="Tile grid size, degrees (defines the fixed grid)")
DATASET_Q = Query(DEFAULT_DATASET, description="SPHEREx data release: dr1 or qr3")


@app.get("/cutout", response_class=FileResponse)
def cutout(
    ra: float = RA_Q,
    dec: float = DEC_Q,
    wavelength: float = Query(..., gt=0, le=10,
                              description="Wanted wavelength, micron"),
    size: float = SIZE_Q,
    dataset: Dataset = DATASET_Q,
    detector: Optional[int] = Query(None, ge=1, le=6),
    tolerance_factor: float = Query(
        TOLERANCE_FACTOR, gt=0, le=1,
        description="Stop searching once |found - wanted| <= factor * wanted "
                    "at the tile centre (0.02 = within 2 %)."),
    max_probes: int = Query(
        MAX_PROBES, ge=1, le=2000,
        description="Max new exposures to probe in this request."),
    strict: bool = Query(
        False, description="404 instead of returning the nearest image when "
                           "nothing within tolerance was found."),
    refresh: bool = Query(False, description="Ignore cache and start over"),
):
    """PNG of the grid tile containing (ra, dec), plus where the point is in it."""
    collection = resolve_collection(dataset)

    r = find_cutout(ra, dec, size, collection, detector, wavelength,
                    tolerance_factor, max_probes, refresh)

    wl = r["meta"]["wavelength_um"]          # wavelength at the TILE CENTRE
    if strict and not r["match"]:
        raise HTTPException(
            404,
            f"Nothing within {tolerance_factor:.1%} of {wavelength} um "
            f"after {r['probes']} probes; closest is {wl:.4f} um.",
        )

    loc = r["located"]
    pt, img, tile = loc["point"], loc["image"], r["tile"]

    metadata = {
        "requested": {"ra": ra, "dec": dec, "wavelength_um": wavelength},
        "tile": tile,
        "match": {
            "within_tolerance_at_tile_center": r["match"],
            "tile_center_wavelength_um": wl,
            "tile_center_bandwidth_um": r["meta"]["bandwidth_um"],
            "tolerance_um": r["tolerance_um"],
            "source_image": r["source"],
        },
        "image": img,
        "point": pt,
    }

    def fmt(v, spec=".4f"):
        return "" if v is None else format(v, spec)

    return FileResponse(
        r["path"],
        media_type="image/png",
        headers={
            "X-Cache": "HIT" if r["cached"] else "MISS",
            "X-Match": "within-tolerance" if r["match"] else "nearest",
            "X-Probes": str(r["probes"]),
            "X-Tolerance-um": f"{r['tolerance_um']:.4f}",
            "X-Wavelength-um": f"{wl:.4f}",
            "X-Bandwidth-um": f"{r['meta']['bandwidth_um']:.4f}",
            "X-Wavelength-Offset-um": f"{wl - wavelength:+.4f}",
            "X-Source-Image": r["source"],
            "X-Tile-Id": tile["id"],
            "X-Image-Width": str(img["width"]),
            "X-Image-Height": str(img["height"]),
            "X-Point-X": fmt(pt["png_x"], ".3f"),
            "X-Point-Y": fmt(pt["png_y"], ".3f"),
            "X-Point-Fraction-X": fmt(pt["fraction_x"], ".6f"),
            "X-Point-Fraction-Y": fmt(pt["fraction_y"], ".6f"),
            "X-Point-In-Image": "true" if pt["in_image"] else "false",
            "X-Point-Wavelength-um": fmt(pt["wavelength_um"]),
            "X-Metadata": json.dumps(metadata, separators=(",", ":")),
        },
    )


@app.get("/wavelengths")
def wavelengths(
    ra: float = RA_Q,
    dec: float = DEC_Q,
    size: float = SIZE_Q,
    dataset: Dataset = DATASET_Q,
    detector: Optional[int] = Query(None, ge=1, le=6),
):
    """Tile-centre wavelengths already discovered for the tile containing
    (ra, dec). Never hits the network."""
    collection = resolve_collection(dataset)
    tile = tile_for(ra, dec, size)
    state = read_state(cache_key(tile, collection, detector))
    if state is None:
        return {"cached": False, "tile": tile, "candidates": 0, "probed": 0,
                "wavelengths_um": []}

    found = sorted(
        m["wavelength_um"] for m in state["probes"].values()
        if "wavelength_um" in m
    )
    return {
        "cached": True,
        "tile": tile,
        "candidates": len(state["candidates"]),
        "probed": len(state["probes"]),
        "failed": sum(1 for m in state["probes"].values() if "error" in m),
        "wavelengths_um": found,
    }


@app.get("/datasets")
def datasets():
    """Configured datasets and whether IRSA actually offers their collection."""
    try:
        cols = Irsa.list_collections(servicetype="SIA")
        available = {str(n) for n in cols["collection"]}
        spherex = sorted(n for n in available if "spherex" in n.lower())
    except Exception as exc:
        raise HTTPException(502, f"Could not list collections: {exc}")

    return {
        "default": DEFAULT_DATASET,
        "datasets": {
            name: {"collection": col, "available_at_irsa": col in available}
            for name, col in DATASETS.items()
        },
        "all_spherex_collections_at_irsa": spherex,
    }


@app.get("/collections")
def collections():
    """SPHEREx collections available through IRSA SIA."""
    try:
        cols = Irsa.list_collections(servicetype="SIA")
        return [str(n) for n in cols["collection"] if "spherex" in str(n).lower()]
    except Exception as exc:
        raise HTTPException(502, f"Could not list collections: {exc}")


@app.delete("/cache")
def clear_cache(
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    size: Optional[float] = None,
    dataset: Dataset = DATASET_Q,
    detector: Optional[int] = None,
):
    """Delete the cached tile containing (ra, dec) (give ra, dec, size) or the
    whole cache."""
    import shutil

    collection = resolve_collection(dataset)

    if ra is not None and dec is not None and size is not None:
        tile = tile_for(ra, dec, size)
        key = cache_key(tile, collection, detector)
        with _lock_for(key):
            shutil.rmtree(key_dir(key), ignore_errors=True)
        return {"cleared": key, "tile": tile["id"]}

    for d in CACHE_DIR.iterdir():
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
    return {"cleared": "all"}


@app.get("/health")
def health():
    return {"status": "ok"}