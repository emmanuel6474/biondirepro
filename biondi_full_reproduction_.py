from __future__ import annotations
import argparse
import itertools
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import textwrap
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from PIL import Image
from scipy import ndimage, signal
try:
    import fitz
except Exception as exc:
    raise SystemExit('PyMuPDF is required: pip install pymupdf') from exc
try:
    import requests
except Exception:
    requests = None
try:
    import cv2
except Exception:
    cv2 = None
RNG_SEED = 20260814

def mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def jdump(obj, p: Path):

    def conv(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x))
    p.write_text(json.dumps(obj, indent=2, default=conv), encoding='utf-8')

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def safe_corr(a, b) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float('nan')
    a = a[ok] - np.mean(a[ok])
    b = b[ok] - np.mean(b[ok])
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else float('nan')

def download(url: str, dest: Path, timeout=60) -> bool:
    if requests is None:
        print(f'[skip] requests unavailable; cannot download {url}')
        return False
    try:
        r = requests.get(url, timeout=timeout, headers={'User-Agent': 'Mozilla/5.0 research-replication'})
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        print(f'[download failed] {url}: {e}')
        return False

def pdf_text(path: Path) -> str:
    doc = fitz.open(path)
    return '\n'.join((p.get_text('text') for p in doc))

def extract_embedded_images(pdf_path: Path, page_1based: int, outdir: Path) -> List[Path]:
    doc = fitz.open(pdf_path)
    p = doc[page_1based - 1]
    outs = []
    for i, item in enumerate(p.get_images(full=True)):
        info = doc.extract_image(item[0])
        out = outdir / f"page{page_1based:02d}_img{i:02d}_xref{item[0]}.{info['ext']}"
        out.write_bytes(info['image'])
        outs.append(out)
    return outs

def audit_luna_zip(zip_path: Path, outdir: Path) -> dict:
    out = mkdir(outdir / 'luna_audit')
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(td)
        roots = [p for p in Path(td).iterdir() if p.is_dir()]
        root = roots[0] if len(roots) == 1 else Path(td)

        def read(rel):
            p = root / rel
            return p.read_text(errors='replace') if p.exists() else ''
        run_suite = read('orchestration/run_benchmark_suite.py')
        scorer = read('orchestration/score_endtoend_synthetic.py')
        generator = read('orchestration/generate_endtoend_synthetic_slc.py')
        pipe = read('cuda/src/sar_tomo_pipeline.cu')
        kern = read('cuda/src/sar_tomo_kernels.cu')
        types = read('cuda/include/sar_tomo_types.hpp')
        freq_truth_injected = bool(re.search('gen_meta\\.get\\(["\\\']vib_freqs["\\\'].*?investigation_freq\\s*=\\s*float', run_suite, flags=re.S))
        freq_override_as_detected = 'cfg.investigationFreq > 0.0f' in pipe and 'vr.detectedInvestigationFreqHz = investigationFreq' in pipe
        maxz_score = '.max(axis=1)' in scorer
        has_depth_truth = bool(re.search('["\\\'](?:z_true|depth_m|depth_true)["\\\']', generator))
        z_asym = 'adaptiveZBelowM = 180.0' in types and 'adaptiveZAboveM = 32.0' in types
        row_major_cov = 'row = (int)(local / (size_t)nPairs)' in kern and 'col = (int)(local % (size_t)nPairs)' in kern and ('R[idx] = make_cuComplex' in kern)
        uses_cublas_solver = 'cublasCgetrfBatched' in pipe or 'cublasCgetrfBatched' in read('cuda/src/sar_tomo_inversion.cu')
        uses_cusolver = 'cusolverDnCheevd' in pipe or 'cusolverDnCheevd' in read('cuda/src/sar_tomo_inversion.cu')
        three_x_in_code = any((tok in '\n'.join([run_suite, scorer, generator, pipe, kern, types]) for tok in ['3x', '3×', 'pre-register', 'preregister']))
        rng = np.random.default_rng(RNG_SEED)
        nobs = 64
        nz = 161
        kz = np.linspace(-0.08, 0.08, nobs)
        zgrid = np.linspace(-100, 100, nz)
        A = np.exp(1j * np.outer(kz, zgrid))
        Am = np.exp(1j * np.outer(-kz, zgrid))
        y = (rng.normal(size=nobs) + 1j * rng.normal(size=nobs)) / np.sqrt(2)
        h = np.linalg.pinv(A, rcond=1e-10) @ y
        hm = np.linalg.pinv(Am, rcond=1e-10) @ y
        mirror_relerr = float(np.linalg.norm(np.abs(hm) - np.abs(h)[::-1]) / np.linalg.norm(np.abs(h)))
        x = np.arange(128)
        depth = np.arange(64)
        H1 = np.zeros((len(x), len(depth)))
        H2 = np.zeros_like(H1)
        target_x = 64
        H1[target_x, 12] = 30
        H2[target_x, 47] = 30
        H1[:, :] += 1
        H2[:, :] += 1
        c1 = H1.max(axis=1)[target_x] / np.median(np.delete(H1.max(axis=1), target_x))
        c2 = H2.max(axis=1)[target_x] / np.median(np.delete(H2.max(axis=1), target_x))
        profile_corr = safe_corr(H1[target_x], H2[target_x])
        kz2 = np.linspace(-1.1, 1.1, 40)
        zg = np.linspace(-8, 8, 641)
        z0 = 3.75
        Avec = np.exp(1j * np.outer(kz2, zg))
        y0 = np.exp(1j * kz2 * z0)
        R = np.outer(y0, np.conj(y0)) + 0.001 * np.eye(len(y0))
        w, V = np.linalg.eigh(R)
        wT, VT = np.linalg.eigh(R.T)
        e = V[:, -1]
        eT = VT[:, -1]
        p = np.abs(np.conj(e) @ Avec) ** 2
        pT = np.abs(np.conj(eT) @ Avec) ** 2
        z_correct = float(zg[np.argmax(p)])
        z_transposed = float(zg[np.argmax(pT)])
        report_pdf = root / 'docs/report.pdf'
        result = {'repository_zip_sha256': sha256(zip_path), 'embedded_report_sha256': sha256(report_pdf) if report_pdf.exists() else None, 'static_code_checks': {'synthetic_truth_frequency_is_injected_into_benchmark': freq_truth_injected, 'positive_investigation_frequency_is_recorded_as_detected': freq_override_as_detected, 'end_to_end_generator_contains_explicit_depth_ground_truth': has_depth_truth, 'score_collapses_depth_with_max_axis_1': maxz_score, 'default_adaptive_z_grid_is_asymmetric_180_below_32_above': z_asym, 'capon_covariance_kernel_writes_row_major': row_major_cov, 'dense_solver_calls_present': bool(uses_cublas_solver or uses_cusolver), '3x_or_preregistration_marker_found_in_code': three_x_in_code}, 'numerical_checks': {'negate_kz_random_noise_magnitude_mirror_relative_error': mirror_relerr, 'maxz_contrast_before': float(c1), 'maxz_contrast_after_depth_shift': float(c2), 'shifted_target_depth_profile_correlation': profile_corr, 'known_complex_source_depth_m': z0, 'music_proxy_peak_correct_covariance_m': z_correct, 'music_proxy_peak_transposed_covariance_m': z_transposed}}
        jdump(result, out / 'luna_audit.json')
        return result

@dataclass
class TomoParams:
    name: str
    wavelength_m: float
    slant_range_m: float = 650000.0
    aperture_m: float = 42000.0
    theta_deg: float = 35.0
    n_obs: int = 96

    @property
    def nominal_dz(self):
        return self.wavelength_m * self.slant_range_m / (2 * self.aperture_m)

def steering_matrix(params: TomoParams, z: np.ndarray, n_obs: Optional[int]=None) -> Tuple[np.ndarray, np.ndarray]:
    n = n_obs or params.n_obs
    theta = np.deg2rad(params.theta_deg)
    s = np.linspace(-params.aperture_m / 2, params.aperture_m / 2, n)
    bperp = s * np.sin(theta)
    kz = 4 * np.pi * bperp / (params.wavelength_m * params.slant_range_m * np.sin(theta))
    A = np.exp(1j * np.outer(kz, z))
    return (A, kz)

def complex_noise(shape, rng):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)

def add_awgn(Y, snr_db, rng):
    p = np.mean(np.abs(Y) ** 2)
    npow = p / 10 ** (snr_db / 10)
    return Y + np.sqrt(npow) * complex_noise(Y.shape, rng)

def make_test_scene(z, nx=231):
    z = np.asarray(z, float)
    x = np.linspace(0, 230, nx)
    X, Z = np.meshgrid(x, z)
    zn = (Z - z.min()) / max(z.max() - z.min(), 1e-12)
    xn = X / max(x.max(), 1e-12)
    H = np.zeros_like(X, dtype=complex)
    shallow = ((xn - 0.23) / 0.16) ** 2 + ((zn - 0.16) / 0.075) ** 2 <= 1
    H[shallow] += 0.72 * np.exp(1j * 0.35)
    mid = ((xn - 0.55) / 0.20) ** 2 + ((zn - 0.42) / 0.10) ** 2 <= 1
    H[mid] += 0.88 * np.exp(1j * 1.15)
    deep = ((xn - 0.43) / 0.27) ** 2 + ((zn - 0.78) / 0.12) ** 2 <= 1
    H[deep] += 0.78 * np.exp(1j * 2.05)
    conduit_center = 0.45 + 0.07 * (zn - 0.38)
    conduit = (np.abs(xn - conduit_center) <= 0.022) & (zn >= 0.12) & (zn <= 0.68)
    H[conduit] += 1.00 * np.exp(1j * 0.72)
    branch_center = 0.53 + 0.42 * (0.62 - zn)
    branch = (np.abs(xn - branch_center) <= 0.018) & (zn >= 0.42) & (zn <= 0.62)
    H[branch] += 0.62 * np.exp(1j * 1.62)
    core = ((xn - 0.50) / 0.018) ** 2 + ((zn - 0.50) / 0.018) ** 2 <= 1
    H[core] += 0.55 * np.exp(1j * 2.55)
    return x, H


def operator_level_test(params: TomoParams, zmin, zmax, outdir: Path, label: str, dz_override=None) -> dict:
    out = mkdir(outdir / f'operator_{label}')
    dz = float(dz_override if dz_override is not None else params.nominal_dz)
    z = np.arange(zmin, zmax + dz / 2, dz)
    A, kz = steering_matrix(params, z)
    x, H = make_test_scene(z)
    Y = A @ H
    rng = np.random.default_rng(RNG_SEED)
    cond = float(np.linalg.cond(A))
    An = A / np.linalg.norm(A, axis=0, keepdims=True)
    G = np.abs(An.conj().T @ An)
    np.fill_diagonal(G, 0)
    coherence = float(np.max(G))
    runs = {}
    H20 = None
    for snr in [40, 20, 10, 0, -5]:
        Yn = add_awgn(Y, snr, rng)
        Hh = np.linalg.pinv(A, rcond=1e-10) @ Yn
        if snr == 20:
            H20 = Hh
        runs[str(snr)] = {'relative_complex_l2': float(np.linalg.norm(Hh - H) / np.linalg.norm(H)), 'magnitude_correlation': safe_corr(np.abs(Hh), np.abs(H))}
    result = {'params': params.__dict__, 'depth_step_m': dz, 'depth_bins': len(z), 'condition_number': cond, 'max_column_coherence': coherence, 'snr_runs': runs}
    jdump(result, out / 'results.json')
    extent = [x[0], x[-1], z[-1], z[0]]
    vmax = float(np.quantile(np.abs(H), 0.999))
    err = np.abs(H20 - H) if H20 is not None else np.zeros_like(np.abs(H))
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)
    im0 = ax[0].imshow(np.abs(H), aspect='auto', origin='upper', extent=extent, vmin=0, vmax=vmax)
    ax[0].set_title('Known synthetic reflectivity')
    ax[0].set_xlabel('Position along line')
    ax[0].set_ylabel('Depth z (m)')
    im1 = ax[1].imshow(np.abs(H20), aspect='auto', origin='upper', extent=extent, vmin=0, vmax=vmax)
    ax[1].set_title('Reconstruction at 20 dB SNR')
    ax[1].set_xlabel('Position along line')
    im2 = ax[2].imshow(err, aspect='auto', origin='upper', extent=extent)
    ax[2].set_title('Absolute reconstruction error')
    ax[2].set_xlabel('Position along line')
    fig.colorbar(im1, ax=ax[:2], shrink=0.88, label='Magnitude')
    fig.colorbar(im2, ax=ax[2], shrink=0.88, label='Absolute error')
    fig.savefig(out / 'truth_reconstruction_error.png', dpi=220)
    plt.close(fig)
    if H20 is not None:
        picks = [0.23, 0.50, 0.78]
        fig, ax = plt.subplots(1, 3, figsize=(11.5, 4.1), sharey=True, constrained_layout=True)
        for j, frac in enumerate(picks):
            ix = int(round(frac * (len(x) - 1)))
            ax[j].plot(np.abs(H[:, ix]), z, linewidth=1.8, label='Known')
            ax[j].plot(np.abs(H20[:, ix]), z, linewidth=1.2, label='Recovered')
            ax[j].invert_yaxis()
            ax[j].set_title(f'Profile at x={x[ix]:.0f}')
            ax[j].set_xlabel('Magnitude')
            if j == 0:
                ax[j].set_ylabel('Depth z (m)')
            ax[j].legend(fontsize=8)
        fig.savefig(out / 'depth_profile_checks.png', dpi=220)
        plt.close(fig)
    return result


def known_depth_calibration_sensitivity(params: TomoParams, outdir: Path, z0=64.0) -> dict:
    out = mkdir(outdir / 'giza_calibration_sensitivity')
    dz = params.nominal_dz
    z = np.arange(-40, 150 + dz / 2, dz)
    A, _ = steering_matrix(params, z)
    iz = int(np.argmin(np.abs(z - z0)))
    h = np.zeros(len(z), complex)
    h[iz] = 1
    y = A @ h
    rows = []
    for frac in [-0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1]:
        p2 = TomoParams(params.name, params.wavelength_m * (1 + frac), params.slant_range_m, params.aperture_m, params.theta_deg, params.n_obs)
        Am, _ = steering_matrix(p2, z)
        hh = np.linalg.pinv(Am, rcond=1e-10) @ y
        ip = int(np.argmax(np.abs(hh)))
        a = Am[:, iz]
        focus = float(abs(np.vdot(a, y) / np.vdot(a, a)))
        rows.append({'lambda_error_pct': 100 * frac, 'peak_z_m': float(z[ip]), 'depth_bias_m': float(z[ip] - z[iz]), 'focus_at_true_slice': focus})
    pd.DataFrame(rows).to_csv(out / 'sensitivity.csv', index=False)
    plt.figure(figsize=(7, 4.2))
    plt.plot([r['lambda_error_pct'] for r in rows], [r['depth_bias_m'] for r in rows], marker='o')
    plt.xlabel('Wavelength calibration error (%)')
    plt.ylabel('Depth-coordinate bias (m)')
    plt.tight_layout()
    plt.savefig(out / 'depth_bias.png', dpi=160)
    plt.close()
    return {'true_grid_depth_m': float(z[iz]), 'rows': rows}

def ar1_cov(n, rho):
    i = np.arange(n)
    return rho ** np.abs(i[:, None] - i[None, :])

def output_noise_factor(A, rho_look):
    P = np.linalg.pinv(A, rcond=1e-10)
    C = P @ ar1_cov(A.shape[0], rho_look) @ P.conj().T
    sd = np.sqrt(np.maximum(np.real(np.diag(C)), 1e-15))
    C = C / sd[:, None] / sd[None, :]
    C = (C + C.conj().T) / 2
    w, V = np.linalg.eigh(C)
    return (V @ np.diag(np.sqrt(np.maximum(w, 0))), C)

def has_blob(mask, minspan=3, minarea=6):
    conn = np.ones((3, 3), int)
    lab, n = ndimage.label(mask, structure=conn)
    for j, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        pts = np.argwhere(lab[sl] == j)
        if len(pts) >= minarea and np.ptp(pts[:, 0]) + 1 >= minspan:
            return True
    return False

def vesuvius_null_tests(params: TomoParams, outdir: Path, trials=500) -> dict:
    out = mkdir(outdir / 'vesuvius_nulls')
    rng = np.random.default_rng(RNG_SEED)
    nx = 160
    rows = []
    for dz_label, dz in [('paper_rounded_36m', 36.0), ('formula_exact', params.nominal_dz)]:
        z = np.arange(0, 3000 + dz / 2, dz)
        for k in [64, 96, 128]:
            A, _ = steering_matrix(params, z, n_obs=k)
            cond = float(np.linalg.cond(A))
            An = A / np.linalg.norm(A, axis=0, keepdims=True)
            G = np.abs(An.conj().T @ An)
            np.fill_diagonal(G, 0)
            coh = float(np.max(G))
            L, C = output_noise_factor(A, 0.5)
            adj = float(np.median(np.abs(np.diag(C, 1)))) if len(z) > 1 else np.nan
            ntr = min(trials, 350)
            Z = complex_noise((len(z), nx * ntr), rng)
            B = (L @ Z).reshape(len(z), nx, ntr).transpose(2, 0, 1)
            for tail in [0.001, 0.0001]:
                thr = math.sqrt(-math.log(tail))
                p = float(np.mean([has_blob(np.abs(m) > thr) for m in B]))
                rows.append({'dz_mode': dz_label, 'dz_m': dz, 'k': k, 'condition_number': cond, 'max_column_coherence': coh, 'adjacent_output_noise_corr': adj, 'tail_per_coefficient': tail, 'P_any_blob_span3_area6': p})
    df = pd.DataFrame(rows)
    df.to_csv(out / 'single_view_null_and_conditioning.csv', index=False)
    dz = params.nominal_dz
    z = np.arange(0, 3000 + dz / 2, dz)
    A, _ = steering_matrix(params, z, n_obs=96)
    L, C = output_noise_factor(A, 0.5)
    ntr = min(trials, 500)
    B1 = (L @ complex_noise((len(z), nx * ntr), rng)).reshape(len(z), nx, ntr).transpose(2, 0, 1)
    Bi = (L @ complex_noise((len(z), nx * ntr), rng)).reshape(len(z), nx, ntr).transpose(2, 0, 1)
    thr = math.sqrt(-math.log(0.001))
    conn = np.ones((3, 3), int)
    cross = []
    for rv in [0, 0.25, 0.5, 0.75, 0.9, 0.97]:
        B2 = rv * B1 + np.sqrt(1 - rv ** 2) * Bi
        stable = 0
        for m1, m2 in zip(B1, B2):
            q1 = np.abs(m1) > thr
            q2 = np.abs(m2) > thr
            lab, n = ndimage.label(q1, structure=conn)
            d2 = ndimage.binary_dilation(q2, structure=conn, iterations=1)
            ok = False
            for j, sl in enumerate(ndimage.find_objects(lab), 1):
                if sl is None:
                    continue
                comp = lab == j
                pts = np.argwhere(comp)
                if len(pts) >= 6 and np.ptp(pts[:, 0]) + 1 >= 3 and (np.count_nonzero(comp & d2) >= 4):
                    ok = True
                    break
            stable += ok
        cross.append({'cross_view_noise_corr': rv, 'P_false_stable_blob': stable / ntr})
    pd.DataFrame(cross).to_csv(out / 'cross_view_null.csv', index=False)
    plt.figure(figsize=(7, 4.2))
    plt.plot([r['cross_view_noise_corr'] for r in cross], [r['P_false_stable_blob'] for r in cross], marker='o')
    plt.xlabel('Cross-view noise correlation')
    plt.ylabel('P(false persistent feature)')
    plt.title('Vesuvius null, 3 independent depth cells')
    plt.tight_layout()
    plt.savefig(out / 'cross_view_null.png', dpi=160)
    plt.close()
    result = {'single_view': rows, 'cross_view': cross}
    jdump(result, out / 'results.json')
    return result

def parse_vesuvius_paper(pdf_path: Path, outdir: Path) -> Tuple[dict, dict]:
    out = mkdir(outdir / 'vesuvius_paper')
    text = pdf_text(pdf_path)
    flat = re.sub('\\s+', ' ', text)
    result = {'pdf_sha256': sha256(pdf_path), 'table1_has_17_january_2022': bool(re.search('17\\s+January\\s+2022', flat, re.I)), 'text_says_february_is_month_sar_acquired': bool(re.search('February\\s+2022,\\s+which\\s+is\\s+the\\s+month\\s+in\\s+which\\s+the\\s+SAR\\s+data\\s+was\\s+acquired', flat, re.I)), 'doppler_bandwidth_22khz_present': bool(re.search('22\\s*kHz', flat, re.I)), 'prf_twice_doppler_bandwidth_present': bool(re.search('PRF\\s+2\\s*[·x×]?\\s*\\(?Doppler\\s+bandwidth', flat, re.I)), 'vc_re_station_present': 'IV-VCRE' in flat, 'vbkn_station_present': 'IV-VBKN' in flat, 'one_khz_detail_present': bool(re.search('1\\s*kHz', flat, re.I)), 'frequency_200hz_present': bool(re.search('200\\s*Hz', flat, re.I)), 'lambda_4_86m_present': bool(re.search('4\\.86\\s*m', flat, re.I))}
    result['computed_from_table1'] = {'prf_hz_if_2x22khz': 44000.0, 'nyquist_hz_if_prf_44khz': 22000.0, 'delta_z_from_4_86m_650km_42km_m': 4.86 * 650000 / (2 * 42000)}
    imgs = {}
    for pg in [15, 18, 19, 20]:
        imgs[pg] = extract_embedded_images(pdf_path, pg, out)
    jdump(result, out / 'paper_text_checks.json')
    return (result, imgs)

def _trace_from_color(img, bounds, color, tol=110):
    x0, x1, y0, y1 = bounds
    roi = img[y0:y1, x0:x1].astype(float)
    d = np.linalg.norm(roi - np.asarray(color, float), axis=2)
    m = d < tol
    tr = np.full(roi.shape[1], np.nan)
    for x in range(roi.shape[1]):
        ys = np.where(m[:, x])[0]
        if len(ys):
            tr[x] = np.median(ys)
    idx = np.arange(len(tr))
    good = np.isfinite(tr)
    if good.sum() > 5:
        tr = np.interp(idx, idx[good], tr[good])
    return tr

def published_curve_digitization(imgs: dict, outdir: Path) -> dict:
    out = mkdir(outdir / 'paper_curve_digitization')
    p18 = imgs.get(18, [])
    p19 = imgs.get(19, [])
    if len(p18) < 3 or len(p19) < 1:
        return {'status': 'missing expected embedded images'}
    im20 = np.array(Image.open(p18[0]).convert('RGB'))
    im21 = np.array(Image.open(p18[1]).convert('RGB'))
    im22 = np.array(Image.open(p18[2]).convert('RGB'))
    im23 = np.array(Image.open(p19[0]).convert('RGB'))
    blue = (0, 114, 189)
    orange = (217, 83, 25)
    configs = {'20a': (im20, (80, 623, 35, 466)), '20b': (im20, (703, 1246, 35, 466)), '20c': (im20, (1337, 1878, 35, 466)), '21a': (im21, (80, 625, 35, 465)), '21b': (im21, (725, 1268, 35, 466)), '21c_error': (im21, (1354, 1896, 35, 466)), '22a': (im22, (82, 627, 35, 463)), '22b': (im22, (709, 1252, 35, 463)), '22c': (im22, (1338, 1880, 35, 470)), '23': (im23, (80, 623, 40, 465))}
    rows = []
    for name, (im, b) in configs.items():
        btr = _trace_from_color(im, b, blue)
        otr = _trace_from_color(im, b, orange)
        raw = safe_corr(btr, otr)
        sm = safe_corr(ndimage.gaussian_filter1d(btr, 15), ndimage.gaussian_filter1d(otr, 15))
        rmse = float(np.sqrt(np.nanmean((btr - otr) ** 2)) / (b[3] - b[2]))
        rows.append({'panel': name, 'digitized_trace_corr': raw, 'smoothed_envelope_corr': sm, 'normalized_vertical_rmse': rmse})
    pd.DataFrame(rows).to_csv(out / 'published_curve_correlations.csv', index=False)
    return {'note': 'approximate RGB digitization of published curves; not raw arrays', 'panels': rows}

def dem_overlay_energy_test(imgs: dict, outdir: Path) -> dict:
    out = mkdir(outdir / 'paper_dem_overlay')
    p15 = imgs.get(15, [])
    if len(p15) < 2:
        return {'status': 'missing Figure 15/16 images'}
    rows = []
    for fig, path, line_rgb in [(15, p15[0], (255, 255, 0)), (16, p15[1], (255, 0, 0))]:
        im = np.array(Image.open(path).convert('RGB'))
        h, w = im.shape[:2]
        y0 = int(0.37 * h) if fig == 15 else int(0.39 * h)
        roi = im[y0:int(0.94 * h), int(0.06 * w):int(0.93 * w)]
        target = np.asarray(line_rgb, float)
        dm = np.linalg.norm(roi.astype(float) - target, axis=2)
        line = dm < 90
        yl = np.full(roi.shape[1], np.nan)
        for x in range(roi.shape[1]):
            yy = np.where(line[:, x])[0]
            if len(yy):
                yl[x] = np.median(yy)
        good = np.isfinite(yl)
        if good.sum() < 100:
            rows.append({'figure': fig, 'status': 'line extraction failed', 'line_columns': int(good.sum())})
            continue
        idx = np.arange(len(yl))
        yl = np.interp(idx, idx[good], yl[good])
        a = roi.astype(float) / 255
        warm = a[:, :, 0] + 0.55 * a[:, :, 1] - 1.15 * a[:, :, 2]

        def band_score(offset):
            vals = []
            for x, y in enumerate(yl.astype(int)):
                yc = y + offset
                lo = max(0, yc - 5)
                hi = min(warm.shape[0], yc + 6)
                if hi > lo:
                    col = warm[lo:hi, x]
                    lm = line[lo:hi, x]
                    if np.any(~lm):
                        vals.extend(col[~lm].tolist())
            return float(np.median(vals)) if vals else np.nan
        scores = {str(o): band_score(o) for o in [-80, -40, -20, 0, 20, 40, 80]}
        vals = np.array(list(scores.values()), float)
        central = scores['0']
        rank = int(np.sum(vals > central) + 1)
        rows.append({'figure': fig, 'line_columns': int(good.sum()), 'band_warmness_by_pixel_offset': scores, 'central_rank_1_is_hottest': rank})
    jdump(rows, out / 'dem_overlay_energy.json')
    return {'figures': rows}

def extract_troiano_fig2(troiano_pdf: Path, outdir: Path) -> Optional[Path]:
    imgs = extract_embedded_images(troiano_pdf, 3, outdir)
    return imgs[0] if imgs else None

def crop_nonwhite(a, thr=245):
    m = np.any(a < thr, axis=2)
    if not np.any(m):
        return a
    ys, xs = np.where(m)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def resistivity_scalar(a):
    a = a.astype(np.float32) / 255
    s = a[:, :, 0] - a[:, :, 2]
    if cv2 is not None:
        s = cv2.GaussianBlur(s, (0, 0), 5)
    else:
        s = ndimage.gaussian_filter(s, 5)
    return (s - s.mean()) / (s.std() + 1e-06)

def _best_template_match(T0, sources, scales):
    if cv2 is None:
        return None
    best = (-1, None)
    for sname, src in sources.items():
        S = (src if getattr(src, 'ndim', 0) == 2 else resistivity_scalar(src)).astype(np.float32)
        for scale in scales:
            nw = int(T0.shape[1] * scale)
            nh = int(T0.shape[0] * scale)
            if nw < 10 or nh < 10 or nw > S.shape[1] or (nh > S.shape[0]):
                continue
            T = cv2.resize(T0, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
            res = cv2.matchTemplate(S, T, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if mx > best[0]:
                best = (float(mx), (sname, float(scale), tuple(map(int, loc)), (nw, nh)))
    return best

def _tile_shuffle(T, n, rng):
    if cv2 is None:
        return T
    h, w = T.shape
    H = h // n * n
    W = w // n * n
    Tc = cv2.resize(T, (W, H))
    th, tw = (H // n, W // n)
    tiles = [Tc[i * th:(i + 1) * th, j * tw:(j + 1) * tw].copy() for i in range(n) for j in range(n)]
    rng.shuffle(tiles)
    o = np.empty_like(Tc)
    for idx, tile in enumerate(tiles):
        i, j = divmod(idx, n)
        o[i * th:(i + 1) * th, j * tw:(j + 1) * tw] = tile
    return cv2.resize(o, (w, h))

def match_biondi_mt_to_troiano(imgs: dict, troiano_pdf: Path, outdir: Path, permutations=80) -> dict:
    out = mkdir(outdir / 'troiano_match')
    if cv2 is None:
        return {'status': 'opencv unavailable'}
    p20 = imgs.get(20, [])
    if not p20:
        return {'status': 'Figure 25 image missing'}
    b = np.array(Image.open(p20[0]).convert('RGB'))
    fig2 = extract_troiano_fig2(troiano_pdf, out)
    if fig2 is None:
        return {'status': 'Troiano Figure 2 unavailable'}
    t = np.array(Image.open(fig2).convert('RGB'))
    targets = {'25c': crop_nonwhite(b[5:415, 930:1365]), '25f': crop_nonwhite(b[530:1370, 930:1365]), '25i': crop_nonwhite(b[1450:1845, 945:1335])}
    sources_rgb = {'Troiano_NS': t[15:320, 65:995], 'Troiano_WE': t[335:675, 150:970]}
    sources = {k: resistivity_scalar(v).astype(np.float32) for k, v in sources_rgb.items()}
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    scales = np.linspace(0.18, 0.9, 20)
    null_scales = np.linspace(0.18, 0.9, 10)
    for name, tg in targets.items():
        T0 = resistivity_scalar(tg)
        actual = _best_template_match(T0, sources, scales)
        vals = []
        for _ in range(permutations):
            sh = _tile_shuffle(T0, 4, rng)
            vals.append(_best_template_match(sh, sources, null_scales)[0])
        p = (np.sum(np.asarray(vals) >= actual[0]) + 1) / (len(vals) + 1)
        rows.append({'panel': name, 'best_match_corr': actual[0], 'match': actual[1], 'tile_shuffle_null_p': float(p), 'null_95': float(np.quantile(vals, 0.95)), 'null_99': float(np.quantile(vals, 0.99))})
    jdump(rows, out / 'mt_patch_matches.json')
    return {'note': 'image-level provenance/consistency check against Troiano Fig.2, not original MT numeric grid', 'matches': rows}
INGV_BASE = 'https://webservices.ingv.it/fdsnws'

def get_text_url(url, params, timeout=60):
    if requests is None:
        raise RuntimeError('requests unavailable')
    r = requests.get(url, params=params, timeout=timeout, headers={'User-Agent': 'Mozilla/5.0 research-replication'})
    r.raise_for_status()
    return (r.text, r.url)

def ingv_station_channel_metadata(station: str, outdir: Path) -> dict:
    params = {'network': 'IV', 'station': station, 'channel': 'HH?', 'starttime': '2022-01-01', 'endtime': '2022-03-01', 'level': 'channel', 'format': 'text', 'nodata': 404}
    text, url = get_text_url(f'{INGV_BASE}/station/1/query', params)
    (outdir / f'{station}_station_channels.txt').write_text(text)
    rates = []
    rows = []
    for ln in text.splitlines():
        if not ln or ln.startswith('#'):
            continue
        p = ln.split('|')
        if len(p) >= 15:
            try:
                rate = float(p[14])
                rates.append(rate)
            except:
                rate = np.nan
            rows.append({'network': p[0], 'station': p[1], 'location': p[2], 'channel': p[3], 'sample_rate_hz': rate})
    return {'request_url': url, 'rows': rows, 'unique_sample_rates_hz': sorted(set((r for r in rates if np.isfinite(r))))}

def download_mseed(station, start, end, outdir: Path) -> Optional[Path]:
    if requests is None:
        return None
    params = {'network': 'IV', 'station': station, 'location': '*', 'channel': 'HH?', 'starttime': start, 'endtime': end, 'nodata': 404}
    try:
        r = requests.get(f'{INGV_BASE}/dataselect/1/query', params=params, timeout=90, headers={'User-Agent': 'Mozilla/5.0 research-replication'})
        r.raise_for_status()
        p = outdir / f"{station}_{start.replace(':', '').replace('-', '')}.mseed"
        p.write_bytes(r.content)
        return p
    except Exception as e:
        print(f'[waveform] {station} {start}: {e}')
        return None

def analyze_mseed(path: Path, outdir: Path) -> dict:
    try:
        from obspy import read
    except Exception:
        return {'file': str(path), 'bytes': path.stat().st_size, 'status': 'saved; install obspy to decode'}
    st = read(str(path))
    rows = []
    for tr in st:
        tr = tr.copy()
        tr.detrend('linear')
        tr.detrend('demean')
        fs = float(tr.stats.sampling_rate)
        x = tr.data.astype(float)
        f, P = signal.welch(x, fs=fs, nperseg=min(len(x), max(32, int(fs))))
        rows.append({'id': tr.id, 'sample_rate_hz': fs, 'nyquist_hz': fs / 2, 'npts': int(tr.stats.npts), 'peak_psd_freq_hz': float(f[np.argmax(P)]) if len(f) else np.nan})
    return {'file': str(path), 'streams': rows}

def ingv_events_2022(outdir: Path) -> dict:
    params = {'starttime': '2022-01-01T00:00:00', 'endtime': '2022-03-01T00:00:00', 'minlat': 40.72, 'maxlat': 40.92, 'minlon': 14.28, 'maxlon': 14.58, 'orderby': 'time-asc', 'format': 'text', 'limit': 10000}
    txt, url = get_text_url(f'{INGV_BASE}/event/1/query', params)
    (outdir / 'ingv_events_jan_feb_2022.txt').write_text(txt)
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    header = None
    for ln in lines:
        if ln.startswith('#') and 'Time' in ln and ('Latitude' in ln):
            header = ln.lstrip('#').split('|')
            break
    data = [ln.split('|') for ln in lines if not ln.startswith('#')]
    if header is None and data:
        header = ['EventID', 'Time', 'Latitude', 'Longitude', 'Depth/Km', 'Author', 'Catalog', 'Contributor', 'ContributorID', 'MagType', 'Magnitude', 'MagAuthor', 'EventLocationName']
    df = pd.DataFrame(data, columns=header[:len(data[0])] if data else header)
    for col in ['Latitude', 'Longitude', 'Depth/Km', 'Magnitude']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'Time' in df:
        df['month'] = df['Time'].astype(str).str.slice(0, 7)
    df.to_csv(outdir / 'ingv_events_jan_feb_2022.csv', index=False)
    counts = df['month'].value_counts().to_dict() if 'month' in df else {}
    if all((c in df for c in ['Longitude', 'Depth/Km', 'month'])):
        plt.figure(figsize=(7.5, 4.5))
        for m, g in df.groupby('month'):
            plt.scatter(g['Longitude'], -g['Depth/Km'], s=15, label=m, alpha=0.75)
        plt.xlabel('Longitude')
        plt.ylabel('Depth coordinate (km; negative downward)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / 'ingv_events_lon_depth.png', dpi=160)
        plt.close()
    return {'request_url': url, 'counts_by_month': counts, 'n_events': len(df)}

def srtm_tile_online(outdir: Path) -> dict:
    if requests is None:
        return {'status': 'requests unavailable'}
    url = 'https://s3.amazonaws.com/elevation-tiles-prod/skadi/N40/N40E014.hgt.gz'
    try:
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        raw = gzip.decompress(r.content)
        n = int(round(math.sqrt(len(raw) / 2)))
        arr = np.frombuffer(raw, dtype='>i2').reshape(n, n).astype(float)
        lat0, lon0 = (40.0, 14.0)

        def rc(lat, lon):
            row = int(round((41 - lat) * (n - 1)))
            col = int(round((lon - 14) * (n - 1)))
            return (row, col)
        r1, c1 = rc(40.9, 14.3)
        r2, c2 = rc(40.74, 14.58)
        rr = sorted([r1, r2])
        cc = sorted([c1, c2])
        crop = arr[rr[0]:rr[1] + 1, cc[0]:cc[1] + 1]
        np.save(outdir / 'srtm_N40E014_crop.npy', crop)
        plt.figure(figsize=(6.5, 5))
        plt.imshow(crop, origin='upper', extent=[14.3, 14.58, 40.74, 40.9], aspect='auto')
        plt.scatter([14.431419, 14.429881], [40.818999, 40.829959], marker='x')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.colorbar(label='SRTM elevation (m)')
        plt.tight_layout()
        plt.savefig(outdir / 'srtm_context.png', dpi=160)
        plt.close()
        rs, cs = rc(40.8214, 14.4265)
        return {'source_url': url, 'tile_size': n, 'summit_area_sample_m': float(arr[rs, cs]), 'note': 'public SRTM-derived mirror; not exact author file'}
    except Exception as e:
        return {'status': f'download failed: {e}'}

def upsampling_information_test(fs=100.0, target_fs=44000.0, outdir: Optional[Path]=None) -> dict:
    duration = 2.0
    t = np.arange(int(fs * duration)) / fs
    x = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 40 * t)
    n = int(round(len(x) * target_fs / fs))
    y = signal.resample(x, n)
    f = np.fft.rfftfreq(n, 1 / target_fs)
    P = np.abs(np.fft.rfft(y)) ** 2
    low = P[f <= fs / 2 + 1e-12].sum()
    high = P[f > fs / 2 + 1e-12].sum()
    y_back = signal.resample(y, len(x))
    result = {'original_fs_hz': fs, 'original_nyquist_hz': fs / 2, 'upsampled_fs_hz': target_fs, 'input_tones_hz': [10.0, 40.0], 'high_frequency_energy_fraction_after_bandlimited_resampling': float(high / (low + high)), 'roundtrip_original_sample_correlation': float(np.corrcoef(x, y_back)[0, 1])}
    if outdir:
        plt.figure(figsize=(7, 4))
        plt.semilogy(f + 1e-09, P + 1e-20)
        plt.axvline(fs / 2, ls='--')
        plt.xlim(0, min(2000, target_fs / 2))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.tight_layout()
        plt.savefig(outdir / 'upsampling_does_not_create_bandwidth.png', dpi=160)
        plt.close()
    return result

def online_raw_tests(outdir: Path, sar_utc: Optional[str]) -> dict:
    out = mkdir(outdir / 'online_raw')
    result = {}
    for sta in ['VCRE', 'VBKN']:
        try:
            result[f'station_{sta}'] = ingv_station_channel_metadata(sta, out)
        except Exception as e:
            result[f'station_{sta}'] = {'error': str(e)}
    try:
        result['events'] = ingv_events_2022(out)
    except Exception as e:
        result['events'] = {'error': str(e)}
    times = []
    if sar_utc:
        import datetime as dt
        t = dt.datetime.fromisoformat(sar_utc.replace('Z', '+00:00'))
        times = [(t.isoformat().replace('+00:00', 'Z'), (t + dt.timedelta(seconds=2)).isoformat().replace('+00:00', 'Z'), 'exact_user_supplied')]
    else:
        times = [('2022-01-17T12:00:00', '2022-01-17T12:00:10', 'representative_not_synchronized'), ('2022-02-17T12:00:00', '2022-02-17T12:00:10', 'representative_not_synchronized')]
    wave = []
    for sta in ['VCRE', 'VBKN']:
        for start, end, status in times:
            p = download_mseed(sta, start, end, out)
            if p:
                a = analyze_mseed(p, out)
                a['timing_status'] = status
                a['start'] = start
                wave.append(a)
    result['waveforms'] = wave
    result['srtm'] = srtm_tile_online(out)
    rates = []
    for sta in ['VCRE', 'VBKN']:
        rates += result.get(f'station_{sta}', {}).get('unique_sample_rates_hz', [])
    if rates:
        fs = float(np.median(rates))
        result['bandwidth_consistency'] = {'median_station_sample_rate_hz': fs, 'station_nyquist_hz': fs / 2, 'paper_sar_prf_from_table_hz': 44000, 'paper_figure_full_frequency_halfspan_hz_approx': 22000, 'ratio_paper_axis_halfspan_to_station_nyquist': 22000 / (fs / 2), 'interpretation': 'If the blue station curve is plotted on the SAR Doppler-frequency axis, frequencies above station Nyquist cannot be independent seismometer information; a remapping/resampling step must be documented.'}
        result['upsampling_test'] = upsampling_information_test(fs, 44000, out)
    jdump(result, out / 'online_raw_results.json')
    return result

def build_markdown(summary: dict, out: Path):
    lines = ['# Reproduction summary', '', f'Generated by `{Path(__file__).name}`.', '']
    if 'paper' in summary:
        p = summary['paper']
        lines += ['## Vesuvius paper checks', f"- Table 1 says 17 January 2022: **{p.get('table1_has_17_january_2022')}**", f"- Text says February 2022 is the acquisition month: **{p.get('text_says_february_is_month_sar_acquired')}**", f"- Table-parameter PRF = 2 x 22 kHz => **{p.get('computed_from_table1', {}).get('prf_hz_if_2x22khz')} Hz**", f"- Resolution from 4.86 m, 650 km, 42 km => **{p.get('computed_from_table1', {}).get('delta_z_from_4_86m_650km_42km_m'):.3f} m** (paper rounds to ~36 m).", '']
    if 'luna' in summary:
        s = summary['luna']['static_code_checks']
        lines += ['## Luna code audit', f"- Synthetic truth frequency injected into benchmark: **{s['synthetic_truth_frequency_is_injected_into_benchmark']}**", f"- Explicit depth ground truth in end-to-end generator: **{s['end_to_end_generator_contains_explicit_depth_ground_truth']}**", f"- Score collapses depth with max(axis=1): **{s['score_collapses_depth_with_max_axis_1']}**", f"- Row-major covariance write detected: **{s['capon_covariance_kernel_writes_row_major']}**", '']
    if 'giza_operator' in summary:
        g = summary['giza_operator']
        lines += ['## Operator-level Y=A h', f"- Giza condition number: **{g['condition_number']:.3g}**", f"- Giza 20 dB magnitude correlation: **{g['snr_runs']['20']['magnitude_correlation']:.6f}**", '']
    if 'vesuvius_operator' in summary:
        v = summary['vesuvius_operator']
        lines += [f"- Vesuvius exact-formula dz: **{v['depth_step_m']:.3f} m**", f"- Vesuvius condition number (configured look count): **{v['condition_number']:.3g}**", f"- Vesuvius 20 dB magnitude correlation: **{v['snr_runs']['20']['magnitude_correlation']:.6f}**", '']
    if 'vesuvius_operator_36m' in summary:
        v36 = summary['vesuvius_operator_36m']
        lines += [f"- Vesuvius on the paper-rounded 36 m grid: condition number **{v36['condition_number']:.3g}**, 20 dB magnitude correlation **{v36['snr_runs']['20']['magnitude_correlation']:.6f}**.", "  This is a sensitivity result, not an exact reconstruction of the authors' undisclosed internal depth/look grid.", '']
    if 'vesuvius_nulls' in summary:
        rows = summary['vesuvius_nulls'].get('single_view', [])

        def pick(mode, k, tail):
            for r in rows:
                if r.get('dz_mode') == mode and r.get('k') == k and (abs(r.get('tail_per_coefficient', 0) - tail) < 1e-12):
                    return r
            return None
        ex = pick('formula_exact', 96, 0.001)
        rd = pick('paper_rounded_36m', 96, 0.001)
        lines += ['## Vesuvius persistence/null sensitivity']
        if ex:
            lines.append(f"- Exact-formula 37.607 m grid, k=96: P(noise blob spanning >=3 cells, area >=6) = **{ex['P_any_blob_span3_area6']:.4g}** in the Monte Carlo run.")
        if rd:
            lines.append(f"- Rounded 36 m grid, k=96: the same false-blob rate becomes **{rd['P_any_blob_span3_area6']:.4g}**, with adjacent reconstructed-noise correlation **{rd['adjacent_output_noise_corr']:.4f}**.")
        lines += ["- Therefore the persistence null is highly sensitive to the exact z grid and number of tomographic looks; the paper does not expose enough numerical implementation detail to identify the authors' exact k/grid from the manuscript alone.", '']
    if 'curve_digitization' in summary:
        lines += ['## Published figure digitization', 'Approximate image-level correlations, not raw arrays:']
        for r in summary['curve_digitization'].get('panels', []):
            lines.append(f"- {r['panel']}: raw r={r['digitized_trace_corr']:.3f}, smoothed-envelope r={r['smoothed_envelope_corr']:.3f}")
        lines.append('')
    if 'mt_match' in summary:
        lines += ['## Troiano 2008 consistency check']
        for r in summary['mt_match'].get('matches', []):
            lines.append(f"- {r['panel']}: best Troiano Fig.2 patch corr={r['best_match_corr']:.3f}, tile-shuffle p={r['tile_shuffle_null_p']:.4g}")
        lines.append('')
    lines += ['## Interpretation guardrails', '- Passing Y=A h confirms the inverse operator behaves correctly under its assumed observation model.', '- Noise-null failure rates do not exclude stable systematic SAR artifacts.', '- Exact replication of the Vesuvius SAR result still requires the original COSMO-SkyMed SLC/product metadata and the exact acquisition UTC.', '- The paper-image tests quantify what is visible in published figures; they are not substitutes for unreleased numerical arrays.']
    out.write_text('\n'.join(lines), encoding='utf-8')

def jet_scalar_image(a, size=(300, 160)):
    if cv2 is None:
        raise RuntimeError('opencv unavailable')
    from matplotlib import colormaps
    from scipy.spatial import cKDTree
    a = cv2.resize(np.asarray(a, dtype=np.uint8), size, interpolation=cv2.INTER_AREA).astype(np.float32)
    pal = (colormaps['jet'](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.float32)
    idx = cKDTree(pal).query(a.reshape(-1, 3), k=1, workers=1)[1]
    return (idx.astype(np.float32) / 255.0).reshape(size[1], size[0])


def norm01(a):
    a = np.asarray(a, float)
    lo, hi = np.quantile(a, [0.02, 0.98])
    return np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)

def blob_descriptors(a, q=0.9, min_area=0.02, blur=0.0):
    b = np.asarray(a, float) if blur <= 0 else ndimage.gaussian_filter(np.asarray(a, float), blur)
    m = b >= np.quantile(b, q)
    m = ndimage.binary_opening(m, np.ones((2, 2), bool))
    m = ndimage.binary_closing(m, np.ones((3, 3), bool))
    lab, n = ndimage.label(m, np.ones((3, 3), int))
    H, W = b.shape
    out = []
    for j in range(1, n + 1):
        yy, xx = np.where(lab == j)
        if len(xx) / (H * W) < min_area:
            continue
        x0, x1 = (xx.min(), xx.max())
        y0, y1 = (yy.min(), yy.max())
        if len(xx) > 2:
            cov = np.cov(np.stack([xx / W, yy / H]))
            ev = np.linalg.eigvalsh(cov)
            elong = float(np.sqrt(max(ev[-1], 1e-12) / max(ev[0], 1e-12)))
        else:
            elong = 1.0
        out.append({'area': len(xx) / (H * W), 'cx': float(xx.mean() / W), 'cy': float(yy.mean() / H), 'w': float((x1 - x0 + 1) / W), 'h': float((y1 - y0 + 1) / H), 'elong': elong, 'pixels': int(len(xx))})
    return sorted(out, key=lambda x: x['area'], reverse=True)

def blob_score(a, b, ctol=0.16):
    d = math.hypot(a['cx'] - b['cx'], a['cy'] - b['cy'])
    return float(math.exp(-(d / ctol) ** 2 - 0.45 * abs(math.log((a['area'] + 1e-12) / (b['area'] + 1e-12))) - 0.22 * (abs(math.log((a['w'] + 1e-12) / (b['w'] + 1e-12))) + abs(math.log((a['h'] + 1e-12) / (b['h'] + 1e-12)))) - 0.08 * abs(math.log((a['elong'] + 1e-12) / (b['elong'] + 1e-12)))))

def best_blob_score(a, b, q=0.9, min_area=0.02, ctol=0.16):
    A = blob_descriptors(a, q, min_area)
    B = blob_descriptors(b, q, min_area)
    if not A or not B:
        return (0.0, None, None)
    best = max(((blob_score(x, y, ctol), x, y) for x in A for y in B))
    return best

def rank_match(y, ref):
    vals = np.sort(np.asarray(ref).ravel())
    idx = np.argsort(np.asarray(y).ravel())
    o = np.empty(len(idx), float)
    o[idx] = vals
    return o.reshape(np.asarray(y).shape)

def phase_surrogate_pair(a, b, rho, rng):
    Fa = np.fft.rfft2(a - np.mean(a))
    Fb = np.fft.rfft2(b - np.mean(b))
    shp = Fa.shape

    def unit():
        z = rng.standard_normal(shp) + 1j * rng.standard_normal(shp)
        return z / np.maximum(np.abs(z), 1e-12)
    c = unit()
    e1 = unit()
    e2 = unit()
    u1 = rho * c + np.sqrt(max(0, 1 - rho * rho)) * e1
    u2 = rho * c + np.sqrt(max(0, 1 - rho * rho)) * e2
    u1 /= np.maximum(np.abs(u1), 1e-12)
    u2 /= np.maximum(np.abs(u2), 1e-12)
    x = np.fft.irfft2(np.abs(Fa) * u1, s=a.shape)
    y = np.fft.irfft2(np.abs(Fb) * u2, s=b.shape)
    return (x, y)

def extract_all_paper_images(pdf_path, outdir):
    out = mkdir(outdir / 'paper_images')
    doc = fitz.open(pdf_path)
    rows = []
    paths = {}
    for pi, p in enumerate(doc, 1):
        for ii, item in enumerate(p.get_images(full=True)):
            info = doc.extract_image(item[0])
            q = out / f"p{pi:02d}_i{ii}_x{item[0]}.{info['ext']}"
            q.write_bytes(info['image'])
            try:
                im = Image.open(q)
                w, h = im.size
            except Exception:
                w = h = 0
            rows.append({'page': pi, 'index': ii, 'xref': item[0], 'width': w, 'height': h, 'path': str(q)})
            paths[pi, ii] = q
    pd.DataFrame(rows).to_csv(out / 'index.csv', index=False)
    return (paths, rows)

def figure11_12_morphology(paths, outdir, trials=500):
    out = mkdir(outdir / 'figure11_12_persistence')
    p11 = paths.get((12, 0))
    p12 = paths.get((12, 1))
    if not p11 or not p12:
        return {'status': 'Figure 11/12 embedded images unavailable'}
    i11 = Image.open(p11).convert('RGB').crop((182, 889, 1613, 1652))
    i12 = Image.open(p12).convert('RGB').crop((175, 73, 1606, 836))
    a = norm01(ndimage.gaussian_filter(jet_scalar_image(np.asarray(i11)), 1.8)[:90])
    b = norm01(ndimage.gaussian_filter(jet_scalar_image(np.asarray(i12)), 1.8)[:90])
    qs = [0.9, 0.92, 0.94, 0.96]
    matches = []
    for q in qs:
        min_area = 0.018 if q <= 0.92 else 0.008
        sc, x, y = best_blob_score(a, b, q, min_area, 0.16)
        matches.append({'q': q, 'score': sc, 'figure11_blob': x, 'figure12_blob': y})
    observed = float(np.median([r['score'] for r in matches]))
    phase_coh = float(abs(np.sum(np.fft.rfft2(a - a.mean()) * np.conj(np.fft.rfft2(b - b.mean())))) / np.sqrt(np.sum(abs(np.fft.rfft2(a - a.mean())) ** 2) * np.sum(abs(np.fft.rfft2(b - b.mean())) ** 2)))
    rng = np.random.default_rng(RNG_SEED)
    null = []
    for rho in [0, 0.5, phase_coh, 0.9, 0.99]:
        vals = []
        for _ in range(trials):
            x, y = phase_surrogate_pair(a, b, rho, rng)
            ss = []
            for q in qs:
                min_area = 0.018 if q <= 0.92 else 0.008
                ss.append(best_blob_score(x, y, q, min_area, 0.16)[0])
            vals.append(float(np.median(ss)))
        vals = np.asarray(vals)
        null.append({'phase_correlation': float(rho), 'p_ge_observed': float((np.sum(vals >= observed) + 1) / (len(vals) + 1)), 'null_median': float(np.median(vals)), 'null_95': float(np.quantile(vals, 0.95)), 'null_99': float(np.quantile(vals, 0.99))})
    plt.figure(figsize=(9, 3.8))
    plt.subplot(1, 2, 1)
    plt.imshow(a, aspect='auto')
    plt.title('Figure 11')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(b, aspect='auto')
    plt.title('Figure 12')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out / 'preprocessed_maps.png', dpi=180)
    plt.close()
    result = {'pixel_correlation_after_lowpass': safe_corr(a, b), 'fourier_complex_coherence': phase_coh, 'multi_threshold_blob_match_median': observed, 'threshold_matches': matches, 'phase_surrogate_null': null, 'interpretation': 'object-level morphology; no exact pixel overlap is required'}
    jdump(result, out / 'results.json')
    return result

def line_ridge_test(path, fig, outdir, permutations=1000):
    out = mkdir(outdir / 'dem_overlay_ridge')
    im = Image.open(path).convert('RGB')
    if fig == 15:
        roi = np.asarray(im.crop((140, 666, 1564, 1525)))
        size = (700, 420)
        color = 'yellow'
    else:
        roi = np.asarray(im.crop((96, 729, 1473, 1488)))
        size = (700, 386)
        color = 'red'
    a = cv2.resize(roi, size, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    if color == 'yellow':
        lm = (hsv[:, :, 0] > 20) & (hsv[:, :, 0] < 40) & (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 150)
    else:
        lm = ((hsv[:, :, 0] < 8) | (hsv[:, :, 0] > 172)) & (hsv[:, :, 1] > 150) & (hsv[:, :, 2] > 140)
    yl = np.full(a.shape[1], np.nan)
    for x in range(a.shape[1]):
        yy = np.where(lm[:, x])[0]
        if len(yy):
            yl[x] = np.median(yy)
    good = np.isfinite(yl)
    yl = np.interp(np.arange(len(yl)), np.where(good)[0], yl[good])
    v = jet_scalar_image(a, size=size)
    vin = cv2.inpaint((v * 255).astype(np.uint8), (lm * 255).astype(np.uint8), 3, cv2.INPAINT_TELEA) / 255.0
    sm = ndimage.gaussian_filter(vin, 2)
    gy = np.gradient(sm, axis=0)
    top = int(0.67 * a.shape[0])

    def score(line, win=5):
        vals = []
        offs = []
        for x, y0 in enumerate(line):
            lo = max(0, int(round(y0)) - win)
            hi = min(top, int(round(y0)) + win + 1)
            if hi <= lo:
                continue
            q = np.abs(gy[lo:hi, x])
            j = lo + int(np.argmax(q))
            vals.append(float(np.max(q)))
            offs.append(float(j - y0))
        if len(vals) < 0.6 * len(line):
            return (np.nan, np.asarray(offs))
        return (float(np.mean(vals)), np.asarray(offs))
    observed, offs = score(yl)
    rng = np.random.default_rng(RNG_SEED + fig)
    vals = []
    maxshift = int(0.3 * a.shape[0])
    for _ in range(permutations):
        sh = int(rng.integers(-maxshift, maxshift + 1))
        if abs(sh) < 15:
            sh = sh + 25 if sh >= 0 else sh - 25
        s, _ = score(yl + sh)
        if np.isfinite(s):
            vals.append(s)
    vals = np.asarray(vals)
    p = float((np.sum(vals >= observed) + 1) / (len(vals) + 1)) if len(vals) else np.nan
    meters_per_px = (3500.0 / size[1]) if fig == 15 else (4000.0 / size[1])
    medpx = float(np.median(np.abs(offs)))
    p90px = float(np.quantile(np.abs(offs), 0.9))
    result = {'figure': fig, 'overlay_line_coverage': float(good.mean()), 'mean_local_tomogram_ridge_strength': observed, 'median_ridge_to_dem_line_px': medpx, 'p90_ridge_to_dem_line_px': p90px, 'estimated_meters_per_resized_pixel': meters_per_px, 'median_ridge_to_dem_line_m': medpx * meters_per_px, 'p90_ridge_to_dem_line_m': p90px * meters_per_px, 'random_vertical_shift_p': p, 'null_99_strength': float(np.quantile(vals, 0.99)) if len(vals) else np.nan}
    jdump(result, out / f'figure{fig}.json')
    return result

def figure25_panel_anomaly(a, q=0.72):
    a = cv2.resize(np.asarray(a, dtype=np.uint8), (180, 180), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(float)
    for c in range(3):
        lab[:, :, c] = ndimage.gaussian_filter(lab[:, :, c], 3)
    H, W = lab.shape[:2]
    d = int(0.12 * min(H, W))
    border = np.zeros((H, W), bool)
    border[:d] = 1
    border[-d:] = 1
    border[:, :d] = 1
    border[:, -d:] = 1
    bg = np.median(lab[border], axis=0)
    s = np.linalg.norm(lab - bg, axis=2)
    valid = np.ones((H, W), bool)
    e = int(0.06 * H)
    valid[:e] = False
    valid[-e:] = False
    valid[:, :e] = False
    valid[:, -e:] = False
    m = (s >= np.quantile(s[valid], q)) & valid
    m = ndimage.binary_opening(m, np.ones((3, 3), bool))
    m = ndimage.binary_closing(m, np.ones((7, 7), bool))
    labm, n = ndimage.label(m, np.ones((3, 3), int))
    best = None
    for j in range(1, n + 1):
        yy, xx = np.where(labm == j)
        if len(xx) < 0.005 * H * W:
            continue
        area = len(xx) / (H * W)
        cx = float(xx.mean() / W)
        cy = float(yy.mean() / H)
        central = math.exp(-((cx - 0.5) / 0.35) ** 2 - ((cy - 0.5) / 0.4) ** 2)
        v = area * central
        if best is None or v > best[0]:
            cov = np.cov(np.stack([xx / W, yy / H]))
            ev = np.linalg.eigvalsh(cov)
            elong = float(np.sqrt(max(ev[-1], 1e-12) / max(ev[0], 1e-12)))
            best = (v, {'area': area, 'cx': cx, 'cy': cy, 'w': float((xx.max() - xx.min() + 1) / W), 'h': float((yy.max() - yy.min() + 1) / H), 'elong': elong})
    return best[1] if best else None

def figure25_object_validation(path, outdir, permutations=5000):
    out = mkdir(outdir / 'figure25_object_validation')
    im = Image.open(path).convert('RGB')
    boxes = {'a': (0, 5, 435, 415), 'c': (930, 5, 1365, 415), 'd': (0, 530, 435, 1370), 'f': (930, 530, 1365, 1370), 'g': (0, 1450, 435, 1845), 'i': (945, 1450, 1335, 1845)}
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for p, q in [('a', 'c'), ('d', 'f'), ('g', 'i')]:
        x = figure25_panel_anomaly(np.asarray(im.crop(boxes[p])))
        y = figure25_panel_anomaly(np.asarray(im.crop(boxes[q])))
        if x is None or y is None:
            rows.append({'pair': p + '-' + q, 'status': 'component unavailable'})
            continue
        obs = blob_score(x, y, 0.2)
        vals = []
        for _ in range(permutations):
            yy = dict(y)
            yy['cx'] = float(rng.uniform(y['w'] / 2, max(y['w'] / 2, 1 - y['w'] / 2)))
            yy['cy'] = float(rng.uniform(y['h'] / 2, max(y['h'] / 2, 1 - y['h'] / 2)))
            vals.append(blob_score(x, yy, 0.2))
        vals = np.asarray(vals)
        rows.append({'pair': p + '-' + q, 'sar_blob': x, 'mt_blob': y, 'object_match_score': obs, 'random_location_p': float((np.sum(vals >= obs) + 1) / (len(vals) + 1)), 'null_95': float(np.quantile(vals, 0.95)), 'null_99': float(np.quantile(vals, 0.99))})
    pd.DataFrame([{k: v for k, v in r.items() if k not in ('sar_blob', 'mt_blob')} for r in rows]).to_csv(out / 'scores.csv', index=False)
    result = {'note': 'large-object location/extent comparison; color polarity and exact pixels are not required', 'pairs': rows}
    jdump(result, out / 'results.json')
    return result

def vesuvius_object_null(outdir, trials=500):
    out = mkdir(outdir / 'vesuvius_object_null')
    lam = 4.86
    R = 650000.0
    dz = 36.0
    aperture = lam * R / (2 * dz)
    z = np.arange(0, 3000 + dz / 2, dz)
    k = 96
    p = TomoParams('Vesuvius_object', lam, R, aperture, 35.0, k)
    A, _ = steering_matrix(p, z, k)
    P = np.linalg.pinv(A, rcond=1e-10)
    nx = 160
    x = np.arange(nx)
    H = np.zeros((len(z), nx), complex)
    H[np.ix_((z >= 300) & (z <= 2200), np.abs(x - 82) <= 4)] += np.exp(1j * 0.4)
    H[np.ix_((z >= 800) & (z <= 1150), (x >= 64) & (x <= 103))] += 0.8 * np.exp(1j * 1.2)
    X, Z = np.meshgrid(x, z)
    body = ((X - 82) / 31) ** 2 + ((Z - 2600) / 320) ** 2 <= 1
    H[body] += 0.75 * np.exp(1j * 2.0)
    Y = A @ H
    rng = np.random.default_rng(RNG_SEED)
    sp = np.mean(np.abs(Y) ** 2)

    def reconstruct(snr):
        npow = sp / 10 ** (snr / 10)
        n = np.sqrt(npow) * complex_noise(Y.shape, rng)
        return P @ (Y + n)
    target = []
    for snr in [-10, -5, 0, 5, 10]:
        vals = []
        for _ in range(max(40, trials // 5)):
            a = norm01(np.abs(reconstruct(snr)))
            b = norm01(np.abs(reconstruct(snr)))
            vals.append(best_blob_score(a, b, 0.9, 0.02, 0.16)[0])
        target.append({'snr_db': snr, 'median_large_blob_match': float(np.median(vals)), 'p10': float(np.quantile(vals, 0.1)), 'p90': float(np.quantile(vals, 0.9))})
    null = []
    for rho in [0, 0.5, 0.9, 0.97]:
        vals = []
        for _ in range(trials):
            n1 = complex_noise((k, nx), rng)
            ni = complex_noise((k, nx), rng)
            n2 = rho * n1 + np.sqrt(1 - rho * rho) * ni
            a = norm01(np.abs(P @ n1))
            b = norm01(np.abs(P @ n2))
            vals.append(best_blob_score(a, b, 0.9, 0.02, 0.16)[0])
        vals = np.asarray(vals)
        null.append({'noise_view_corr': rho, 'P_any_large_blob_match_ge_0_75': float(np.mean(vals >= 0.75)), 'max_score': float(np.max(vals)), 'p99_score': float(np.quantile(vals, 0.99))})
    result = {'paper_resolution_m': dz, 'effective_aperture_consistent_with_36m_m': aperture, 'paper_aperture_approx_m': 42000.0, 'n_depth_bins': len(z), 'n_looks_used': k, 'condition_number': float(np.linalg.cond(A)), 'target': target, 'null': null}
    jdump(result, out / 'results.json')
    return result

def vesuvius_null_tests(params, outdir, trials=500):
    return vesuvius_object_null(outdir, trials)

def parse_vesuvius_paper(pdf_path, outdir):
    out = mkdir(outdir / 'vesuvius_paper')
    text = pdf_text(pdf_path)
    flat = re.sub('\\s+', ' ', text)
    result = {'pdf_sha256': sha256(pdf_path), 'table1_has_17_january_2022': bool(re.search('17\\s+January\\s+2022', flat, re.I)), 'text_says_february_is_month_sar_acquired': bool(re.search('February\\s+2022,\\s+which\\s+is\\s+the\\s+month\\s+in\\s+which\\s+the\\s+SAR\\s+data\\s+was\\s+acquired', flat, re.I)), 'doppler_bandwidth_22khz_present': bool(re.search('22\\s*kHz', flat, re.I)), 'prf_twice_doppler_bandwidth_present': bool(re.search('PRF\\s+2\\s*[·x×]?\\s*\\(?Doppler\\s+bandwidth', flat, re.I)), 'vc_re_station_present': 'IV-VCRE' in flat, 'vbkn_station_present': 'IV-VBKN' in flat, 'one_khz_detail_present': bool(re.search('1\\s*kHz', flat, re.I)), 'frequency_200hz_present': bool(re.search('200\\s*Hz', flat, re.I)), 'lambda_4_86m_present': bool(re.search('4\\.86\\s*m', flat, re.I)), 'lower_frequency_repeat_claim': bool(re.search('same results.*lower vibrational frequency', flat, re.I)), 'computed_from_table1': {'prf_hz_if_2x22khz': 44000.0, 'nyquist_hz_if_prf_44khz': 22000.0, 'delta_z_from_4_86m_650km_42km_m': 4.86 * 650000 / (2 * 42000), 'effective_aperture_for_36m_m': 4.86 * 650000 / (2 * 36)}}
    paths, rows = extract_all_paper_images(pdf_path, out)
    imgs = {}
    for pg in [12, 15, 18, 19, 20, 21]:
        imgs[pg] = [Path(r['path']) for r in rows if r['page'] == pg]
    jdump(result, out / 'paper_text_checks.json')
    return (result, imgs, paths)

def dem_overlay_energy_test(imgs, outdir):
    out = mkdir(outdir / 'paper_dem_overlay')
    p = imgs.get(15, [])
    if len(p) < 2:
        return {'status': 'missing Figure 15/16 images'}
    r15 = line_ridge_test(p[0], 15, outdir, 1000)
    r16 = line_ridge_test(p[1], 16, outdir, 1000)
    result = {'figure15': r15, 'figure16': r16}
    jdump(result, out / 'results.json')
    return result

def normalized_xcorr_scan(x, tpl):
    x = np.asarray(x, float)
    t = np.asarray(tpl, float)
    t = (t - t.mean()) / (t.std() + 1e-12)
    n = len(t)
    num = signal.fftconvolve(x, t[::-1], mode='valid')
    cs = np.concatenate([[0], np.cumsum(x)])
    cs2 = np.concatenate([[0], np.cumsum(x * x)])
    sm = cs[n:] - cs[:-n]
    sm2 = cs2[n:] - cs2[:-n]
    var = np.maximum(sm2 - sm * sm / n, 1e-12)
    den = np.sqrt(var) * np.linalg.norm(t)
    return num / den

def paper_time_templates(imgs):
    p18 = imgs.get(18, [])
    p19 = imgs.get(19, [])
    out = []
    if len(p18) >= 2:
        im = np.array(Image.open(p18[1]).convert('RGB'))
        for name, b in [('VCRE_F21a', (80, 625, 35, 465)), ('VCRE_F21b', (725, 1268, 35, 466))]:
            tr = _trace_from_color(im, b, (0, 114, 189))
            tr = -(tr - np.nanmean(tr)) / (np.nanstd(tr) + 1e-12)
            out.append((name, tr))
    if p19:
        im = np.array(Image.open(p19[0]).convert('RGB'))
        tr = _trace_from_color(im, (80, 623, 40, 465), (0, 114, 189))
        tr = -(tr - np.nanmean(tr)) / (np.nanstd(tr) + 1e-12)
        out.append(('F23_blue', tr))
    return out

def scan_mseed_against_paper(path, templates):
    try:
        from obspy import read
    except Exception:
        return {'status': 'obspy unavailable', 'file': str(path)}
    st = read(str(path))
    rows = []
    for tr in st:
        q = tr.copy()
        q.detrend('linear')
        q.detrend('demean')
        fs = float(q.stats.sampling_rate)
        raw = np.asarray(q.data, float)
        variants = [('raw', raw)]
        for lo, hi in [(0.5, 10), (1, 20), (5, min(40, 0.45 * fs))]:
            if hi > lo and hi < fs / 2:
                try:
                    sos = signal.butter(4, [lo, hi], btype='bandpass', fs=fs, output='sos')
                    variants.append((f'band_{lo}_{hi}', signal.sosfiltfilt(sos, raw)))
                except Exception:
                    pass
        for tn, tpl0 in templates:
            n = max(16, int(round(fs)))
            tpl = signal.resample(tpl0, n)
            for vn, x in variants:
                c = normalized_xcorr_scan(x, tpl)
                if not len(c):
                    continue
                ii = int(np.argmax(np.abs(c)))
                t0 = q.stats.starttime + ii / fs
                rows.append({'trace': q.id, 'template': tn, 'variant': vn, 'sample_rate_hz': fs, 'best_abs_corr': float(abs(c[ii])), 'signed_corr': float(c[ii]), 'match_time_utc': str(t0)})
    return {'file': str(path), 'matches': sorted(rows, key=lambda r: r['best_abs_corr'], reverse=True)}

def srtm_tile_online(outdir):
    if requests is None:
        return {'status': 'requests unavailable'}
    urls = ['https://step.esa.int/auxdata/dem/SRTMGL1/N40E014.SRTMGL1.hgt.zip', 'https://s3.amazonaws.com/elevation-tiles-prod/skadi/N40/N40E014.hgt.gz']
    raw = None
    used = None
    for url in urls:
        try:
            r = requests.get(url, timeout=120, headers={'User-Agent': 'Mozilla/5.0 research-replication'})
            r.raise_for_status()
            if url.endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                    nm = [n for n in zf.namelist() if n.lower().endswith('.hgt')][0]
                    raw = zf.read(nm)
            else:
                raw = gzip.decompress(r.content)
            used = url
            break
        except Exception:
            pass
    if raw is None:
        return {'status': 'SRTM download failed'}
    n = int(round(math.sqrt(len(raw) / 2)))
    a = np.frombuffer(raw, dtype='>i2').reshape(n, n).astype(float)

    def rc(lat, lon):
        return (int(round((41 - lat) * (n - 1))), int(round((lon - 14) * (n - 1))))
    r1, c1 = rc(40.9, 14.3)
    r2, c2 = rc(40.74, 14.58)
    rr = sorted([r1, r2])
    cc = sorted([c1, c2])
    crop = a[rr[0]:rr[1] + 1, cc[0]:cc[1] + 1]
    np.save(outdir / 'srtm_N40E014_crop.npy', crop)
    plt.figure(figsize=(6.5, 5))
    plt.imshow(crop, origin='upper', extent=[14.3, 14.58, 40.74, 40.9], aspect='auto')
    plt.scatter([14.431419, 14.429881], [40.818999, 40.829959], marker='x')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.colorbar(label='Elevation (m)')
    plt.tight_layout()
    plt.savefig(outdir / 'srtm_context.png', dpi=160)
    plt.close()
    return {'source_url': used, 'tile_size': n, 'crop_shape': list(crop.shape)}

def online_raw_tests(outdir, sar_utc, imgs=None):
    out = mkdir(outdir / 'online_raw')
    result = {}
    for sta in ['VCRE', 'VBKN']:
        try:
            result[f'station_{sta}'] = ingv_station_channel_metadata(sta, out)
        except Exception as e:
            result[f'station_{sta}'] = {'error': str(e)}
    try:
        result['events'] = ingv_events_2022(out)
    except Exception as e:
        result['events'] = {'error': str(e)}
    result['srtm'] = srtm_tile_online(out)
    templates = paper_time_templates(imgs or {})
    dates = ['2022-01-17', '2022-02-02', '2022-02-18']
    if sar_utc:
        dates = [sar_utc[:10]]
    wave = []
    for sta in ['VCRE', 'VBKN']:
        for d in dates:
            start = d + 'T00:00:00'
            end = d + 'T23:59:59.999'
            p = download_mseed(sta, start, end, out)
            if p:
                a = analyze_mseed(p, out)
                a['date'] = d
                if templates:
                    a['paper_curve_scan'] = scan_mseed_against_paper(p, templates)
                wave.append(a)
    result['waveforms'] = wave
    rates = []
    for sta in ['VCRE', 'VBKN']:
        rates += result.get(f'station_{sta}', {}).get('unique_sample_rates_hz', [])
    if rates:
        fs = float(np.median(rates))
        result['bandwidth_consistency'] = {'median_station_sample_rate_hz': fs, 'station_nyquist_hz': fs / 2, 'paper_sar_prf_hz': 44000.0, 'paper_approximately_1khz_station_comparison': True, 'upsampling_control': upsampling_information_test(fs, 44000, out)}
    jdump(result, out / 'online_raw_results.json')
    return result

def match_biondi_mt_to_troiano(imgs, troiano_pdf, outdir, permutations=80):
    out = mkdir(outdir / 'troiano_match')
    if cv2 is None:
        return {'status': 'opencv unavailable'}
    cv2.setNumThreads(1)
    p20 = imgs.get(20, [])
    if not p20:
        return {'status': 'Figure 25 image missing'}
    b = np.array(Image.open(p20[0]).convert('RGB'))
    fig2 = extract_troiano_fig2(troiano_pdf, out)
    if fig2 is None:
        return {'status': 'Troiano Figure 2 unavailable'}
    t = np.array(Image.open(fig2).convert('RGB'))
    targets = {'25c': crop_nonwhite(b[5:415, 930:1365]), '25f': crop_nonwhite(b[530:1370, 930:1365]), '25i': crop_nonwhite(b[1450:1845, 945:1335])}
    sources = {'Troiano_NS': resistivity_scalar(t[15:320, 65:995]).astype(np.float32), 'Troiano_WE': resistivity_scalar(t[335:675, 150:970]).astype(np.float32)}
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for name, tg in targets.items():
        T0 = resistivity_scalar(tg)
        actual = _best_template_match(T0, sources, np.linspace(0.18, 0.9, 20))
        srcname, scale, loc, shape = actual[1]
        S = sources[srcname]
        vals = []
        for _ in range(permutations):
            sh = _tile_shuffle(T0, 4, rng)
            best = -1.0
            for sc in [scale * 0.9, scale, scale * 1.1]:
                nw = int(sh.shape[1] * sc); nh = int(sh.shape[0] * sc)
                if nw < 10 or nh < 10 or nw > S.shape[1] or nh > S.shape[0]:
                    continue
                T = cv2.resize(sh, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
                r = cv2.matchTemplate(S, T, cv2.TM_CCOEFF_NORMED)
                best = max(best, float(np.max(r)))
            vals.append(best)
        vals = np.asarray(vals)
        rows.append({'panel': name, 'best_match_corr': actual[0], 'match': actual[1], 'tile_shuffle_null_p': float((np.sum(vals >= actual[0]) + 1) / (len(vals) + 1)), 'null_95': float(np.quantile(vals, 0.95)), 'null_99': float(np.quantile(vals, 0.99))})
    jdump(rows, out / 'mt_patch_matches.json')
    return {'note': 'image-level provenance/consistency check against Troiano Fig.2, not original MT numeric grid', 'matches': rows}


def build_markdown(summary, out):
    p = summary.get('paper', {})
    f = summary.get('fig11_12', {})
    v = summary.get('vesuvius_object_null', {})
    d = summary.get('dem_overlay', {})
    mt = summary.get('figure25_objects', {})
    lines = ['# Reproduction summary', '', f"Vesuvius paper date check: Table 1 January = {p.get('table1_has_17_january_2022')}; February statement = {p.get('text_says_february_is_month_sar_acquired')}", f"Paper resolution used in the main Vesuvius tests: {v.get('paper_resolution_m')} m", f"Effective aperture implied by lambda=4.86 m, R=650 km, dz=36 m: {v.get('effective_aperture_consistent_with_36m_m')} m", f"Vesuvius Y=Ah operator condition number: {summary.get('vesuvius_operator', {}).get('condition_number')}", f"Vesuvius Y=Ah 20 dB correlation: {summary.get('vesuvius_operator', {}).get('snr_runs', {}).get('20', {}).get('magnitude_correlation')}", '', f"Figure 11/12 low-pass pixel correlation: {f.get('pixel_correlation_after_lowpass')}", f"Figure 11/12 large-object multi-threshold match: {f.get('multi_threshold_blob_match_median')}", f"Figure 11/12 measured Fourier coherence: {f.get('fourier_complex_coherence')}", '']
    for r in f.get('phase_surrogate_null', []):
        lines.append(f"Phase-surrogate rho={r['phase_correlation']:.3f}: p(object match >= observed)={r['p_ge_observed']:.4g}")
    lines.append('')
    if d:
        for k in ['figure15', 'figure16']:
            r = d.get(k, {})
            lines.append(f"{k}: median DEM-to-local-ridge distance={r.get('median_ridge_to_dem_line_px')} px; shift-null p={r.get('random_vertical_shift_p')}")
    lines.append('')
    for r in mt.get('pairs', []):
        if 'object_match_score' in r:
            lines.append(f"Figure 25 {r['pair']}: large-object score={r['object_match_score']:.3f}; random-location p={r['random_location_p']:.4g}")
    lines.append('')
    for r in summary.get('mt_match', {}).get('matches', []):
        lines.append(f"Troiano provenance {r['panel']}: r={r['best_match_corr']:.3f}; shuffle p={r['tile_shuffle_null_p']:.4g}")
    lines.append('')
    severe = summary.get('vesuvius_severe_noise', {})
    if severe:
        sm = severe.get('signal_medians', {})
        lines.append(f"Severe-noise stress test at {severe.get('snr_db')} dB: median pixel r={sm.get('pixel_correlation')}; object score={sm.get('object_score')}; support Dice A/B={sm.get('view_a_dice_to_truth')}/{sm.get('view_b_dice_to_truth')}.")
        for r in severe.get('nulls', []):
            if r.get('cross_view_correlation') == 0.97:
                lines.append(f"Severe-noise fixed-support null {r.get('null_model')} rho=0.97: max Dice={r.get('max_dice_to_truth_any_view')}; false recoveries={r.get('false_recovery_count')}/{r.get('trials')}.")
        lines.append('')
    extg = summary.get('gotcha_external_height', {})
    if extg:
        rows = extg.get('height_noise', [])
        r10 = next((r for r in rows if r.get('snr_db') == 10), None)
        if r10:
            lines.append(f"External GOTCHA geometry control: at 10 dB the median/p90 height error is {r10['median_abs_error_m']:.3f}/{r10['p90_abs_error_m']:.3f} m across published calibration-target heights.")
        lines.append(f"GOTCHA Kz sign flip median mirror error: {extg.get('sign_flip_median_mirror_error_m')} m; planned-vs-actual geometry median error: {extg.get('geometry_mismatch_median_abs_error_m'):.3f} m.")
    rr = summary.get('rollo_external_targets', {})
    if rr:
        lines.append(f"Real known-target control: Umbra 1.08 mm shaker p={rr['Umbra_1_08mm']['p_shift']:.4g}; bright stationary fence p={rr['stationary_fence']['p_shift']:.4g}.")
    gd = summary.get('gotcha_durango_control', {})
    if gd:
        lines.append(f"GOTCHA real moving-target image control: random-location p={gd.get('p_random_location'):.4g} for the GPS/PAUX-predicted Doppler-shifted ROI.")
    lines.append('The morphology tests compare large connected objects by centroid, area, extent and elongation. Exact pixel persistence is not required.')
    lines.append('The online branch queries INGV station metadata, raw miniSEED, January-February 2022 events and SRTM, then searches full candidate CSG-1 repeat-cycle days against digitized one-second paper traces when ObsPy is available.')
    out.write_text('\n'.join(lines), encoding='utf-8')

def vesuvius_depth_localization_validation(outdir, trials=120):
    out = mkdir(outdir / 'vesuvius_depth_localization')
    lam = 4.86
    R = 650000.0
    dz = 36.0
    aperture = lam * R / (2 * dz)
    z = np.arange(0, 3000 + dz / 2, dz)
    p = TomoParams('Vesuvius_depth_validation', lam, R, aperture, 35.0, 96)
    A, _ = steering_matrix(p, z, 96)
    P = np.linalg.pinv(A, rcond=1e-10)
    rng = np.random.default_rng(RNG_SEED + 101)
    target_depths = [360, 900, 1512, 2088, 2700]
    rows = []
    for snr in [20, 10, 0, -5, -10]:
        errors = []
        for z0 in target_depths:
            iz = int(np.argmin(np.abs(z - z0)))
            h = np.zeros(len(z), complex)
            h[iz] = 1.0
            y = A @ h
            npow = np.mean(np.abs(y) ** 2) / 10 ** (snr / 10)
            for _ in range(trials):
                yn = y + np.sqrt(npow) * complex_noise(y.shape, rng)
                hh = P @ yn
                zp = float(z[int(np.argmax(np.abs(hh)))])
                errors.append(abs(zp - float(z[iz])))
        e = np.asarray(errors)
        rows.append({'snr_db': snr, 'median_abs_depth_error_m': float(np.median(e)), 'p90_abs_depth_error_m': float(np.quantile(e, 0.9)), 'fraction_exact_depth_bin': float(np.mean(e == 0)), 'fraction_within_one_36m_cell': float(np.mean(e <= 36.0))})
    pd.DataFrame(rows).to_csv(out / 'depth_localization.csv', index=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot([r['snr_db'] for r in rows], [r['p90_abs_depth_error_m'] for r in rows], marker='o', label='90th percentile error')
    ax.plot([r['snr_db'] for r in rows], [r['median_abs_depth_error_m'] for r in rows], marker='s', label='Median error')
    ax.axhline(36.0, linewidth=1, linestyle='--', label='One 36 m cell')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Absolute depth error (m)')
    ax.set_title('Known-depth localization across noise levels')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'depth_localization_validation.png', dpi=220)
    plt.close(fig)
    result = {'target_depths_m': target_depths, 'trials_per_target_per_snr': trials, 'rows': rows}
    jdump(result, out / 'results.json')
    return result

def vesuvius_wavelength_ablation(outdir, z0=1512.0):
    out = mkdir(outdir / 'vesuvius_wavelength_ablation')
    lam = 4.86
    R = 650000.0
    dz = 36.0
    aperture = lam * R / (2 * dz)
    z = np.arange(0, 3000 + dz / 2, dz)
    p = TomoParams('Vesuvius_wavelength_ablation', lam, R, aperture, 35.0, 96)
    A, _ = steering_matrix(p, z, 96)
    iz = int(np.argmin(np.abs(z - z0)))
    h = np.zeros(len(z), complex)
    h[iz] = 1.0
    y = A @ h
    rows = []
    for frac in [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]:
        pm = TomoParams('mismatch', lam * (1 + frac), R, aperture, 35.0, 96)
        Am, _ = steering_matrix(pm, z, 96)
        hh = np.linalg.pinv(Am, rcond=1e-10) @ y
        zp = float(z[int(np.argmax(np.abs(hh)))])
        rows.append({'wavelength_error_pct': float(100 * frac), 'recovered_depth_m': zp, 'depth_bias_m': float(zp - z[iz]), 'peak_magnitude': float(np.max(np.abs(hh)))})
    pd.DataFrame(rows).to_csv(out / 'wavelength_ablation.csv', index=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot([r['wavelength_error_pct'] for r in rows], [r['depth_bias_m'] for r in rows], marker='o')
    ax.axhline(0, linewidth=1)
    ax.set_xlabel('Wavelength calibration error (%)')
    ax.set_ylabel('Recovered depth bias (m)')
    ax.set_title('Depth-scale response to wavelength calibration')
    fig.tight_layout()
    fig.savefig(out / 'wavelength_ablation.png', dpi=220)
    plt.close(fig)
    result = {'true_grid_depth_m': float(z[iz]), 'rows': rows}
    jdump(result, out / 'results.json')
    return result

def macro_envelope(z, x):
    X, Z = np.meshgrid(x, z)
    e = np.zeros_like(X, dtype=float)
    e += 1.00 * np.exp(-(((X - 0.48 * x.max()) / (0.17 * x.max())) ** 2 + ((Z - 900) / 330) ** 2))
    e += 0.76 * np.exp(-(((X - 0.60 * x.max()) / (0.25 * x.max())) ** 2 + ((Z - 1650) / 430) ** 2))
    center = 0.44 * x.max() + 0.010 * (Z - 500)
    e += 0.68 * np.exp(-((X - center) / (0.052 * x.max())) ** 2) * np.exp(-((Z - 1300) / 950) ** 8)
    e += 0.90 * np.exp(-(((X - 0.40 * x.max()) / (0.29 * x.max())) ** 2 + ((Z - 2400) / 310) ** 2))
    return e / max(float(e.max()), 1e-12)

def textured_macro_view(env, rng, texture_strength=0.55, clutter=0.08):
    t = ndimage.gaussian_filter(rng.standard_normal(env.shape), 2.0)
    t = (t - t.mean()) / (t.std() + 1e-12)
    amp = env * np.clip(1 + texture_strength * t, 0.05, 2.5)
    ph = ndimage.gaussian_filter(rng.standard_normal(env.shape), 1.2)
    ph = 2 * np.pi * (ph - ph.min()) / (ph.max() - ph.min() + 1e-12)
    h = amp * np.exp(1j * ph)
    c = ndimage.gaussian_filter(complex_noise(env.shape, rng), 1.0)
    h += clutter * c / np.sqrt(np.mean(np.abs(c) ** 2) + 1e-12)
    return h

def smooth_complex_field(shape, rng, sigma=(1.7, 5.0)):
    a = ndimage.gaussian_filter(rng.standard_normal(shape), sigma)
    b = ndimage.gaussian_filter(rng.standard_normal(shape), sigma)
    z = a + 1j * b
    return z / np.sqrt(np.mean(np.abs(z) ** 2) + 1e-12)

def fast_blob_descriptors(a, q=0.72, min_area=0.02, size=(100, 64)):
    if cv2 is None:
        return blob_descriptors(ndimage.gaussian_filter(norm01(a), 2.0), q, min_area)
    b = cv2.resize(np.asarray(norm01(a), np.float32), size, interpolation=cv2.INTER_AREA)
    b = cv2.GaussianBlur(b, (0, 0), 1.5)
    m = (b >= np.quantile(b, q)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    H, W = b.shape
    out = []
    for j in range(1, n):
        area_px = int(stats[j, cv2.CC_STAT_AREA])
        if area_px / (H * W) < min_area:
            continue
        yy, xx = np.where(lab == j)
        if len(xx) > 2:
            cov = np.cov(np.stack([xx / W, yy / H]))
            ev = np.linalg.eigvalsh(cov)
            elong = float(np.sqrt(max(ev[-1], 1e-12) / max(ev[0], 1e-12)))
        else:
            elong = 1.0
        out.append({'area': area_px / (H * W), 'cx': float(cent[j, 0] / W), 'cy': float(cent[j, 1] / H), 'w': float(stats[j, cv2.CC_STAT_WIDTH] / W), 'h': float(stats[j, cv2.CC_STAT_HEIGHT] / H), 'elong': elong})
    return sorted(out, key=lambda x: x['area'], reverse=True)

def macro_match(a, b):
    A = fast_blob_descriptors(np.abs(a))
    B = fast_blob_descriptors(np.abs(b))
    if not A or not B:
        return 0.0
    return float(max(blob_score(x, y, 0.20) for x in A for y in B))

def vesuvius_object_null(outdir, trials=500):
    out = mkdir(outdir / 'vesuvius_object_null')
    lam = 4.86
    R = 650000.0
    dz = 36.0
    aperture = lam * R / (2 * dz)
    z = np.arange(0, 3000 + dz / 2, dz)
    k = 96
    p = TomoParams('Vesuvius_object', lam, R, aperture, 35.0, k)
    A, _ = steering_matrix(p, z, k)
    P = np.linalg.pinv(A, rcond=1e-10)
    nx = 140
    x = np.arange(nx)
    env = macro_envelope(z, x)
    rng = np.random.default_rng(RNG_SEED + 202)
    target = []
    example = None
    example_candidates = []
    n_target = max(80, min(160, trials // 3))
    for snr in [-15, -10, -5, 0, 5, 10]:
        pix = []
        obj = []
        for j in range(n_target):
            H1 = textured_macro_view(env, rng)
            H2 = textured_macro_view(env, rng)
            Y1 = A @ H1
            Y2 = A @ H2
            sp = 0.5 * (np.mean(np.abs(Y1) ** 2) + np.mean(np.abs(Y2) ** 2))
            npow = sp / 10 ** (snr / 10)
            R1 = P @ (Y1 + np.sqrt(npow) * complex_noise(Y1.shape, rng))
            R2 = P @ (Y2 + np.sqrt(npow) * complex_noise(Y2.shape, rng))
            a = norm01(np.abs(R1))
            b = norm01(np.abs(R2))
            pix.append(safe_corr(a, b))
            obj.append(macro_match(R1, R2))
            if snr == -5:
                example_candidates.append((a, b, abs(a - b), float(pix[-1]), float(obj[-1])))
        row = {'snr_db': snr, 'median_pixel_correlation': float(np.median(pix)), 'pixel_corr_p10': float(np.quantile(pix, 0.1)), 'pixel_corr_p90': float(np.quantile(pix, 0.9)), 'median_large_object_score': float(np.median(obj)), 'object_score_p10': float(np.quantile(obj, 0.1)), 'object_score_p90': float(np.quantile(obj, 0.9))}
        target.append(row)
        if snr == -5 and example_candidates:
            example = min(example_candidates, key=lambda q: abs(q[4] - row['median_large_object_score']) + 0.5 * abs(q[3] - row['median_pixel_correlation']))
    null = []
    rhos = [0.0, 0.5, 0.71, 0.9, 0.97]
    for model in ['operator_shaped', 'smooth_colored']:
        for rho in rhos:
            vals = []
            for _ in range(trials):
                if model == 'operator_shaped':
                    n1 = complex_noise((k, nx), rng)
                    ni = complex_noise((k, nx), rng)
                    n2 = rho * n1 + np.sqrt(max(0, 1 - rho * rho)) * ni
                    a = P @ n1
                    b = P @ n2
                else:
                    f1 = smooth_complex_field((len(z), nx), rng)
                    fi = smooth_complex_field((len(z), nx), rng)
                    a = f1
                    b = rho * f1 + np.sqrt(max(0, 1 - rho * rho)) * fi
                vals.append(macro_match(a, b))
            vals = np.asarray(vals)
            null.append({'null_model': model, 'cross_view_correlation': rho, 'median_score': float(np.median(vals)), 'p95_score': float(np.quantile(vals, 0.95)), 'p99_score': float(np.quantile(vals, 0.99)), 'P_score_ge_0_75': float(np.mean(vals >= 0.75)), 'P_score_ge_0_90': float(np.mean(vals >= 0.90))})
    if example is not None:
        a, b, diff, pr, os = example
        extent = [x[0], x[-1], z[-1], z[0]]
        fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.5), constrained_layout=True)
        im0 = ax[0].imshow(a, aspect='auto', origin='upper', extent=extent, vmin=0, vmax=1)
        ax[0].set_title('View A: independent internal texture')
        ax[0].set_xlabel('Position along line')
        ax[0].set_ylabel('Depth z (m)')
        ax[1].imshow(b, aspect='auto', origin='upper', extent=extent, vmin=0, vmax=1)
        ax[1].set_title('View B: same macroscopic object')
        ax[1].set_xlabel('Position along line')
        im2 = ax[2].imshow(diff, aspect='auto', origin='upper', extent=extent, vmin=0, vmax=1)
        ax[2].set_title(f'Pixel difference, r={pr:.2f}; object score={os:.2f}')
        ax[2].set_xlabel('Position along line')
        fig.colorbar(im0, ax=ax[:2], shrink=0.86, label='Normalized magnitude')
        fig.colorbar(im2, ax=ax[2], shrink=0.86, label='Absolute pixel difference')
        fig.savefig(out / 'persistence_example.png', dpi=220)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    snr = np.array([r['snr_db'] for r in target])
    pm = np.array([r['median_pixel_correlation'] for r in target])
    pl = np.array([r['pixel_corr_p10'] for r in target])
    ph = np.array([r['pixel_corr_p90'] for r in target])
    om = np.array([r['median_large_object_score'] for r in target])
    ol = np.array([r['object_score_p10'] for r in target])
    oh = np.array([r['object_score_p90'] for r in target])
    ax.plot(snr, pm, marker='o', label='Pixel-by-pixel correlation')
    ax.fill_between(snr, pl, ph, alpha=0.16)
    ax.plot(snr, om, marker='s', label='Macroscopic morphology score')
    ax.fill_between(snr, ol, oh, alpha=0.16)
    ax.set_ylim(-0.05, 1.02)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Similarity')
    ax.set_title('Macroscopic persistence despite changing pixels')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'pixel_vs_object_persistence.png', dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for model, marker in [('operator_shaped', 'o'), ('smooth_colored', 's')]:
        rr = [r for r in null if r['null_model'] == model]
        ax.plot([r['cross_view_correlation'] for r in rr], [r['P_score_ge_0_75'] for r in rr], marker=marker, label=f'{model.replace("_", " ")}: S>=0.75')
        ax.plot([r['cross_view_correlation'] for r in rr], [r['P_score_ge_0_90'] for r in rr], marker=marker, linestyle='--', label=f'{model.replace("_", " ")}: S>=0.90')
    ax.set_xlabel('Imposed cross-view correlation')
    ax.set_ylabel('False macroscopic match probability')
    ax.set_ylim(-0.02, 1.02)
    ax.set_title('Hard nulls: when shared structure can mimic persistence')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'null_false_match_rates.png', dpi=220)
    plt.close(fig)
    pd.DataFrame(target).to_csv(out / 'target_persistence.csv', index=False)
    pd.DataFrame(null).to_csv(out / 'structured_nulls.csv', index=False)
    result = {'paper_resolution_m': dz, 'effective_aperture_consistent_with_36m_m': aperture, 'paper_aperture_approx_m': 42000.0, 'n_depth_bins': len(z), 'n_looks_used': k, 'condition_number': float(np.linalg.cond(A)), 'target': target, 'null': null, 'target_trials_per_snr': n_target, 'null_trials_per_condition': trials}
    jdump(result, out / 'results.json')
    return result

def figure11_12_morphology(paths, outdir, trials=500):
    out = mkdir(outdir / 'figure11_12_persistence')
    p11 = paths.get((12, 0))
    p12 = paths.get((12, 1))
    if not p11 or not p12:
        return {'status': 'Figure 11/12 embedded images unavailable'}
    i11 = Image.open(p11).convert('RGB').crop((182, 889, 1613, 1652))
    i12 = Image.open(p12).convert('RGB').crop((175, 73, 1606, 836))
    a = norm01(ndimage.gaussian_filter(jet_scalar_image(np.asarray(i11)), 1.8)[:90])
    b = norm01(ndimage.gaussian_filter(jet_scalar_image(np.asarray(i12)), 1.8)[:90])
    qs = [0.90, 0.92, 0.94, 0.96]
    matches = []
    for q in qs:
        min_area = 0.018 if q <= 0.92 else 0.008
        sc, x, y = best_blob_score(a, b, q, min_area, 0.16)
        matches.append({'q': q, 'score': sc, 'figure11_blob': x, 'figure12_blob': y})
    observed = float(np.median([r['score'] for r in matches]))
    phase_coh = float(abs(np.sum(np.fft.rfft2(a - a.mean()) * np.conj(np.fft.rfft2(b - b.mean())))) / np.sqrt(np.sum(abs(np.fft.rfft2(a - a.mean())) ** 2) * np.sum(abs(np.fft.rfft2(b - b.mean())) ** 2)))
    rng = np.random.default_rng(RNG_SEED)
    null = []
    for rho in [0, 0.5, phase_coh, 0.9, 0.99]:
        vals = []
        for _ in range(trials):
            x, y = phase_surrogate_pair(a, b, rho, rng)
            ss = []
            for q in qs:
                min_area = 0.018 if q <= 0.92 else 0.008
                ss.append(best_blob_score(x, y, q, min_area, 0.16)[0])
            vals.append(float(np.median(ss)))
        vals = np.asarray(vals)
        null.append({'phase_correlation': float(rho), 'p_ge_observed': float((np.sum(vals >= observed) + 1) / (len(vals) + 1)), 'null_median': float(np.median(vals)), 'null_95': float(np.quantile(vals, 0.95)), 'null_99': float(np.quantile(vals, 0.99))})
    ablation = []
    for width in [260, 300, 340]:
        for blur in [1.2, 1.8, 2.4]:
            for rows_kept in [80, 90, 100]:
                aa = norm01(ndimage.gaussian_filter(jet_scalar_image(np.asarray(i11), size=(width, 160)), blur)[:rows_kept])
                bb = norm01(ndimage.gaussian_filter(jet_scalar_image(np.asarray(i12), size=(width, 160)), blur)[:rows_kept])
                ss = []
                for q in [0.88, 0.90, 0.92, 0.94, 0.96]:
                    min_area = 0.018 if q <= 0.92 else 0.008
                    ss.append(best_blob_score(aa, bb, q, min_area, 0.16)[0])
                ablation.append({'width_px': width, 'blur_sigma': blur, 'rows_kept': rows_kept, 'median_multi_threshold_score': float(np.median(ss)), 'pixel_correlation': safe_corr(aa, bb)})
    av = np.asarray([r['median_multi_threshold_score'] for r in ablation])
    ablation_summary = {'n_configurations': len(ablation), 'median_score': float(np.median(av)), 'q10_score': float(np.quantile(av, 0.1)), 'q90_score': float(np.quantile(av, 0.9)), 'fraction_score_ge_0_70': float(np.mean(av >= 0.70)), 'fraction_score_ge_0_75': float(np.mean(av >= 0.75)), 'fraction_score_ge_0_80': float(np.mean(av >= 0.80))}
    pd.DataFrame(ablation).to_csv(out / 'preprocessing_ablation.csv', index=False)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.9), constrained_layout=True)
    ax[0].imshow(a, aspect='auto')
    ax[0].set_title('Nominal-frequency published map')
    ax[0].axis('off')
    ax[1].imshow(b, aspect='auto')
    ax[1].set_title('Lower-frequency published repeat')
    ax[1].axis('off')
    fig.savefig(out / 'preprocessed_maps.png', dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    groups = []
    labels = []
    for blur in [1.2, 1.8, 2.4]:
        groups.append([r['median_multi_threshold_score'] for r in ablation if r['blur_sigma'] == blur])
        labels.append(f'blur={blur}')
    ax.boxplot(groups, labels=labels, showmeans=True)
    ax.axhline(observed, linestyle='--', linewidth=1.2, label='Primary configuration')
    ax.set_ylabel('Median multi-threshold object score')
    ax.set_ylim(0, 1)
    ax.set_title('Published-repeat preprocessing ablation')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'preprocessing_ablation.png', dpi=220)
    plt.close(fig)
    result = {'pixel_correlation_after_lowpass': safe_corr(a, b), 'fourier_complex_coherence': phase_coh, 'multi_threshold_blob_match_median': observed, 'threshold_matches': matches, 'phase_surrogate_null': null, 'preprocessing_ablation': ablation, 'preprocessing_ablation_summary': ablation_summary, 'interpretation': 'object-level morphology; no exact pixel overlap is required'}
    jdump(result, out / 'results.json')
    return result


def high_noise_support_metrics(rec, truth_mask, outer_mask):
    mag = norm01(np.abs(rec))
    smooth = ndimage.gaussian_filter(mag, (1.2, 2.0))
    q = 1.0 - float(np.mean(truth_mask))
    pred = smooth >= np.quantile(smooth, q)
    inter = int(np.count_nonzero(pred & truth_mask))
    dice = float(2 * inter / max(np.count_nonzero(pred) + np.count_nonzero(truth_mask), 1))
    contrast = float(np.mean(smooth[truth_mask]) / max(np.mean(smooth[~outer_mask]), 1e-12))
    return {'magnitude': mag, 'smooth': smooth, 'dice_to_truth': dice, 'truth_region_contrast': contrast}

def shift_depth_field(a, bins):
    out = np.zeros_like(a)
    if bins > 0:
        out[bins:] = a[:-bins]
    elif bins < 0:
        out[:bins] = a[-bins:]
    else:
        out[:] = a
    return out

def vesuvius_severe_noise_validation(outdir, trials=500, snr_db=-10.0):
    out = mkdir(outdir / 'vesuvius_severe_noise')
    lam = 4.86
    R = 650000.0
    dz = 36.0
    aperture = lam * R / (2 * dz)
    z = np.arange(0, 3000 + dz / 2, dz)
    k = 96
    p = TomoParams('Vesuvius_severe_noise', lam, R, aperture, 35.0, k)
    A, _ = steering_matrix(p, z, k)
    P = np.linalg.pinv(A, rcond=1e-10)
    nx = 140
    x = np.arange(nx)
    env = macro_envelope(z, x)
    truth_mask = env >= 0.30
    outer_mask = ndimage.binary_dilation(truth_mask, iterations=3)
    rng = np.random.default_rng(RNG_SEED + 909)
    n_signal = max(160, min(320, trials))
    signal_rows = []
    candidates = []
    reference_power = None
    for j in range(n_signal):
        rec = []
        metrics = []
        for _ in range(2):
            H = textured_macro_view(env, rng)
            Y = A @ H
            if reference_power is None:
                reference_power = float(np.mean(np.abs(Y) ** 2))
            npow = float(np.mean(np.abs(Y) ** 2) / 10 ** (snr_db / 10))
            Rv = P @ (Y + np.sqrt(npow) * complex_noise(Y.shape, rng))
            rec.append(Rv)
            metrics.append(high_noise_support_metrics(Rv, truth_mask, outer_mask))
        pixel_r = safe_corr(metrics[0]['magnitude'], metrics[1]['magnitude'])
        object_s = macro_match(rec[0], rec[1])
        row = {'pixel_correlation': pixel_r, 'object_score': object_s, 'view_a_dice_to_truth': metrics[0]['dice_to_truth'], 'view_b_dice_to_truth': metrics[1]['dice_to_truth'], 'view_a_truth_contrast': metrics[0]['truth_region_contrast'], 'view_b_truth_contrast': metrics[1]['truth_region_contrast']}
        signal_rows.append(row)
        candidates.append((rec[0], rec[1], metrics[0], metrics[1], row))
    med = {k0: float(np.median([r[k0] for r in signal_rows])) for k0 in signal_rows[0]}
    representative = min(candidates, key=lambda c: abs(c[4]['pixel_correlation'] - med['pixel_correlation']) + abs(c[4]['object_score'] - med['object_score']) + 0.5 * abs(c[4]['view_a_dice_to_truth'] - med['view_a_dice_to_truth']) + 0.5 * abs(c[4]['view_b_dice_to_truth'] - med['view_b_dice_to_truth']))
    ablation_rows = []
    n_ablation = max(120, min(220, trials // 2))
    shifted = shift_depth_field(env, 14)
    for case, env_b in [('same object', env), ('object shifted 504 m', shifted), ('object absent in view B', np.zeros_like(env))]:
        vals = []
        for _ in range(n_ablation):
            H1 = textured_macro_view(env, rng)
            H2 = textured_macro_view(env_b, rng)
            Y1 = A @ H1
            Y2 = A @ H2
            npow = reference_power / 10 ** (snr_db / 10)
            R1 = P @ (Y1 + np.sqrt(npow) * complex_noise(Y1.shape, rng))
            R2 = P @ (Y2 + np.sqrt(npow) * complex_noise(Y2.shape, rng))
            m1 = high_noise_support_metrics(R1, truth_mask, outer_mask)
            m2 = high_noise_support_metrics(R2, truth_mask, outer_mask)
            vals.append((safe_corr(m1['magnitude'], m2['magnitude']), macro_match(R1, R2), m1['dice_to_truth'], m2['dice_to_truth'], m1['truth_region_contrast'], m2['truth_region_contrast']))
        v = np.asarray(vals, float)
        ablation_rows.append({'case': case, 'median_pixel_correlation': float(np.median(v[:, 0])), 'median_pair_object_score': float(np.median(v[:, 1])), 'median_view_a_dice_to_original_truth': float(np.median(v[:, 2])), 'median_view_b_dice_to_original_truth': float(np.median(v[:, 3])), 'median_view_a_truth_contrast': float(np.median(v[:, 4])), 'median_view_b_truth_contrast': float(np.median(v[:, 5]))})
    null_rows = []
    for model in ['operator_shaped', 'smooth_colored']:
        for rho in [0.0, 0.5, 0.9, 0.97]:
            vals = []
            false_recovery = 0
            for _ in range(trials):
                if model == 'operator_shaped':
                    n1 = complex_noise((k, nx), rng)
                    ni = complex_noise((k, nx), rng)
                    n2 = rho * n1 + np.sqrt(max(0, 1 - rho * rho)) * ni
                    R1 = P @ n1
                    R2 = P @ n2
                else:
                    f1 = smooth_complex_field((len(z), nx), rng)
                    fi = smooth_complex_field((len(z), nx), rng)
                    R1 = f1
                    R2 = rho * f1 + np.sqrt(max(0, 1 - rho * rho)) * fi
                m1 = high_noise_support_metrics(R1, truth_mask, outer_mask)
                m2 = high_noise_support_metrics(R2, truth_mask, outer_mask)
                pixel_r = safe_corr(m1['magnitude'], m2['magnitude'])
                object_s = macro_match(R1, R2)
                vals.append((pixel_r, object_s, m1['dice_to_truth'], m2['dice_to_truth'], m1['truth_region_contrast'], m2['truth_region_contrast']))
                if m1['dice_to_truth'] >= 0.50 and m2['dice_to_truth'] >= 0.50 and m1['truth_region_contrast'] >= 1.15 and m2['truth_region_contrast'] >= 1.15:
                    false_recovery += 1
            v = np.asarray(vals, float)
            null_rows.append({'null_model': model, 'cross_view_correlation': rho, 'median_pixel_correlation': float(np.median(v[:, 0])), 'median_pair_object_score': float(np.median(v[:, 1])), 'median_view_a_dice_to_truth': float(np.median(v[:, 2])), 'median_view_b_dice_to_truth': float(np.median(v[:, 3])), 'max_dice_to_truth_any_view': float(np.max(v[:, 2:4])), 'false_recovery_count': int(false_recovery), 'trials': int(trials), 'false_recovery_fraction': float(false_recovery / trials)})
    R1, R2, M1, M2, rep = representative
    da = norm01(np.log1p(12 * np.abs(R1)))
    db = norm01(np.log1p(12 * np.abs(R2)))
    consensus = np.sqrt(norm01(ndimage.gaussian_filter(np.abs(R1), (1.2, 2.0))) * norm01(ndimage.gaussian_filter(np.abs(R2), (1.2, 2.0))))
    extent = [x[0], x[-1], z[-1], z[0]]
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 9.0), constrained_layout=True)
    im0 = ax[0, 0].imshow(env, aspect='auto', origin='upper', extent=extent, vmin=0, vmax=1, cmap='jet')
    ax[0, 0].set_title('Known macroscopic envelope')
    ax[0, 0].set_ylabel('Depth z (m)')
    ax[0, 0].set_xlabel('Position along line')
    ax[0, 1].imshow(da, aspect='auto', origin='upper', extent=extent, vmin=0, vmax=1, cmap='jet')
    ax[0, 1].set_title(f'Independent reconstruction A, {snr_db:.0f} dB')
    ax[0, 1].set_xlabel('Position along line')
    ax[1, 0].imshow(db, aspect='auto', origin='upper', extent=extent, vmin=0, vmax=1, cmap='jet')
    ax[1, 0].set_title(f'Independent reconstruction B, {snr_db:.0f} dB')
    ax[1, 0].set_ylabel('Depth z (m)')
    ax[1, 0].set_xlabel('Position along line')
    ax[1, 1].imshow(consensus, aspect='auto', origin='upper', extent=extent, vmin=0, vmax=1, cmap='jet')
    ax[1, 1].contour(x, z, truth_mask.astype(float), levels=[0.5], linewidths=1.2, colors='white')
    ax[1, 1].set_title('Cross-view consensus; white line = known support')
    ax[1, 1].set_xlabel('Position along line')
    fig.colorbar(im0, ax=ax.ravel().tolist(), shrink=0.74, label='Normalized/log-compressed magnitude')
    fig.savefig(out / 'severe_noise_echogram_stress_test.png', dpi=240)
    plt.close(fig)
    pd.DataFrame(signal_rows).to_csv(out / 'severe_noise_signal_trials.csv', index=False)
    pd.DataFrame(ablation_rows).to_csv(out / 'severe_noise_ablations.csv', index=False)
    pd.DataFrame(null_rows).to_csv(out / 'severe_noise_location_nulls.csv', index=False)
    result = {'snr_db': float(snr_db), 'signal_trials': int(n_signal), 'signal_medians': med, 'representative_example': rep, 'ablation_trials_per_case': int(n_ablation), 'ablations': ablation_rows, 'nulls': null_rows, 'support_definition': 'macro_envelope >= 0.30', 'false_recovery_rule': 'both views Dice>=0.50 and truth-region contrast>=1.15'}
    jdump(result, out / 'results.json')
    return result


def save_validation_visuals(paths, imgs, summary, troiano_pdf, outdir):
    out = mkdir(outdir / 'validation_visuals')
    from matplotlib.patches import Rectangle
    p11 = paths.get((12, 0))
    p12 = paths.get((12, 1))
    if p11 and p12:
        src1 = Image.open(p11).convert('RGB').crop((182, 889, 1613, 1652))
        src2 = Image.open(p12).convert('RGB').crop((175, 73, 1606, 836))
        a = norm01(ndimage.gaussian_filter(jet_scalar_image(np.asarray(src1)), 1.8)[:90])
        b = norm01(ndimage.gaussian_filter(jet_scalar_image(np.asarray(src2)), 1.8)[:90])
        fig, ax = plt.subplots(2, 2, figsize=(10.5, 6.8))
        ax[0, 0].imshow(src1); ax[0, 0].set_title('Published nominal-frequency echogram'); ax[0, 0].axis('off')
        ax[0, 1].imshow(src2); ax[0, 1].set_title('Published lower-frequency repeat'); ax[0, 1].axis('off')
        ax[1, 0].imshow(a, aspect='auto'); ax[1, 0].set_title('Audit preprocessing: nominal'); ax[1, 0].axis('off')
        ax[1, 1].imshow(b, aspect='auto'); ax[1, 1].set_title('Audit preprocessing: lower frequency'); ax[1, 1].axis('off')
        fig.suptitle('Published repeat and the maps actually used by the morphology test')
        fig.tight_layout(); fig.savefig(out / 'repeat_source_to_analysis.png', dpi=200, bbox_inches='tight'); plt.close(fig)
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
        for aa, im, name in zip(ax, [a, b], ['Nominal-frequency map', 'Lower-frequency map']):
            aa.imshow(im, aspect='auto')
            for q in [0.90, 0.92]:
                ds = blob_descriptors(im, q, 0.018)
                if ds:
                    d = ds[0]; H, W = im.shape; x0 = (d['cx'] - d['w'] / 2) * W; y0 = (d['cy'] - d['h'] / 2) * H
                    aa.add_patch(Rectangle((x0, y0), d['w'] * W, d['h'] * H, fill=False, linewidth=2))
                    aa.plot(d['cx'] * W, d['cy'] * H, marker='x', markersize=8); aa.text(x0, max(0, y0 - 2), f'q={q:.2f}', fontsize=8)
            aa.set_title(name); aa.axis('off')
        fig.suptitle('Dominant-object localization at the two primary thresholds')
        fig.tight_layout(); fig.savefig(out / 'repeat_object_localization.png', dpi=200, bbox_inches='tight'); plt.close(fig)
    p15 = imgs.get(15, [])
    if len(p15) >= 2:
        payload = []
        for figid, path in [(15, p15[0]), (16, p15[1])]:
            im = Image.open(path).convert('RGB')
            if figid == 15:
                roi = np.asarray(im.crop((140, 666, 1564, 1525))); size = (700, 420); color = 'yellow'; mpp = 3500 / 420
            else:
                roi = np.asarray(im.crop((96, 729, 1473, 1488))); size = (700, 386); color = 'red'; mpp = 4000 / 386
            arr = cv2.resize(roi, size, interpolation=cv2.INTER_AREA); hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
            if color == 'yellow': lm = (hsv[:, :, 0] > 20) & (hsv[:, :, 0] < 40) & (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 150)
            else: lm = ((hsv[:, :, 0] < 8) | (hsv[:, :, 0] > 172)) & (hsv[:, :, 1] > 150) & (hsv[:, :, 2] > 140)
            yl = np.full(arr.shape[1], np.nan)
            for x in range(arr.shape[1]):
                yy = np.where(lm[:, x])[0]
                if len(yy): yl[x] = np.median(yy)
            good = np.isfinite(yl); yl = np.interp(np.arange(len(yl)), np.where(good)[0], yl[good])
            v = jet_scalar_image(arr, size=size); vin = cv2.inpaint((v * 255).astype(np.uint8), (lm * 255).astype(np.uint8), 3, cv2.INPAINT_TELEA) / 255.0
            gy = np.gradient(ndimage.gaussian_filter(vin, 2), axis=0); top = int(0.67 * arr.shape[0]); ridge = []; offs = []
            for x, y0 in enumerate(yl):
                lo = max(0, int(round(y0)) - 5); hi = min(top, int(round(y0)) + 6)
                if hi <= lo: ridge.append(np.nan); continue
                q = np.abs(gy[lo:hi, x]); j = lo + int(np.argmax(q)); ridge.append(j); offs.append(j - y0)
            payload.append((figid, arr, yl, np.asarray(ridge), np.asarray(offs) * mpp))
        fig, ax = plt.subplots(2, 2, figsize=(10.5, 6.6), gridspec_kw={'height_ratios': [2, 1]})
        for j, (figid, arr, yl, ridge, off) in enumerate(payload):
            ax[0, j].imshow(arr); ax[0, j].plot(np.arange(len(yl)), yl, linewidth=1.5, label='published DEM line'); ax[0, j].plot(np.arange(len(ridge)), ridge, linewidth=1.2, label='re-estimated tomographic ridge'); ax[0, j].set_title(f'Published section {figid}: line removal and ridge re-estimation'); ax[0, j].axis('off'); ax[0, j].legend(fontsize=7, loc='lower right')
            med = np.median(np.abs(off)); ax[1, j].plot(np.arange(len(off)), np.abs(off), linewidth=.8); ax[1, j].axhline(med, linestyle='--', label=f'median {med:.1f} m'); ax[1, j].axhline(36, linestyle=':', label='36 m nominal cell'); ax[1, j].set_xlabel('Horizontal sample'); ax[1, j].set_ylabel('|ridge - DEM| (m)'); ax[1, j].legend(fontsize=7)
        fig.suptitle('Independent surface-coordinate extraction from the published SRTM overlays')
        fig.tight_layout(); fig.savefig(out / 'srtm_ridge_diagnostic.png', dpi=200, bbox_inches='tight'); plt.close(fig)
    p18 = imgs.get(18, []); p19 = imgs.get(19, [])
    if len(p18) >= 3 and len(p19) >= 1:
        im20 = np.array(Image.open(p18[0]).convert('RGB')); im21 = np.array(Image.open(p18[1]).convert('RGB')); im22 = np.array(Image.open(p18[2]).convert('RGB')); im23 = np.array(Image.open(p19[0]).convert('RGB'))
        blue = (0, 114, 189); orange = (217, 83, 25)
        configs = {'20a': (im20, (80, 623, 35, 466)), '20b': (im20, (703, 1246, 35, 466)), '20c': (im20, (1337, 1878, 35, 466)), '21a': (im21, (80, 625, 35, 465)), '21b': (im21, (725, 1268, 35, 466)), '21c_error': (im21, (1354, 1896, 35, 466)), '22a': (im22, (82, 627, 35, 463)), '22b': (im22, (709, 1252, 35, 463)), '22c': (im22, (1338, 1880, 35, 470)), '23': (im23, (80, 623, 40, 465))}
        rows = []; traces = {}
        for name, (im, bb) in configs.items():
            t1 = _trace_from_color(im, bb, blue); t2 = _trace_from_color(im, bb, orange); rows.append((name, safe_corr(t1, t2), safe_corr(ndimage.gaussian_filter1d(t1, 15), ndimage.gaussian_filter1d(t2, 15)))); traces[name] = (t1, t2)
        fig = plt.figure(figsize=(10.5, 5.4)); gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1]); aa = fig.add_subplot(gs[:, 0]); aa.imshow(im21); aa.axis('off'); aa.set_title('Published Figure 21: synchronized comparison')
        for k, name in enumerate(['21b', '21c_error']):
            aa = fig.add_subplot(gs[k, 1]); t1, t2 = traces[name]; x = np.linspace(0, 1, len(t1)); z1 = (t1 - np.mean(t1)) / (np.std(t1) + 1e-9); z2 = (t2 - np.mean(t2)) / (np.std(t2) + 1e-9); rr = next(r for r in rows if r[0] == name); aa.plot(x, z1, linewidth=.9, label='digitized SAR'); aa.plot(x, z2, linewidth=.9, label='digitized INGV'); aa.set_title(f'Audit extraction {name}: raw r={rr[1]:.3f}, smoothed r={rr[2]:.3f}'); aa.set_xlabel('Normalized horizontal coordinate'); aa.set_ylabel('Standardized trace')
            if k == 0: aa.legend(fontsize=7)
        fig.suptitle('Published SAR-versus-INGV comparison and independent trace extraction'); fig.tight_layout(); fig.savefig(out / 'ingv_source_to_analysis.png', dpi=200, bbox_inches='tight'); plt.close(fig)
        fig = plt.figure(figsize=(10.5, 7.4)); gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15])
        for k, name in enumerate(['20c', '21b', '21c_error', '22b']):
            aa = fig.add_subplot(gs[k // 2, k % 2]); t1, t2 = traces[name]; x = np.linspace(0, 1, len(t1)); z1 = (t1 - np.mean(t1)) / (np.std(t1) + 1e-9); z2 = (t2 - np.mean(t2)) / (np.std(t2) + 1e-9); rr = next(r for r in rows if r[0] == name); aa.plot(x, z1, linewidth=.9, label='digitized SAR'); aa.plot(x, z2, linewidth=.9, label='digitized INGV'); aa.set_title(f'{name}: raw r={rr[1]:.3f}, smoothed r={rr[2]:.3f}'); aa.set_xlabel('Normalized horizontal coordinate'); aa.set_ylabel('Standardized trace')
            if k == 0: aa.legend(fontsize=7)
        aa = fig.add_subplot(gs[2, :]); names = [r[0] for r in rows]; x = np.arange(len(names)); w = .36; aa.bar(x - w / 2, [r[1] for r in rows], width=w, label='raw trace r'); aa.bar(x + w / 2, [r[2] for r in rows], width=w, label='smoothed r'); aa.set_xticks(x); aa.set_xticklabels(names, rotation=30, ha='right'); aa.set_ylim(-.1, 1); aa.set_ylabel('Correlation'); aa.legend(); fig.suptitle('Independent RGB digitization of the published SAR-versus-INGV curves'); fig.tight_layout(); fig.savefig(out / 'ingv_digitized_validation.png', dpi=200, bbox_inches='tight'); plt.close(fig)
    p20 = imgs.get(20, [])
    if p20:
        mtimg = Image.open(p20[0]).convert('RGB'); boxes = {'a': (0, 5, 435, 415), 'c': (930, 5, 1365, 415), 'd': (0, 530, 435, 1370), 'f': (930, 530, 1365, 1370), 'g': (0, 1450, 435, 1845), 'i': (945, 1450, 1335, 1845)}
        fig, ax = plt.subplots(3, 2, figsize=(8.3, 10.0))
        for i, (p, q) in enumerate([('a', 'c'), ('d', 'f'), ('g', 'i')]):
            c1 = mtimg.crop(boxes[p]); c2 = mtimg.crop(boxes[q]); d1 = figure25_panel_anomaly(np.asarray(c1)); d2 = figure25_panel_anomaly(np.asarray(c2)); arr1 = cv2.resize(np.asarray(c1, dtype=np.uint8), (180, 180), interpolation=cv2.INTER_AREA); arr2 = cv2.resize(np.asarray(c2, dtype=np.uint8), (180, 180), interpolation=cv2.INTER_AREA)
            for aa, arr, d, txt in [(ax[i, 0], arr1, d1, f'SAR panel {p}'), (ax[i, 1], arr2, d2, f'MT panel {q}')]:
                aa.imshow(arr)
                if d:
                    x0 = (d['cx'] - d['w'] / 2) * 180; y0 = (d['cy'] - d['h'] / 2) * 180; aa.add_patch(Rectangle((x0, y0), d['w'] * 180, d['h'] * 180, fill=False, linewidth=2)); aa.plot(d['cx'] * 180, d['cy'] * 180, marker='x', markersize=8)
                aa.set_title(txt); aa.axis('off')
            sc = blob_score(d1, d2, .2) if d1 and d2 else np.nan; ax[i, 1].text(.02, .05, f'object score = {sc:.3f}', transform=ax[i, 1].transAxes, fontsize=9, bbox=dict(boxstyle='round', alpha=.8))
        fig.suptitle('Object descriptors used for the direct SAR-to-magnetotelluric comparison'); fig.tight_layout(); fig.savefig(out / 'mt_object_localization.png', dpi=200, bbox_inches='tight'); plt.close(fig)
    if summary.get('figure25_objects') and summary.get('mt_match'):
        drows = [r for r in summary['figure25_objects'].get('pairs', []) if 'object_match_score' in r]; prows = summary['mt_match'].get('matches', [])
        if drows and prows:
            fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.5)); x = np.arange(len(drows)); w = .34; ax[0].bar(x - w / 2, [r['object_match_score'] for r in drows], width=w, label='observed object score'); ax[0].bar(x + w / 2, [r['null_95'] for r in drows], width=w, label='random-location 95th pct.'); ax[0].set_xticks(x); ax[0].set_xticklabels([r['pair'] for r in drows]); ax[0].set_ylim(0, 1); ax[0].set_title('Direct SAR-to-MT geometry'); ax[0].set_ylabel('Score'); ax[0].legend(fontsize=7)
            x = np.arange(len(prows)); ax[1].bar(x - w / 2, [r['best_match_corr'] for r in prows], width=w, label='template correlation'); ax[1].bar(x + w / 2, [r['null_99'] for r in prows], width=w, label='tile-shuffle 99th pct.'); ax[1].set_xticks(x); ax[1].set_xticklabels([r['panel'] for r in prows]); ax[1].set_ylim(0, 1); ax[1].set_title('MT source provenance'); ax[1].legend(fontsize=7); fig.suptitle('Two different questions in the magnetotelluric validation'); fig.tight_layout(); fig.savefig(out / 'mt_direct_vs_provenance.png', dpi=200, bbox_inches='tight'); plt.close(fig)
            if troiano_pdf:
                fig2 = extract_troiano_fig2(troiano_pdf, out)
                if fig2:
                    src = Image.open(fig2).convert('RGB'); fig = plt.figure(figsize=(10.5, 5.2)); gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1]); aa = fig.add_subplot(gs[0, 0]); aa.imshow(src); aa.axis('off'); aa.set_title('Troiano et al. (2008) resistivity sections'); aa = fig.add_subplot(gs[0, 1]); x = np.arange(len(prows)); aa.bar(x - w / 2, [r['best_match_corr'] for r in prows], width=w, label='best template correlation'); aa.bar(x + w / 2, [r['null_99'] for r in prows], width=w, label='tile-shuffle 99th percentile'); aa.set_xticks(x); aa.set_xticklabels([r['panel'] for r in prows]); aa.set_ylim(0, 1); aa.set_ylabel('Correlation'); aa.legend(fontsize=8)
                    for i, r in enumerate(prows): aa.text(i, max(r['best_match_corr'], r['null_99']) + .025, f"p={r['tile_shuffle_null_p']:.4f}", ha='center', fontsize=9)
                    aa.set_title('Audit traceability test'); fig.suptitle('Independent MT source and quantitative provenance check'); fig.tight_layout(); fig.savefig(out / 'troiano_source_to_analysis.png', dpi=200, bbox_inches='tight'); plt.close(fig)
    p21 = imgs.get(21, [])
    if p21:
        seis = Image.open(p21[0]).convert('RGB'); fig = plt.figure(figsize=(10.5, 5.5)); gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1]); aa = fig.add_subplot(gs[0, 0]); aa.imshow(seis); aa.axis('off'); aa.set_title('Published 3-D seismic-event comparison'); aa = fig.add_subplot(gs[0, 1]); aa.axis('off'); aa.set_title('Independent replay status'); items = [('Hypocentre coordinates', 'Independent INGV catalogue'), ('Depth coordinate', 'Directly testable once the subset is fixed'), ('Event magnitude', 'Independent catalogue field'), ('Acquisition subset', 'Ambiguous: January table vs February text'), ('Event-by-event replay', 'Not asserted without exact SAR timestamp')]; y = .88
        for a0, b0 in items:
            aa.text(.02, y, a0, fontsize=10, fontweight='bold', va='top'); aa.text(.02, y - .06, b0, fontsize=9, va='top', wrap=True); y -= .17
        fig.suptitle('Seismic-event validation: published geometry and what can be reproduced independently'); fig.tight_layout(); fig.savefig(out / 'seismic_source_and_audit.png', dpi=200, bbox_inches='tight'); plt.close(fig)
    return {'outdir': str(out)}


GOTCHA_C = 299792458.0
GOTCHA_FC = 9.6e9
GOTCHA_LAMBDA = GOTCHA_C / GOTCHA_FC
GOTCHA_PLANNED_ELEV_DEG = np.array([45.00, 44.82, 44.64, 44.45, 44.27, 44.08, 43.89, 43.71], float)
GOTCHA_ACTUAL_MEAN_ELEV_DEG = np.array([45.66, 44.01, 43.92, 44.18, 44.14, 43.53, 43.01, 43.06], float)
GOTCHA_TARGETS = pd.DataFrame([
    ('15TR-01C', -32.14, 42.54, -0.53), ('15TR-03C', -28.09, 38.67, -0.42),
    ('15TR-04C', -13.86, 37.70, -0.05), ('15TR-05C', -24.39, 32.96, -0.33),
    ('15TR-06C', -32.50, 33.41, -0.57), ('15TR-07C', -5.12, 22.98, -0.05),
    ('27TR-01C', -7.51, 51.47, -0.09), ('DR-01C', -15.55, 42.96, -0.13),
    ('DR-02C', -26.16, 45.64, -0.43), ('DR-03C', -18.58, 33.53, -0.18),
    ('DR-04C', -20.88, 27.10, -0.23), ('DR-05C', -13.24, 32.09, -0.09),
    ('DR-06C', -29.27, 24.48, -0.48), ('DR-07C', -26.15, 17.50, -0.44),
], columns=['id', 'x_m', 'y_m', 'z_true_m'])
GOTCHA_ZGRID = np.linspace(-1.5, 1.5, 3001)

def gotcha_height_kz(angles_deg, wavelength=GOTCHA_LAMBDA):
    th = np.deg2rad(np.asarray(angles_deg, float))
    th0 = th.mean()
    return 4 * np.pi * (th - th0) / (wavelength * np.cos(th0))

def gotcha_height_profile(y, kz, z=GOTCHA_ZGRID):
    A = np.exp(1j * np.outer(kz, z))
    return np.abs(A.conj().T @ y) / (np.sqrt(len(kz)) * np.linalg.norm(y))

def gotcha_psf_metrics(angles_deg):
    kz = gotcha_height_kz(angles_deg)
    z = np.linspace(-1.5, 1.5, 12001)
    p = np.abs(np.exp(-1j * np.outer(kz, z)).sum(axis=0)) / len(kz)
    i0 = int(np.argmax(p))
    def width(level):
        m = p >= level
        lo = i0
        while lo > 0 and m[lo - 1]:
            lo -= 1
        hi = i0
        while hi + 1 < len(z) and m[hi + 1]:
            hi += 1
        return float(z[hi] - z[lo])
    w50 = width(0.5)
    m = (np.abs(z) > w50) & (np.abs(z) < 1.5)
    return {'minus3db_width_m': width(1 / np.sqrt(2)), 'half_amplitude_width_m': w50, 'largest_central_sidelobe': float(np.max(p[m])), 'z': z, 'psf': p}

def gotcha_height_noise_validation(outdir, trials=500):
    out = mkdir(outdir / 'external_gotcha_height')
    angles = GOTCHA_ACTUAL_MEAN_ELEV_DEG
    kz = gotcha_height_kz(angles)
    A = np.exp(1j * np.outer(kz, GOTCHA_ZGRID))
    AH = A.conj().T
    rng = np.random.default_rng(RNG_SEED + 600)
    rows = []
    for snr in [40, 20, 10, 0, -5]:
        errs = []
        cohs = []
        scale = np.sqrt(10 ** (-snr / 10))
        for zt in GOTCHA_TARGETS.z_true_m:
            y0 = np.exp(1j * kz * float(zt))[:, None]
            n = (rng.standard_normal((len(kz), trials)) + 1j * rng.standard_normal((len(kz), trials))) / np.sqrt(2)
            Y = y0 + scale * n
            P = np.abs(AH @ Y) / (np.sqrt(len(kz)) * np.linalg.norm(Y, axis=0)[None, :])
            ii = np.argmax(P, axis=0)
            zh = GOTCHA_ZGRID[ii]
            errs.extend(np.abs(zh - float(zt)).tolist())
            cohs.extend(P[ii, np.arange(trials)].tolist())
        e = np.asarray(errs)
        rows.append({'snr_db': snr, 'median_abs_error_m': float(np.median(e)), 'p90_abs_error_m': float(np.quantile(e, .9)), 'within_0_25': float(np.mean(e <= .25)), 'within_0_5': float(np.mean(e <= .5)), 'median_peak_coherence': float(np.median(cohs))})
    df = pd.DataFrame(rows)
    df.to_csv(out / 'height_noise.csv', index=False)
    planned = gotcha_psf_metrics(GOTCHA_PLANNED_ELEV_DEG)
    actual = gotcha_psf_metrics(GOTCHA_ACTUAL_MEAN_ELEV_DEG)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(planned['z'], planned['psf'], label='planned elevations')
    ax.plot(actual['z'], actual['psf'], label='published mean actual elevations')
    ax.set_xlim(-1.5, 1.5)
    ax.set_xlabel('Height offset (m)')
    ax.set_ylabel('Normalized coherent response')
    ax.set_title('GOTCHA height focus from independent elevation sampling')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'height_psf.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.plot(df.snr_db, df.median_abs_error_m, marker='o', label='median')
    ax.plot(df.snr_db, df.p90_abs_error_m, marker='o', label='90th percentile')
    ax.axhline(.49, linestyle='--', linewidth=1, label='published ideal ~0.49 m height resolution')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Absolute height error (m)')
    ax.set_title('Known calibration-target heights under additive noise')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'height_noise.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    sign_rows = []
    Af = np.exp(1j * np.outer(-kz, GOTCHA_ZGRID))
    for _, r in GOTCHA_TARGETS.iterrows():
        y = np.exp(1j * kz * float(r.z_true_m))
        z1 = float(GOTCHA_ZGRID[int(np.argmax(np.abs(AH @ y)))])
        z2 = float(GOTCHA_ZGRID[int(np.argmax(np.abs(Af.conj().T @ y)))])
        sign_rows.append({'id': r.id, 'z_true_m': float(r.z_true_m), 'z_correct_m': z1, 'z_kz_flipped_m': z2, 'mirror_error_m': abs(z2 + float(r.z_true_m))})
    sdf = pd.DataFrame(sign_rows)
    sdf.to_csv(out / 'kz_sign_flip.csv', index=False)
    kt = kz
    geo_rows = []
    km = gotcha_height_kz(GOTCHA_PLANNED_ELEV_DEG)
    Am = np.exp(1j * np.outer(km, GOTCHA_ZGRID))
    for _, r in GOTCHA_TARGETS.iterrows():
        y = np.exp(1j * kt * float(r.z_true_m))
        p = np.abs(Am.conj().T @ y) / (np.sqrt(len(km)) * np.linalg.norm(y))
        zh = float(GOTCHA_ZGRID[int(np.argmax(p))])
        geo_rows.append({'id': r.id, 'z_true_m': float(r.z_true_m), 'z_hat_planned_m': zh, 'abs_error_m': abs(zh - float(r.z_true_m)), 'peak_coherence': float(np.max(p))})
    gdf = pd.DataFrame(geo_rows)
    gdf.to_csv(out / 'planned_vs_actual_geometry.csv', index=False)
    pass_rows = []
    for snr in [10, 0]:
        rr = np.random.default_rng(RNG_SEED + 610 + snr)
        for k in [8, 7, 6, 5, 4, 3]:
            subs = list(itertools.combinations(range(8), k))
            all_err = []
            med_sub = []
            for sub in subs:
                idx = np.asarray(sub)
                kk = gotcha_height_kz(angles[idx])
                AA = np.exp(1j * np.outer(kk, GOTCHA_ZGRID))
                AAH = AA.conj().T
                ee = []
                for zt in GOTCHA_TARGETS.z_true_m:
                    y0 = np.exp(1j * kk * float(zt))[:, None]
                    n = (rr.standard_normal((k, 20)) + 1j * rr.standard_normal((k, 20))) / np.sqrt(2)
                    Y = y0 + np.sqrt(10 ** (-snr / 10)) * n
                    P = np.abs(AAH @ Y) / (np.sqrt(k) * np.linalg.norm(Y, axis=0)[None, :])
                    zh = GOTCHA_ZGRID[np.argmax(P, axis=0)]
                    e = np.abs(zh - float(zt))
                    ee.extend(e.tolist())
                    all_err.extend(e.tolist())
                med_sub.append(float(np.median(ee)))
            pass_rows.append({'snr_db': snr, 'n_passes': k, 'median_abs_error_m': float(np.median(all_err)), 'p90_abs_error_m': float(np.quantile(all_err, .9)), 'worst_subset_median_error_m': float(np.max(med_sub))})
    pdf = pd.DataFrame(pass_rows)
    pdf.to_csv(out / 'pass_dropout.csv', index=False)
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    for snr in [10, 0]:
        d = pdf[pdf.snr_db == snr]
        ax.plot(d.n_passes, d.median_abs_error_m, marker='o', label=f'{snr} dB median')
        ax.plot(d.n_passes, d.p90_abs_error_m, marker='o', linestyle='--', label=f'{snr} dB 90th pct.')
    ax.invert_xaxis()
    ax.set_xlabel('Elevation passes retained')
    ax.set_ylabel('Height error (m)')
    ax.set_title('Ablation: removing independent GOTCHA elevation views')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'pass_dropout.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    result = {'source': 'GOTCHA published calibration geometry, not raw phase-history replay', 'center_frequency_hz': GOTCHA_FC, 'wavelength_m': GOTCHA_LAMBDA, 'planned_psf': {k: v for k, v in planned.items() if k not in ['z', 'psf']}, 'actual_mean_psf': {k: v for k, v in actual.items() if k not in ['z', 'psf']}, 'height_noise': rows, 'sign_flip_median_mirror_error_m': float(np.median(sdf.mirror_error_m)), 'geometry_mismatch_median_abs_error_m': float(np.median(gdf.abs_error_m)), 'geometry_mismatch_max_abs_error_m': float(np.max(gdf.abs_error_m)), 'pass_dropout': pass_rows}
    jdump(result, out / 'results.json')
    return result

def _external_jet_inverse(crop):
    jet = (colormaps['jet'](np.arange(256))[:, :3] * 255).astype(np.float32)
    a = crop.astype(np.float32).reshape(-1, 3)
    out = np.empty(len(a), np.float32)
    for i in range(0, len(a), 50000):
        b = a[i:i + 50000]
        d = ((b[:, None, :] - jet[None, :, :]) ** 2).sum(axis=2)
        out[i:i + len(b)] = np.argmin(d, axis=1) / 255.0
    return out.reshape(crop.shape[:2])

def _external_shift_null(maps, q=.95, n=5000, seed=RNG_SEED):
    bs = [m >= np.quantile(m, q) for m in maps]
    c = np.sum(bs, axis=0) >= 2
    lab, nn = ndimage.label(c, np.ones((3, 3), int))
    obs = int(np.max(np.bincount(lab.ravel())[1:])) if nn else 0
    h, w = c.shape
    rng = np.random.default_rng(seed)
    vals = np.empty(n, int)
    for i in range(n):
        s = [np.roll(np.roll(b, int(rng.integers(h)), 0), int(rng.integers(w)), 1) for b in bs]
        cc = np.sum(s, axis=0) >= 2
        ll, n2 = ndimage.label(cc, np.ones((3, 3), int))
        vals[i] = int(np.max(np.bincount(ll.ravel())[1:])) if n2 else 0
    return {'observed_area_px': obs, 'p_shift': float((np.sum(vals >= obs) + 1) / (n + 1)), 'null_median_px': float(np.median(vals)), 'null95_px': float(np.quantile(vals, .95)), 'null99_px': float(np.quantile(vals, .99)), 'null_values': vals, 'consensus': c}

def _external_fixed_roi_null(maps, q=.95, n=20000, frac=.2, seed=RNG_SEED):
    bs = [m >= np.quantile(m, q) for m in maps]
    h, w = bs[0].shape
    rh, rw = int(round(frac * h)), int(round(frac * w))
    y0, x0 = (h - rh) // 2, (w - rw) // 2
    c = np.sum(bs, axis=0) >= 2
    obs = int(c[y0:y0 + rh, x0:x0 + rw].sum())
    rng = np.random.default_rng(seed)
    vals = np.empty(n, int)
    for i in range(n):
        s = [np.roll(np.roll(b, int(rng.integers(h)), 0), int(rng.integers(w)), 1) for b in bs]
        cc = np.sum(s, axis=0) >= 2
        vals[i] = int(cc[y0:y0 + rh, x0:x0 + rw].sum())
    return {'observed_consensus_px': obs, 'p_shift': float((np.sum(vals >= obs) + 1) / (n + 1)), 'null_median_px': float(np.median(vals)), 'null95_px': float(np.quantile(vals, .95)), 'null99_px': float(np.quantile(vals, .99)), 'null_values': vals, 'consensus': c, 'roi': [y0, y0 + rh, x0, x0 + rw]}

def rollo_real_target_validation(page4, page5, outdir, shift_trials=5000, fixed_trials=20000):
    out = mkdir(outdir / 'external_rollo_targets')
    p4 = np.asarray(Image.open(page4).convert('RGB'))
    p5 = np.asarray(Image.open(page5).convert('RGB'))
    tsx = [_external_jet_inverse(p4[178:298, 1357:1624]), _external_jet_inverse(p4[359:479, 990:1257]), _external_jet_inverse(p4[359:479, 1357:1624])]
    umb = [_external_jet_inverse(p4[919:1040, 1366:1624]), _external_jet_inverse(p4[1099:1220, 998:1256]), _external_jet_inverse(p4[1099:1220, 1366:1624])]
    shaker = [_external_jet_inverse(p5[295:399, 174:388]), _external_jet_inverse(p5[447:551, 174:388]), _external_jet_inverse(p5[600:704, 174:388])]
    fence = [_external_jet_inverse(p5[295:399, 479:693]), _external_jet_inverse(p5[447:551, 479:693]), _external_jet_inverse(p5[600:704, 479:693])]
    rt = _external_shift_null(tsx, .95, shift_trials, RNG_SEED + 700)
    ru = _external_shift_null(umb, .95, shift_trials, RNG_SEED + 701)
    rs = _external_fixed_roi_null(shaker, .95, fixed_trials, .2, RNG_SEED + 702)
    rf = _external_fixed_roi_null(fence, .95, fixed_trials, .2, RNG_SEED + 703)
    pd.DataFrame([['TerraSAR-X', 16.13, rt['observed_area_px'], rt['null99_px'], rt['p_shift']], ['Umbra', 1.08, ru['observed_area_px'], ru['null99_px'], ru['p_shift']]], columns=['sensor', 'known_motion_mm', 'observed_area_px', 'null99_px', 'p_shift']).to_csv(out / 'known_target_null.csv', index=False)
    pd.DataFrame([['vibrating shaker', rs['observed_consensus_px'], rs['null99_px'], rs['p_shift']], ['stationary fence', rf['observed_consensus_px'], rf['null99_px'], rf['p_shift']]], columns=['control', 'observed_consensus_px', 'null99_px', 'p_shift']).to_csv(out / 'shaker_stationary_control.csv', index=False)
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.5))
    for j, m in enumerate(umb):
        ax[0, j].imshow(m, aspect='auto', origin='lower')
        ax[0, j].set_title(['Umbra h1,2', 'Umbra h1,3', 'Umbra h2,3'][j])
        ax[0, j].axis('off')
    ax[1, 0].imshow(ru['consensus'], aspect='auto', origin='lower')
    ax[1, 0].set_title('2-of-3 persistent object')
    ax[1, 0].axis('off')
    ax[1, 1].hist(ru['null_values'], bins=55)
    ax[1, 1].axvline(ru['observed_area_px'], linewidth=2)
    ax[1, 1].set_title(f"shift null, p={ru['p_shift']:.4g}")
    ax[1, 1].set_xlabel('Largest consensus object (px)')
    ax[1, 2].axis('off')
    ax[1, 2].text(.02, .78, 'Known physical shaker', fontweight='bold', fontsize=13)
    ax[1, 2].text(.02, .58, '2 Hz, 1.08 mm', fontsize=12)
    ax[1, 2].text(.02, .38, f"Observed: {ru['observed_area_px']} px", fontsize=11)
    ax[1, 2].text(.02, .20, f"Null 99th: {ru['null99_px']:.0f} px", fontsize=11)
    ax[1, 2].text(.02, .04, f"p = {ru['p_shift']:.4g}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / 'umbra_known_target.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.5))
    for row, (name, maps, r) in enumerate([('Vibrating shaker', shaker, rs), ('Stationary fence', fence, rf)]):
        ax[row, 0].imshow(maps[0], aspect='auto', origin='lower')
        ax[row, 0].set_title(name + ' h1,2')
        ax[row, 0].axis('off')
        ax[row, 1].imshow(r['consensus'], aspect='auto', origin='lower')
        y0, y1, x0, x1 = r['roi']
        ax[row, 1].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], linewidth=1.5)
        ax[row, 1].set_title('2-of-3 consensus, fixed ROI')
        ax[row, 1].axis('off')
        ax[row, 2].hist(r['null_values'], bins=50)
        ax[row, 2].axvline(r['observed_consensus_px'], linewidth=2)
        ax[row, 2].set_title(f"p={r['p_shift']:.4g}")
        ax[row, 2].set_xlabel('Consensus pixels in ROI')
    fig.tight_layout()
    fig.savefig(out / 'shaker_vs_stationary.png', dpi=220, bbox_inches='tight')
    result = {'TerraSAR_X_16_13mm': {k: v for k, v in rt.items() if k not in ['null_values', 'consensus']}, 'Umbra_1_08mm': {k: v for k, v in ru.items() if k not in ['null_values', 'consensus']}, 'vibrating_shaker': {k: v for k, v in rs.items() if k not in ['null_values', 'consensus']}, 'stationary_fence': {k: v for k, v in rf.items() if k not in ['null_values', 'consensus']}}
    jdump(result, out / 'results.json')
    return result

def gotcha_durango_image_validation(pdf_path, outdir, trials=20000):
    out = mkdir(outdir / 'external_gotcha_durango')
    doc = fitz.open(pdf_path)
    page = doc[5]
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
    q = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    panel = q[1634:1913, 1044:1396].astype(float)
    lum = .2126 * panel[:, :, 0] + .7152 * panel[:, :, 1] + .0722 * panel[:, :, 2]
    x0, x1 = 1183 - 1044 + 5, 1183 - 1044 + 75 - 5
    y0, y1 = 1756 - 1634 + 5, 1756 - 1634 + 34 - 5
    roi = lum[y0:y1, x0:x1]
    obs = float(np.quantile(roi, .9))
    rh, rw = roi.shape
    rng = np.random.default_rng(RNG_SEED + 720)
    vals = []
    tc = np.array([(y0 + y1) / 2, (x0 + x1) / 2])
    while len(vals) < trials:
        yy = int(rng.integers(5, lum.shape[0] - rh - 5))
        xx = int(rng.integers(5, lum.shape[1] - rw - 5))
        cc = np.array([yy + rh / 2, xx + rw / 2])
        if abs(cc[0] - tc[0]) < rh and abs(cc[1] - tc[1]) < rw:
            continue
        vals.append(float(np.quantile(lum[yy:yy + rh, xx:xx + rw], .9)))
    vals = np.asarray(vals)
    p = float((np.sum(vals >= obs) + 1) / (len(vals) + 1))
    pd.DataFrame([['46 s NCCD GPS/PAUX-predicted Doppler-shifted ROI', obs, p, np.median(vals), np.quantile(vals, .95), np.quantile(vals, .99)]], columns=['test', 'observed_p90_luminance', 'p_random_location', 'null_median', 'null95', 'null99']).to_csv(out / 'image_control.csv', index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3))
    ax[0].imshow(panel.astype(np.uint8))
    ax[0].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], linewidth=2)
    ax[0].set_title('GOTCHA Durango, 46 s NCCD')
    ax[0].axis('off')
    ax[1].hist(vals, bins=60)
    ax[1].axvline(obs, linewidth=2)
    ax[1].set_title(f'Random-location null, p={p:.4g}')
    ax[1].set_xlabel('90th-percentile intensity in matched ROI')
    fig.tight_layout()
    fig.savefig(out / 'durango_real_control.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    result = {'observed_p90_luminance': obs, 'p_random_location': p, 'null_median': float(np.median(vals)), 'null95': float(np.quantile(vals, .95)), 'null99': float(np.quantile(vals, .99)), 'trials': int(trials), 'status': 'image-level real-motion localization control using the GPS/PAUX-predicted ROI published by AFRL; not raw phase-history replay'}
    jdump(result, out / 'results.json')
    return result

def resolve_input(v, candidates):
    if v and Path(v).exists():
        return Path(v)
    for x in candidates:
        p = Path(x)
        if p.exists():
            return p
    return None

def main():
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1).__enter__()
    except Exception:
        pass
    if cv2 is not None:
        cv2.setNumThreads(1)
    ap = argparse.ArgumentParser()
    ap.add_argument('--luna-zip', type=Path)
    ap.add_argument('--vesuvius-paper', type=Path)
    ap.add_argument('--troiano-paper', type=Path)
    ap.add_argument('--outdir', type=Path, default=Path('biondi_complete_retest'))
    ap.add_argument('--online', action='store_true')
    ap.add_argument('--sar-utc', type=str, default=None)
    ap.add_argument('--phase-trials', type=int, default=200)
    ap.add_argument('--null-trials', type=int, default=500)
    ap.add_argument('--mt-permutations', type=int, default=120)
    ap.add_argument('--rollo-page4', type=Path)
    ap.add_argument('--rollo-page5', type=Path)
    ap.add_argument('--gotcha-gmti-pdf', type=Path)
    ap.add_argument('--external-trials', type=int, default=500)
    args = ap.parse_args()
    args.luna_zip = resolve_input(args.luna_zip, ['sar-doppler-tomography-main.zip', '/mnt/data/sar-doppler-tomography-main.zip'])
    args.vesuvius_paper = resolve_input(args.vesuvius_paper, ['vesuvius_paper.pdf', '/mnt/data/vesuvius_paper.pdf'])
    args.troiano_paper = resolve_input(args.troiano_paper, ['troiano_2008_vesuvius_resistivity.pdf', '/mnt/data/troiano_2008_vesuvius_resistivity.pdf'])
    args.rollo_page4 = resolve_input(args.rollo_page4, ['rollo_page4.png'])
    args.rollo_page5 = resolve_input(args.rollo_page5, ['rollo_page5.png'])
    args.gotcha_gmti_pdf = resolve_input(args.gotcha_gmti_pdf, ['GOTCHA_GMTI_challenge.pdf'])
    out = mkdir(args.outdir)
    summary = {'rng_seed': RNG_SEED}
    if args.luna_zip:
        summary['luna'] = audit_luna_zip(args.luna_zip, out)
    giza = TomoParams('Giza', 0.48, n_obs=64)
    summary['giza_operator'] = operator_level_test(giza, -40, 150, out, 'giza')
    summary['giza_calibration'] = known_depth_calibration_sensitivity(giza, out)
    eff = 4.86 * 650000 / (2 * 36)
    ves = TomoParams('Vesuvius', 4.86, 650000, eff, 35, 96)
    summary['vesuvius_operator'] = operator_level_test(ves, 0, 3000, out, 'vesuvius_36m', dz_override=36.0)
    summary['vesuvius_depth_localization'] = vesuvius_depth_localization_validation(out)
    summary['vesuvius_wavelength_ablation'] = vesuvius_wavelength_ablation(out)
    summary['vesuvius_object_null'] = vesuvius_object_null(out, args.null_trials)
    summary['vesuvius_severe_noise'] = vesuvius_severe_noise_validation(out, args.null_trials, -10.0)
    summary['gotcha_external_height'] = gotcha_height_noise_validation(out, args.external_trials)
    if args.rollo_page4 and args.rollo_page5:
        summary['rollo_external_targets'] = rollo_real_target_validation(args.rollo_page4, args.rollo_page5, out)
    if args.gotcha_gmti_pdf:
        summary['gotcha_durango_control'] = gotcha_durango_image_validation(args.gotcha_gmti_pdf, out)
    imgs = {}
    paths = {}
    if args.vesuvius_paper:
        summary['paper'], imgs, paths = parse_vesuvius_paper(args.vesuvius_paper, out)
        summary['curve_digitization'] = published_curve_digitization(imgs, out)
        summary['dem_overlay'] = dem_overlay_energy_test(imgs, out)
        summary['fig11_12'] = figure11_12_morphology(paths, out, args.phase_trials)
        f25 = paths.get((20, 0))
        if f25:
            summary['figure25_objects'] = figure25_object_validation(f25, out, 5000)
    if args.troiano_paper and imgs:
        summary['mt_match'] = match_biondi_mt_to_troiano(imgs, args.troiano_paper, out, args.mt_permutations)
    if imgs and paths:
        summary['validation_visuals'] = save_validation_visuals(paths, imgs, summary, args.troiano_paper, out)
    summary['upsampling_control_100_to_44k'] = upsampling_information_test(100, 44000, out)
    if args.online:
        try:
            summary['online_raw'] = online_raw_tests(out, args.sar_utc, imgs)
        except Exception as e:
            summary['online_raw'] = {'error': str(e)}
    jdump(summary, out / 'ALL_RESULTS.json')
    build_markdown(summary, out / 'SUMMARY.md')
    print(json.dumps({'outdir': str(out.resolve()), 'summary': str((out / 'SUMMARY.md').resolve()), 'json': str((out / 'ALL_RESULTS.json').resolve())}, indent=2))
if __name__ == '__main__':
    main()
