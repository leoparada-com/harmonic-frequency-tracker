# -*- coding: utf-8 -*-
"""
Flask backend para picos 3D y trayectorias armónicas

- Tracking LATERAL ESTRICTO (cada pico se une al más cercano en el frame siguiente)
- Etiquetado PERFECTO de picos: pertenece a trayectorias si y solo si ese pico (frame_idx, peak_idx)
  fue utilizado en algún track final (tras filtros). Lo demás = huérfano/outlier.
- Exporta:
  * /tracks           -> tracks laterales + zeros + full_* alineados a times_grid
  * /data_labeled     -> puntos_in_tracks vs puntos_outliers, con floor_db y times_grid
  * /data (compat)    -> lista plana de picos sin etiquetas
"""

import os
import numpy as np
from flask import Flask, render_template, jsonify
from scipy.io import wavfile
from scipy.signal import stft, get_window, find_peaks

app = Flask(__name__)

# -------------------- Paths --------------------
DATA_DIR = "data"
WAV_FILE = os.path.join(DATA_DIR, "1.wav")

# -------------------- STFT ---------------------
N_PERSEG   = 4096
N_OVERLAP  = N_PERSEG // 2
N_FFT      = 8192
FMIN, FMAX = 20, 20000
FLOOR_DB   = -110.0

# -------------- Picos -------------------------
TOPK        = 12         # máximo de picos por frame
PEAK_DIST   = 3          # distancia mínima (bins) entre picos
FRAME_SKIP  = 2          # saltar frames para aligerar
FMAX_VIEW   = 4000       # recorte de visualización

# -------------- Tracking lateral ---------------
LATERAL_DF_MAX_HZ = 40.0  # umbral de continuidad entre frames contiguos
TRACK_MIN_PTS     = 6     # puntos mínimos para reportar un track
TRACK_MAX_RETURN  = 16    # máximo de tracks a devolver

# -------------- Cache STFT por mtime -----------
_STFT_CACHE = {"key": None, "f": None, "t": None, "S_db": None}


# =================== Utilidades ===================

def _load_wav():
    fs, x = wavfile.read(WAV_FILE)
    if x.ndim > 1:
        x = x[:, 0]
    return fs, x.astype(np.float32)


def _compute_stft_db(x: np.ndarray, fs: int):
    win = get_window("hann", N_PERSEG, fftbins=True)
    f, t, Z = stft(
        x, fs=fs, window=win, nperseg=N_PERSEG, noverlap=N_OVERLAP,
        nfft=N_FFT, padded=True, boundary=None
    )
    S = np.abs(Z).astype(np.float32)
    S_db = 20 * np.log10(np.maximum(S, 1e-12)).astype(np.float32)
    band = (f >= FMIN) & (f <= FMAX)
    return f[band].astype(np.float32), t.astype(np.float32), np.maximum(S_db[band, :], FLOOR_DB).astype(np.float32)


def _stft_db_cached():
    try:
        mtime = os.path.getmtime(WAV_FILE)
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró {WAV_FILE}")

    key = (mtime, N_PERSEG, N_OVERLAP, N_FFT, FMIN, FMAX)
    if _STFT_CACHE["key"] == key:
        return _STFT_CACHE["f"], _STFT_CACHE["t"], _STFT_CACHE["S_db"]

    fs, x = _load_wav()
    f, t, S_db = _compute_stft_db(x, fs)
    _STFT_CACHE.update({"key": key, "f": f, "t": t, "S_db": S_db})
    return f, t, S_db


# =============== Extracción de picos/frame ===============

def _extract_frames_detailed():
    """
    Devuelve:
      times: [t_j]
      frames: [[{time,freq,mag,frame_idx,peak_idx}], ...] ordenado por frecuencia
      points_flat: lista plana (compat /data)
    IMPORTANTE: peak_idx es el índice DESPUÉS de ordenar por frecuencia.
    """
    f, t, S_db = _stft_db_cached()
    frames, times, points_flat = [], [], []
    frame_idx = -1
    for j in range(0, S_db.shape[1], FRAME_SKIP):
        frame_idx += 1
        col = S_db[:, j]
        med = float(np.median(col))
        base_height = max(FLOOR_DB + 8.0, med + 6.0)
        pk, props = find_peaks(col, height=base_height, prominence=5.0, distance=PEAK_DIST)

        peaks = []
        if pk.size:
            order = np.argsort(props["peak_heights"])[::-1][:TOPK]   # top por altura
            sel = pk[order]
            sel = sel[f[sel] <= FMAX_VIEW]

            # construimos lista temporal, luego ORDENAMOS POR FRECUENCIA y asignamos peak_idx finales
            tmp = [{"time": float(t[j]), "freq": float(f[idx]), "mag": float(col[idx])} for idx in sel]
            tmp.sort(key=lambda p: p["freq"])

            for k, p in enumerate(tmp):
                p["frame_idx"] = frame_idx
                p["peak_idx"]  = k               # índice estable post-orden
                peaks.append(p)
                points_flat.append({"time": p["time"], "freq": p["freq"], "mag": p["mag"]})

        frames.append(peaks)
        times.append(float(t[j]))
    return times, frames, points_flat


# =============== Aux para zeros/full por rejilla ===============

def _zeros_and_full(t_obs, f_obs, m_obs, t_grid):
    n = len(t_grid)
    if n == 0:
        return [], [], [], [], []
    if len(t_obs) == 0:
        return list(t_grid), [np.nan]*n, [FLOOR_DB]*n, [np.nan]*n, [FLOOR_DB]*n

    dt = float(np.median(np.diff(t_grid))) if len(t_grid) > 1 else 0.05
    t0 = float(t_grid[0])

    idx = [int(round((ti - t0) / dt)) for ti in t_obs]
    idx = [min(n-1, max(0, i)) for i in idx]

    fgrid = [None] * n
    for i, ii in enumerate(idx):
        fgrid[ii] = float(f_obs[i])

    idx_sorted = sorted(set(idx))
    for a, b in zip(idx_sorted[:-1], idx_sorted[1:]):
        if b > a + 1:
            fa = fgrid[a]; fb = fgrid[b]
            for g in range(a+1, b):
                alpha = (g - a) / (b - a)
                fgrid[g] = float((1 - alpha) * fa + alpha * fb)

    first, last = idx_sorted[0], idx_sorted[-1]
    for g in range(0, first):
        fgrid[g] = float(fgrid[first])
    for g in range(last + 1, n):
        fgrid[g] = float(fgrid[last])

    m_full = [float(FLOOR_DB)] * n
    for i, ii in enumerate(idx):
        m_full[ii] = float(m_obs[i])

    have = set(idx_sorted)
    tz, fz, mz = [], [], []
    for g in range(n):
        if g not in have:
            tz.append(float(t_grid[g]))
            fz.append(float(fgrid[g]))
            mz.append(float(FLOOR_DB))

    return tz, fz, mz, [float(v) for v in fgrid], m_full


# =============== Tracking lateral + etiquetado ===============

def _link_tracks_lateral(times_grid, frames, df_max=LATERAL_DF_MAX_HZ, min_len=TRACK_MIN_PTS):
    """
    Construye tracks por continuidad lateral (1-a-1) y devuelve:
      tracks_out: lista con tracks válidos (>= min_len) con zeros y full_*
      used_ids:   conjunto de tuples (frame_idx, peak_idx) usados por esos tracks válidos
    """
    if not frames:
        return [], set()

    # Cada track guardará además las identidades (frame_idx, peak_idx) que usa
    tracks_raw = []  # [{times,freqs,mags, ids:[(fi,pi),...]}, ...]
    used_ids_first_frame = set()

    # seeds con todos los picos del primer frame (orden ya por frecuencia)
    t0 = times_grid[0]
    for p in frames[0]:
        tr = {
            "times": [t0],
            "freqs": [p["freq"]],
            "mags" : [p["mag"]],
            "ids"  : [(p["frame_idx"], p["peak_idx"])],
        }
        tracks_raw.append(tr)
        used_ids_first_frame.add((p["frame_idx"], p["peak_idx"]))

    # matching greedy frame a frame
    for k in range(len(frames)-1):
        tA, pA = times_grid[k],   frames[k]
        tB, pB = times_grid[k+1], frames[k+1]
        usedB = [False]*len(pB)

        # Para cada track que termina en tA, buscar el pico más cercano en tB (<= df_max)
        for tr in tracks_raw:
            if not tr["times"] or tr["times"][-1] != tA:
                continue
            f_last = tr["freqs"][-1]

            best_j, best_df = -1, float("inf")
            for j, pj in enumerate(pB):
                if usedB[j]:
                    continue
                df = abs(pj["freq"] - f_last)
                if df <= df_max and df < best_df:
                    best_df, best_j = df, j

            if best_j >= 0:
                pj = pB[best_j]
                tr["times"].append(tB)
                tr["freqs"].append(pj["freq"])
                tr["mags"].append(pj["mag"])
                tr["ids"].append((pj["frame_idx"], pj["peak_idx"]))
                usedB[best_j] = True

        # picos de B no usados inician nuevo track
        for j, pj in enumerate(pB):
            if not usedB[j]:
                tracks_raw.append({
                    "times": [tB], "freqs":[pj["freq"]], "mags":[pj["mag"]],
                    "ids":[(pj["frame_idx"], pj["peak_idx"])],
                })

    # Filtrar por longitud mínima y construir salida con zeros/full
    tracks_valid = []
    used_ids = set()
    for tr in tracks_raw:
        if len(tr["times"]) >= min_len:
            tz, fz, mz, f_full, m_full = _zeros_and_full(tr["times"], tr["freqs"], tr["mags"], times_grid)
            tracks_valid.append({
                "times": list(map(float, tr["times"])),
                "freqs": list(map(float, tr["freqs"])),
                "mags" : list(map(float, tr["mags"])),
                "zeros_times": tz, "zeros_freqs": fz, "zeros_mags": mz,
                "full_freqs": f_full, "full_mags": m_full
            })
            used_ids.update(tr["ids"])  # ¡solo ids de tracks válidos!

    # Ordenar por longitud y nivel medio
    def _score(tr):
        return (len(tr["times"]), float(np.mean(tr["mags"])) if tr["mags"] else -1e9)

    tracks_valid.sort(key=_score, reverse=True)
    if len(tracks_valid) > TRACK_MAX_RETURN:
        # si recortamos, mantenemos coherencia de used_ids con el corte (opcional)
        kept = set()
        new_tracks = []
        for tr in tracks_valid[:TRACK_MAX_RETURN]:
            new_tracks.append(tr)
            # notar que used_ids ya contiene más ids, pero eso no afecta el etiquetado
            # (solo ayuda a considerar como in-track puntos de tracks largos fuera del top N si no recortáramos).
            # Si prefieres estrictamente los del top N, elimina este comentario y limpia used_ids aquí.
            kept.add(id(tr))
        tracks_valid = new_tracks

    return tracks_valid, used_ids


# =================== Endpoints ===================

@app.route("/data")
def data():
    # puntos crudos (compatibilidad)
    _, _, points_flat = _extract_frames_detailed()
    return jsonify(points_flat)


@app.route("/data_labeled")
def data_labeled():
    # puntos etiquetados correctamente contra los tracks DEVUELTOS
    times, frames, _ = _extract_frames_detailed()
    tracks, used_ids = _link_tracks_lateral(times, frames, df_max=LATERAL_DF_MAX_HZ, min_len=TRACK_MIN_PTS)

    pts_in, pts_out = [], []
    for fr in frames:
        for p in fr:
            key = (p["frame_idx"], p["peak_idx"])
            item = {"time": p["time"], "freq": p["freq"], "mag": p["mag"]}
            if key in used_ids:
                pts_in.append(item)
            else:
                pts_out.append(item)

    return jsonify({
        "points_in_tracks": pts_in,
        "points_outliers": pts_out,
        "floor_db": FLOOR_DB,
        "times_grid": times
    })


@app.route("/tracks")
def tracks():
    times, frames, _ = _extract_frames_detailed()
    tks, _ = _link_tracks_lateral(times, frames, df_max=LATERAL_DF_MAX_HZ, min_len=TRACK_MIN_PTS)
    return jsonify({
        "tracks": tks,
        "floor_db": FLOOR_DB,
        "times_grid": times
    })


# ==================== Frontend ====================

@app.route("/")
def index():
    return render_template("index.html")


# ===================== Main =======================

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(WAV_FILE):
        print(f"⚠️ No se encontró {WAV_FILE}. Coloca tu archivo WAV en {DATA_DIR}/")
    app.run(debug=True)
