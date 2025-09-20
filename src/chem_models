# src/chem_models.py
import numpy as np

# Picos alineados con espectros demo (ver make_demo_spectrum en spectra.py)
_PVDF_PEAKS = [765, 973, 1180, 1402]
_NOMEX_PEAKS = [1540, 1640, 3300]

def _peak_score(x, y, centers, tol=40):
    x = np.asarray(x); y = np.asarray(y)
    s = 0.0
    for c in centers:
        m = (x >= c - tol) & (x <= c + tol)
        if m.any():
            s += float(np.maximum(y[m], 0).mean())
    return s

def predict_components(x, y):
    x = np.asarray(x); y = np.asarray(y)
    if y.size == 0:
        return {"family": "Desconocido", "probs": {"otros": 1.0}, "peaks_hint": False}

    s_pvdf  = _peak_score(x, y, _PVDF_PEAKS)
    s_nomex = _peak_score(x, y, _NOMEX_PEAKS)
    total = s_pvdf + s_nomex

    if total < 1e-6:
        idx = int(np.argmax(y))
        if idx < len(y)//3 and np.max(y) > (y.mean() + 2*y.std()):
            return {"family": "Plásticos", "probs": {"PVDF": 0.85, "otros": 0.15}, "peaks_hint": False}
        if idx > 2*len(y)//3 and np.max(y) > (y.mean() + 2*y.std()):
            return {"family": "Textiles", "probs": {"Nomex": 0.85, "otros": 0.15}, "peaks_hint": False}
        return {"family": "Desconocido", "probs": {"otros": 1.0}, "peaks_hint": False}

    p_pvdf = float(s_pvdf / (total + 1e-9))
    p_nomex = 1.0 - p_pvdf
    fam = "Plásticos" if p_pvdf >= p_nomex else "Textiles"
    return {
        "family": fam,
        "probs": {"PVDF": p_pvdf, "Nomex": p_nomex},
        "peaks_hint": True,
        "peaks_used": {"PVDF": _PVDF_PEAKS, "Nomex": _NOMEX_PEAKS},
    }

def predict_compatibility(a_proc, b_proc):
    (xa, ya) = a_proc; (xb, yb) = b_proc
    n = min(len(ya), len(yb))
    if n == 0:
        return 0.5, (0.45, 0.55)
    ya = np.asarray(ya[:n]); yb = np.asarray(yb[:n])
    ya = ya / (np.linalg.norm(ya) + 1e-6)
    yb = yb / (np.linalg.norm(yb) + 1e-6)
    score = float(np.clip(ya @ yb, 0.0, 1.0))
    w = 0.10 * (1.0 - score) + 0.05
    return score, (max(0.0, score - w), min(1.0, score + w))

