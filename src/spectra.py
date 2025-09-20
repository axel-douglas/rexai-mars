import numpy as np
import pandas as pd

def load_spectrum_file(file):
    name = getattr(file, "name", "uploaded")
    if name.lower().endswith(".csv"):
        df = pd.read_csv(file)
        xcol = [c for c in df.columns if "x" in c.lower() or "wn" in c.lower() or "mz" in c.lower()][0]
        ycol = [c for c in df.columns if "y" in c.lower() or "abs" in c.lower() or "int" in c.lower()][0]
        x = df[xcol].values
        y = df[ycol].values
        return x, y, {"name": name, "format": "csv"}
    # (stub) otros formatos
    raise ValueError("Formato no soportado en demo (usa CSV)")

def preprocess_spectrum(x, y, baseline=True, snv=True, sg_window=15, sg_order=3):
    xp = np.array(x, dtype=float)
    yp = np.array(y, dtype=float)
    if baseline:
        yp = yp - np.percentile(yp, 5)
    if snv:
        yp = (yp - yp.mean()) / (yp.std() + 1e-6)
    # Savitzky-Golay simple (demostrativo)
    try:
        from scipy.signal import savgol_filter
        yp = savgol_filter(yp, max(5, sg_window|1), sg_order)
    except Exception:
        pass
    return xp, yp, {"baseline": baseline, "snv": snv, "sg_window": sg_window, "sg_order": sg_order}

def make_demo_spectrum(kind="pvdf", n=1200, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(600, 3500, n)
    y = np.zeros_like(x)

    def bump(mu, sigma, amp):
        return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    if kind.lower() == "pvdf":
        for mu in [765, 973, 1180, 1402]:
            y += bump(mu, 22, 0.45)
    else:  # nomex
        for mu in [1540, 1640, 3300]:
            y += bump(mu, 28, 0.50)

    y += 0.03 * rng.normal(size=n)  # ruido
    y = (y - y.min()) / (y.max() - y.min() + 1e-9)
    return x.tolist(), y.tolist()
