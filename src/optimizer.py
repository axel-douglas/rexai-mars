from __future__ import annotations
import numpy as np
import torch

from .features import build_feature_vector_from_props
from .eval import evaluate_advanced_solution

def _ensure_tensor(x):
    return x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32)

def _valid_props(props, allow_families):
    """
    Si allow_families está vacío → todo permitido.
    Si no, sólo acepta proporciones >0 en las familias permitidas.
    Orden props: (Plásticos, Textiles, Metales, Compuestos)
    """
    if not allow_families:
        return True
    fams = ["Plásticos", "Textiles", "Metales", "Compuestos"]
    for i, fam in enumerate(fams):
        if fam not in allow_families and props[i] > 1e-9:
            return False
    return True

def optimize_recipes(model, data, allow_families=None, step=0.10, top_k=5):
    """
    Grid simple en pasos 'step' sobre P,T,M y C=1-P-T-M (si >=0).
    Evalúa con el modelo (para performance) y con evaluate_advanced_solution.
    Devuelve Top-K por REX-Score.
    """
    allow_families = allow_families or []
    step = float(step)

    candidates = []
    grid = np.arange(0.0, 1.0 + 1e-9, step)

    for p in grid:
        for t in grid:
            for m in grid:
                c = 1.0 - (p + t + m)
                if c < -1e-9:
                    continue
                c = max(0.0, float(c))
                props = (float(p), float(t), float(m), c)
                if not _valid_props(props, allow_families):
                    continue

                # Features para el modelo
                x = build_feature_vector_from_props(data, props)  # → 4 features
                x = _ensure_tensor(x).reshape(1, 1, -1)           # (1,1,4)
                fb = _ensure_tensor([[[0.85, 0.90, 0.80, 0.95]]]) # (1,1,4)

                props_pred, perf = model(x, fb)

                # Evaluación: usamos la mezcla candidata (props) y la perf del modelo
                ev = evaluate_advanced_solution(
                    proportions=list(props),
                    performance=perf,
                    mars_data=data,
                    target_material="Optimizer",
                    total_kg=0.5 * data.total_waste,
                    scenario="Grid",
                    selected_waste=[]
                )

                candidates.append({
                    "props": props,
                    "score": ev["rex_score"],
                    "profile": {
                        "energy_kwh_kg": ev["energy_kwh_kg"],
                        "water_l_kg":    ev["water_l_kg"],
                        "crew_min_kg":   ev["crew_min_kg"],
                    },
                })

    candidates.sort(key=lambda r: r["score"], reverse=True)
    return candidates[: top_k]
