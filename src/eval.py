from __future__ import annotations
import numpy as np

def _to_list4(x):
    # Acepta list/tuple/np/torch y devuelve lista de 4 normalizada
    try:
        import torch
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    arr = np.array(x).astype(float).reshape(-1)
    if arr.size >= 4:
        arr = arr[:4]
    else:
        arr = np.pad(arr, (0, 4 - arr.size))
    s = float(arr.sum())
    if s > 0:
        arr = arr / s
    return arr.tolist()

def _to_perf3(perf):
    try:
        import torch
        if hasattr(perf, "detach"):
            perf = perf.detach().cpu().numpy()
    except Exception:
        pass
    v = np.array(perf).astype(float).reshape(-1)
    if v.size < 3:
        v = np.pad(v, (0, 3 - v.size))
    return float(v[0]), float(v[1]), float(v[2])

def evaluate_advanced_solution(proportions, performance, mars_data,
                               target_material: str, total_kg: float,
                               scenario: str, selected_waste):
    # proporciones P,T,M,C
    props = _to_list4(proportions)
    p, t, m, c = props

    # perfil de proceso (demo) → energía/agua/crew base
    route = mars_data.process_route_for_mix((p, t, m, c))
    e = w = cr = 0.0
    for op in route:
        costs = mars_data.op_costs(op)
        e += costs["energia"]
        w += costs["agua"]
        cr += costs["crew"]

    # performance del modelo como ajuste fino
    e_m, w_m, cr_m = _to_perf3(performance)
    energy_kwh_kg = e + 0.2 * e_m
    water_l_kg    = w + 0.2 * w_m
    crew_min_kg   = cr + 0.2 * cr_m

    # score simple: más polímero + ISRU + menos consumo
    base = 60.0 * (p + c) + 10.0 * (1.0 - energy_kwh_kg) + 10.0 * (1.0 - water_l_kg) + 10.0 * (1.0 - 0.1 * crew_min_kg)
    rex_score = max(0.0, min(100.0, base))

    return {
        "proportions": props,
        "process": {"decomposition": route[0], "purify": route[1], "extrude": route[2], "manufacture": route[3]},
        "energy_kwh_kg": energy_kwh_kg,
        "water_l_kg": water_l_kg,
        "crew_min_kg": crew_min_kg,
        "rex_score": rex_score,
        "risk_microplastics": 0.15 * (p + c),  # demo
        "management": {"Recycle": "Water-closed loop • No incineration"},
    }
