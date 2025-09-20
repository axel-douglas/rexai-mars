# src/exporter.py
import io
import pandas as pd

def _route_string(ev):
    # usa las descripciones de proceso guardadas en ev["process"]
    order = ["decomposition","purify","extrude","manufacture"]
    steps = []
    for k in order:
        if ev.get("process", {}).get(k):
            steps.append(f"{k}:{ev['process'][k]}")
    return " | ".join(steps) if steps else ""

def evaluation_to_df(ev, data):
    """Convierte la evaluación actual a un DataFrame de una fila (export)."""
    p,t,m,c = ev["proportions"]
    row = {
        "scenario": ev.get("scenario",""),
        "target": ev.get("target_name",""),
        "P_plastic": round(p,4),
        "T_textile": round(t,4),
        "M_metal": round(m,4),
        "C_composite": round(c,4),
        "rex_score_%": round(ev["rex_score"],2),
        "energy_kwh_kg": round(ev["energy_kwh_kg"],3),
        "water_l_kg": round(ev["water_l_kg"],3),
        "crew_min_kg": round(ev["crew_min_kg"],3),
        "h_mass_frac_%": round(100.0*ev.get("h_mass_frac",0.0),2),
        "risk_microplastics": round(ev.get("risk_microplastics",0.0),3),
        "policy_recycle": ev.get("management",{}).get("Recycle",""),
        "process_route": _route_string(ev),
    }
    # metas del target (si existen)
    tgt = data.get_target(ev.get("target_name","") or "")
    if tgt:
        row.update({
            "target_e_obj_gpa": tgt.e_obj,
            "target_sigma_obj_mpa": tgt.sigma_obj,
            "target_k_obj_w_mk": tgt.k_obj,
            "target_masa_max_kg": tgt.masa_max,
            "target_notas": tgt.notas,
        })
    return pd.DataFrame([row])

def evaluation_to_csv_bytes(ev, data, filename_hint="rexai_run.csv"):
    df = evaluation_to_df(ev, data)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8"), filename_hint

def runs_table_to_csv_bytes(runs):
    df = pd.DataFrame(runs) if not isinstance(runs, pd.DataFrame) else runs
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8"), "rexai_runs_history.csv"
