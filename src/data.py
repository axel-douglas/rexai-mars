# src/data.py
from __future__ import annotations
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass

DATA_DIR_ENV = "REXAI_DATA_DIR"

def _read_csv(fname: str) -> pd.DataFrame:
    base1 = os.path.join(os.getcwd(), "data", fname)
    base2 = os.path.join(os.getenv(DATA_DIR_ENV, os.getcwd()), fname)
    for path in (base1, base2, fname):
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            return df
    raise FileNotFoundError(f"No se encontró {fname} en ./data/ o ruta env {DATA_DIR_ENV}")

def _safe_col(df: pd.DataFrame, name: str, default=None):
    return df[name] if name in df.columns else default

@dataclass
class TargetProfile:
    target_id: str
    nombre: str
    uso: str
    e_obj: float | None
    sigma_obj: float | None
    k_obj: float | None
    masa_max: float | None
    notas: str | None

class MarsRecyclingData:
    def __init__(self):
        self.waste_catalog = _read_csv("waste_catalog.csv")
        self.process_ledger = _read_csv("process_ledger.csv")
        self.targets_df = _read_csv("targets.csv")

        # --- normalizaciones de columnas ---
        # waste_catalog: item_id, nombre, familia, subfamilia, pct_masa, densidad, tm_tg_c, ...
        rename_wc = {}
        if "porc_masa" in self.waste_catalog.columns and "pct_masa" not in self.waste_catalog.columns:
            rename_wc["porc_masa"] = "pct_masa"
        if "%masa" in self.waste_catalog.columns and "pct_masa" not in self.waste_catalog.columns:
            rename_wc["%masa"] = "pct_masa"
        if "porcentaje_masa" in self.waste_catalog.columns and "pct_masa" not in self.waste_catalog.columns:
            rename_wc["porcentaje_masa"] = "pct_masa"
        if rename_wc:
            self.waste_catalog = self.waste_catalog.rename(columns=rename_wc)

        if "familia" not in self.waste_catalog.columns:
            raise ValueError("waste_catalog.csv debe incluir columna 'familia'.")

        # targets: admite e_obj_gpa, sigma_obj_mpa, k_obj_w_mk, masa_max_kg
        t = self.targets_df
        if "e_obj" not in t.columns and "e_obj_gpa" in t.columns:
            t = t.rename(columns={"e_obj_gpa": "e_obj"})
        if "sigma_obj" not in t.columns and "sigma_obj_mpa" in t.columns:
            t = t.rename(columns={"sigma_obj_mpa": "sigma_obj"})
        if "k_obj" not in t.columns and "k_obj_w_mk" in t.columns:
            t = t.rename(columns={"k_obj_w_mk": "k_obj"})
        if "masa_max" not in t.columns and "masa_max_kg" in t.columns:
            t = t.rename(columns={"masa_max_kg": "masa_max"})
        self.targets_df = t

        # procesa targets
        self.targets = []
        for i, r in self.targets_df.iterrows():
            self.targets.append(TargetProfile(
                target_id=str(_safe_col(self.targets_df, "target_id", pd.Series([f"t{i}"]))[i]),
                nombre=str(_safe_col(self.targets_df, "nombre", pd.Series([f"target_{i}"]))[i]),
                uso=str(_safe_col(self.targets_df, "uso", pd.Series([""]))[i]),
                e_obj=float(_safe_col(self.targets_df, "e_obj", pd.Series([np.nan]))[i]) if not pd.isna(
                    _safe_col(self.targets_df, "e_obj", pd.Series([np.nan]))[i]) else None,
                sigma_obj=float(_safe_col(self.targets_df, "sigma_obj", pd.Series([np.nan]))[i]) if not pd.isna(
                    _safe_col(self.targets_df, "sigma_obj", pd.Series([np.nan]))[i]) else None,
                k_obj=float(_safe_col(self.targets_df, "k_obj", pd.Series([np.nan]))[i]) if not pd.isna(
                    _safe_col(self.targets_df, "k_obj", pd.Series([np.nan]))[i]) else None,
                masa_max=float(_safe_col(self.targets_df, "masa_max", pd.Series([np.nan]))[i]) if not pd.isna(
                    _safe_col(self.targets_df, "masa_max", pd.Series([np.nan]))[i]) else None,
                notas=str(_safe_col(self.targets_df, "notas", pd.Series([""]))[i]),
            ))

        # composición por familia (normalizada)
        fam = self.waste_catalog["familia"].astype(str).str.strip().str.title()
        frac = _safe_col(self.waste_catalog, "pct_masa", pd.Series([np.nan] * len(self.waste_catalog)))
        if frac.isna().all():
            frac = pd.Series([1.0] * len(self.waste_catalog))
        agg = pd.DataFrame({"familia": fam, "pct": frac}).groupby("familia")["pct"].sum()
        agg = agg / max(agg.sum(), 1e-9)
        comp = {
            "Plásticos": float(agg.get("Plástico", 0.0) + agg.get("Plásticos", 0.0) + agg.get("Foam Packaging", 0.0)),
            "Textiles": float(agg.get("Textil", 0.0) + agg.get("Textiles", 0.0)),
            "Metales": float(agg.get("Metal", 0.0) + agg.get("Metales", 0.0)),
            "Compuestos": float(agg.get("Compuesto", 0.0) + agg.get("Compuestos", 0.0)),
        }
        s = sum(comp.values()) or 1.0
        self.waste_composition = {k: v / s for k, v in comp.items()}

        self.total_waste = 1000.0  # kg

    @property
    def families(self) -> list[str]:
        return ["Plásticos", "Textiles", "Metales", "Compuestos"]

    def get_target_names(self) -> list[str]:
        return [t.nombre for t in self.targets]

    def get_target(self, nombre: str) -> TargetProfile | None:
        for t in self.targets:
            if t.nombre == nombre:
                return t
        return None

    def op_costs(self, operacion: str) -> dict:
        df = self.process_ledger
        # soporta 'operacion' y 'operación'
        if "operacion" in df.columns:
            mask = df["operacion"].fillna("").str.lower().eq(operacion.lower())
        elif "operación" in df.columns:
            mask = df["operación"].fillna("").str.lower().eq(operacion.lower())
        else:
            mask = pd.Series([False] * len(df))
        row = df[mask].head(1)
        if row.empty:
            return {"energia": 0.0, "agua": 0.0, "crew": 0.0}

        def _g(c1, default=0.0):
            return float(row.iloc[0][c1]) if c1 in row.columns else default

        return {
            "energia": _g("energia_kwh_kg"),
            "agua": _g("agua_l_kg"),
            "crew": _g("crew_min_kg"),
        }

    def process_route_for_mix(self, mix: tuple[float, float, float, float]) -> list[str]:
        p, t, m, c = mix
        dom = int(np.argmax([p, t, m, c]))
        if dom == 2:
            return ["triturado", "separación", "fusion_al", "laminado"]
        elif dom == 0:
            return ["triturado", "lavado", "extrusion", "manufactura"]
        elif dom == 1:
            return ["triturado", "lavado", "aglomerado/prensado", "manufactura"]
        else:
            return ["triturado", "lavado", "sinterizado_foam", "manufactura"]
