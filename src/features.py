import numpy as np

def build_feature_vector(data, selected_families=None):
    """
    Devuelve vector de 4 features [Plásticos, Textiles, Metales, Compuestos]
    normalizado. Si se pasa `selected_families`, limita y renormaliza.
    """
    comp = data.waste_composition.copy()
    if selected_families:
        s = sum(comp.get(fam,0.0) for fam in selected_families) or 1.0
        vec = [
            comp.get("Plásticos",0.0) if "Plásticos" in selected_families else 0.0,
            comp.get("Textiles",0.0)  if "Textiles"  in selected_families else 0.0,
            comp.get("Metales",0.0)   if "Metales"   in selected_families else 0.0,
            comp.get("Compuestos",0.0)if "Compuestos"in selected_families else 0.0,
        ]
        vec = [v/s for v in vec]
    else:
        vec = [comp["Plásticos"], comp["Textiles"], comp["Metales"], comp["Compuestos"]]
    # salida 2D o 3D no obligatoria; app/model ya normalizan
    return np.array([vec], dtype=np.float32)

def build_feature_vector_from_props(data, props_tuple):
    p,t,m,c = props_tuple
    return np.array([[p,t,m,c]], dtype=np.float32)
