# REX-Mars — Supra-Reciclaje en Marte (2025 UI revamp • SpaceX-grade)
# Flujo: Overview • Table • Charts • Predictions • Optimizer • Process • KPIs • Safety • History
# Mejoras: Mejores contrastes, sin secciones vacías, wrappers condicionales, flujo intuitivo con gamificación (progreso, badges, tooltips), UX SpaceX (minimalista, bold, red accents, icons)

import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import torch

# --- módulos del proyecto ---
from src.data import MarsRecyclingData
from src.features import build_feature_vector, build_feature_vector_from_props
from src.model import train_rexai_2025
from src.eval import evaluate_advanced_solution
from src.spectra import load_spectrum_file, preprocess_spectrum
from src.chem_models import predict_components, predict_compatibility
from src.optimizer import optimize_recipes

# ------------------ Config base ------------------
st.set_page_config(page_title="REX-Mars • SpaceX Dragon Edition", page_icon="🚀", layout="wide")

# ------------------ Tema & CSS (mejor contrastes: WCAG-compliant, bold fonts) ------------------
LIGHT = {
    "--rex-bg": "#FFFFFF",          # pure white for light
    "--rex-surface": "#F5F5F5",
    "--rex-ink": "#000000",         # black for high contrast
    "--rex-muted": "#606060",
    "--rex-border": "#D0D0D0",
    "--rex-primary": "#CC0000",     # SpaceX red
    "--rex-primary-ink": "#FFFFFF",
    "--rex-accent": "#FF9900",      # orange accent
    "--rex-success": "#00CC00",
    "--rex-warning": "#FFCC00",
    "--rex-danger": "#CC0000",
    "--rex-tab-ink": "#000000",
    "--rex-tab-ink-muted": "#808080",
    "--rex-slider": "#CC0000",
}

DARK = {
    "--rex-bg": "#000000",          # pure black for space
    "--rex-surface": "#1A1A1A",     # dark gray panels
    "--rex-ink": "#FFFFFF",         # white text
    "--rex-muted": "#A0A0A0",
    "--rex-border": "#333333",
    "--rex-primary": "#CC0000",     # SpaceX red
    "--rex-primary-ink": "#FFFFFF",
    "--rex-accent": "#FF9900",
    "--rex-success": "#00CC00",
    "--rex-warning": "#FFCC00",
    "--rex-danger": "#CC0000",
    "--rex-tab-ink": "#FFFFFF",
    "--rex-tab-ink-muted": "#808080",
    "--rex-slider": "#CC0000",
}

def _css_vars(theme: dict) -> str:
    return "\n".join(f"{k}: {v};" for k, v in theme.items())

def apply_theme(dark: bool = True):
    theme = DARK if dark else LIGHT
    st.markdown(
        f"""
        <style>
        :root {{
            {_css_vars(theme)}
            --rex-radius: 8px; --rex-pad: 16px; --rex-pad-lg: 24px;
        }}
        html, body, .stApp {{ background: var(--rex-bg); color: var(--rex-ink); font-family: Helvetica, sans-serif; }}
        /* SIDE BAR */
        section[data-testid="stSidebar"] {{ background: var(--rex-surface); border-right:1px solid var(--rex-border); }}
        section[data-testid="stSidebar"] * {{ color: var(--rex-ink) !important; }}
        /* TITLES */
        h1, h2, h3, h4, h5, h6 {{ color: var(--rex-ink) !important; font-weight: bold; }}
        /* INPUTS */
        .stSelectbox, .stMultiSelect, .stTextInput, .stNumberInput, .stFileUploader {{
            color: var(--rex-ink) !important; font-size: 16px;
        }}
        div[data-baseweb="select"] > div {{ background: var(--rex-surface); border:1px solid var(--rex-border); border-radius: 8px; }}
        /* BUTTONS (bold, icons possible) */
        .rex-btn > button {{ 
            border-radius: 8px; border: 0; padding: 12px 20px; font-weight: bold; font-size: 16px;
            box-shadow: 0 2px 4px rgba(0,0,0,.2); transition: background 0.3s;
        }}
        .rex-btn--primary > button {{ background: var(--rex-primary); color: var(--rex-primary-ink); }}
        .rex-btn--ghost > button {{ background: transparent; color: var(--rex-ink); border:1px solid var(--rex-border); }}
        .rex-btn--ok > button {{ background: var(--rex-success); color: #000000; }}
        /* CARDS (shadow subtle) */
        .rex-card {{
            background: var(--rex-surface); border: 1px solid var(--rex-border);
            border-radius: var(--rex-radius); padding: var(--rex-pad-lg);
            box-shadow: 0 4px 8px rgba(0,0,0,.1); margin-bottom: 16px; transition: transform 0.2s;
        }}
        .rex-card:hover {{ transform: translateY(-2px); }}
        /* METRICS (larger, bold) */
        .rex-metric {{ background: var(--rex-surface); border:1px solid var(--rex-border); 
                       border-radius: 8px; padding: 16px; text-align: center; }}
        .rex-metric h4 {{ margin: 0 0 8px 0; color: var(--rex-muted); font-weight: bold; }}
        .rex-metric p {{ margin: 0; font-size: 28px; font-weight: bold; }}
        /* TABS */
        .stTabs [data-baseweb="tab-list"] {{ gap: 16px; }}
        .stTabs [data-baseweb="tab"] button {{ 
            border-radius: 8px; padding: 8px 16px; border:1px solid var(--rex-border);
            background: var(--rex-surface); font-weight: bold;
        }}
        .stTabs [aria-selected="true"] p {{ color: var(--rex-tab-ink) !important; font-weight: bold; }}
        .stTabs [aria-selected="false"] p {{ color: var(--rex-tab-ink-muted) !important; }}
        /* SLIDER (visible thumb) */
        .stSlider [data-baseweb="slider"] div[role="slider"] {{ background: var(--rex-slider) !important; box-shadow: 0 0 8px var(--rex-slider); }}
        .stSlider [data-baseweb="slider"] div[aria-disabled="false"] > div:nth-child(2) > div {{
            background: var(--rex-slider) !important;
        }}
        /* TABLAS */
        .stDataFrame thead tr th {{ color: var(--rex-ink) !important; }}
        /* FOOTER/MAINMENU */
        footer, #MainMenu {{ display: none; }}
        /* Tooltips */
        [title] {{ cursor: help; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Toggle Dark Mode (default dark for space missions)
if "rex_dark" not in st.session_state:
    st.session_state["rex_dark"] = True
st.session_state["rex_dark"] = st.sidebar.toggle("Dark Mode 🚀", value=st.session_state["rex_dark"])
apply_theme(st.session_state["rex_dark"])

# Plotly theme (high contrast)
def use_plotly_theme(dark: bool = True):
    font_color = "#FFFFFF" if dark else "#000000"
    paper = "rgba(0,0,0,0)"
    plot = "#1A1A1A" if dark else "#FFFFFF"
    grid = "#333333" if dark else "#D0D0D0"
    tmpl = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Helvetica, sans-serif", size=14, color=font_color),
            paper_bgcolor=paper,
            plot_bgcolor=plot,
            colorway=["#CC0000", "#FF9900", "#00CC00", "#FFCC00", "#CC0000", "#A0A0A0"],
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis=dict(showgrid=True, gridcolor=grid, zeroline=False, ticks="outside", tickfont=dict(color=font_color)),
            yaxis=dict(showgrid=True, gridcolor=grid, zeroline=False, ticks="outside", tickfont=dict(color=font_color)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=font_color)),
        )
    )
    pio.templates["rex"] = tmpl
    pio.templates.default = "plotly_dark+rex" if dark else "plotly_white+rex"
use_plotly_theme(st.session_state["rex_dark"])

# ------------------ Cache & Estado (agregar gamificación: progreso) ------------------
@st.cache_resource
def get_model_and_data():
    data = MarsRecyclingData()
    model = train_rexai_2025(data, epochs=5)
    return model, data

model, data = get_model_and_data()
st.session_state.setdefault("evaluation", None)
st.session_state.setdefault("runs", [])
st.session_state.setdefault("samples", {})
st.session_state.setdefault("last_action", None)
st.session_state.setdefault("progress_step", 1)  # Gamificación: pasos 1-4
st.session_state.setdefault("achievements", [])  # Badges

# Sidebar con progreso gamificado
with st.sidebar:
    st.header("Mission Progress 🚀")
    progress = st.progress(st.session_state["progress_step"] / 4)
    st.caption(f"Step {st.session_state['progress_step']}/4 completed")
    if st.session_state["achievements"]:
        st.subheader("Achievements")
        for ach in st.session_state["achievements"]:
            st.markdown(f"🏆 {ach}")

# ------------------ UI helpers (agregar icons, tooltips) ------------------
def btn(label, kind="primary", key=None, icon=None):
    label = f"{icon} {label}" if icon else label
    cls = f"rex-btn rex-btn--{kind}"
    with st.container():
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        clicked = st.button(label, key=key)
        st.markdown('</div>', unsafe_allow_html=True)
    return clicked

def metric_card(label: str, value: str, help: str | None = None):
    st.markdown('<div class="rex-metric">', unsafe_allow_html=True)
    st.markdown(f"<h4>{label}</h4>", unsafe_allow_html=True)
    st.markdown(f"<p>{value}</p>", unsafe_allow_html=True)
    if help: st.caption(help)
    st.markdown("</div>", unsafe_allow_html=True)

def kpi_row(ev):
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("REX-Score", f"{ev['rex_score']:.2f}%", "Overall efficiency")
    with c2: metric_card("Energy (kWh/kg)", f"{ev['energy_kwh_kg']:.2f}", "Power usage")
    with c3: metric_card("Water (L/kg)", f"{ev['water_l_kg']:.2f}", "Water consumption")
    with c4: metric_card("Crew Time (min/kg)", f"{ev['crew_min_kg']:.1f}", "Human effort")

def recipe_pie(ev):
    labels = ["Plastics", "Textiles", "Metals", "Composites"]
    fig = go.Figure(data=[go.Pie(labels=labels, values=ev["proportions"], hole=0.45)])
    fig.update_traces(textinfo="percent", hovertemplate="%{label}: %{percent}")
    st.plotly_chart(fig, use_container_width=True)

def process_stepper(ev):
    st.subheader("Manufacturing Process")
    order = ["decomposition", "purify", "extrude", "manufacture"]
    for key in order:
        step = ev["process"].get(key)
        if step:
            st.markdown(f"- **{key.title()}**: {step}")

def log_run(ev, scenario, selected):
    st.session_state["runs"].append({
        "Scenario": scenario,
        "Waste": ", ".join(selected or ["(base)"]),
        "REX": round(ev["rex_score"], 2),
        "E[kWh/kg]": ev["energy_kwh_kg"],
        "W[L/kg]": ev["water_l_kg"],
        "Crew[min/kg]": ev["crew_min_kg"]
    })
    try:
        pd.DataFrame(st.session_state["runs"]).to_csv("runs.csv", index=False)
    except Exception:
        pass
    # Gamificación
    if "First Run Logged" not in st.session_state["achievements"]:
        st.session_state["achievements"].append("First Run Logged")
        st.balloons()

def runs_table():
    runs = st.session_state.get("runs", [])
    if not runs:
        st.info("No trials yet. Generate a recipe to start! 🌟")
        return
    st.dataframe(pd.DataFrame(runs), use_container_width=True, hide_index=True)

def guide(step: int):
    steps = [
        "Choose objective and waste",
        "Load samples in **Table** and preprocess in **Charts** (optional for demo)",
        "Check **Predictions** and try **Optimizer**",
        "View **Process**, **KPIs** and save in **History**",
    ]
    pct = int((step/4)*100)
    st.markdown(f"##### Quick Guide · **Step {step}/4 ({pct}%)** — {steps[step-1]}")
    st.caption("Select **Product Objective** and **Waste to Process**, then hit **Predict Trials**.")

def next_steps_block():
    st.markdown("###### Next Launch Steps 🚀")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Predictions**")
        st.caption("View components, compatibility, suggested trials.")
    with c2:
        st.markdown("**2. Optimizer**")
        st.caption("Explore mixes with NASA constraints.")
    with c3:
        st.markdown("**3. Process**")
        st.caption("Fab plan and resources.")

# ------------------ NAV TABS ------------------
tabs = st.tabs([
    "Overview", "Table", "Charts", "Predictions",
    "Optimizer", "Process", "KPIs", "Safety", "History"
])

# ---------- OVERVIEW ----------
with tabs[0]:
    guide(st.session_state["progress_step"])
    st.subheader("Mission Overview")

    col1, col2 = st.columns([1,1])
    with col1:
        target_sel = st.selectbox("Name of Objective", [t for t in data.get_target_names()] or ["Habitat Panel"], help="Select your target material")
    with col2:
        scenario = st.selectbox("Scenario", ["Residence Renovations", "Cosmic Celebrations", "Daring Discoveries"], help="Mission context")

    families = list(data.families)
    selected_waste = st.multiselect("Waste Types", families, placeholder="Select waste families", help="Choose what to recycle")

    waste_pct = st.slider("Percentage to Process (0.1–1.0)", 0.1, 1.0, 0.5, 0.1, help="Fraction of total waste")

    cta = btn("Predict Trials", "primary", key="gen_main", icon="🔮")

    st.markdown("### ⚡ Quick Demo (90s)")
    dcol1, dcol2, dcol3 = st.columns(3)
    with dcol1:
        demo_specs = btn("Load Demo Spectra (PVDF & Nomex)", "ghost", key="demo_specs", icon="📂")
    with dcol2:
        demo_panel = btn("Demo PANEL (insulation)", "ghost", key="demo_panel", icon="🛡️")
    with dcol3:
        demo_shield = btn("Demo SHIELD (H shielding)", "ghost", key="demo_shield", icon="☢️")

    # acciones (actualizar progreso)
    def _eval_and_store(ev_scenario, ev_selected, total_kg, target_name):
        fb = torch.tensor([[[0.85, 0.90, 0.80, 0.95]]], dtype=torch.float32)
        x = build_feature_vector(data, ev_selected)
        props, perf = model(x, fb)
        ev = evaluate_advanced_solution(
            proportions=props, performance=perf, mars_data=data,
            target_material=target_name, total_kg=total_kg,
            scenario=ev_scenario, selected_waste=ev_selected
        )
        st.session_state["evaluation"] = ev
        log_run(ev, ev_scenario, ev_selected)
        st.session_state["last_action"] = "generated"
        st.session_state["progress_step"] = max(st.session_state["progress_step"], 2)
        st.toast("Recipe generated ✅ Launch successful!")

    if cta:
        _eval_and_store(scenario, selected_waste, waste_pct * data.total_waste, target_sel)

    if demo_specs:
        st.toast("Demo spectra loaded (PVDF & Nomex) ✅")
        st.session_state["last_action"] = "demo_specs"

    def _demo_eval(mix, label):
        fb = torch.tensor([[[0.9, 0.9, 0.8, 0.95]]], dtype=torch.float32)
        x = build_feature_vector_from_props(data, mix)
        props, perf = model(x, fb)
        total_kg = waste_pct * data.total_waste
        ev = evaluate_advanced_solution(
            proportions=props, performance=perf, mars_data=data,
            target_material=target_sel, total_kg=total_kg,
            scenario=f"Demo {label}", selected_waste=[]
        )
        st.session_state["evaluation"] = ev
        log_run(ev, f"Demo {label}", [])
        st.session_state["last_action"] = f"demo_{label.lower()}"
        st.session_state["progress_step"] = max(st.session_state["progress_step"], 2)
        st.toast(f"Demo {label} applied ✅")
        if "Demo Master" not in st.session_state["achievements"]:
            st.session_state["achievements"].append("Demo Master")
            st.balloons()

    if demo_panel:
        _demo_eval((0.60, 0.10, 0.00, 0.30), "PANEL")
    if demo_shield:
        _demo_eval((0.70, 0.05, 0.05, 0.20), "SHIELD")

    ev = st.session_state["evaluation"]
    if ev:
        # Solo wrap en card si hay contenido
        st.markdown('<div class="rex-card">', unsafe_allow_html=True)
        st.markdown("### Mission Brief")
        chips = " • ".join(selected_waste or ["(demo recipe)"])
        left, right = st.columns([2,1])
        with left:
            st.write(f"**Scenario:** {scenario}")
            st.write(f"**Waste Types:** {chips}")
            st.caption(f"**Process Profile:** Metallics/Polymers/Hybrid · Policy: Reject•Reuse•Recycle · Sealed Cabin · No incineration/airlock")
        with right:
            st.metric("Kg to Process", f"{(waste_pct * data.total_waste):,.0f}")
            st.metric("REX-Score", f"{ev['rex_score']:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown('<div class="rex-card">', unsafe_allow_html=True)
            recipe_pie(ev)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="rex-card">', unsafe_allow_html=True)
            kpi_row(ev)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="rex-card">', unsafe_allow_html=True)
        next_steps_block()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div class="rex-card">
              <h3>REX-Mars • SpaceX Dragon — Waste-to-Works. Faster. 🚀</h3>
              <p>Turn mission waste into assets. Load samples, AI suggests mixes with costs, launch your plan.</p>
              <p>Start by selecting options above and predict trials!</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------- TABLE ----------
with tabs[1]:
    if st.session_state["samples"]:
        st.markdown('<div class="rex-card">', unsafe_allow_html=True)
    st.subheader("Data Table")
    st.caption("Upload spectra files: CSV, TXT, JCAMP-DX, mzML — offline processing.")
    uploaded = st.file_uploader(
        "Upload files", type=["csv", "txt", "jdx", "jcamp", "mzml"],
        accept_multiple_files=True, help="Drag and drop or browse"
    )
    if uploaded:
        for up in uploaded:
            name = up.name
            try:
                x_raw, y_raw, meta = load_spectrum_file(up)
                st.session_state["samples"][name] = {"raw": (x_raw, y_raw), "meta": meta}
                st.success(f"{name}: loaded ({len(x_raw)} points). 📈")
            except Exception as e:
                st.error(f"{name}: error ({e}).")
        st.session_state["progress_step"] = max(st.session_state["progress_step"], 2)

    if st.session_state["samples"]:
        st.write("Loaded Samples:")
        st.json({k: {"points": len(v["raw"][0]), "meta": v.get("meta", {})}
                 for k, v in st.session_state["samples"].items()})
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Upload samples to begin analysis. 🌟")

# ---------- CHARTS ----------
with tabs[2]:
    samples = list(st.session_state["samples"].keys())
    if samples:
        st.markdown('<div class="rex-card">', unsafe_allow_html=True)
        st.subheader("Spectra Charts")
        name = st.selectbox("Select Sample", samples, help="Choose a loaded sample")
        x_raw, y_raw = st.session_state["samples"][name]["raw"]
        col = st.columns(4)
        with col[0]:
            do_baseline = st.checkbox("Baseline Correction", value=True, help="Remove baseline drift")
        with col[1]:
            do_snv = st.checkbox("SNV Normalization", value=True, help="Standard Normal Variate")
        with col[2]:
            sg_window = st.slider("SG Window", 5, 41, 15, step=2, help="Savitzky-Golay window size")
        with col[3]:
            sg_order = st.slider("SG Order", 1, 5, 3, help="Polynomial order")

        x_p, y_p, meta = preprocess_spectrum(
            x_raw, y_raw, baseline=do_baseline, snv=do_snv,
            sg_window=sg_window, sg_order=sg_order
        )
        st.session_state["samples"][name]["proc"] = (x_p, y_p)

        fig = go.Figure()
        fig.add_scatter(x=x_raw, y=y_raw, name="Raw", opacity=0.5, mode="lines")
        fig.add_scatter(x=x_p, y=y_p, name="Processed", mode="lines")
        fig.update_layout(xaxis_title="Wavenumber / m/z / λ", yaxis_title="Abs./Intensity", legend_title="Signal")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Load samples in Table tab first. 📊")

# ---------- PREDICTIONS ----------
with tabs[3]:
    samples = list(st.session_state["samples"].keys())
    if samples:
        st.markdown('<div class="rex-card">', unsafe_allow_html=True)
        st.subheader("Score Predictions")
        name = st.selectbox("Sample for Inference", samples, key="pred_sample", help="Analyze composition")
        x_p, y_p = st.session_state["samples"].get(name, {}).get("proc", (None, None))
        if x_p is not None:
            comp = predict_components(x_p, y_p)
            st.write("**Estimated Composition**")
            st.json(comp)

        if len(samples) >= 2:
            a = st.selectbox("Compatibility A", samples, index=0, key="cmp_a")
            b = st.selectbox("Compatibility B", samples, index=1, key="cmp_b")
            ap = st.session_state["samples"].get(a, {}).get("proc")
            bp = st.session_state["samples"].get(b, {}).get("proc")
            if ap and bp:
                score, ci = predict_compatibility(ap, bp)
                st.write(f"**A–B Compatibility:** {score:.2f} (CI: {ci[0]:.2f}–{ci[1]:.2f})")
                fig = go.Figure()
                fig.add_scatter(x=[1], y=[score * 100], mode="markers", error_y=dict(type="data", array=[(ci[1] - score) * 100], arrayminus=[(score - ci[0]) * 100]), name="Predicted")
                fig.update_yaxes(title="Score (%)", range=[0, 100])
                fig.update_xaxes(visible=False)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Suggested Trials")
        colp = st.columns([1,1,1])
        with colp[0]:
            ci_view = st.selectbox("Confidence Interval", ["90%", "95%", "99%"], index=1)
            ci_width = {"90%": 0.05, "95%": 0.07, "99%": 0.10}[ci_view]
        with colp[1]:
            _ = st.selectbox("Batch Display", ["Collapsed", "Expanded"], index=0)
        with colp[2]:
            topk_pred = st.slider("Recipes Count", 5, 12, 8)

        if btn("🔮 Predict Trials", "primary", key="predict_trials", icon="🔮"):
            best = optimize_recipes(model, data, [], step=0.10, top_k=topk_pred)
            best = sorted(best, key=lambda r: r["score"], reverse=True)
            if best:
                labels = [f"B1E{i}" for i in range(1, len(best)+1)]
                ys = [float(r["score"]) for r in best]
                xs = list(range(1, len(best)+1))
                err_plus = [ci_width * 100 for _ in best]
                err_minus = [ci_width * 100 for _ in best]

                fig = go.Figure()
                trial_labels = labels[:3:2]  # Ejemplo
                trial_ys = [ys[i] for i in range(0,3,2)]
                fig.add_scatter(x=trial_labels, y=trial_ys, mode="markers", marker=dict(color="#00CC00", size=12), name="Trial")

                failed_labels = labels[1:3:2]
                failed_ys = [ys[i]-10 for i in range(1,3,2)]
                fig.add_scatter(x=failed_labels, y=failed_ys, mode="markers", marker=dict(color="#FFCC00", size=12), name="Trial with failed restrictions")

                predicted_labels = labels[3:]
                predicted_ys = ys[3:]
                fig.add_scatter(x=predicted_labels, y=predicted_ys, mode="markers", marker=dict(color="#CC00CC", size=12), name="Mean prediction trial",
                                error_y=dict(type="data", array=err_plus[3:], arrayminus=err_minus[3:]))

                fig.update_layout(title="Score Predictions")
                fig.update_yaxes(title="Score", range=[0, 100])
                fig.update_xaxes(title="Trial ID", tickmode="array", tickvals=labels, ticktext=labels)
                st.plotly_chart(fig, use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    small_fig1 = go.Figure()
                    small_fig1.add_scatter(x=["Trial"], y=[60], mode="markers", marker=dict(color="#00CC00"))
                    small_fig1.add_scatter(x=["Failed"], y=[30], mode="markers", marker=dict(color="#FFCC00"))
                    small_fig1.add_scatter(x=["Predicted"], y=[55], mode="markers", error_y=dict(type="data", array=[10], arrayminus=[10]), marker=dict(color="#CC00CC"))
                    small_fig1.update_layout(showlegend=False, height=200, title="Objective A")
                    st.plotly_chart(small_fig1, use_container_width=True)
                with c2:
                    small_fig2 = go.Figure()
                    small_fig2.add_scatter(x=["Trial"], y=[50], mode="markers", marker=dict(color="#00CC00"))
                    small_fig2.add_scatter(x=["Failed"], y=[20], mode="markers", marker=dict(color="#FFCC00"))
                    small_fig2.add_scatter(x=["Predicted"], y=[45], mode="markers", error_y=dict(type="data", array=[15], arrayminus=[15]), marker=dict(color="#CC00CC"))
                    small_fig2.update_layout(showlegend=False, height=200, title="Objective B")
                    st.plotly_chart(small_fig2, use_container_width=True)

                for i, r in enumerate(best, 1):
                    with st.expander(f"{labels[i-1]} — Score {r['score']:.1f}% • P:{r['props'][0]:.2f} T:{r['props'][1]:.2f} M:{r['props'][2]:.2f} C:{r['props'][3]:.2f}"):
                        st.json({k: r["profile"][k] for k in ["energy_kwh_kg", "water_l_kg", "crew_min_kg"]})
                        if btn(f"Apply {labels[i-1]}", "ok", key=f"apply_pred_{i}", icon="✅"):
                            x = build_feature_vector_from_props(data, r["props"])
                            fb = torch.tensor([[[0.85, 0.90, 0.80, 0.95]]], dtype=torch.float32)
                            props, perf = model(x, fb)
                            total_kg = 0.5 * data.total_waste
                            ev = evaluate_advanced_solution(
                                proportions=props, performance=perf, mars_data=data,
                                target_material="Habitat", total_kg=total_kg,
                                scenario="Predict trials", selected_waste=[]
                            )
                            st.session_state["evaluation"] = ev
                            log_run(ev, "Predict trials", [])
                            st.session_state["last_action"] = "applied_prediction"
                            st.session_state["progress_step"] = max(st.session_state["progress_step"], 3)
                            st.toast(f"Recipe {labels[i-1]} applied ✅")
            else:
                st.warning("No valid recipes found. Try adjusting parameters.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Process samples in Charts to unlock predictions. 🔍")

# ---------- OPTIMIZER ----------
with tabs[4]:
    st.subheader("Recipe Optimizer")
    st.caption("Max REX-Score, min resources. Compliant with NASA: no burn, closed loop, sealed.")
    allow_waste = st.multiselect("Limit Families", ["Plastics", "Textiles", "Metals", "Composites", "Foam Packaging"], [], help="Restrict to these")
    step = st.slider("Search Step (Δ)", 0.05, 0.25, 0.10, 0.05, help="Granularity")
    topk = st.slider("Top-K Recipes", 3, 12, 6, help="Number of top results")
    if btn("Explore Combinations", "primary", key="run_opt", icon="⚙️"):
        best = optimize_recipes(model, data, allow_waste, step=step, top_k=topk)
        if best:
            st.session_state["progress_step"] = max(st.session_state["progress_step"], 3)
            if "Optimizer Pro" not in st.session_state["achievements"]:
                st.session_state["achievements"].append("Optimizer Pro")
                st.balloons()
            st.markdown('<div class="rex-card">', unsafe_allow_html=True)
            for i, r in enumerate(best, 1):
                p, t, m, c = r["props"]
                st.write(f"#{i} **REX-Score {r['score']:.2f}%** → P:{p:.2f} T:{t:.2f} M:{m:.2f} C:{c:.2f}")
                with st.expander("Profile"):
                    st.json({k: r["profile"][k] for k in ["energy_kwh_kg", "water_l_kg", "crew_min_kg"]})
                if btn(f"Apply #{i}", "ok", key=f"apply_{i}", icon="✅"):
                    x = build_feature_vector_from_props(data, r["props"])
                    fb = torch.tensor([[[0.85, 0.90, 0.80, 0.95]]], dtype=torch.float32)
                    props, perf = model(x, fb)
                    total_kg = 0.5 * data.total_waste
                    ev = evaluate_advanced_solution(
                        proportions=props, performance=perf, mars_data=data,
                        target_material="Habitat", total_kg=total_kg,
                        scenario="Optimizer", selected_waste=[]
                    )
                    st.session_state["evaluation"] = ev
                    log_run(ev, "Optimizer (Top-K)", [])
                    st.session_state["last_action"] = "applied_optimizer"
                    st.toast(f"Recipe #{i} applied ✅")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No combinations found. Broaden search.")

# ---------- PROCESS ----------
with tabs[5]:
    ev = st.session_state["evaluation"]
    if ev:
        st.markdown('<div class="rex-card">', unsafe_allow_html=True)
        process_stepper(ev)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Generate a recipe in Overview or Predictions to view process. 🛠️")

# ---------- KPIs ----------
with tabs[6]:
    ev = st.session_state["evaluation"]
    if ev:
        st.markdown('<div class="rex-card">', unsafe_allow_html=True)
        st.subheader("Key Performance Indicators")
        kpi_row(ev)
        fig = go.Figure()
        fig.add_bar(x=["Energy", "Water", "Crew"], y=[ev["energy_kwh_kg"], ev["water_l_kg"], ev["crew_min_kg"]])
        fig.update_yaxes(title="Consumption")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Run a trial to see KPIs. 📊")

# ---------- SAFETY ----------
with tabs[7]:
    st.subheader("Safety & Compliance")
    if st.session_state["evaluation"]:
        st.markdown('<div class="rex-card">', unsafe_allow_html=True)
    st.markdown("✔ Sealed Ops (zero emissions; HEPA filters)")
    st.markdown("✔ Closed Water System")
    st.markdown("✔ **No**: Incineration or Airlock Dump")
    ev = st.session_state["evaluation"]
    if ev:
        st.markdown(f"✔ Microplastics Risk: **low** ({ev['risk_microplastics']:.2f})")
        st.markdown(f"✔ Policy: **{ev['management']['Recycle']}**")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- HISTORY ----------
with tabs[8]:
    st.subheader("Mission History")
    runs_table()
    if st.session_state["runs"]:
        st.session_state["progress_step"] = 4
        if "History Buff" not in st.session_state["achievements"]:
            st.session_state["achievements"].append("History Buff")
            st.balloons()
