# ui/components.py
import streamlit as st
import plotly.graph_objects as go

def metric_card(label: str, value: str, help: str | None = None):
    with st.container():
        st.markdown('<div class="rex-card rex-metric">', unsafe_allow_html=True)
        st.markdown(f"<h4>{label}</h4>", unsafe_allow_html=True)
        st.markdown(f"<p>{value}</p>", unsafe_allow_html=True)
        if help:
            st.caption(help)
        st.markdown("</div>", unsafe_allow_html=True)

def kpi_row(ev):
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("REX-Score", f"{ev['rex_score']:.2f}%")
    with c2: metric_card("Energía (kWh/kg)", f"{ev['energy_kwh_kg']:.2f}")
    with c3: metric_card("Agua (L/kg)", f"{ev['water_l_kg']:.2f}")
    with c4: metric_card("Tiempo Crew (min/kg)", f"{ev['crew_min_kg']:.1f}")

def brief_card(ev, scenario, selected_waste, total_kg, profile_name):
    chips = " • ".join(selected_waste or ["(base)"])
    with st.container():
        st.markdown('<div class="rex-card">', unsafe_allow_html=True)
        st.markdown("### Brief")
        st.write(f"**Escenario:** {scenario}")
        st.write(f"**Residuos:** {chips}")
        st.caption(
            f"**Perfil de proceso:** {profile_name} | Política: Reject•Reuse•Recycle | "
            "Cabina sellada | Sin incineración/airlock"
        )
        st.markdown("</div>", unsafe_allow_html=True)

def recipe_pie(ev):
    labels = ["Plásticos", "Textiles", "Metales", "Compuestos"]
    fig = go.Figure(data=[go.Pie(labels=labels, values=ev["proportions"], hole=0.45)])
    fig.update_traces(textinfo="percent", hovertemplate="%{label}: %{percent}")
    return fig

def process_stepper(ev):
    order = ["decomposition", "purify", "extrude", "manufacture"]
    st.markdown('<div class="rex-card">', unsafe_allow_html=True)
    st.subheader("Proceso de fabricación")
    for key in order:
        step = ev["process"].get(key)
        if step:
            st.markdown(f"- **{key.title()}**: {step}")
    st.markdown("</div>", unsafe_allow_html=True)
