import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from rexai_model import RexAI_2025, MarsRecyclingData  # Importar módulos

# Configuración página estilo SpaceX
st.set_page_config(
    page_title="REX-AI 2025 - SpaceTrash Hack",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #FF4500, #FF6347);
        padding: 2rem;
        text-align: center;
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-header"><h1>🚀 REX-AI 2025</h1><p>Reciclaje Disruptivo para Marte - SpaceTrash Hack Winner</p></div>', unsafe_allow_html=True)
    
    # Sidebar controles
    with st.sidebar:
        st.header("🎛️ Controles Misión")
        material = st.selectbox("Material Objetivo", ["Hábitat", "Herramientas", "Aislamiento", "Electrónica"])
        waste_percent = st.slider("Porcentaje Basura", 10, 100, 50)
        feedback = st.slider("Feedback Humano", 0.5, 1.0, 0.85)
        if st.button("🔄 Actualizar IoT", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    # Cargar datos y modelo
    @st.cache_resource
    def load_rexai():
        mars_data = MarsRecyclingData()
        model = train_rexai_optimized(mars_data)
        return model, mars_data
    
    model, mars_data = load_rexai()
    
    # Generar solución
    if st.button("🪐 Generar Solución Rex-AI", type="primary", use_container_width=True):
        with st.spinner("🔬 Generando solución óptima..."):
            # Lógica de generación (similar al Colab)
            total_kg = (waste_percent / 100) * mars_data.total_waste
            
            # Simular resultados
            proportions = np.random.dirichlet(np.ones(4))
            performance = np.random.uniform(0.7, 0.95, 4)
            evaluation = evaluate_solution(
                torch.tensor([proportions]), 
                torch.tensor([performance]), 
                mars_data, material, total_kg
            )
            
            # Mostrar resultados
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Rex-Score</h3>
                    <h2>{evaluation['rex_score']:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Kg Procesados</h3>
                    <h2>{evaluation['kg_processed']:.0f}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>CO₂ Ahorrado</h3>
                    <h2>{evaluation['co2_saved']:.0f}kg</h2>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Eficiencia</h3>
                    <h2>{evaluation['efficiency']:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Gráficos
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=list(mars_data.waste_composition.keys()),
                values=[p*100 for p in proportions],
                marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("*Desarrollado con ❤️ por OneRex | Arquitectura basada en Rex-AI 2025*")

if __name__ == "__main__":
    main()
