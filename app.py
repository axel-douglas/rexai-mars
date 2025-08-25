import streamlit as st
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import optuna
import time
import random
from datetime import datetime

# Configuración página estilo SpaceX
st.set_page_config(
    page_title="REX-AI 2025 - SpaceTrash Hack Winner 🇦🇷",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado minimalista
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #0a2540, #1a3c72);
        padding: 2rem;
        text-align: center;
        color: white;
        border-radius: 10px;
        margin-bottom: 1rem;
        font-family: 'Orbitron', monospace;
    }
    .metric-box {
        background: #FF4500;
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 🪐 Clase MarsRecyclingData (Optimizada)
class MarsRecyclingData:
    def __init__(self):
        self.total_waste = 12600
        self.waste_composition = {
            'Plásticos': {'density': 1.2, 'strength': 50},
            'Textiles': {'density': 0.8, 'strength': 20},
            'Metales': {'density': 2.7, 'strength': 200},
            'Compuestos': {'density': 1.5, 'strength': 80}
        }
        self.target_materials = {
            'Hábitat': {'strength': 250},
            'Herramientas': {'strength': 150},
            'Aislamiento': {'strength': 80},
            'Electrónica': {'strength': 100}
        }
        self.mars_conditions = {'temp_range': (-140, 20), 'gravity': 0.38}
        self.iot_data = {}
        self.model = None
        self.scaler = StandardScaler()

    def update_iot_data(self):
        self.iot_data = {
            'current_temp': np.random.uniform(*self.mars_conditions['temp_range']),
            'gravity': self.mars_conditions['gravity'],
            'timestamp': datetime.now().isoformat()
        }
        return self.iot_data

    def generate_training_data(self, n_samples=500):
        X = []
        y_proportions = []
        y_performance = []
        for _ in range(n_samples):
            weights = np.random.dirichlet(np.ones(4))
            feat_vec = [self.iot_data.get('current_temp', 0) / 100, self.iot_data.get('gravity', 0.38)]
            for props in self.waste_composition.values():
                feat_vec.extend([props['density'], props['strength'] / 250])
            feat_vec = feat_vec[:12]  # 12 dims simplificados
            X.append(feat_vec)
            y_proportions.append(weights)
            y_performance.append(np.random.uniform(0.7, 0.95, 4))
        return np.array(X), np.array(y_proportions), np.array(y_performance)

# 🧠 RexAI 2025 Ligero
class RexAI_2025_Light:
    def __init__(self, input_dim=12):
        self.input_dim = input_dim
        self.model_prop = MLPRegressor(hidden_layer_sizes=(64,), max_iter=100)
        self.model_perf = MLPRegressor(hidden_layer_sizes=(64,), max_iter=100)
        self.scaler = StandardScaler()

    def fit(self, X, y_proportions, y_performance):
        X_scaled = self.scaler.fit_transform(X)
        self.model_prop.fit(X_scaled, y_proportions)
        self.model_perf.fit(X_scaled, y_performance)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model_prop.predict(X_scaled), self.model_perf.predict(X_scaled)

# 🔍 Optimización Ligera
def optimize_rexai(trial):
    hidden_size = trial.suggest_int('hidden_size', 32, 128, step=32)
    return -(hidden_size / 100)

# 📊 Evaluación Simplificada
def evaluate_solution(proportions, performance, mars_data, target_material, total_kg, iot_data):
    target_strength = mars_data.target_materials[target_material]['strength']
    efficiency = performance[0][0]
    rex_score = (efficiency * 0.5 + min(performance[0][1], 1) * 0.5) * 100
    kg_processed = total_kg * efficiency
    kg_saved = kg_processed * 0.95
    return {
        'rex_score': rex_score,
        'kg_processed': kg_processed,
        'kg_saved': kg_saved,
        'efficiency': efficiency * 100
    }

# 🎛️ Interfaz Principal
def main():
    st.markdown('<div class="main-header"><h1>🚀 REX-AI 2025</h1><p>Reciclaje para Marte - SpaceTrash Hack</p></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("🎛️ CONTROLES")
        material = st.selectbox("Material", ["Hábitat", "Herramientas", "Aislamiento", "Electrónica"])
        waste_percent = st.slider("Basura %", 10, 100, 50)
        if st.button("🔄 Actualizar IoT"):
            st.cache_data.clear()
            st.rerun()

    @st.cache_resource
    def load_rexai_model():
        mars_data = MarsRecyclingData()
        X, y_props, y_perf = mars_data.generate_training_data(300)
        study = optuna.create_study(direction='minimize')
        study.optimize(optimize_rexai, n_trials=3)
        model = RexAI_2025_Light(input_dim=12)
        model.fit(X, y_props, y_perf)
        return model, mars_data

    with st.spinner("🚀 Inicializando..."):
        model, mars_data = load_rexai_model()

    if st.button("🪐 Generar Solución"):
        with st.spinner("🔬 Procesando..."):
            iot_data = mars_data.update_iot_data()
            total_kg = (waste_percent / 100) * mars_data.total_waste
            input_features = [iot_data['current_temp'] / 100, iot_data['gravity']]
            for props in mars_data.waste_composition.values():
                input_features.extend([props['density'], props['strength'] / 250])
            input_array = np.array([input_features[:12]])

            proportions, performance = model.predict(input_array)
            evaluation = evaluate_solution(proportions, performance, mars_data, material, total_kg, iot_data)

            st.markdown(f'<div class="metric-box"><h3>Rex-Score</h3><h4>{evaluation["rex_score"]:.1f}%</h4></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-box"><h3>Kg Procesados</h3><h4>{evaluation["kg_processed"]:.0f}kg</h4></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-box"><h3>Kg Ahorrados</h3><h4>{evaluation["kg_saved"]:.0f}kg</h4></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-box"><h3>Eficiencia</h3><h4>{evaluation["efficiency"]:.1f}%</h4></div>', unsafe_allow_html=True)

            st.success(f"✅ Solución generada para {material} | Temp: {iot_data['current_temp']:.1f}°C")

    st.markdown("---")
    st.markdown("<p>🌟 Desarrollado por OneRex Argentina | SpaceTrash Hack Winner</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
