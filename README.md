# REX-Mars — Supra-Reciclaje en Marte (Lab-ready MVP)

**Objetivo:** ayudar a ingenieros en un hábitat marciano a **convertir basura inorgánica** (plásticos, textiles, metales, compuestos y *Foam Packaging*) en **productos útiles**, eligiendo **recetas** y **procesos** con **bajo consumo** de energía/agua/tiempo de tripulación y **cumplimiento** de reglas NASA (operación sellada, sin incineración/airlock, agua en lazo cerrado).

## ¿Qué hace?
- **Modelo IA** (PyTorch: Transformer+GRU) que sugiere **proporciones** y **performance** de mezcla.
- **Selector de proceso** (Polymers / Metallics / Hybrid) y **KPIs**: kWh/kg, L/kg, min/kg.
- **Módulo laboratorio**:
  - **Samples**: ingesta de espectros (CSV, TXT, JCAMP-DX, mzML).
  - **Spectra Lab**: baseline, SNV, Savitzky–Golay, visualizador.
  - **Predictions**: estimación de **familias** y **compatibilidad** entre muestras (score + intervalo).
  - **Optimizer**: búsqueda Top-K de recetas (grid/BO) maximizando **REX-Score** con restricciones.
- **Seguridad**: checklist **NASA-compliant**.
- **Historial**: guarda corridas para trazabilidad.

## REX-Score (cómo evaluamos)
Puntaje 0–100 que combina:
- **Desvío de vertedero** (35%),
- **Energía (kWh/kg)**, **Agua (L/kg)**, **Tiempo tripulación (min/kg)** (25%, 20%, 20%),
- Penalización por **riesgo de microplásticos**.
> Las constantes de proceso vienen de perfiles **Polymers/Metallics** (póster MSFC) y se ponderan en modo **Hybrid**.

## Cumplimiento
- Operación sellada (cero humo; captura de finos/HEPA).
- Agua en lazo cerrado.
- **Prohibido**: incineración / expulsión por airlock.
- Uso de **MGS-1** como filler (mineralogía documentada).

## Estructura de carpetas
