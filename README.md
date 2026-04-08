# SmartQC AI – QC Anomaly Detection with Isolation Forest

**SmartQC AI** es un prototipo funcional orientado a la detección proactiva de anomalías en procesos de control de calidad, integrando reglas estadísticas de referencia, aprendizaje automático con **Isolation Forest** y una interfaz interactiva desarrollada en **Streamlit**.

Este repositorio reúne tanto la parte analítica del proyecto como el prototipo demostrativo del sistema.

---

## Descripción general

El proyecto parte de la necesidad de fortalecer el monitoreo de control de calidad mediante un enfoque complementario al uso de reglas estadísticas tradicionales. Para ello, se desarrolló una solución que permite:

- analizar series temporales de resultados de control
- comparar alertas estadísticas con anomalías detectadas por IA
- generar datasets sintéticos para validación
- simular una consulta a LIS (Sistema de Información de Laboratorio)
- visualizar hallazgos en un prototipo funcional con enfoque de producto

---

## Funcionalidades del prototipo

La aplicación **SmartQC AI** permite:

- usar un **dataset de ejemplo**
- **subir archivos CSV**
- **generar datasets sintéticos**
- ejecutar una **simulación de consulta a LIS**
- detectar anomalías mediante **Isolation Forest**
- visualizar resultados con media y límites de control
- comparar eventos detectados por IA con reglas estadísticas base
- generar un **resumen ejecutivo automático**

---

## Estructura del repositorio

```text
qc-anomaly-detection-isolation-forest/
├── app/                  # Prototipo funcional en Streamlit
│   ├── app.py
│   └── assets/
│       ├── smartqc_logo.png
│       └── unir_logo_white.png
├── notebooks/            # Notebooks de análisis, modelado y validación
├── data/
│   └── sample/           # Datasets de ejemplo o sintéticos
├── docs/
│   ├── demo/             # Materiales de apoyo para la demostración
│   └── screenshots/      # Capturas del prototipo
├── .streamlit/
│   └── config.toml       # Configuración visual de Streamlit
├── requirements.txt      # Dependencias del proyecto
├── .gitignore
└── README.md
