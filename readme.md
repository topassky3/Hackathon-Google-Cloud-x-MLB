# Hackathon MLB - Predicción de Prospectos ⚾📊

![MLB Logo](https://upload.wikimedia.org/wikipedia/commons/a/a6/Major_League_Baseball_logo.svg)
![Google Cloud](https://logowik.com/content/uploads/images/google-cloud.jpg)

Solución para el **Desafío #5: Prospect Prediction** de la Hackathon Google Cloud x MLB™

## 📋 Tabla de Contenidos
- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Arquitectura de la Solución](#-arquitectura-de-la-solución)
- [Flujo de Trabajo](#-flujo-de-trabajo)
- [Instalación y Uso](#-instalación-y-uso)
- [Resultados Clave](#-resultados-clave)
- [Mejoras Futuras](#-mejoras-futuras)
- [Licencia](#-licencia)

## 🚀 Descripción del Proyecto
Pipeline completo para predecir el potencial de prospectos de béisbol utilizando:

```python
# Core Components
def main_components():
    return [
        "📂 Extracción de datos históricos (1901-Presente)",
        "🔍 Enriquecimiento con GUMBO Feed",
        "☁️ Almacenamiento en BigQuery", 
        "🤖 Modelo RandomForest (MAE: 0.1304)",
        "📝 Generación de informes con Gemini",
        "🕰️ Sistema de comparación histórica"
    ]

## 📐 Arquitectura de la Solución

graph LR
    A[API MLB] --> B[Extractor de Datos]
    B --> C{{GUMBO Feed}}
    C --> D[BigQuery]
    D --> E[Feature Engineering]
    E --> F[RandomForest]
    F --> G{{Gemini}}
    F --> H[Visualizador]
    G --> I[Reporte Predictivo]


