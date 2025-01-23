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
```
## 🔄 Flujo de Trabajo

### 1. Generación de GamePKs (Identificadores Únicos de Juegos)
```python fetch_game_data.py```

# Output: mlb_game_data.json con lista de 218,743 gamePks

### 2. Pipeline Principal 
```python pipeline_mlb.py```

Resultados:

Archivo	Registros	Columnas	Descripción
mlb_games_stats.csv	1,111	14	Estadísticas por juego
mlb_games_boxscore.csv	51,151	11	Stats individuales de jugadores

### 3. Carga a BigQuery
```python upload_bigquery.py```
Requisito: Archivo credenciales.json válido en Google Cloud.

### 4. Entrenamiento del Modelo & Reportes
Script: `train_model.py**
Requisito: Archivo credenciales.json válido en Google Cloud.
Clave de gemini: <API KEY>

Salidas:

modelo_prospectos.pkl: Modelo serializado

reporte_gemini.md: Análisis interpretativo generado por IA

curvas_proyeccion.png: Visualización de escenarios
