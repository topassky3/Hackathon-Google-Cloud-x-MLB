import os
import numpy as np
import pandas as pd
from google.cloud import bigquery
import google.api_core.exceptions

# Librería para ML
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Librería para visualización
import matplotlib.pyplot as plt

# -------------------------
# 1. Importar google-generativeai
# -------------------------
import google.generativeai as genai

# -------------------------------------------------------------------
# A) CONFIGURACIÓN DE CREDENCIALES
# -------------------------------------------------------------------
def set_credentials(credentials_path):
    """
    Configura la variable de entorno GOOGLE_APPLICATION_CREDENTIALS para usar el archivo JSON de credenciales.
    """
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    print(f"[INFO] Credenciales configuradas desde: {credentials_path}")


def configure_gemini(api_key):
    """
    Configura la API Key para Google Generative AI (Gemini).
    """
    genai.configure(api_key=api_key)
    print("[INFO] Gemini configurado con la API Key provista.")


# -------------------------------------------------------------------
# B) CREAR DATASET EN BIGQUERY (SI NO EXISTE)
# -------------------------------------------------------------------
def create_dataset_if_not_exists(dataset_id, location="US"):
    """
    Verifica si un conjunto de datos existe en BigQuery, y lo crea si no existe.
    Args:
        dataset_id (str): ID del dataset en formato "project_id.dataset_id".
        location (str): Ubicación del dataset (por defecto "US").
    """
    client = bigquery.Client()
    project_id, dataset_name = dataset_id.split('.')
    
    dataset_ref = bigquery.DatasetReference(project_id, dataset_name)
    dataset = bigquery.Dataset(dataset_ref)

    try:
        # Verifica si el conjunto de datos ya existe
        client.get_dataset(dataset_ref)
        print(f"[INFO] El conjunto de datos '{dataset_name}' ya existe en el proyecto '{project_id}'.")
    except google.api_core.exceptions.NotFound:
        # Si no existe, crea el conjunto de datos
        print(f"[INFO] Creando el conjunto de datos '{dataset_name}' en el proyecto '{project_id}'...")
        dataset.location = location
        client.create_dataset(dataset)
        print(f"[OK] Conjunto de datos '{dataset_name}' creado exitosamente.")


# -------------------------------------------------------------------
# C) LECTURA DE DATOS DESDE BIGQUERY
# -------------------------------------------------------------------
def fetch_data_from_bigquery(project_id, dataset_id, table_name):
    """
    Consulta datos de la tabla indicada en BigQuery y los devuelve como un DataFrame de pandas.
    """
    client = bigquery.Client(project=project_id)
    query = f"""
        SELECT *
        FROM `{project_id}.{dataset_id}.{table_name}`
    """
    df = client.query(query).to_dataframe()
    print(f"[INFO] Datos extraídos de '{table_name}': {df.shape[0]} filas, {df.shape[1]} columnas.")
    return df


# -------------------------------------------------------------------
# D) COMBINACIÓN Y FEATURE ENGINEERING
# -------------------------------------------------------------------
def prepare_dataset(df_box, df_stats):
    """
    Combina datos de boxscore (df_box) y de stats (df_stats) a nivel de juego.
    Crea la métrica 'current_avg' y la variable objetivo 'next_avg'.

    - current_avg: hits / atBats para cada jugador en el juego.
    - next_avg: el current_avg del siguiente juego del mismo jugador.
    """
    # 1. Unir ambas tablas por 'gamePk'
    df_merged = pd.merge(
        df_box,
        df_stats[['gamePk','away_runs','home_runs','away_hits','home_hits']],
        on='gamePk',
        how='left'
    )

    # 2. Calcular 'current_avg' de cada jugador en ese juego (hits / atBats)
    df_merged['current_avg'] = df_merged.apply(
        lambda row: row['hits'] / row['atBats'] if row['atBats'] else 0,
        axis=1
    )

    # 3. Convertir gameDate a tipo fecha
    df_merged['gameDate'] = pd.to_datetime(df_merged['gameDate'], errors='coerce')

    # 4. year_of_game (para comparar jugadores del mismo año)
    df_merged['year_of_game'] = df_merged['gameDate'].dt.year

    # 5. Ordenar datos por jugador + fecha
    df_merged.sort_values(by=['player_id','gameDate'], inplace=True)

    # 6. Crear la columna 'next_avg' (target), que es el current_avg del siguiente juego
    df_merged['next_avg'] = df_merged.groupby('player_id')['current_avg'].shift(-1)

    # 7. Remover filas donde next_avg sea NaN (jugadores sin juego posterior)
    df_merged.dropna(subset=['next_avg'], inplace=True)

    print(f"[INFO] DataFrame combinado: {df_merged.shape[0]} registros después de feature engineering.")
    return df_merged


# -------------------------------------------------------------------
# E) CREACIÓN DE FEATURES (X) Y TARGET (y)
# -------------------------------------------------------------------
def create_features_and_target(df):
    """
    Selecciona las columnas que servirán como features (X) y define la variable target (y).
    En este caso, la variable objetivo es 'next_avg'.
    """
    # Ejemplo de features. Puedes añadir más (ERA, side, position, etc.)
    features = [
        'current_avg',
        'home_runs',  # carreras del equipo local
        'away_runs',  # carreras del equipo visitante
        'hits',       
        'homeRuns',   
        'rbi'         
    ]
    
    # Llenar con 0 los valores faltantes
    df[features] = df[features].fillna(0)

    # Definir X e y
    X = df[features].copy()
    y = df['next_avg'].copy()
    return X, y, df


# -------------------------------------------------------------------
# F) ENTRENAR EL MODELO DE ML (RandomForest)
# -------------------------------------------------------------------
def train_model(X, y):
    """
    Entrena un modelo de regresión (RandomForest) para predecir la métrica 'next_avg'.
    Retorna el modelo entrenado y la métrica de error (MAE).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    print("[INFO] Entrenando modelo RandomForest...")
    model.fit(X_train, y_train)

    # Evaluar con Mean Absolute Error (MAE)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"[RESULT] MAE en conjunto de test: {mae:.4f}")
    return model, mae, X_test, y_test


# -------------------------------------------------------------------
# G) COMPARAR JUGADOR CON HISTÓRICOS
# -------------------------------------------------------------------
def compare_with_historical_players(df, prospect_id, prospect_year, n_similar=5):
    """
    Compara un jugador (prospecto) con otros jugadores que jugaron en el mismo año.
    Utiliza la distancia euclidiana en las mismas features del modelo.
    """
    same_year = df[df['year_of_game'] == prospect_year].copy()
    if same_year.empty:
        print(f"[WARN] No se encontraron datos para el año {prospect_year}.")
        return pd.DataFrame()

    features = ['current_avg', 'home_runs', 'away_runs', 'hits', 'homeRuns', 'rbi']

    # Filtrar filas del prospecto en ese año
    prospect_rows = same_year[same_year['player_id'] == prospect_id]
    if prospect_rows.empty:
        print(f"[WARN] El prospecto {prospect_id} no tiene registros en {prospect_year}.")
        return pd.DataFrame()

    # Promedio de estadísticas del prospecto
    prospect_data = prospect_rows[features].mean().values

    def euclidean_dist(row):
        return np.linalg.norm(row.values - prospect_data)

    # Calculamos la distancia para todos en el mismo año
    same_year['similarity'] = same_year[features].apply(euclidean_dist, axis=1)

    # Excluir prospecto
    same_year = same_year[same_year['player_id'] != prospect_id]

    # Ordenar por similitud y retornar top n_similar
    similar_players = same_year.sort_values(by='similarity').head(n_similar)
    return similar_players


# -------------------------------------------------------------------
# H) VISUALIZACIÓN DE PROYECCIONES (ESCENARIOS)
# -------------------------------------------------------------------
def plot_projection_scenarios(model, X_test, y_test, n_points=30):
    """
    Muestra una curva de proyección (predicción) y escenarios
    (optimista, promedio, pesimista) para un subconjunto de datos.
    """
    # Subconjunto de X_test
    X_subset = X_test.iloc[:n_points].copy()
    y_real = y_test.iloc[:n_points].values  # valores reales
    y_pred = model.predict(X_subset)        # predicción base

    # Definimos un "delta" para optimista/pesimista
    delta = 0.05
    y_optimista = y_pred + delta
    y_pesimista = y_pred - delta

    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(range(n_points), y_real, label='Real (next_avg)', marker='o')
    plt.plot(range(n_points), y_pred, label='Escenario Promedio (predicción)', marker='o')
    plt.plot(range(n_points), y_optimista, label='Escenario Optimista', linestyle='--')
    plt.plot(range(n_points), y_pesimista, label='Escenario Pesimista', linestyle='--')

    plt.title("Curvas de Proyección y Escenarios (Optimista, Promedio, Pesimista)")
    plt.xlabel("Índice de juego (subset de test)")
    plt.ylabel("Valor de next_avg (promedio de bateo)")
    plt.legend()
    plt.grid(True)
    plt.show()


# -------------------------------------------------------------------
# I) GENERAR TEXTO CON GEMINI
# -------------------------------------------------------------------
def generate_gemini_summary(
    model_name,
    mae,
    prospect_id,
    prospect_year,
    similar_players,
    max_tokens=1024
):
    generation_config = {
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": max_tokens,
        "response_mime_type": "text/plain",
    }

    gemini_model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        system_instruction=(
            "Eres un experto en análisis de béisbol. Tu tarea es "
            "describir los resultados de un modelo de predicción y "
            "comparar jugadores."
        )
    )

    if not similar_players.empty and 'player_name' in similar_players.columns:
        list_similar = similar_players['player_name'].unique().tolist()
    else:
        list_similar = similar_players['player_id'].unique().tolist()

    prompt = f"""
    Hemos entrenado un modelo RandomForestRegressor para predecir el 'next_avg' de un jugador en MLB.
    El MAE obtenido es de {mae:.4f}.
    El prospecto analizado es el jugador con ID {prospect_id} durante el año {prospect_year}.
    Encontramos que los jugadores más similares son: {list_similar}.

    Redacta un breve resumen en español explicando:
    1) ¿Cómo interpretamos el MAE y la métrica 'next_avg'?
    2) ¿Por qué {prospect_id} se parece a esos jugadores similares?
    3) ¿Qué proyecciones o consideraciones harías sobre su potencial?
    """

    # --- La parte importante: pasar el prompt como argumento posicional ---
    response = gemini_model.generate_content(prompt)

     # Retornar el contenido correcto:
    if response.candidates:
        return response.candidates[0].content
    else:
        return "No hay contenido en la respuesta de Gemini."

# -------------------------------------------------------------------
# J) FUNCIÓN PRINCIPAL
# -------------------------------------------------------------------
def main():
    """
    1. Configura credenciales y dataset en BigQuery.
    2. Lee las tablas 'games_boxscore' y 'games_stats' desde BigQuery.
    3. Combina datos y realiza feature engineering.
    4. Entrena un modelo de ML (RandomForest) para predecir 'next_avg'.
    5. Muestra curvas de proyección (optimista, promedio, pesimista).
    6. Compara un prospecto con jugadores históricos del mismo año.
    7. Genera resumen narrativo con Gemini.
    """
    # 1. Configurar credenciales
    credentials_path = "credenciales.json"   # Ajusta la ruta a tu archivo JSON
    set_credentials(credentials_path)

    # 2. Configurar la API key de Gemini (reemplaza con la tuya)
    configure_gemini("AIzaSyCW-_8k6OacvjTUSzDiKtijy6W2y1ZhaYk")

    # 3. Verificar o crear dataset en BigQuery
    project_id = "maps-3d-439423"
    dataset_id = "mlb_dataset"
    create_dataset_if_not_exists(f"{project_id}.{dataset_id}")

    # 4. Cargar datos desde BigQuery
    table_boxscore = "games_boxscore"
    table_stats = "games_stats"

    print("[INFO] Leyendo datos de 'games_boxscore' desde BigQuery...")
    df_box = fetch_data_from_bigquery(project_id, dataset_id, table_boxscore)

    print("[INFO] Leyendo datos de 'games_stats' desde BigQuery...")
    df_stats = fetch_data_from_bigquery(project_id, dataset_id, table_stats)

    # 5. Preparar dataset (join + feature engineering)
    df_combined = prepare_dataset(df_box, df_stats)

    # 6. Crear features (X) y target (y)
    X, y, df_final = create_features_and_target(df_combined)
    print(f"[INFO] Tamaño final para entrenamiento: {X.shape[0]} filas, {X.shape[1]} features.")

    # 7. Entrenar el modelo
    model, mae, X_test, y_test = train_model(X, y)

    # 8. Visualización de escenarios
    plot_projection_scenarios(model, X_test, y_test, n_points=30)

    # 9. Comparar prospecto con jugadores históricos
    prospect_id = 114879   # Ejemplo
    prospect_year = 1901
    n_similar = 5
    similar_players = compare_with_historical_players(df_final, prospect_id, prospect_year, n_similar)

    print("==============================================================")
    if not similar_players.empty:
        print(f"[INFO] Jugadores más parecidos al prospecto {prospect_id} en el año {prospect_year}:")
        cols_to_show = ['player_id','player_name','year_of_game','current_avg','similarity']
        print(similar_players[cols_to_show].reset_index(drop=True))
    else:
        print("[INFO] No se encontraron comparaciones para este prospecto y año.")
    print("==============================================================")

    # 10. (Opcional) Generar un resumen narrativo con Gemini
    #     Ajusta el 'model_name' según la versión que tengas acceso
    gemini_model_name = "gemini-2.0-flash-exp"  # <--- Usa uno de los que tienes disponibles
    summary = generate_gemini_summary(
        model_name=gemini_model_name,
        mae=mae,
        prospect_id=prospect_id,
        prospect_year=prospect_year,
        similar_players=similar_players,
        max_tokens=512
    )
    print("\n[Gemini Summary]\n", summary)
    print("==============================================================")

    print("[INFO] Proceso completo. Se ha entrenado el modelo, mostrado la curva de proyección, realizado la comparación y generado un texto de Gemini.")


# -------------------------------------------------------------------
# K) EJECUTAR SCRIPT
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
