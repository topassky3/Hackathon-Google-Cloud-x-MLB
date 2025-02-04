import pandas as pd
from google.cloud import bigquery
import google.api_core.exceptions  # Asegúrate de importar las excepciones
import os

# -------------------------------------------------------------------
# Configuración inicial
# -------------------------------------------------------------------
def set_credentials(credentials_path):
    """
    Configura la variable de entorno GOOGLE_APPLICATION_CREDENTIALS para usar el archivo JSON de credenciales.
    
    Args:
        credentials_path (str): Ruta al archivo JSON de credenciales.
    """
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    print(f"[INFO] Credenciales configuradas desde: {credentials_path}")

# -------------------------------------------------------------------
# Crear conjunto de datos si no existe
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
    
    # Crear la referencia del dataset correctamente
    dataset_ref = bigquery.DatasetReference(project_id, dataset_name)
    dataset = bigquery.Dataset(dataset_ref)  # Crear objeto Dataset

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
# Función para subir un DataFrame a BigQuery
# -------------------------------------------------------------------
def upload_to_bigquery(df, table_id):
    """
    Sube un DataFrame de pandas a BigQuery.
    
    Args:
        df (pd.DataFrame): DataFrame a subir.
        table_id (str): ID de la tabla de BigQuery (formato: project.dataset.table).
    """
    client = bigquery.Client()
    
    # Configuración del job de carga
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,  # Sobrescribe los datos existentes
        autodetect=True  # Detecta automáticamente el esquema de las columnas
    )
    
    # Subir el DataFrame
    print(f"[INFO] Subiendo datos a BigQuery en la tabla: {table_id}...")
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()  # Espera a que el job termine
    
    print(f"[OK] Datos subidos exitosamente a: {table_id}")

# -------------------------------------------------------------------
# Función principal
# -------------------------------------------------------------------
def main():
    # Ruta al archivo de credenciales JSON
    credentials_path = "credenciales.json"  # Cambia esto por la ruta a tu archivo JSON
    set_credentials(credentials_path)
    
    # Archivos CSV
    boxscore_file = "mlb_games_boxscore.csv"  # Archivo boxscore
    stats_file = "mlb_games_stats.csv"  # Archivo stats
    
    # IDs de las tablas y dataset en BigQuery
    project_id = "#id_project"  # El ID del proyecto en las credenciales
    dataset_id = "id_dataset"  # Cambia esto al dataset que configuraste en BigQuery
    table_id_boxscore = f"{project_id}.{dataset_id}.games_boxscore"
    table_id_stats = f"{project_id}.{dataset_id}.games_stats"
    
    # Verificar y crear el dataset si no existe
    create_dataset_if_not_exists(f"{project_id}.{dataset_id}")
    
    # Leer los CSV como DataFrames
    print("[INFO] Leyendo archivos CSV...")
    df_boxscore = pd.read_csv(boxscore_file)
    df_stats = pd.read_csv(stats_file)
    
    # Subir a BigQuery
    print("[INFO] Subiendo archivo 'mlb_games_boxscore.csv' a BigQuery...")
    upload_to_bigquery(df_boxscore, table_id_boxscore)
    
    print("[INFO] Subiendo archivo 'mlb_games_stats.csv' a BigQuery...")
    upload_to_bigquery(df_stats, table_id_stats)
    
    print("[INFO] ¡Proceso completado con éxito!")

# -------------------------------------------------------------------
# Ejecutar el script
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
