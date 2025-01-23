import json
import requests
import pandas as pd
import time

from google.cloud import bigquery

# ------------------------------------------------------------------------------
# 1. Lectura de la Lista de Juegos desde mlb_game_data.json
# ------------------------------------------------------------------------------
def read_local_game_data(json_file_path):
    """
    Lee el archivo JSON que contiene la lista de juegos (gamePk, fechas, equipos, etc.)
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data

# ------------------------------------------------------------------------------
# 2. GUMBO Data: Estadísticas de Juego
# ------------------------------------------------------------------------------
def fetch_game_gumbo_data(game_pk):
    """
    Endpoint: https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live
    Obtiene el feed GUMBO completo de un juego específico.
    """
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"[ERROR] No se pudo obtener datos GUMBO para gamePk={game_pk}. Code: {resp.status_code}")
        return {}
    return resp.json()

def parse_gumbo_stats(gumbo_data):
    """
    Extrae estadísticas relevantes del feed GUMBO, por ejemplo:
    - runs, hits, errors
    - equipo ganador/perdedor
    - duración del juego
    - (Opcional) boxscore detallado por jugador
    """
    if not gumbo_data or 'liveData' not in gumbo_data:
        return {}
    
    linescore = gumbo_data['liveData'].get('linescore', {})
    teams_line = linescore.get('teams', {})
    
    away_runs, away_hits, away_errors = None, None, None
    home_runs, home_hits, home_errors = None, None, None
    
    if 'away' in teams_line:
        away_runs = teams_line['away'].get('runs')
        away_hits = teams_line['away'].get('hits')
        away_errors = teams_line['away'].get('errors')
    if 'home' in teams_line:
        home_runs = teams_line['home'].get('runs')
        home_hits = teams_line['home'].get('hits')
        home_errors = teams_line['home'].get('errors')
    
    decisions = gumbo_data['liveData'].get('decisions', {})
    winner_team = decisions.get('winner', {}).get('team', {}).get('name')
    loser_team = decisions.get('loser', {}).get('team', {}).get('name')
    
    # Duración del juego (si está disponible)
    gameData = gumbo_data.get('gameData', {})
    gameInfo = gameData.get('gameInfo', {})
    game_duration = None
    if 'duration' in gameInfo:
        duration_dict = gameInfo['duration']
        hours = duration_dict.get('hours', 0)
        minutes = duration_dict.get('minutes', 0)
        game_duration = f"{hours}h {minutes}m"
    
    # Ejemplo de un dict con stats de equipo
    return {
        'away_runs': away_runs,
        'away_hits': away_hits,
        'away_errors': away_errors,
        'home_runs': home_runs,
        'home_hits': home_hits,
        'home_errors': home_errors,
        'winner_team': winner_team,
        'loser_team': loser_team,
        'game_duration': game_duration
    }

# ------------------------------------------------------------------------------
# 3. Información del Boxscore y Estadísticas de Jugadores
#    (Opcional) para obtener stats detalladas de cada jugador en el juego
# ------------------------------------------------------------------------------
def parse_player_stats_from_boxscore(gumbo_data):
    """
    Recorre el boxscore dentro de 'liveData/boxscore/teams' para obtener estadísticas
    por jugador, por ejemplo: hits, at-bats, era, strikeouts, etc.
    Devuelve una lista de dicts con stats de cada jugador (puedes ampliarlo a tu gusto).
    """
    if not gumbo_data or 'liveData' not in gumbo_data:
        return []
    
    boxscore = gumbo_data['liveData'].get('boxscore', {})
    teams_box = boxscore.get('teams', {})
    
    # Retorno final
    players_stats = []
    
    for side in ['away', 'home']:
        team_data = teams_box.get(side, {})
        team_id = team_data.get('team', {}).get('id')
        team_name = team_data.get('team', {}).get('name')
        
        players = team_data.get('players', {})
        for player_key, player_info in players.items():
            person = player_info.get('person', {})
            stats = player_info.get('stats', {})
            
            player_id = person.get('id')
            player_name = person.get('fullName')
            position = player_info.get('position', {}).get('abbreviation')  # Ej: 'P', 'C', '1B', etc.
            
            # Ejemplo de algunas stats de pitcheo/bateo
            # Ajustar según necesites
            batting_stats = stats.get('batting', {})
            pitching_stats = stats.get('pitching', {})
            
            players_stats.append({
                'team_id': team_id,
                'team_name': team_name,
                'player_id': player_id,
                'player_name': player_name,
                'position': position,
                'side': side,  # away/home
                'hits': batting_stats.get('hits'),
                'atBats': batting_stats.get('atBats'),
                'homeRuns': batting_stats.get('homeRuns'),
                'rbi': batting_stats.get('rbi'),
                'era': pitching_stats.get('era'),
                'strikeOuts': pitching_stats.get('strikeOuts')
            })
    
    return players_stats

# ------------------------------------------------------------------------------
# 4. Información (bio) de Jugadores y Stats de Temporada
# ------------------------------------------------------------------------------
def fetch_player_info(player_id):
    """
    Endpoint: https://statsapi.mlb.com/api/v1/people/{player_id}
    Devuelve info biográfica básica (nombre, apellido, fecha nacimiento, altura, peso, etc.)
    """
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"[WARN] No se pudo obtener info para player_id={player_id}. Code: {resp.status_code}")
        return {}
    
    data = resp.json()
    people = data.get('people', [])
    if not people:
        return {}
    
    person = people[0]
    return {
        'player_id': player_id,
        'fullName': person.get('fullName'),
        'birthDate': person.get('birthDate'),
        'height': person.get('height'),
        'weight': person.get('weight'),
        'primaryPosition': person.get('primaryPosition', {}).get('abbreviation'),
        'mlbDebutDate': person.get('mlbDebutDate')
        # Puedes agregar más campos si los necesitas
    }

def fetch_player_season_stats(player_id, season):
    """
    Endpoint: https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=season&season={season}
    Para obtener estadísticas de temporada (AVG, ERA, HR, etc.)
    """
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=season&season={season}"
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"[WARN] No se pudo obtener estadísticas temporada para player_id={player_id}, season={season}")
        return {}
    
    data = resp.json()
    stats_list = data.get('stats', [])
    if not stats_list:
        return {}
    
    splits = stats_list[0].get('splits', [])
    if not splits:
        return {}
    
    # Extraemos la primera (puede haber más splits por equipo, etc.)
    stat = splits[0].get('stat', {})
    return {
        'avg': stat.get('avg'),
        'obp': stat.get('obp'),
        'slg': stat.get('slg'),
        'ops': stat.get('ops'),
        'homeRuns': stat.get('homeRuns'),
        'rbi': stat.get('rbi'),
        'era': stat.get('era'),
        'wins': stat.get('wins'),
        'losses': stat.get('losses'),
        'strikeOuts': stat.get('strikeOuts'),
        'war': stat.get('war')
        # Añade más según necesites
    }

# ------------------------------------------------------------------------------
# 5. Construir DataFrames Combinados
# ------------------------------------------------------------------------------
def build_enriched_dataframe(games_list, fetch_boxscore=True):
    """
    1) Para cada gamePk en games_list, obtiene datos GUMBO (stats de juego).
    2) (Opcional) Si fetch_boxscore=True, parsea estadisticas de cada jugador en el boxscore.
    3) Devuelve 2 DataFrames:
       - df_games: stats a nivel de juego
       - df_boxscore: stats a nivel de jugador (opcional)
    """
    all_games = []
    all_boxscore_rows = []
    
    for i, game in enumerate(games_list, start=1):
        game_pk = game.get('gamePk')
        game_date = game.get('gameDate')
        teams = game.get('teams', {})
        away_team = teams.get('away', None)
        home_team = teams.get('home', None)
        venue = game.get('venue')
        
        # GUMBO para el juego
        gumbo_data = fetch_game_gumbo_data(game_pk)
        game_stats = parse_gumbo_stats(gumbo_data)
        
        # Construir registro a nivel de juego
        row_game = {
            'gamePk': game_pk,
            'gameDate': game_date,
            'away_team': away_team,
            'home_team': home_team,
            'venue': venue,
            **game_stats
        }
        all_games.append(row_game)
        
        # Si deseas recolectar boxscore detallado, entra aquí
        if fetch_boxscore:
            player_stats_list = parse_player_stats_from_boxscore(gumbo_data)
            for pstats in player_stats_list:
                # Añade gamePk, gameDate para enlazarlo con el juego
                pstats['gamePk'] = game_pk
                pstats['gameDate'] = game_date
                all_boxscore_rows.append(pstats)
        
        time.sleep(0.2)  # Para no saturar la API en muchos juegos
        if i % 50 == 0:
            print(f"[INFO] Procesados {i} juegos...")
    
    df_games = pd.DataFrame(all_games)
    df_boxscore = pd.DataFrame(all_boxscore_rows) if fetch_boxscore else pd.DataFrame()
    return df_games, df_boxscore

# ------------------------------------------------------------------------------
# 6. Guardar en CSV y Subir a BigQuery
# ------------------------------------------------------------------------------
def save_to_csv(df, filename):
    df.to_csv(filename, index=False)
    print(f"[OK] CSV guardado: {filename}")

def upload_to_bigquery(df, table_id):
    client = bigquery.Client()
    job = client.load_table_from_dataframe(df, table_id)
    job.result()
    print(f"[OK] Datos subidos a BigQuery: {table_id}")

# ------------------------------------------------------------------------------
# 7. Ejecución Principal
# ------------------------------------------------------------------------------
def main():
    # Ajusta la ruta si mlb_game_data.json está en otra ubicación
    json_file_path = "mlb_game_data.json"
    games_list = read_local_game_data(json_file_path)
    
    if not games_list:
        print("[ERROR] No se encontró/leyó información de mlb_game_data.json.")
        return
    
    print("[INFO] Construyendo DataFrame con estadísticas de juego + boxscore...")
    df_games, df_boxscore = build_enriched_dataframe(games_list, fetch_boxscore=True)
    
    # Ejemplo: guardamos ambos DF
    save_to_csv(df_games, "mlb_games_stats.csv")
    if not df_boxscore.empty:
        save_to_csv(df_boxscore, "mlb_games_boxscore.csv")
    
    # Subir a BigQuery
    # Ajusta a tu proyecto/dataset/tabla
    #table_id_games = "mi-proyecto-mlb.mlb_dataset.games_stats"
    #table_id_boxscore = "mi-proyecto-mlb.mlb_dataset.games_boxscore"
    """
    upload_to_bigquery(df_games, table_id_games)
    if not df_boxscore.empty:
        upload_to_bigquery(df_boxscore, table_id_boxscore)
    """
    print("[INFO] ¡Proceso completado con éxito!")
    print("[INFO] Ahora podrás combinar estos datos con la información de jugadores (edad, WAR histórico, etc.).")

if __name__ == "__main__":
    main()
