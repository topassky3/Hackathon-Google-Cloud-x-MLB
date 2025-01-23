import requests
import json
from datetime import datetime

def fetch_mlb_schedule(start_year, end_year, sport_id='1', game_type='R'):
    """Fetch the MLB regular season schedule for a range of years."""
    all_games = []
    for year in range(start_year, end_year + 1):
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId={sport_id}&season={year}&gameType={game_type}"
        response = requests.get(url)
        if response.status_code == 200:
            schedule = response.json()
            games = schedule.get('dates', [])
            for date in games:
                for game in date.get('games', []):
                    game_data = {
                        'gamePk': game['gamePk'],
                        'gameDate': game['officialDate'],
                        'teams': {
                            'away': game['teams']['away']['team']['name'],
                            'home': game['teams']['home']['team']['name']
                        },
                        'venue': game['venue']['name'] if 'venue' in game else 'Unknown'
                    }
                    all_games.append(game_data)
    return all_games

def save_to_json(data, filename):
    """Save data to a JSON file."""
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data successfully saved to {filename}")

# Set the range of years
start_year = 1901  # Adjust as needed, based on available data
end_year = datetime.now().year  # Or set to the last available year you want to include

# Fetch and save game data
game_data = fetch_mlb_schedule(start_year, end_year)
save_to_json(game_data, 'mlb_game_data.json')
