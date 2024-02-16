from dotenv.main import load_dotenv
import requests
from datetime import datetime
import os

class OddsAPICall:
    load_dotenv()
    def __init__(self, api_key):
        self.api_key = os.getenv("ODDS_API_KEY")
        
    def fetch_odds(self, sport, regions, markets, bookmakers):
        url = "https://api.the-odds-api.com/v4/sports/{}/odds/".format(sport)
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "bookmakers": bookmakers
        }
        response = requests.get(url, params=params)
        return response.json() if response.status_code == 200 else None

    def parse_games(self, odds_data):
        parsed_games = []
        for game in odds_data:
            if not game['bookmakers']:
                continue
            game_info = {
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "commence_time": datetime.strptime(game["commence_time"], "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S"),
                "odds": []
            }
            for market in game['bookmakers'][0]['markets']:
                if market['key'] == 'totals':
                    for outcome in market['outcomes']:
                        game_info["odds"].append({
                            "name": outcome['name'],
                            "price": outcome['price'],
                            "point": outcome['point']
                        })
            parsed_games.append(game_info)
        return parsed_games
