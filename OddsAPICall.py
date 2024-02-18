from dotenv.main import load_dotenv
import requests
from datetime import datetime
import os
import pytz

class OddsAPICall:
    load_dotenv()
    def __init__(self, api_key):
        self.api_key = os.getenv("ODDS_API_KEY")
        
    def fetch_odds(self, sport, regions, markets, primary_bookmaker, fallback_bookmaker=None):
        url = "https://api.the-odds-api.com/v4/sports/{}/odds/".format(sport)
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "bookmakers": primary_bookmaker
        }
        response = requests.get(url, params=params)
        odds_data = response.json() if response.status_code == 200 else None

        if odds_data and all(not game['bookmakers'] for game in odds_data) and fallback_bookmaker:
            print(f"No data from primary bookmaker {primary_bookmaker}. Trying fallback bookmaker {fallback_bookmaker}.")
            params["bookmakers"] = fallback_bookmaker
            response = requests.get(url, params=params)
            odds_data = response.json() if response.status_code == 200 else None

        return odds_data

    def parse_games(self, odds_data):
        parsed_games = []
        utc_zone = pytz.utc
        eastern_zone = pytz.timezone('US/Eastern')
        for game in odds_data:
            if not game['bookmakers']:
                continue
            bookmaker_name = game['bookmakers'][0]['title']
            utc_time = datetime.strptime(game["commence_time"], "%Y-%m-%dT%H:%M:%SZ")
            eastern_time = utc_time.replace(tzinfo=utc_zone).astimezone(eastern_zone)
            game_info = {
                "bookmaker": bookmaker_name,
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "commence_time": eastern_time.strftime("%m-%d-%Y %I:%M:%S %p %Z"),
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
