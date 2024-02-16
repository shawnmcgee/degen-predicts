from dotenv import load_dotenv
from jinja2.bccache import Bucket
import pandas as pd
import xgboost as xgb
import pickle
import datetime
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
import requests
from bs4 import BeautifulSoup
from datetime import timedelta
import numpy as np
import os
from tqdm import tqdm
import boto3
from urllib.parse import urlparse
import io

def fetch_nhl_data(base_url, params):
    results = []
    start = 0
    limit = params['limit']
    total_records = float('inf')  # Initial placeholder

    while start < total_records:
        params['start'] = start
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            page_results = data['data']
            results.extend(page_results)
            
            if total_records == float('inf'):
                total_records = data['total']
            start += limit
            tqdm.write(f"Fetched {len(page_results)} records. Total so far: {len(results)}")
        else:
            print(f"Failed to retrieve data: {response.status_code}")
            break

    # Return results as a DataFrame
    return pd.DataFrame(results)

def fetch_nhl_schedule(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'lxml')
        table = soup.find('table', {'id': 'games'})  # Adjust if needed

        headers = ['Date', 'Visitor', 'G', 'Home', 'G.1', 'SO', 'Att.', 'LOG', 'Notes']
        data = []
        rows = table.find_all('tr')[1:]  # Skipping header row
        for row in rows:
            date_cell = row.find('th').get_text(strip=True)
            cells = row.find_all('td')
            row_data = [date_cell] + [cell.get_text(strip=True) for cell in cells]
            if len(row_data) == 7:  # Handling missing notes
                row_data.append('')
            data.append(row_data)

        return pd.DataFrame(data, columns=headers)
    else:
        print(f"Failed to retrieve schedule: {response.status_code}")
        return pd.DataFrame()  # Return an empty DataFrame in case of failure


skater_params = {
    "isAggregate": "false",
    "isGame": "false",
    "sort": '[{"property":"points","direction":"DESC"},{"property":"goals","direction":"DESC"},{"property":"assists","direction":"DESC"},{"property":"playerId","direction":"ASC"}]',
    "start": 0,
    "limit": 100,  
    "factCayenneExp": "gamesPlayed>=1",
    "cayenneExp": "gameTypeId=2 and seasonId<=20232024 and seasonId>=20232024"
}

goalie_params = {
    "isAggregate": "false",
    "isGame": "false",
    "sort": '[{"property":"wins","direction":"DESC"},{"property":"savePct","direction":"DESC"},{"property":"playerId","direction":"ASC"}]',
    "start": 0,
    "limit": 100,
    "factCayenneExp": "gamesPlayed>=1",
    "cayenneExp": "gameTypeId=2 and seasonId<=20232024 and seasonId>=20232024"
}


team_params = {
    "isAggregate": "false",
    "isGame": "false",
    "sort": '[{"property":"points","direction":"DESC"},{"property":"wins","direction":"DESC"},{"property":"teamId","direction":"ASC"}]',
    "start": 0,
    "limit": 50,
    "factCayenneExp": "gamesPlayed>=1",
    "cayenneExp": "gameTypeId=2 and seasonId<=20232024 and seasonId>=20232024"
}

# Base API endpoint URL
base_skater_url = "https://api.nhle.com/stats/rest/en/skater/summary"
base_goalie_url = "https://api.nhle.com/stats/rest/en/goalie/summary"
base_teams_url = "https://api.nhle.com/stats/rest/en/team/summary"
nhl_schedule_url = "https://www.hockey-reference.com/leagues/NHL_2024_games.html"



team_name_mapping = {
    "ANA": "Anaheim Ducks",
    "ARI": "Arizona Coyotes",
    "BOS": "Boston Bruins",
    "BUF": "Buffalo Sabres",
    "CGY": "Calgary Flames",
    "CAR": "Carolina Hurricanes",
    "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche",
    "CBJ": "Columbus Blue Jackets",
    "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings",
    "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings",
    "MIN": "Minnesota Wild",
    "MTL": "Montreal Canadiens",
    "NSH": "Nashville Predators",
    "NJD": "New Jersey Devils",
    "NYI": "New York Islanders",
    "NYR": "New York Rangers",
    "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins",
    "SJS": "San Jose Sharks",
    "SEA": "Seattle Kraken",
    "STL": "St. Louis Blues",
    "TBL": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs",
    "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights",
    "WPG": "Winnipeg Jets",
    "WSH": "Washington Capitals"
}


required_columns = [
    'Home_Team', 'Visitor_Team', 'Home_GF/GP', 'Visitor_GF/GP', 
    'Home_GA/GP', 'Visitor_GA/GP', 'Home_PP%', 'Visitor_PP%', 
    'Home_PK%', 'Visitor_PK%', 'Home_FOW%_y', 'Visitor_FOW%_y'
]



# Function to apply team name mapping
def map_team_names(df, column_name):
    if column_name in df.columns:
        df[column_name] = df[column_name].map(team_name_mapping).fillna(df[column_name])
    else:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    return df



def prepare_nhl_data(goalie_stats_df, nhl_schedule_df, skater_stats_df, teams_stats_df):
    # Instead of loading CSVs, use the provided DataFrames directly
    nhl_schedule_df['Date'] = pd.to_datetime(nhl_schedule_df['Date'])
    merged_home = pd.merge(nhl_schedule_df, teams_stats_df, how='left', left_on='Home', right_on='teamFullName')
    merged_full = pd.merge(merged_home, teams_stats_df, how='left', left_on='Visitor', right_on='teamFullName', suffixes=('_Home', '_Visitor'))
    columns_to_drop = ['Unnamed: 5', 'Home_Att.', 'Home_LOG', 'Home_Notes', 'seasonId']
    merged_full.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    merged_full.fillna(merged_full.median(numeric_only=True), inplace=True)

    # Map team names in the schedule to full names as in teams_df
    nhl_schedule_df['Home'] = nhl_schedule_df['Home'].map(team_name_mapping).fillna(nhl_schedule_df['Home'])
    nhl_schedule_df['Visitor'] = nhl_schedule_df['Visitor'].map(team_name_mapping).fillna(nhl_schedule_df['Visitor'])
    
    # Merge team stats into the schedule DataFrame for both home and visitor teams
    merged_df = nhl_schedule_df.merge(teams_stats_df, left_on='Home', right_on='teamFullName', how='left', suffixes=('', '_Home'))
    merged_df = merged_df.merge(teams_stats_df, left_on='Visitor', right_on='teamFullName', how='left', suffixes=('', '_Visitor'))

    # Drop unnecessary columns from the merge operation
    columns_to_drop = ['teamFullName', 'teamFullName_Visitor']
    merged_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Calculate new features like differences and special teams index
    merged_df['GF/GP_Diff'] = merged_df['goalsForPerGame'] - merged_df['goalsForPerGame_Visitor']
    merged_df['GA/GP_Diff'] = merged_df['goalsAgainstPerGame'] - merged_df['goalsAgainstPerGame_Visitor']
    merged_df['Home_Special_Teams_Index'] = merged_df['powerPlayPct'] + merged_df['penaltyKillPct']
    merged_df['Visitor_Special_Teams_Index'] = merged_df['powerPlayPct_Visitor'] + merged_df['penaltyKillPct_Visitor']

    return merged_df
    # Add additional calculations for features like recent form, head-to-head stats, team fatigue, etc., based on your initial script  

# Fetch and prepare the dataset
goalie_data = fetch_nhl_data(base_goalie_url, goalie_params)
schedule_data = fetch_nhl_schedule(nhl_schedule_url)
skater_data = fetch_nhl_data(base_skater_url, skater_params)
teams_data = fetch_nhl_data(base_teams_url, team_params)
data = prepare_nhl_data(goalie_data, schedule_data, skater_data, teams_data)

print(data.columns)
# Define the features to be used and the target variable
features = [
    'Home_GF/GP', 'Visitor_GF/GP', 'Home_GA/GP', 'Visitor_GA/GP', 
    'Home_PP%', 'Visitor_PP%', 'Home_PK%', 'Visitor_PK%', 'Home_FOW%_y', 
    'Visitor_FOW%_y', 'GF/GP_Diff', 'GA/GP_Diff', 'Home_Special_Teams_Index', 
    'Visitor_Special_Teams_Index', 'Home_Recent_Form_Goals_Scored', 
    'Home_Recent_Form_Goals_Conceded', 'Visitor_Recent_Form_Goals_Scored', 
    'Visitor_Recent_Form_Goals_Conceded', 'H2H_Wins_Home', 'H2H_Wins_Visitor', 
    'Home_Team_Fatigue', 'Visitor_Team_Fatigue', 'Home_PP%_Avg', 'Home_PK%_Avg', 
    'Visitor_PP%_Avg', 'Visitor_PK%_Avg'
]

for col in required_columns:
    if col not in schedule_data.columns:
        schedule_data[col] = 0

# Applying the mapping to the datasets
season_results_df = map_team_names(schedule_data, 'Home')
season_results_df = map_team_names(schedule_data, 'Visitor')
skaters_df = map_team_names(skater_data, 'teamAbbrevs')
goalies_df = map_team_names(goalie_data, 'teamAbbrevs')
print(teams_data.columns)
print(skaters_df.columns)
print(goalies_df.columns)

teams_rename_map = {
    'goalsForPerGame': 'Team_GF/GP', 
    'goalsAgainstPerGame': 'Team_GA/GP', 
    'penaltyKillPct': 'Team_PK%', 
    'powerPlayPct': 'Team_PP%', 
    'faceoffWinPct': 'Team_FOW%'
}
teams_data.rename(columns=teams_rename_map, inplace=True)

season_results_df = season_results_df.merge(
    teams_data[['teamFullName', 'Team_GF/GP', 'Team_GA/GP', 'Team_PP%', 'Team_PK%', 'Team_FOW%']],
    left_on='Home',
    right_on='teamFullName',
    how='left',
    suffixes=('', '_Home')
).drop(columns='teamFullName')

season_results_df = season_results_df.merge(
    teams_data[['teamFullName', 'Team_GF/GP', 'Team_GA/GP', 'Team_PP%', 'Team_PK%', 'Team_FOW%']],
    left_on='Visitor',
    right_on='teamFullName',
    how='left',
    suffixes=('', '_Visitor')
).drop(columns='teamFullName')

# Calculate differences and special teams index
season_results_df['GF/GP_Diff'] = season_results_df['Home_GF/GP'] - season_results_df['Visitor_GF/GP']
season_results_df['GA/GP_Diff'] = season_results_df['Home_GA/GP'] - season_results_df['Visitor_GA/GP']
season_results_df['Home_Special_Teams_Index'] = season_results_df['Home_PP%'] + season_results_df['Home_PK%']
season_results_df['Visitor_Special_Teams_Index'] = season_results_df['Visitor_PP%'] + season_results_df['Visitor_PK%']

columns_to_convert = ['Home_GF/GP', 'Home_GA/GP', 'Home_PP%', 'Home_PK%', 'Home_FOW%_y',
                      'Visitor_GF/GP', 'Visitor_GA/GP', 'Visitor_PP%', 'Visitor_PK%', 'Visitor_FOW%_y']
for col in columns_to_convert:
    if col in season_results_df.columns:
        season_results_df[col] = season_results_df[col].astype(float)

teams_rename_map = {
    'goalsForPerGame': 'Team_GF/GP', 
    'goalsAgainstPerGame': 'Team_GA/GP', 
    'penaltyKillPct': 'Team_PK%', 
    'powerPlayPct': 'Team_PP%', 
    'faceoffWinPct': 'Team_FOW%'
}
teams_data.rename(columns=teams_rename_map, inplace=True)

def calculate_recent_form(team, df, last_n_games=5):
    # Ensure 'G' and 'G.1' are numeric, converting non-numeric values to NaN
    df['G'] = pd.to_numeric(df['G'], errors='coerce')
    df['G.1'] = pd.to_numeric(df['G.1'], errors='coerce')
    
    # Filter for games involving the team and where goal data is available
    team_games = df[((df['Home'] == team) | (df['Visitor'] == team)) & 
                    (df['G'].notna()) & (df['G.1'].notna())].tail(last_n_games)

    # Calculate goals scored and conceded
    team_games['Goals_Scored'] = team_games.apply(
        lambda row: row['G.1'] if row['Home'] == team else row['G'], axis=1)
    team_games['Goals_Conceded'] = team_games.apply(
        lambda row: row['G'] if row['Home'] == team else row['G.1'], axis=1)

    # Calculate averages
    avg_goals_scored = team_games['Goals_Scored'].mean()
    avg_goals_conceded = team_games['Goals_Conceded'].mean()

    return avg_goals_scored, avg_goals_conceded


def aggregate_player_stats(team, skaters_df, goalies_df):
    # Filter skater and goalie data for the specified team
        name_to_abbrev = {v: k for k, v in team_name_mapping.items()}

    # Get the team abbreviation for the provided team name
        team_abbrev = name_to_abbrev.get(team, None)
        if team_abbrev is None:
            raise ValueError(f"Team abbreviation for {team} not found.")

    # Filter skater and goalie data for the specified team by abbreviation
        team_skaters = skaters_df[skaters_df['teamAbbrevs'] == team_abbrev]      

    # Aggregate skater stats (modify as needed based on available columns)
        skater_stats_aggregated = team_skaters[['goals', 'assists', 'points', 'plusMinus', 'penaltyMinutes']].mean()
        
        team_goalies = goalies_df[goalies_df['teamAbbrevs'] == team_abbrev]

    # Aggregate goalie stats (modify as needed based on available columns)
        goalie_stats_aggregated = team_goalies[['wins', 'losses', 'goalsAgainstAverage', 'savePct']].mean()

    # Combine skater and goalie stats
        combined_stats = pd.concat([skater_stats_aggregated, goalie_stats_aggregated])

        return combined_stats

def calculate_head_to_head_stats(team1, team2, df):
    # Filter matches where either team1 is home and team2 is visitor, or vice versa
    head_to_head_games = df[((df['Home'] == team1) & (df['Visitor'] == team2)) |
                            ((df['Home'] == team2) & (df['Visitor'] == team1))]

    # Calculate wins for each team
    wins_team1 = len(head_to_head_games[(head_to_head_games['Home'] == team1) & (head_to_head_games['G.1'] > head_to_head_games['G'])]) + \
                 len(head_to_head_games[(head_to_head_games['Visitor'] == team1) & (head_to_head_games['G'] > head_to_head_games['G.1'])])

    wins_team2 = len(head_to_head_games) - wins_team1

    # Calculate average goals scored by each team
    avg_goals_scored_team1 = head_to_head_games.apply(lambda row: row['G.1'] if row['Home'] == team1 else row['G'], axis=1).mean()
    avg_goals_scored_team2 = head_to_head_games.apply(lambda row: row['G.1'] if row['Home'] == team2 else row['G'], axis=1).mean()

    return wins_team1, wins_team2, avg_goals_scored_team1, avg_goals_scored_team2

def calculate_team_fatigue(team, date, df, period_days=7):
    # Convert the date string to a datetime object if it's not already
    if isinstance(date, str):
        date = pd.to_datetime(date)
    
    # Define the start date for the period to calculate fatigue
    start_date = date - timedelta(days=period_days)

    # Filter games for the specified team within the given time period
    recent_games = df[((df['Home'] == team) | (df['Visitor'] == team)) &
                      (df['Date'] >= start_date) & (df['Date'] < date)]
    
    # The fatigue metric could be the number of games played in this period
    fatigue_metric = len(recent_games)

    return fatigue_metric

def calculate_special_teams_performance(team, df):
    # Filter data for the specified team
    team_data = df[df['teamFullName'] == team]

    # Calculate the average power play percentage (PP%) and penalty kill percentage (PK%)
    # Assuming 'PP%' and 'PK%' are columns in your dataset
    pp_avg = team_data['Team_PP%'].mean()
    pk_avg = team_data['Team_PK%'].mean()

    return pp_avg, pk_avg
    # For example, calculating recent form (you'll need to implement the logic within these function calls)
    # This is a placeholder for how you might begin to integrate such calculations
    # merged_df['Home_Recent_Form_Goals_Scored'] = merged_df['Home'].apply(lambda team: calculate_recent_form(team, merged_df))
    # Note: Actual implementation of calculate_recent_form and other functions will depend on the specific logic and data available
    
    # Fill NaN values in the DataFrame after all calculations
    merged_df.fillna(merged_df.median(numeric_only=True), inplace=True)
    
    return merged_df

# Updating the final dataset with the latest team statistics and new features
for i, row in season_results_df.iterrows():
    if 'Home_Team' in season_results_df.columns and 'Visitor_Team' in season_results_df.columns:
        home_team = row['Home']
        visitor_team = row['Visitor']
        game_date = pd.to_datetime(row['Date'])  # Ensure date is in datetime format

        home_stats_df = teams_data[teams_data['teamFullName'] == home_team]
        if not home_stats_df.empty:
            home_stats = home_stats_df.iloc[-1]
            season_results_df.at[i, 'Home_GF/GP'] = home_stats['Team_GF/GP']
            season_results_df.at[i, 'Home_GA/GP'] = home_stats['Team_GA/GP']
            season_results_df.at[i, 'Home_PP%'] = home_stats['Team_PP%']
            season_results_df.at[i, 'Home_PK%'] = home_stats['Team_PK%']
            season_results_df.at[i, 'Home_FOW%_y'] = home_stats['Team_FOW%']

        # Update visitor team statistics
        visitor_stats_df = teams_data[teams_data['teamFullName'] == visitor_team]
        if not visitor_stats_df.empty:
            visitor_stats = visitor_stats_df.iloc[-1]
            season_results_df.at[i, 'Visitor_GF/GP'] = visitor_stats['Team_GF/GP']
            season_results_df.at[i, 'Visitor_GA/GP'] = visitor_stats['Team_GA/GP']
            season_results_df.at[i, 'Visitor_PP%'] = visitor_stats['Team_PP%']
            season_results_df.at[i, 'Visitor_PK%'] = visitor_stats['Team_PK%']
            season_results_df.at[i, 'Visitor_FOW%_y'] = visitor_stats['Team_FOW%']

        # Calculate and update new features
        # Recent form
        # Player stats
        # Head-to-head stats
        # Team fatigue
        # Special teams performance
        # Add these calculations here...
        # Assuming you have loaded your datasets and defined all necessary functions

        # Calculate recent form for home and visitor teams
        home_recent_form_scored, home_recent_form_conceded = calculate_recent_form(home_team, season_results_df)
        visitor_recent_form_scored, visitor_recent_form_conceded = calculate_recent_form(visitor_team, season_results_df)

        # Aggregate player stats for home and visitor teams
        home_player_stats = aggregate_player_stats(home_team, skaters_df, goalies_df)
        visitor_player_stats = aggregate_player_stats(visitor_team, skaters_df, goalies_df)

        # Calculate head-to-head statistics
        h2h_wins_home, h2h_wins_visitor, h2h_avg_goals_home, h2h_avg_goals_visitor = calculate_head_to_head_stats(home_team, visitor_team, season_results_df)

        # Calculate team fatigue
        home_team_fatigue = calculate_team_fatigue(home_team, game_date, season_results_df)
        visitor_team_fatigue = calculate_team_fatigue(visitor_team, game_date, season_results_df)

        # Calculate special teams performance
        home_pp_avg, home_pk_avg = calculate_special_teams_performance(home_team, teams_data)
        season_results_df.at[i, 'Home_PP%_Avg'] = home_pp_avg
        season_results_df.at[i, 'Home_PK%_Avg'] = home_pk_avg

        away_pp_avg, away_pk_avg = calculate_special_teams_performance(visitor_team, teams_data)
        season_results_df.at[i, 'Away_PP%_Avg'] = away_pp_avg
        season_results_df.at[i, 'Away_PK%_Avg'] = away_pk_avg

        # Assign calculated values to the final dataset
        season_results_df.at[i, 'Home_Recent_Form_Goals_Scored'] = home_recent_form_scored
        season_results_df.at[i, 'Home_Recent_Form_Goals_Conceded'] = home_recent_form_conceded
        season_results_df.at[i, 'Visitor_Recent_Form_Goals_Scored'] = visitor_recent_form_scored
        season_results_df.at[i, 'Visitor_Recent_Form_Goals_Conceded'] = visitor_recent_form_conceded

        # (Similarly for player stats, head-to-head, fatigue, and special teams metrics)
        # Player stats (assuming the aggregate_player_stats function returns a Series or dict)
        home_player_stats = aggregate_player_stats(home_team, skaters_df, goalies_df)
        visitor_player_stats = aggregate_player_stats(visitor_team, skaters_df, goalies_df)
        for stat, value in home_player_stats.items():
            season_results_df.at[i, f'Home_{stat}'] = value
        for stat, value in visitor_player_stats.items():
            season_results_df.at[i, f'Visitor_{stat}'] = value

        # Head-to-head statistics
        h2h_wins_home, h2h_wins_visitor, h2h_avg_goals_home, h2h_avg_goals_visitor = calculate_head_to_head_stats(home_team, visitor_team, season_results_df)
        season_results_df.at[i, 'H2H_Wins_Home'] = h2h_wins_home
        season_results_df.at[i, 'H2H_Wins_Visitor'] = h2h_wins_visitor
        season_results_df.at[i, 'H2H_Avg_Goals_Scored_Home'] = h2h_avg_goals_home
        season_results_df.at[i, 'H2H_Avg_Goals_Scored_Visitor'] = h2h_avg_goals_visitor

        # Team fatigue
        home_team_fatigue = calculate_team_fatigue(home_team, game_date, season_results_df)
        visitor_team_fatigue = calculate_team_fatigue(visitor_team, game_date, season_results_df)
        season_results_df.at[i, 'Home_Team_Fatigue'] = home_team_fatigue
        season_results_df.at[i, 'Visitor_Team_Fatigue'] = visitor_team_fatigue

        # Special teams performance
        home_pp_avg, home_pk_avg = calculate_special_teams_performance(home_team, teams_data)
        visitor_pp_avg, visitor_pk_avg = calculate_special_teams_performance(visitor_team, teams_data)
        season_results_df.at[i, 'Home_PP%_Avg'] = home_pp_avg
        season_results_df.at[i, 'Home_PK%_Avg'] = home_pk_avg
        season_results_df.at[i, 'Visitor_PP%_Avg'] = visitor_pp_avg
        season_results_df.at[i, 'Visitor_PK%_Avg'] = visitor_pk_avg

        # Optional: Recalculate differences and special teams index if needed
# ..    .
        season_results_df['GF/GP_Diff'] = season_results_df['Home_GF/GP'] - season_results_df['Visitor_GF/GP']
        season_results_df['GA/GP_Diff'] = season_results_df['Home_GA/GP'] - season_results_df['Visitor_GA/GP']
        season_results_df['Home_Special_Teams_Index'] = season_results_df['Home_PP%'] + season_results_df['Home_PK%']
        season_results_df['Visitor_Special_Teams_Index'] = season_results_df['Visitor_PP%'] + season_results_df['Visitor_PK%']

load_dotenv()  
cloudcube_base_path = 'moygkytojg0o'
bucket_name='cloud-cube-us2'

s3_client = boto3.client(
's3',
aws_access_key_id=os.getenv('CLOUDCUBE_ACCESS_KEY_ID'),
aws_secret_access_key=os.getenv('CLOUDCUBE_SECRET_ACCESS_KEY'),
region_name='us-east-1'
)

print(season_results_df.columns)
print("Starting model training...")
season_results_df['Total_Goals'] = season_results_df['G'] + season_results_df['G.1']
X = season_results_df[features]
y = season_results_df['Total_Goals']

y.fillna(y.median(), inplace=True)

y.replace([np.inf, -np.inf], y.median(), inplace=True)

assert np.all(np.isfinite(y)), "There are still non-finite values in y."

X.replace([np.inf, -np.inf], np.nan, inplace=True)

X.fillna(X.median(), inplace=True)

assert np.all(np.isfinite(X)), "There are still non-finite values in X."

# Model training and evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(objective ='reg:squarederror')
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.05],
    'max_depth': [2, 3, 4],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8, 0.9]
}
grid_cv = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0)
grid_cv.fit(X_train, y_train)
print("Model training completed.")
best_model = grid_cv.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test set: ", mse)

# Optionally save the model or return it
# For example, to save the model to a variable for later use:
trained_model = best_model

model_file = io.BytesIO()
pickle.dump(trained_model, model_file)
model_file.seek(0)

team_stats_file = io.BytesIO()
pickle.dump(season_results_df, team_stats_file)
team_stats_file.seek(0)

model_key = f"{cloudcube_base_path}/model.pkl"
team_stats_key = f"{cloudcube_base_path}/team_stats.pkl"

s3_client.upload_fileobj(model_file, bucket_name, model_key)
s3_client.upload_fileobj(team_stats_file, bucket_name, team_stats_key)

print("Model and team stats uploaded to CloudCube.")

# Calculate and print default feature values
default_values = X.median().to_dict()
print("Default feature values:", default_values)