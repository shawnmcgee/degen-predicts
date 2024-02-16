from telnetlib import GA
import unicodedata
from flask import g, current_app
import pandas as pd
import pickle
import boto3
from urllib.parse import urlparse
import os
import io

class NHLGoalsPredictor:
    
    def __init__(self, model, team_stats_df):
         # Initialize S3 client
        cloudcube_url = urlparse(os.getenv('CLOUDCUBE_URL'))
        bucket_name = cloudcube_url.netloc.split('.')[0]
        s3_client = boto3.client(
            's3',
            aws_access_key_id=cloudcube_url.username,
            aws_secret_access_key=cloudcube_url.password,
            region_name='us-east-1'
        )

        # Download and load the model
        model_key = f"{cloudcube_url.path.lstrip('/')}/model.pkl"
        model_file = io.BytesIO()
        s3_client.download_fileobj(bucket_name, model_key, model_file)
        model_file.seek(0)
        self.model = pickle.load(model_file)

        # Download and load the team stats DataFrame
        team_stats_key = f"{cloudcube_url.path.lstrip('/')}/team_stats.pkl"
        team_stats_file = io.BytesIO()
        s3_client.download_fileobj(bucket_name, team_stats_key, team_stats_file)
        team_stats_file.seek(0)
        self.team_stats_df = pickle.load(team_stats_file)      

        # List of features used in the model
        self.features = [
        'Home_GF/GP', 'Visitor_GF/GP', 'Home_GA/GP', 'Visitor_GA/GP', 
        'Home_PP%', 'Visitor_PP%', 'Home_PK%', 'Visitor_PK%', 'Home_FOW%_y', 
        'Visitor_FOW%_y', 'GF/GP_Diff', 'GA/GP_Diff', 'Home_Special_Teams_Index', 
        'Visitor_Special_Teams_Index', 'Home_Recent_Form_Goals_Scored', 
        'Home_Recent_Form_Goals_Conceded', 'Visitor_Recent_Form_Goals_Scored', 
        'Visitor_Recent_Form_Goals_Conceded', 'H2H_Wins_Home', 'H2H_Wins_Visitor', 
        'Home_Team_Fatigue', 'Visitor_Team_Fatigue', 'Home_PP%_Avg', 'Home_PK%_Avg', 
        'Visitor_PP%_Avg', 'Visitor_PK%_Avg'
        ]        

        # Calculate median values for these features
        self.default_values = self.team_stats_df[self.features].median().to_dict()
        print(team_stats_df.columns)
        
    def normalize_team_name(self, name): 
        name = unicodedata.normalize('NFD', name)
        name = name.encode('ascii', 'ignore').decode('ascii')
        name = name.replace('.', '')
        return name

    def verify_team_names(self, team_name):
        normalized_team_name = self.normalize_team_name(team_name)
        
        # Normalize names in 'Home' and 'Visitor' columns before comparison
        normalized_home_names = self.team_stats_df['Home'].apply(self.normalize_team_name)
        normalized_visitor_names = self.team_stats_df['Visitor'].apply(self.normalize_team_name)

        if normalized_team_name not in normalized_home_names.values and normalized_team_name not in normalized_visitor_names.values:
            raise ValueError(f"Team name '{team_name}' not found in the dataset. Please check for misspellings.")

    def check_feature_consistency(self, game_data):
        missing_features = [feature for feature in self.features if feature not in game_data.columns]
        if missing_features: 
               raise ValueError(f"Missing features in the input data: {missing_features}")

    def get_team_stats_home(self, team_name):
        print(f"Looking up home stats for: {team_name}")
        if team_name not in self.team_stats_df['Home'].unique():
            print(f"Warning: {team_name} not found in Home teams. Using default values.")
        # Retrieve team stats from the dataset
        team_stats = self.team_stats_df[self.team_stats_df['Home'] == team_name]
        print(team_stats)

        if not team_stats.empty:
            return team_stats.iloc[0].to_dict()
        else:
            return self.default_values
        
    def get_team_stats_visitor(self, team_name):
        print(f"Looking up visitor stats for: {team_name}")
        if team_name not in self.team_stats_df['Visitor'].unique():
            print(f"Warning: {team_name} not found in Visitor teams. Using default values.")
        # Retrieve team stats from the dataset
        team_stats = self.team_stats_df[self.team_stats_df['Visitor'] == team_name]
        print(team_stats)

        if not team_stats.empty:
            return team_stats.iloc[0].to_dict()
        else:
            return self.default_values
        
    def predict_goals(self, home_team, away_team):      
        self.verify_team_names(home_team)
        self.verify_team_names(away_team)

        # Retrieve stats for home and away teams
        home_stats = self.get_team_stats_home(home_team)
        away_stats = self.get_team_stats_visitor(away_team)              

        # Preparing the game data for prediction
        game_data = pd.DataFrame([{
            **{k: home_stats[k] for k in home_stats if k in self.features},
            **{k: away_stats[k] for k in away_stats if k in self.features}
        }])
            
        self.check_feature_consistency(game_data)
        game_data = game_data[self.features]       
         
        print(f"Home Team: {home_team}, Visitor Team: {away_team}")
        print(game_data)
        # Predict the total number of goals
        predicted_goals = self.model.predict(game_data)[0]

        return round(predicted_goals, 3)


     