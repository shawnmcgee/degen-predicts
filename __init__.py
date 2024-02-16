import sys
print(sys.path)
from flask import Flask
import os
print(os.getcwd())
from dotenv import load_dotenv
import pickle
from NHLOverUnderWeb.OddsAPICall import OddsAPICall
from NHLOverUnderWeb.NHLGoalsPredictor import NHLGoalsPredictor

sys.path.append('.')
def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)
    
def load_team_stats(team_stats_path):
    with open(team_stats_path, 'rb') as f:
        return pickle.load(f)
    
def create_app():
    app = Flask(__name__)
    app.secret_key = os.urandom(16)
    
    # Initializes API client and model
    load_dotenv()
    api_key=os.getenv('ODDS_API_KEY')
    api_client = OddsAPICall(api_key)

    model_path = os.path.join('SavedModel', 'model.pkl')
    team_stats_path = os.path.join('SavedModel', 'team_stats.pkl')
    model = load_model(model_path) 
    team_stats = load_team_stats(team_stats_path)
    model_predictor = NHLGoalsPredictor(model, team_stats)
    
    import views
    app.register_blueprint(views.bp)
    
    app.config['API_CLIENT'] = api_client
    app.config['MODEL'] = model
    app.config['TEAM_STATS'] = team_stats
    app.config['MODEL_PREDICTOR'] = model_predictor
    
    return app
