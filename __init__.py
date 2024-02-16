import sys
print(sys.path)
from flask import Flask
import os
print(os.getcwd())
from dotenv import load_dotenv
import pickle
from OddsAPICall import OddsAPICall
from NHLGoalsPredictor import NHLGoalsPredictor
import boto3
from io import BytesIO
from urllib.parse import urlparse

sys.path.append('.')

load_dotenv()

cloudcube_url = urlparse(os.getenv('CLOUDCUBE_URL'))
bucket_name = cloudcube_url.netloc.split('.')[0]
base_path = cloudcube_url.path.lstrip('/')

s3_client = boto3.client(
    's3',
    aws_access_key_id = cloudcube_url.username,
    aws_secret_access_key = cloudcube_url.password,
    region_name = 'us-east-1'
)

def download_object_from_s3(bucket, key):
    "Download an object from S3 to a BytesIO buffer."""
    buffer = BytesIO()
    s3_client.download_fileobj(Bucket=bucket, Key=key, Fileobj=buffer)
    buffer.seek(0)
    return buffer

def load_model(model_key):
    model_buffer = download_object_from_s3(bucket_name, model_key)
    return pickle.load(model_buffer)
    
def load_team_stats(team_stats_key):
    team_stats_buffer = download_object_from_s3(bucket_name, team_stats_key)
    return pickle.load(team_stats_buffer)
    
def create_app():
    app = Flask(__name__)
    app.secret_key = os.urandom(16)
    
    # Initializes API client and model
    load_dotenv()
    api_key=os.getenv('ODDS_API_KEY')
    api_client = OddsAPICall(api_key)

    model_key = f"{base_path}/model.pkl"
    team_stats_key = f"{base_path}/team_stats.pkl"

    model = load_model(model_key)
    team_stats = load_team_stats(team_stats_key)
    model_predictor = NHLGoalsPredictor(model, team_stats)
    
    import views
    app.register_blueprint(views.bp)
    
    app.config['API_CLIENT'] = api_client
    app.config['MODEL'] = model
    app.config['TEAM_STATS'] = team_stats
    app.config['MODEL_PREDICTOR'] = model_predictor
    
    return app
