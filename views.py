from datetime import datetime
import json
from dotenv import load_dotenv
import os
from flask import Blueprint, redirect, render_template, request, flash, session, current_app, url_for
from LogoMapping import TEAM_LOGOS

bp = Blueprint('views', __name__, template_folder='templates')

@bp.route('/')
@bp.route('/login', methods=['GET', 'POST'])

def login():
    load_dotenv()
    users_json = os.getenv("USER_LOGIN")
    if users_json is not None:
        users = json.loads(users_json)
    else:
        flash('Login configuration error.', 'danger')
        return render_template('login.html')

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users and users[username] == password:
            session['logged_in'] = True
            #flash('Logged in successfully!', 'success')
            return redirect(url_for('views.show_games'))
        else:
            flash('Invalid Credentials. Please try again.', 'danger')

    # This line should return the login template for GET requests
    return render_template('login.html')

@bp.route('/games')
def show_games():
    # Protect this route
    if 'logged_in' not in session:
        return redirect(url_for('views.login'))
    # Your existing code for showing games
    api_client = current_app.config['API_CLIENT']
    model_predictor = current_app.config['MODEL_PREDICTOR']
    games = api_client.fetch_odds("icehockey_nhl", "us", "totals", "betonlineag")
    if games:
        parsed_games = api_client.parse_games(games)

        # Add predictions to each game
        for game in parsed_games:
            prediction = model_predictor.predict_goals(game["away_team"], game["home_team"])
            game["predicted_total_goals"] = prediction
            game["team1_logo"] = url_for('static', filename=TEAM_LOGOS.get(game["home_team"], "Default_logo.png")) 
            game["team2_logo"] = url_for('static', filename=TEAM_LOGOS.get(game["away_team"], "default_logo.png"))

        return render_template('games.html', games=parsed_games)
    else:
        return "Error fetching data"

@bp.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('views.login'))