import os
import logging
from datetime import datetime

import joblib
import pandas as pd
from functools import wraps
from flask import (
    Flask, request, jsonify, render_template, send_from_directory, session, redirect, url_for, flash
)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
from flask_compress import Compress
from dotenv import load_dotenv


# ==============================================================================
# Configuration & Initialization
# ==============================================================================
load_dotenv()

app = Flask(__name__)
secret_key = os.environ.get('FLASK_SECRET_KEY')
if not secret_key:
    raise RuntimeError(
        "CRITICAL ERROR: FLASK_SECRET_KEY is not set in the environment variables. "
        "Set it in your .env file or export it before running the app."
    )
app.secret_key = secret_key
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

db_uri = os.environ.get('SQLALCHEMY_DATABASE_URI')
if not db_uri:
    raise ValueError("CRITICAL ERROR: SQLALCHEMY_DATABASE_URI is not set in the environment variables. Please provide the PostgreSQL URL.")
app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 280
}
# Initialize extensions
db = SQLAlchemy(app)
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})
compress = Compress(app)


# ==============================================================================
# Database Models
# ==============================================================================
class User(db.Model):
    __tablename__ = 'users'
    UserID = db.Column(db.Integer, primary_key=True)
    Name = db.Column(db.String(100), nullable=False)
    Email = db.Column(db.String(100), unique=True, nullable=False)
    Password = db.Column(db.String(200), nullable=False)
    Role = db.Column(db.String(50), nullable=False, default='User')
    
    weather_data = db.relationship('WeatherData', backref='user', lazy=True)

class WeatherData(db.Model):
    __tablename__ = 'weather_data'
    DataID = db.Column(db.Integer, primary_key=True)
    UserID = db.Column(db.Integer, db.ForeignKey('users.UserID'), nullable=True)
    AnnualRainfall = db.Column(db.Float, nullable=False)
    CloudVisibility = db.Column(db.Float, nullable=False)
    Temperature = db.Column(db.Float, nullable=False)
    Humidity = db.Column(db.Float, nullable=False)
    SeasonalRainfall = db.Column(db.Float, nullable=False)
    
    predictions = db.relationship('PredictionResult', backref='weather_data', lazy=True)

class PredictionResult(db.Model):
    __tablename__ = 'prediction_results'
    PredictionID = db.Column(db.Integer, primary_key=True)
    DataID = db.Column(db.Integer, db.ForeignKey('weather_data.DataID'), nullable=False)
    FloodResult = db.Column(db.String(50), nullable=False)
    FloodProbability = db.Column(db.Float, nullable=False)
    PredictionDate = db.Column(db.DateTime, nullable=False, default=datetime.now)

# Create tables
with app.app_context():
    try:
        db.session.execute(db.text('SELECT 1'))
        logging.info("Database connection verified successfully.")
    except Exception as e:
        logging.critical("Database connection failed: %s", e)
        raise SystemExit("CRITICAL: Cannot connect to database. Check SQLALCHEMY_DATABASE_URI.")
    db.create_all()
    logging.info("Database tables verified/created successfully.")


def save_prediction(inputs, result):
    """Insert a prediction record into the database."""
    try:
        user_id = session.get('user_id')
        total_seasonal = inputs['jan_feb'] + inputs['mar_may'] + inputs['jun_sep'] + inputs['oct_dec']
        weather = WeatherData(
            UserID=user_id,
            AnnualRainfall=inputs['annual'],
            CloudVisibility=inputs['cloud_cover'],
            Temperature=inputs['temp'],
            Humidity=inputs['humidity'],
            SeasonalRainfall=total_seasonal
        )
        db.session.add(weather)
        db.session.commit()
        
        prediction = PredictionResult(
            DataID=weather.DataID,
            FloodResult=result['prediction'],
            FloodProbability=result['probability']
        )
        db.session.add(prediction)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logging.error("Failed to save prediction: %s", e)


def get_prediction_history():
    """Retrieve all prediction records for the current user."""
    user_id = session.get('user_id')
    if user_id:
        results = PredictionResult.query.join(WeatherData).filter(WeatherData.UserID == user_id).order_by(PredictionResult.PredictionID.desc()).all()
    else:
        results = []
    history = []
    for res in results:
        weather = res.weather_data
        history.append({
            'id': res.PredictionID,
            'timestamp': res.PredictionDate.strftime('%Y-%m-%d %H:%M:%S'),
            'temp': weather.Temperature if weather else 0.0,
            'humidity': weather.Humidity if weather else 0.0,
            'cloud_cover': weather.CloudVisibility if weather else 0.0,
            'annual': weather.AnnualRainfall if weather else 0.0,
            'prediction': res.FloodResult,
            'probability': res.FloodProbability,
            'risk_message': 'Flood Risk Detected!' if res.FloodResult == 'HIGH' else 'No flood risk detected.'
        })
    return history


def user_cache_key():
    """Generate a cache key scoped to the current user's session."""
    return f"history_user_{session.get('user_id', 'anon')}"


# ==============================================================================
# Model Loading
# ==============================================================================
model = None
preprocessor = None
feature_names = None


def load_models():
    """Load the trained ML model, scaler, and feature names from disk."""
    global model, preprocessor, feature_names
    try:
        model_path = os.path.join(MODELS_DIR, 'floods.save')
        scaler_path = os.path.join(MODELS_DIR, 'transform.save')
        features_path = os.path.join(MODELS_DIR, 'feature_names.pkl')

        for path, name in [
            (model_path, 'floods.save'),
            (scaler_path, 'transform.save'),
            (features_path, 'feature_names.pkl')
        ]:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Required model artifact '{name}' not found at '{path}'. "
                    "Run train.py to generate model artifacts before starting the app."
                )

        model = joblib.load(model_path)
        preprocessor = joblib.load(scaler_path)
        feature_names = joblib.load(features_path)
        logging.info(
            "Successfully loaded ML model, preprocessor, and feature names.")
    except Exception as e:
        logging.error("Failed to load models: %s", e)
        model = None


# Load on startup
load_models()


# ==============================================================================
# Authentication & Security
# ==============================================================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for('login_page', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(Email=email).first()
        
        if user and check_password_hash(user.Password, password):
            session['user_id'] = user.UserID
            session['user_name'] = user.Name
            flash("Logged in successfully!", "success")
            next_page = request.args.get('next')
            return redirect(next_page or url_for('home'))
        else:
            flash("Invalid email or password.", "danger")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(Email=email).first():
            flash("Email already exists. Please login.", "warning")
            return redirect(url_for('login_page'))
            
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(Name=name, Email=email, Password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash("Registration successful! You can now log in.", "success")
        return redirect(url_for('login_page'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('home'))

# ==============================================================================
# Routes - Pages
# ==============================================================================
@app.route('/')
def home():
    """Renders the landing page."""
    return render_template('home.html')


@app.route('/Predict')
@login_required
def predict_page():
    """Renders the prediction input form."""
    return render_template('index.html')


@app.route('/reports')
def reports():
    """Renders the evaluation reports dashboard."""
    return render_template('reports.html')


@app.route('/history')
@login_required
@cache.cached(timeout=60, key_prefix=user_cache_key)
def history():
    """Renders the prediction history page."""
    records = get_prediction_history()
    return render_template('history.html', records=records)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint to verify API and model status."""
    if model is None or preprocessor is None:
        return jsonify(
            {"status": "error", "message": "Model not loaded"}), 503
    return jsonify({
        "status": "ok",
        "message": "API is running and model is loaded."
    }), 200


@app.route('/reports/<path:filename>')
def serve_report_image(filename):
    """Serves static image files from the reports directory."""
    return send_from_directory(
        os.path.join(BASE_DIR, 'reports'), filename)


# ==============================================================================
# Routes - API
# ==============================================================================
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """API endpoint for predicting flood risk."""

    if model is None or preprocessor is None:
        return jsonify(
            {"error": "Service Unavailable. Model files missing."}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify(
                {"error": "Invalid request. JSON body missing."}), 400

        # Sanitize: strip whitespace from any string values before processing
        for field in data:
            if isinstance(data[field], str):
                data[field] = data[field].strip()

        # 1. Extract Raw Inputs
        required_fields = [
            'temp', 'humidity', 'cloud_cover', 'annual',
            'jan_feb', 'mar_may', 'jun_sep', 'oct_dec'
        ]

        for field in required_fields:
            if field not in data:
                return jsonify(
                    {"error": "Missing required field: %s" % field}), 400

        # Parse and Validate Boundaries
        try:
            temp = float(data['temp'])
            if not (-10 <= temp <= 60):
                raise ValueError(
                    "Temperature must be between -10 and 60 C.")

            humidity = float(data['humidity'])
            if not (0 <= humidity <= 100):
                raise ValueError(
                    "Humidity must be between 0 and 100 percent.")

            cloud_cover = float(data['cloud_cover'])
            if not (0 <= cloud_cover <= 100):
                raise ValueError(
                    "Cloud Cover must be between 0 and 100 percent.")

            annual = float(data['annual'])
            jan_feb = float(data['jan_feb'])
            mar_may = float(data['mar_may'])
            jun_sep = float(data['jun_sep'])
            oct_dec = float(data['oct_dec'])

            if any(val < 0 or val > 20000 for val in [
                   annual, jan_feb, mar_may, jun_sep, oct_dec]):
                raise ValueError(
                    "Rainfall values must be between 0 and 20000 mm.")

        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400

        # 2. Feature Engineering (Match training logic exactly)
        total_seasonal = jan_feb + mar_may + jun_sep + oct_dec
        interaction = (humidity * annual) / 1000.0

        # Assemble dictionary matching training features
        input_features = {
            'Temp': temp,
            'Humidity': humidity,
            'Cloud Cover': cloud_cover,
            'ANNUAL': annual,
            'Jan-Feb': jan_feb,
            'Mar-May': mar_may,
            'Jun-Sep': jun_sep,
            'Oct-Dec': oct_dec,
            'Total_Seasonal_Rainfall': total_seasonal,
            'Humidity_Rainfall_Interaction': interaction
        }

        # Create DataFrame
        input_df = pd.DataFrame([input_features])

        # Ensure correct column order
        try:
            input_df = input_df[feature_names]
        except KeyError as e:
            return jsonify(
                {"error": "Feature mismatch: %s" % str(e)}), 500

        # 3. Preprocessing
        scaled_features = preprocessor.transform(input_df)
        scaled_df = pd.DataFrame(
            scaled_features, columns=feature_names)

        # 4. Prediction
        prediction = model.predict(scaled_df)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(scaled_df)[0][1]
        else:
            probability = 1.0 if prediction == 1 else 0.0

        # 5. Response Formatting
        result_class = "HIGH" if prediction == 1 else "LOW"
        risk_message = (
            "Flood Risk Detected! "
            "Immediate precautionary measures recommended."
            if prediction == 1
            else "No flood risk detected. The area appears safe "
                 "based on current weather parameters."
        )

        response = {
            "prediction": result_class,
            "probability": round(float(probability), 4),
            "message": risk_message
        }

        # 6. Save to Prediction History Database
        raw_inputs = {
            'temp': temp, 'humidity': humidity,
            'cloud_cover': cloud_cover, 'annual': annual,
            'jan_feb': jan_feb, 'mar_may': mar_may,
            'jun_sep': jun_sep, 'oct_dec': oct_dec
        }
        save_prediction(raw_inputs, response)
        cache.delete(user_cache_key())

        return jsonify(response), 200

    except Exception as e:
        logging.error("Prediction Error: %s", e)
        return jsonify(
            {"error": "An internal error occurred: %s" % str(e)}), 500


@app.route('/api/history', methods=['GET'])
@login_required
def api_history():
    """API endpoint to retrieve prediction history as JSON."""
    records = get_prediction_history()
    return jsonify(records), 200


@app.route('/api/history/clear', methods=['POST'])
@login_required
def clear_history():
    """API endpoint to clear prediction history for the current user."""
    try:
        user_id = session['user_id']
        # Delete PredictionResults first (FK dependency on WeatherData)
        user_weather_ids = db.session.query(WeatherData.DataID).filter(
            WeatherData.UserID == user_id
        ).subquery()
        PredictionResult.query.filter(
            PredictionResult.DataID.in_(user_weather_ids)
        ).delete(synchronize_session='fetch')
        # Then delete the user's WeatherData records
        WeatherData.query.filter(WeatherData.UserID == user_id).delete()
        db.session.commit()
        cache.delete(user_cache_key())
        return jsonify({"message": "History cleared successfully."}), 200
    except Exception as e:
        db.session.rollback()
        logging.error("Failed to clear history for user %s: %s", session.get('user_id'), e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
