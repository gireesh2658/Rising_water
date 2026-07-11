# Rising Waters
A web application predicting flood risk from meteorological inputs using an XGBoost classifier.

![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)
![Flask Version](https://img.shields.io/badge/Flask-3.0.3-black.svg?logo=flask)
![XGBoost Version](https://img.shields.io/badge/XGBoost-2.0.3-orange.svg)

## Problem Statement
Built for the SmartBridge AI & ML Internship Track, Rising Waters addresses the need for localized flood risk assessment based on meteorological parameters. The application processes user-submitted weather data (temperature, humidity, cloud cover, and seasonal rainfall) to generate a binary risk classification (HIGH/LOW). It serves as a demonstrative end-to-end machine learning pipeline, moving from offline training with SMOTE balancing to online inference via a Flask REST API and PostgreSQL persistence.

## Architecture Overview

```mermaid
graph TD
    Client[Browser Client] -- POST /predict --> FlaskAPI[Flask Route: predict]
    FlaskAPI -- Read Artifacts --> Model[XGBoost floods.save]
    FlaskAPI -- Insert Log --> Postgres[(PostgreSQL DB)]
    Model -- Return Probability --> FlaskAPI
    FlaskAPI -- JSON Response --> Client
```

| Component | Technology | Role |
| :--- | :--- | :--- |
| **API Server** | Flask | Handles route logic, session auth, and inference orchestration |
| **Database** | PostgreSQL | Persists user credentials and prediction history via SQLAlchemy |
| **Inference Engine** | XGBoost | Generates binary predictions and confidence scores |
| **Client UI** | Jinja2 / JS | Submits asynchronous `fetch` requests to the Flask backend |

## Key Features
- **Risk Prediction (`app.py:predict`)**: Processes 8 meteorological inputs through a serialized XGBoost model (`floods.save`) and StandardScaler (`transform.save`), returning a probability score.
- **User Authentication (`app.py:login_page`)**: Implements session-based authentication with PBKDF2 SHA256 password hashing via `werkzeug.security`.
- **Prediction Auditing (`app.py:history_page`)**: Automatically logs all user predictions to a PostgreSQL `prediction_results` table for historical review and filtering.
- **Model Evaluation (`reports/`)**: Pre-generated visual reports (ROC curve, confusion matrix) served statically on a dedicated dashboard route.
- **Automated Training (`train.py`)**: Standalone pipeline executing mean/median imputation, SMOTE resampling, and hyperparameter tuning (`RandomizedSearchCV`).

## Tech Stack
**Backend**
- Flask (3.0.3)
- Flask-SQLAlchemy (3.1.1)
- Flask-CORS (4.0.0)
- Flask-Caching (2.1.0)
- Flask-Compress (1.15)
- gunicorn (22.0.0)

**Database**
- PostgreSQL (via psycopg2-binary 2.9.9)

**Machine Learning & Data Processing**
- scikit-learn (1.4.2)
- xgboost (2.0.3)
- imbalanced-learn (0.12.2)
- pandas (2.2.2)
- numpy (1.26.4)
- joblib (1.4.2)

**Frontend**
- HTML5 (Jinja2)
- Vanilla CSS (Glassmorphism)
- Vanilla JavaScript

## Project Structure
```text
code files/
├── app.py                  # API endpoints, SQLAlchemy models, and inference logic
├── train.py                # Standalone offline ML training pipeline
├── data_analysis.py        # Exploratory Data Analysis script
├── locustfile.py           # Load testing configuration
├── requirements.txt        # Exact dependency versions
├── .env.example            # Environment variable template
├── data/
│   └── flood_dataset_expanded.csv  # 955-row training dataset
├── models/
│   ├── floods.save         # Serialized XGBoost classifier
│   ├── transform.save      # Fitted StandardScaler
│   └── feature_names.pkl   # Expected feature order
├── reports/                # Static visual evaluation metrics
├── static/                 # CSS, JS, and background assets
└── templates/              # Jinja2 HTML views (index, login, history, etc.)
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gireesh2658/Rising_water.git
   cd "Rising_water/5. Project Development Phase/code files"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   # source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   *Required Variables in `.env`:*
   | Variable | Description | Example |
   | :--- | :--- | :--- |
   | `FLASK_SECRET_KEY` | Key for session signing | `dev-secret-key-super-secure` |
   | `SQLALCHEMY_DATABASE_URI` | PostgreSQL connection string | `postgresql://user:pass@host:5432/dbname?sslmode=require` |

5. **Start the application:**
   ```bash
   python app.py
   ```
   *Note: Database tables are automatically initialized via `db.create_all()` on application startup. The server binds to port `5000` (or the `PORT` env var).*

## Usage / API Reference

| Method | Route | Purpose | Sample Request Body | Sample Response |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/predict` | Submit weather data | `{"temp": 30.5, "humidity": 78.0, "cloud_cover": 42.0, "annual": 3200.5, "jan_feb": 15.2, "mar_may": 210.4, "jun_sep": 2500.5, "oct_dec": 450.3}` | `{"prediction": "HIGH", "probability": 0.8942, "message": "Flood Risk Detected!..."}` |
| `GET` | `/api/history` | Retrieve user prediction history | *(None)* | `[{"id": 12, "timestamp": "2026-07-11 12:00:00", "prediction": "HIGH", "probability": 0.89, ...}]` |
| `POST` | `/api/history/clear` | Purge user history | *(None)* | `{"message": "History cleared successfully."}` |

**Primary User Flow:** Users access the root URL (`/`), register an account, and log in. From the `/Predict` route, users enter current weather parameters. A client-side fetch request posts the data to `/predict`, and the UI dynamically surfaces the calculated flood risk and confidence score.

## Model / ML Pipeline Details
- **Target Variable:** `flood` (binary: 0 = No Flood, 1 = Flood)
- **Base Features:** `Temp`, `Humidity`, `Cloud Cover`, `ANNUAL`, `Jan-Feb`, `Mar-May`, `Jun-Sep`, `Oct-Dec`
- **Engineered Features:** `Total_Seasonal_Rainfall`, `Humidity_Rainfall_Interaction`
- **Final Model Type:** XGBoost Classifier
- **Training Approach (`train.py`):**
  - Missing values imputed (mean for continuous, median for skewed).
  - Outliers capped using the IQR method (1.5x bounds).
  - Features correlated >0.90 dropped.
  - SMOTE applied to balance the training set.
  - Evaluated via 10-Fold Stratified CV across 4 models (Decision Tree, Random Forest, K-Nearest Neighbors (KNN), XGBoost).
  - Best model (XGBoost) tuned with `RandomizedSearchCV` (50 iterations).
- **Performance Metrics:** Accuracy: **0.99**, F1-Score: **0.99** *(measured on a 191-sample 20% holdout set; verified via `reports/classification_report.txt`)*.

## Security Notes
- **Authentication:** Passwords are mathematically hashed (`pbkdf2:sha256`) prior to storage.
- **Session Protection:** Routes handling predictions and history enforce a custom `@login_required` decorator checking the session dictionary.
- **SQL Injection Prevention:** All database operations utilize SQLAlchemy ORM, mapping Python classes to parameterized queries.
- **Secrets Management:** Sensitive configuration is loaded from environment variables rather than hardcoded in the codebase.



## 
- **Author:** gireesh  (SmartBridge AI & ML Internship Track)
