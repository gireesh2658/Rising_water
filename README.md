# Rising Waters — A Machine Learning Approach to Flood Prediction

A full-stack web application that predicts flood risk from meteorological inputs. Users submit weather parameters (temperature, humidity, cloud cover, seasonal rainfall) through a browser-based form, and the system returns a binary flood/no-flood classification with a confidence score. The prediction is powered by an XGBoost classifier trained on a 955-row historical flood dataset, served through a Flask REST API backed by PostgreSQL.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [ML Pipeline](#ml-pipeline)
- [Database Schema](#database-schema)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)

---

## Overview

Floods are among the most destructive natural disasters, often exacerbated by delayed response. Rising Waters addresses this by providing a lightweight, web-accessible prediction tool. A user registers an account, enters current weather conditions for a region, and receives an instant risk assessment. Every prediction is logged to a PostgreSQL database and can be reviewed through a paginated history view.

The ML model is trained offline via a standalone pipeline (`train.py`) and serialized to disk. The Flask application loads these artifacts at startup and serves predictions through a JSON API. The frontend is a set of server-rendered Jinja2 templates with client-side `fetch()` calls for the prediction workflow — no separate frontend build step is required.

---

## Features

Each item below maps to an implemented route or module in the codebase.

- **Flood risk prediction** — accepts 8 weather parameters, engineers 2 derived features, scales inputs, and returns a HIGH/LOW classification with probability (`POST /predict`)
- **User authentication** — registration and login with `pbkdf2:sha256` password hashing, session-based auth, and a `login_required` decorator protecting prediction and history routes
- **Prediction history** — per-user log of all past predictions with timestamp, input parameters, result, and confidence; includes search/filter and a one-click clear-all function
- **Model evaluation dashboard** — serves pre-generated evaluation artifacts (ROC curve, confusion matrix, feature importance chart, classification report) from the `reports/` directory
- **Health check endpoint** — `GET /health` returns model/API status for monitoring
- **Automated ML training pipeline** — `train.py` handles data loading, missing value imputation, IQR outlier capping, feature engineering, correlation thresholding, SMOTE resampling, 10-fold stratified CV across 5 algorithms, hyperparameter tuning via `RandomizedSearchCV`, and report generation
- **Data analysis script** — `data_analysis.py` generates descriptive statistics, univariate distribution/box plots, and a multivariate correlation heatmap
- **Response caching and compression** — Flask-Caching (SimpleCache, 60s TTL on history) and Flask-Compress enabled at the application level
- **Load testing configuration** — a Locust test file (`locustfile.py`) with weighted tasks for `/health`, `/`, and `/history`

---

## Tech Stack

All entries below are sourced from `requirements.txt` and the actual imports in `app.py` / `train.py`.

| Layer | Technology | Version |
|-------|-----------|---------|
| **Backend** | Flask | 3.0.3 |
| | Flask-CORS | 4.0.0 |
| | Flask-SQLAlchemy | 3.1.1 |
| | Flask-Caching | 2.1.0 |
| | Flask-Compress | 1.15 |
| | python-dotenv | 1.0.1 |
| | gunicorn | 22.0.0 |
| **Database** | PostgreSQL (via psycopg2-binary) | 2.9.9 |
| **ML / Data** | scikit-learn | 1.4.2 |
| | XGBoost | 2.0.3 |
| | imbalanced-learn (SMOTE) | 0.12.2 |
| | pandas | 2.2.2 |
| | numpy | 1.26.4 |
| | joblib | 1.4.2 |
| **Visualization** | matplotlib | 3.8.4 |
| | seaborn | 0.13.2 |
| **Frontend** | Jinja2 templates, vanilla CSS, vanilla JS | — |
| **Typography** | Google Fonts (Inter) | — |
| **Runtime** | Python | 3.12.3 (per `.python-version`) |


```

The Flask app loads three serialized artifacts from disk at startup (`flood_model.pkl`, `preprocessor.pkl`, `feature_names.pkl`). On a prediction request, it constructs a DataFrame matching the training feature order, applies the same `StandardScaler` transform, runs `model.predict()` and `model.predict_proba()`, then persists the input and result to PostgreSQL via SQLAlchemy.

---

## ML Pipeline

The training pipeline is fully contained in `train.py` and runs independently of the Flask app.

### Dataset

- **Source file**: `data/flood_dataset_expanded.csv`
- **Size**: 955 rows, 11 columns
- **Raw features**: `Temp`, `Humidity`, `Cloud Cover`, `ANNUAL`, `Jan-Feb`, `Mar-May`, `Jun-Sep`, `Oct-Dec`, `avgjune`, `sub`
- **Target**: `flood` (binary: 0 = no flood, 1 = flood)
- **Class distribution**: approximately 45.7% positive (flood), 54.3% negative

### Pipeline Steps

1. **Missing value handling** — mean imputation for normally distributed numerics, median for skewed (|skewness| > 1), mode for categoricals
2. **Outlier capping** — IQR method (1.5× IQR bounds) applied to all numeric columns except the target
3. **Feature engineering** — two derived features:
   - `Total_Seasonal_Rainfall` = sum of `Jan-Feb`, `Mar-May`, `Jun-Sep`, `Oct-Dec`
   - `Humidity_Rainfall_Interaction` = (`Humidity` × `ANNUAL`) / 1000
4. **Column dropping** — `avgjune` and `sub` are explicitly dropped as unsupported by the frontend
5. **Correlation thresholding** — features correlated above 0.90 are removed, keeping the one more correlated with the target
6. **Train/test split** — 80/20 stratified split (`random_state=42`)
7. **Label encoding** — applied to any remaining categorical columns (fit on train only)
8. **Standard scaling** — `StandardScaler` fit on training data, transform applied to both splits
9. **Feature selection** — Random Forest importance; features with importance < 0.01 are dropped
10. **SMOTE** — applied when the minority class ratio falls below 35%
11. **Model evaluation** — 5 algorithms evaluated via 10-fold Stratified CV:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    - XGBoost
12. **Model selection** — best model chosen by highest mean ROC-AUC
13. **Hyperparameter tuning** — `RandomizedSearchCV` (50 iterations, 5-fold CV, `roc_auc` scoring) on the selected model
14. **Report generation** — confusion matrix heatmap, ROC curve, feature importance bar chart, and classification report text file saved to `reports/`

### Serialized Artifacts

| File | Contents |
|------|----------|
| `models/flood_model.pkl` | Tuned XGBoost classifier |
| `models/preprocessor.pkl` | Fitted `StandardScaler` |
| `models/feature_names.pkl` | Ordered list of feature column names |

### Test Set Performance

From the generated `reports/classification_report.txt` (191 test samples):

|  | Precision | Recall | F1-Score | Support |
|--|-----------|--------|----------|---------|
| Class 0 (No Flood) | 1.00 | 0.99 | 1.00 | 104 |
| Class 1 (Flood) | 0.99 | 1.00 | 0.99 | 87 |
| **Accuracy** | | | **0.99** | **191** |

---

## Database Schema

Four tables are defined as SQLAlchemy models in `app.py`. Tables are auto-created via `db.create_all()` at startup.

### `users`

| Column | Type | Constraints |
|--------|------|-------------|
| `UserID` | Integer | Primary Key |
| `Name` | String(100) | NOT NULL |
| `Email` | String(100) | UNIQUE, NOT NULL |
| `Password` | String(200) | NOT NULL (hashed) |
| `Role` | String(50) | NOT NULL, default `'User'` |

### `weather_data`

| Column | Type | Constraints |
|--------|------|-------------|
| `DataID` | Integer | Primary Key |
| `UserID` | Integer | FK → `users.UserID`, nullable |
| `AnnualRainfall` | Float | NOT NULL |
| `CloudVisibility` | Float | NOT NULL |
| `Temperature` | Float | NOT NULL |
| `Humidity` | Float | NOT NULL |
| `SeasonalRainfall` | Float | NOT NULL |

### `prediction_results`

| Column | Type | Constraints |
|--------|------|-------------|
| `PredictionID` | Integer | Primary Key |
| `DataID` | Integer | FK → `weather_data.DataID`, NOT NULL |
| `ModelID` | Integer | FK → `ml_models.ModelID`, nullable |
| `FloodResult` | String(50) | NOT NULL (`'HIGH'` or `'LOW'`) |
| `FloodProbability` | Float | NOT NULL |
| `PredictionDate` | DateTime | NOT NULL, default `datetime.now` |

### `ml_models`

| Column | Type | Constraints |
|--------|------|-------------|
| `ModelID` | Integer | Primary Key |
| `ModelName` | String(100) | NOT NULL |
| `AlgorithmType` | String(100) | NOT NULL |
| `Accuracy` | Float | nullable |
| `ModelFile` | String(255) | NOT NULL |

> **Note**: The `ml_models` table is defined but not populated by any current code path. `prediction_results.ModelID` is always `NULL` in practice since `save_prediction()` does not set it.

---

## API Endpoints

All routes are defined in `app.py`.

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `GET` | `/` | No | Landing page |
| `GET` | `/Predict` | Yes | Prediction input form |
| `GET` | `/reports` | No | Model evaluation dashboard |
| `GET` | `/history` | Yes | Prediction history page (cached 60s) |
| `GET` | `/login` | No | Login form |
| `POST` | `/login` | No | Authenticate user |
| `GET` | `/register` | No | Registration form |
| `POST` | `/register` | No | Create new user |
| `GET` | `/logout` | No | Clear session |
| `GET` | `/health` | No | API/model status check |
| `POST` | `/predict` | Yes | Submit weather data, receive flood prediction (JSON) |
| `GET` | `/api/history` | Yes | Prediction history as JSON |
| `POST` | `/api/history/clear` | Yes | Delete all predictions for current user |
| `GET` | `/reports/<filename>` | No | Serve report images from `reports/` directory |

### Prediction Request Format

**`POST /predict`** — Content-Type: `application/json`

```json
{
  "temp": 30.5,
  "humidity": 78.0,
  "cloud_cover": 42.0,
  "annual": 3200.5,
  "jan_feb": 15.2,
  "mar_may": 210.4,
  "jun_sep": 2500.5,
  "oct_dec": 450.3
}
```

**Validation rules** (enforced server-side in `app.py`):
- `temp`: -10 to 60 °C
- `humidity`: 0 to 100 %
- `cloud_cover`: 0 to 100 %
- `annual`, `jan_feb`, `mar_may`, `jun_sep`, `oct_dec`: 0 to 20,000 mm

**Response:**

```json
{
  "prediction": "HIGH",
  "probability": 0.8942,
  "message": "Flood Risk Detected! Immediate precautionary measures recommended."
}
```

---

## Project Structure

The repository is organized into numbered project phases. The application code lives under `5. Project Development Phase/code files/`.

```
Rising_water/
├── 1. Brainstorming & Ideation/
│   ├── Brainstorming & Idea Prioritization.pdf
│   ├── Define Problem Statements.pdf
│   └── Empathy Map.pdf
├── 2. Requirement Analysis/
│   ├── Customer Journey Map.pdf
│   ├── Solution Requirements.pdf
│   ├── Technology Stack.pdf
│   └── data flow diagram.pdf
├── 3. Project Design Phase/
│   ├── Problem-Solution Fit.pdf
│   ├── Proposed Solution.pdf
│   └── Solution Architecture.pdf
├── 4. Project Planning Phase/
│   └── Project Planning.pdf
├── 5. Project Development Phase/
│   ├── Code-Layout, Readability and Reusability.pdf
│   ├── Coding & Solution.pdf
│   ├── No. of Functional Features Included in the Solution.pdf
│   └── code files/
│       ├── app.py                      # Flask application (routes, models, prediction logic)
│       ├── train.py                    # Standalone ML training pipeline
│       ├── data_analysis.py            # Descriptive, univariate, multivariate analysis
│       ├── main.py                     # Alternate pipeline entry (references missing src/ modules)
│       ├── locustfile.py               # Locust load testing configuration
│       ├── requirements.txt            # Pinned Python dependencies
│       ├── .env.example                # Environment variable template
│       ├── .python-version             # Python 3.12.3
│       ├── .gitignore
│       ├── data/
│       │   ├── flood_dataset.xlsx      # Original dataset (Excel)
│       │   ├── flood_dataset_expanded.csv   # Expanded dataset (used by train.py)
│       │   └── flood_dataset_expanded.xlsx
│       ├── models/
│       │   ├── flood_model.pkl         # Serialized XGBoost classifier
│       │   ├── preprocessor.pkl        # Serialized StandardScaler
│       │   └── feature_names.pkl       # Feature column order
│       ├── reports/
│       │   ├── classification_report.txt
│       │   ├── confusion_matrix.png
│       │   ├── feature_importance.png
│       │   ├── roc_curve.png
│       │   ├── descriptive_analysis.txt
│       │   ├── multivariate_heatmap.png
│       │   ├── univariate_boxplots.png
│       │   └── univariate_distributions.png
│       ├── static/
│       │   ├── background.png
│       │   ├── home.css
│       │   ├── main.css
│       │   └── main.js
│       ├── templates/
│       │   ├── home.html               # Landing page
│       │   ├── index.html              # Prediction form
│       │   ├── history.html            # Prediction history table
│       │   ├── reports.html            # Model evaluation dashboard
│       │   ├── login.html              # Login form
│       │   ├── register.html           # Registration form
│       │   ├── chance.html             # High-risk result page (unused by current routes)
│       │   └── no_chance.html          # Low-risk result page (unused by current routes)
│       └── performance/               # JMeter, Locust, Postman test configs and reports
├── 6. Project Testing/
│   └── Performance Testing.pdf
├── 7. Project Documentation/
│   └── Sample documentation.pdf
├── 8. Project Demonstration/
│   ├── Communication.pdf
│   ├── Demonstration of Proposed Features.pdf
│   ├── Project Demo Planning.pdf
│   ├── Scalability & Future Plan.pdf
│   └── Team Involvement in Demonstration.pdf
└── README.md
```

---

## Setup and Installation

All commands assume you are working inside the `5. Project Development Phase/code files/` directory.

### 1. Clone the repository

```bash
git clone https://github.com/gireesh2658/Rising_water.git
cd "Rising_water/5. Project Development Phase/code files"
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

Requires Python 3.12+ (per `.python-version`). Install pinned dependencies:

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set **both** required variables:

```
FLASK_SECRET_KEY=<a-secure-random-string>
SQLALCHEMY_DATABASE_URI=postgresql://<user>:<password>@<host>/<database>?sslmode=require
```

> **Important**: The `.env.example` currently only documents `FLASK_SECRET_KEY`. The application will crash at startup without `SQLALCHEMY_DATABASE_URI` — this variable is required by `app.py` (line 41-43).

### 5. Train the model (optional)

Pre-trained artifacts are already present in `models/`. To retrain from scratch:

```bash
python train.py
```

This runs the full pipeline (load → preprocess → engineer features → evaluate 5 models → tune → save) and overwrites files in `models/` and `reports/`.

### 6. Start the Flask server

```bash
python app.py
```

The server starts on `http://0.0.0.0:5000/` (port configurable via `PORT` environment variable).

For production, gunicorn is included in the dependencies:

```bash
gunicorn app:app --bind 0.0.0.0:5000
```

---


*Built for the SmartBridge AI & ML Internship Track.*
