# 🌊 Rising Waters: A Machine Learning Approach to Flood Prediction

![Rising Waters Dashboard](https://via.placeholder.com/1200x600?text=Rising+Waters+Dashboard+Screenshot)

Welcome to the **Rising Waters** machine learning prediction system. This repository contains a production-grade, end-to-end ML pipeline designed to predict flood risk based on historical rainfall and atmospheric data. 

This project was built from the ground up for the **SmartBridge AI & ML Internship Track** and has been rigorously audited to meet senior-level industry standards.

---

## 🏆 Project Features

- **End-to-End ML Pipeline**: A standalone `train.py` script that automatically loads data, handles missing values (Mean/Median/Mode), caps outliers via IQR, engineers derived features, and outputs trained artifacts.
- **Advanced Feature Engineering**: 
  - Automatically derives **Total Seasonal Rainfall** and **Humidity-Rainfall Interaction**.
  - Applies Random Forest Feature Importance and drops weak features.
  - Mitigates multicollinearity by dropping features correlated > 0.90.
- **Class Imbalance Handling**: Automatically applies **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the model doesn't overfit to the majority class.
- **Hyperparameter Tuning**: `RandomizedSearchCV` across 5-fold cross-validation on the best model.
- **Comprehensive Evaluation**: Generates Confusion Matrices, Classification Reports, and ROC-AUC curves stored dynamically in the `/reports` directory.
- **Production Backend**: A fully functioning modular **Flask REST API** (`/predict`, `/health`) serving the exact `joblib` pre-trained model globally in memory.
- **Decoupled Frontend**: A stunning, responsive Glassmorphism UI built with vanilla HTML/CSS/JS that consumes the API via asynchronous JSON `fetch()` requests without reloading the page.

---

## 📊 Model Performance Metrics

Five algorithms were evaluated using **10-fold Stratified Cross-Validation**. The evaluation pipeline automatically selected **XGBoost** as the superior model. 

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.8606 ± 0.0401 | 0.4500 ± 0.4975 | 0.3500 ± 0.4500 | 0.3833 ± 0.4524 | 0.8252 |
| Decision Tree | 0.9136 ± 0.0760 | 0.5333 ± 0.4989 | 0.5000 ± 0.5000 | 0.5167 ± 0.5022 | 0.7323 |
| Random Forest | 0.9220 ± 0.0764 | 0.5000 ± 0.5000 | 0.4500 ± 0.4975 | 0.4667 ± 0.4989 | 0.9621 |
| Gradient Boosting | 0.9220 ± 0.0894 | 0.6000 ± 0.4899 | 0.5000 ± 0.5000 | 0.5333 ± 0.5055 | 0.9470 |
| **XGBoost (Best)** | **0.9576 ± 0.0652** | **0.8000 ± 0.4000** | **0.7000 ± 0.4583** | **0.7333 ± 0.4216** | **0.9621** |

*(Scores represent the Mean ± Std Dev across the 10 folds before final hyperparameter tuning).*

---

## 🛠️ Tech Stack

- **Machine Learning**: `scikit-learn`, `xgboost`, `imbalanced-learn`, `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Backend API**: `Flask`, `Flask-CORS`, `joblib`, `python-dotenv`
- **Frontend UI**: Vanilla HTML5, Vanilla CSS3 (Glassmorphism design), Vanilla JavaScript (`fetch` API)

---

## 📂 Folder Structure

```text
Rising-Waters-Project/
│
├── data/
│   └── flood_dataset.xlsx          # Raw dataset
├── models/
│   ├── flood_model.pkl             # Serialized XGBoost model
│   ├── preprocessor.pkl            # Serialized StandardScaler
│   └── feature_names.pkl           # Saved column order for API
├── reports/
│   ├── classification_report.txt   # Test set metrics
│   ├── confusion_matrix.png        # CM heatmap
│   ├── feature_importance.png      # Top 10 features
│   └── roc_curve.png               # ROC AUC plot
├── static/
│   └── background.png              # UI background assets
├── templates/
│   ├── index.html                  # Main Prediction UI
│   └── reports.html                # Model Dashboard UI
│
├── app.py                          # Flask API and server
├── train.py                        # Standalone ML training pipeline
├── requirements.txt                # Pinned dependencies
├── .env.example                    # Environment variable template
└── README.md                       # This file
```

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/rising-waters.git
cd rising-waters
```

### 2. Install dependencies
Ensure you are using Python 3.9+. Install the exact pinned dependencies:
```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables
```bash
cp .env.example .env
```
Open `.env` and set `FLASK_SECRET_KEY` to a secure random string.

### 4. Train the Model (Optional)
The pre-trained model files are likely already in `models/`. If you want to retrain the pipeline from scratch:
```bash
python train.py
```
*This will process the data, evaluate the 5 models, run RandomizedSearchCV, and overwrite the artifacts in `models/` and `reports/`.*

### 5. Start the Flask Server
```bash
python app.py
```
The server will start on `http://127.0.0.1:5000/`.

---

## 📡 API Usage

The backend provides a RESTful `/predict` endpoint that accepts JSON.

### Sample `cURL` Request
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "temp": 30.5,
           "humidity": 78.0,
           "cloud_cover": 42.0,
           "annual": 3200.5,
           "jan_feb": 15.2,
           "mar_may": 210.4,
           "jun_sep": 2500.5,
           "oct_dec": 450.3
         }'
```

### Sample JSON Response
```json
{
  "prediction": "HIGH",
  "probability": 0.8942,
  "message": "Flood Risk Detected! Immediate precautionary measures recommended."
}
```

---

*Built with ❤️ for the SmartBridge Internship Program.*
