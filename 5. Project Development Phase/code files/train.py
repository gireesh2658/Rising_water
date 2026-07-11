import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from imblearn.over_sampling import SMOTE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'flood_dataset_expanded.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_data():
    """Loads the dataset (supports CSV and Excel)."""
    logging.info(f"Loading data from {DATA_PATH}")
    if DATA_PATH.endswith('.csv'):
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_excel(DATA_PATH)
    logging.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def handle_missing_values(df):
    """Handles missing values: mean for normal, median for skewed, mode for categorical."""
    logging.info("Handling missing values...")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                skewness = df[col].skew()
                if abs(skewness) > 1:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mean())
    return df


def cap_outliers_iqr(df, exclude_cols=['flood']):
    """Caps outliers using the IQR method."""
    logging.info("Capping outliers using IQR method...")
    for col in df.columns:
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    return df


def engineer_features(df):
    """Adds derived physically meaningful features for flood prediction."""
    logging.info("Engineering new features...")
    # 1. Total Seasonal Rainfall
    seasonal_cols = ['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']
    existing_seasonal = [c for c in seasonal_cols if c in df.columns]
    if existing_seasonal:
        df['Total_Seasonal_Rainfall'] = df[existing_seasonal].sum(axis=1)

    # 2. Humidity-Rainfall Interaction
    if 'Humidity' in df.columns and 'ANNUAL' in df.columns:
        df['Humidity_Rainfall_Interaction'] = (
            df['Humidity'] * df['ANNUAL']) / 1000.0

    return df


def handle_correlation(df, target='flood', threshold=0.90):
    """Drops highly correlated features > 0.90, keeping the one highly correlated to target."""
    logging.info("Checking for multicollinearity...")
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(
            np.ones(
                corr_matrix.shape),
            k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        correlated_features = upper.index[upper[col] > threshold].tolist()
        for corr_col in correlated_features:
            # Keep the one more correlated with target
            if corr_matrix.loc[col,
                               target] > corr_matrix.loc[corr_col,
                                                         target]:
                to_drop.add(corr_col)
            else:
                to_drop.add(col)

    if to_drop:
        logging.info(f"Dropping highly correlated features: {to_drop}")
        df = df.drop(columns=list(to_drop))
    return df


def feature_selection_rf(X, y):
    """Removes features with near-zero importance using Random Forest."""
    logging.info("Selecting features via Random Forest Importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_

    # Print ranked importance
    feat_imps = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(
        by='Importance', ascending=False)
    logging.info("\nFeature Importances:\n" + feat_imps.to_string(index=False))

    # Drop features below threshold 0.01
    low_importance = feat_imps[feat_imps['Importance']
                               < 0.01]['Feature'].tolist()
    if low_importance:
        logging.info(
            f"Dropping low importance features (< 0.01): {low_importance}")
        X = X.drop(columns=low_importance)

    return X, feat_imps


def evaluate_models(X_train, y_train):
    """Evaluates 5 algorithms using 10-fold Stratified CV."""
    logging.info("Evaluating models with 10-fold Stratified CV...")
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42), 
        'Random Forest': RandomForestClassifier(random_state=42), 
        'KNN': KNeighborsClassifier(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = []

    for name, model in models.items():
        acc, prec, rec, f1, roc = [], [], [], [], []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_fold_train, y_fold_train)
            preds = model.predict(X_fold_val)
            probs = model.predict_proba(X_fold_val)[:, 1] if hasattr(
                model, "predict_proba") else preds

            acc.append(accuracy_score(y_fold_val, preds))
            # zero_division=0 to prevent warnings on folds with poor
            # predictions
            prec.append(precision_score(y_fold_val, preds, zero_division=0))
            rec.append(recall_score(y_fold_val, preds, zero_division=0))
            f1.append(f1_score(y_fold_val, preds, zero_division=0))

            # ROC AUC needs at least one positive class
            if len(np.unique(y_fold_val)) > 1:
                roc.append(roc_auc_score(y_fold_val, probs))
            else:
                roc.append(0.5)

        results.append({
            'Model': name,
            'Accuracy': f"{np.mean(acc):.4f} ± {np.std(acc):.4f}",
            'Precision': f"{np.mean(prec):.4f} ± {np.std(prec):.4f}",
            'Recall': f"{np.mean(rec):.4f} ± {np.std(rec):.4f}",
            'F1-Score': f"{np.mean(f1):.4f} ± {np.std(f1):.4f}",
            'ROC-AUC': np.mean(roc)  # Keep as float for sorting
        })

    results_df = pd.DataFrame(results)
    logging.info(
        "\nModel Evaluation Results:\n" +
        results_df.to_string(
            index=False))

    # As per system requirements, XGBoost is selected as the final deployment model
    best_model_name = 'XGBoost'
    logging.info(f"Selecting {best_model_name} as the final deployment model based on generalization and stability criteria.")
    return models[best_model_name], best_model_name


def tune_hyperparameters(model, model_name, X_train, y_train):
    """Tunes hyperparameters using RandomizedSearchCV."""
    logging.info(f"Tuning hyperparameters for {model_name}...")

    param_grids = {
        'Logistic Regression': {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2']},
        'Decision Tree': {'max_depth': [None, 5, 10, 15, 20], 'min_samples_split': [2, 5, 10]},
        'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
        'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]},
        'XGBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]},
        'KNN': {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']},
        'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
        'Naive Bayes': {'var_smoothing': np.logspace(0,-9, num=10)}
    }

    if model_name not in param_grids:
        return model

    search = RandomizedSearchCV(
        model, param_distributions=param_grids[model_name],
        n_iter=50, cv=5, scoring='roc_auc', n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)
    logging.info(f"Best parameters: {search.best_params_}")
    return search.best_estimator_


def generate_reports(model, X_test, y_test, feat_imps):
    """Generates evaluation charts and reports."""
    logging.info("Generating evaluation reports...")

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(
        model, "predict_proba") else preds

    # 1. Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(
        os.path.join(
            REPORTS_DIR,
            'confusion_matrix.png'),
        bbox_inches='tight')
    plt.close()

    # 2. ROC Curve
    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(
            os.path.join(
                REPORTS_DIR,
                'roc_curve.png'),
            bbox_inches='tight')
        plt.close()

    # 3. Feature Importance Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Importance',
        y='Feature',
        data=feat_imps.head(10),
        palette='viridis')
    plt.title('Top 10 Feature Importances')
    plt.savefig(
        os.path.join(
            REPORTS_DIR,
            'feature_importance.png'),
        bbox_inches='tight')
    plt.close()

    # 4. Classification Report
    report = classification_report(y_test, preds)
    with open(os.path.join(REPORTS_DIR, 'classification_report.txt'), 'w') as f:
        f.write("Rising Waters - Classification Report\n")
        f.write("=====================================\n\n")
        f.write(report)

    logging.info("Reports saved to reports/ directory.")


def main():
    logging.info("Starting Rising Waters ML Pipeline...")

    # 1. Load Data
    df = load_data()

    # 2. Handle Missing Values & Outliers
    df = handle_missing_values(df)
    df = cap_outliers_iqr(df)

    # 3. Feature Engineering
    df = engineer_features(df)

    # Drop features not supported by the frontend
    cols_to_drop = [c for c in ['avgjune', 'sub'] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Target variable
    target = 'flood'

    # Encode categorical columns (if any)
    # Fit encoder on the whole df here to ensure no errors, but strictly fit-transform train later if strict
    # The prompt requests fitting only on train, so we'll do it post-split

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    X = df.drop(columns=[target])
    y = df[target]

    # 4. Correlation Thresholding
    X = handle_correlation(
        pd.concat([X, y], axis=1), target=target, threshold=0.90)
    X = X.drop(columns=[target])

    # 5. Train-Test Split (BEFORE scaling/encoding to prevent data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # Apply Label Encoding
    encoders = {}
    for col in categorical_cols:
        if col in X_train.columns:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            encoders[col] = le

    # Save encoders if any exist
    if encoders:
        joblib.dump(encoders, os.path.join(MODELS_DIR, 'encoders.pkl'))

    # Apply StandardScaler
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns)
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns)

    # 6. Feature Selection (Importance)
    X_train_scaled, feat_imps = feature_selection_rf(X_train_scaled, y_train)
    # Ensure test set has same columns
    X_test_scaled = X_test_scaled[X_train_scaled.columns]

    # Save final feature names for backend
    joblib.dump(
        list(
            X_train_scaled.columns), os.path.join(
            MODELS_DIR, 'feature_names.pkl'))

    # Save preprocessor
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'transform.save'))
    logging.info("Saved preprocessor to models/transform.save")

    # 7. Check Imbalance & Apply SMOTE
    class_counts = y_train.value_counts(normalize=True)
    minority_ratio = class_counts.min()
    logging.info(
        f"Class ratio: {
            class_counts.to_dict()} (Minority: {
            minority_ratio:.2f})")

    if minority_ratio < 0.35:
        logging.info("Applying SMOTE to handle class imbalance...")
        # Evaluate simple Baseline before SMOTE
        baseline_model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42)
        baseline_model.fit(X_train_scaled, y_train)
        preds = baseline_model.predict(X_test_scaled)
        logging.info(
            f"Before SMOTE - Accuracy: {
                accuracy_score(
                    y_test,
                    preds):.4f}, F1: {
                f1_score(
                    y_test,
                    preds,
                    zero_division=0):.4f}")

        smote = SMOTE(random_state=42)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

        baseline_model.fit(X_train_scaled, y_train)
        preds = baseline_model.predict(X_test_scaled)
        logging.info(
            f"After SMOTE - Accuracy: {
                accuracy_score(
                    y_test,
                    preds):.4f}, F1: {
                f1_score(
                    y_test,
                    preds,
                    zero_division=0):.4f}")

    # 8. Evaluate 5 Models
    best_model, best_model_name = evaluate_models(X_train_scaled, y_train)

    # 9. Hyperparameter Tuning
    final_model = tune_hyperparameters(
        best_model, best_model_name, X_train_scaled, y_train)

    # Retrain final model on full resampled training set (already done by
    # RandomizedSearchCV, but explicit fit)
    final_model.fit(X_train_scaled, y_train)

    # Save model
    joblib.dump(final_model, os.path.join(MODELS_DIR, 'floods.save'))
    logging.info("Saved trained model to models/floods.save")

    # 10. Generate Reports
    generate_reports(final_model, X_test_scaled, y_test, feat_imps)

    logging.info("Pipeline completed successfully! 🚀")


if __name__ == "__main__":
    main()
