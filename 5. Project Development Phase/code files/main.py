import os
from src.data_preprocessing import preprocess_pipeline
from src.model_training import train_and_select_best
from src.model_evaluation import evaluate_and_plot

def main():
    print("=" * 70)
    print("  ULTRA PRO MAX ML PIPELINE EXECUTION")
    print("=" * 70)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'flood_dataset.xlsx')
    VIZ_DIR = os.path.join(BASE_DIR, 'visualizations')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    os.makedirs(VIZ_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Data Preprocessing (Imputation, Outliers, SMOTE, Scaling)
    X_train_scaled, X_test_scaled, y_train_sm, y_test, feature_names = preprocess_pipeline(
        DATASET_PATH, MODEL_DIR
    )
    
    # 2. Model Training (GridSearch CV, Feature Importance)
    models_dict = train_and_select_best(
        X_train_scaled, y_train_sm, feature_names, VIZ_DIR, MODEL_DIR
    )
    
    # 3. Model Evaluation (ROC-AUC, Confusion Matrix, Classification Report)
    evaluate_and_plot(models_dict, X_test_scaled, y_test, VIZ_DIR, MODEL_DIR)
    
    print("=" * 70)
    print("  PIPELINE EXECUTION COMPLETE. READY FOR FLASK API.")
    print("=" * 70)

if __name__ == "__main__":
    main()
