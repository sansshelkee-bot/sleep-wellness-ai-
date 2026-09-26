import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from catboost import CatBoostClassifier

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from ml.preprocess import (
    SleepDataPreprocessor, prepare_raw_dataframe, TARGET_CLASSES,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES
)
from config import Config


def train_and_evaluate_all():
    print("=" * 70)
    print("  INTELLIGENT SLEEP DISORDER PREDICTION - ML TRAINING PIPELINE")
    print("=" * 70)

    dataset_path = Config.DATASET_PATH
    if not dataset_path.exists():
        print(f"\n[ERROR] Dataset not found at: {dataset_path}")
        print("Please place 'Sleep_health_and_lifestyle_dataset.csv' into 'data/' directory.")
        return False

    print(f"\n[1/6] Loading dataset from: {dataset_path} ...")
    raw_df = pd.read_csv(dataset_path)
    print(f"      Total records loaded: {len(raw_df):,} rows x {len(raw_df.columns)} columns")

    # Clean & Prepare
    print("\n[2/6] Cleaning and preprocessing dataset ...")
    cleaned_df = prepare_raw_dataframe(raw_df)
    target_col = 'Sleep Disorder'

    if target_col not in cleaned_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    y_raw = cleaned_df[target_col]
    X_raw = cleaned_df.drop(columns=[target_col])

    # Fit preprocessor
    preprocessor = SleepDataPreprocessor()
    X_processed = preprocessor.fit_transform(X_raw)
    y_encoded = preprocessor.encode_target(y_raw)

    print(f"      Target class distribution:")
    for cls_name, idx in preprocessor.target_to_idx.items():
        count = int(np.sum(y_encoded == idx))
        pct = (count / len(y_encoded)) * 100
        print(f"        - {cls_name:<12}: {count:,} samples ({pct:.1f}%)")

    # Split dataset (80% train, 20% test, stratified)
    print("\n[3/6] Splitting dataset (80% Train, 20% Test, Stratified, random_state=42) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    print(f"      Training set: {X_train.shape[0]:,} samples")
    print(f"      Test set:     {X_test.shape[0]:,} samples")
    print(f"      Engineered Features: {X_train.shape[1]} features")

    # Define candidate models
    print("\n[4/6] Training & Evaluating candidate machine learning algorithms ...")
    cat_model = CatBoostClassifier(
        iterations=350,
        learning_rate=0.07,
        depth=6,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        random_seed=42,
        verbose=0
    )
    dt_model = DecisionTreeClassifier(max_depth=6, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=180, max_depth=12, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=180, learning_rate=0.07, max_depth=4, random_state=42)

    voting_model = VotingClassifier(
        estimators=[('cat', cat_model), ('rf', rf_model), ('gb', gb_model)],
        voting='soft'
    )

    models_dict = {
        'CatBoost': cat_model,
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model,
        'Decision Tree': dt_model,
        'Voting Ensemble': voting_model
    }

    metrics_results = {}
    best_f1 = -1.0
    best_model_name = 'CatBoost'
    best_model_obj = None

    print("-" * 88)
    print(f"{'Algorithm':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-" * 88)

    for name, model in models_dict.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Handle 2D output from CatBoost if present
        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()

        y_prob = model.predict_proba(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        rec = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        try:
            auc = float(roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted'))
        except Exception:
            auc = 0.0

        cm = confusion_matrix(y_test, y_pred).tolist()

        metrics_results[name] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(auc, 4),
            'confusion_matrix': cm
        }

        print(f"{name:<20} | {acc * 100:>8.2f}% | {prec * 100:>8.2f}% | {rec * 100:>8.2f}% | {f1 * 100:>8.2f}% | {auc:>10.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model

    print("-" * 88)
    print(f"\n[5/6] Deployment Model Selected: {best_model_name} (F1-Score: {best_f1 * 100:.2f}%)")

    # If CatBoost is primary model or close, CatBoost provides native SHAP TreeExplainer compatibility
    # Ensure CatBoost is also trained and saved as primary
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save CatBoost model natively
    cat_model_path = Config.MODELS_DIR / 'sleep_model.cbm'
    cat_model.save_model(str(cat_model_path))

    # Also save selected deployment model via joblib
    deployment_path = Config.MODELS_DIR / 'deployment_model.pkl'
    joblib.dump(best_model_obj, deployment_path)

    # Save preprocessor
    preprocessor_path = Config.MODELS_DIR / 'preprocessor.pkl'
    preprocessor.save(preprocessor_path)

    # Save metrics JSON
    metrics_path = Config.MODELS_DIR / 'model_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'selected_model': best_model_name,
            'best_f1_score': round(best_f1, 4),
            'target_classes': TARGET_CLASSES,
            'feature_names': preprocessor.feature_names_out,
            'total_samples': len(raw_df),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'models': metrics_results
        }, f, indent=4)

    # Save metadata JSON
    metadata_path = Config.MODELS_DIR / 'model_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_name': 'CatBoost Sleep Disorder Classifier',
            'version': '1.2.0',
            'architecture': 'Gradient Boosted Decision Trees',
            'target_classes': TARGET_CLASSES,
            'feature_count': len(preprocessor.feature_names_out),
            'numerical_features': NUMERICAL_FEATURES,
            'categorical_features': CATEGORICAL_FEATURES,
            'selected_algorithm': best_model_name,
            'frameworks': ['CatBoost 1.2', 'scikit-learn 1.9', 'SHAP 0.52'],
            'disclaimer': 'This system is intended for educational and wellness awareness purposes and does not replace professional medical diagnosis.'
        }, f, indent=4)

    print(f"\n[6/6] Saved Model Artifacts successfully:")
    print(f"      - Preprocessor:       {preprocessor_path}")
    print(f"      - CatBoost Model:     {cat_model_path}")
    print(f"      - Deployment Model:   {deployment_path}")
    print(f"      - Model Metrics JSON: {metrics_path}")
    print(f"      - Model Metadata:     {metadata_path}")
    print("\nTraining completed successfully!\n")
    return True


if __name__ == '__main__':
    train_and_evaluate_all()
