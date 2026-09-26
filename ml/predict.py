import os
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from config import Config
from ml.preprocess import (
    SleepDataPreprocessor, prepare_raw_dataframe, TARGET_CLASSES,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES
)

class ModelPredictor:
    """
    Singleton predictor that loads the preprocessor and trained CatBoost model
    to perform validated inference and probability scoring.
    """
    _instance = None

    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.is_ready = False
        self._load_artifacts()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelPredictor()
        return cls._instance

    def _load_artifacts(self):
        try:
            if Config.PREPROCESSOR_PATH.exists():
                self.preprocessor = SleepDataPreprocessor.load(Config.PREPROCESSOR_PATH)
            
            # Prefer CatBoost model (.cbm) for native probability and SHAP TreeExplainer
            if Config.MODEL_PATH.exists():
                cat_model = CatBoostClassifier()
                cat_model.load_model(str(Config.MODEL_PATH))
                self.model = cat_model
                self.is_ready = True
            elif (Config.MODELS_DIR / 'deployment_model.pkl').exists():
                self.model = joblib.load(Config.MODELS_DIR / 'deployment_model.pkl')
                self.is_ready = True
            else:
                self.is_ready = False
        except Exception as e:
            print(f"[ModelPredictor] Warning loading artifacts: {e}")
            self.is_ready = False

    def predict(self, input_dict):
        """
        Accepts dictionary containing user assessment features:
        age, gender, occupation, sleep_duration, quality_of_sleep,
        physical_activity_level, stress_level, bmi_category,
        blood_pressure, heart_rate, daily_steps

        Returns:
        {
            "prediction": "Sleep Apnea",
            "confidence": 0.92,
            "probabilities": {
                "None": 0.03,
                "Insomnia": 0.05,
                "Sleep Apnea": 0.92
            },
            "class_index": 2
        }
        """
        if not self.is_ready:
            self._load_artifacts()
            if not self.is_ready:
                raise RuntimeError("Machine Learning models are not trained yet. Run python ml/train_model.py first.")

        # Standardize keys to match preprocessor expected format
        norm_dict = {
            'Age': float(input_dict.get('age', input_dict.get('Age', 35))),
            'Gender': str(input_dict.get('gender', input_dict.get('Gender', 'Male'))).title(),
            'Occupation': str(input_dict.get('occupation', input_dict.get('Occupation', 'Doctor'))),
            'Sleep Duration': float(input_dict.get('sleep_duration', input_dict.get('Sleep Duration', 7.0))),
            'Quality of Sleep': int(input_dict.get('quality_of_sleep', input_dict.get('Quality of Sleep', 7))),
            'Physical Activity Level': float(input_dict.get('physical_activity_level', input_dict.get('Physical Activity Level', 50))),
            'Stress Level': int(input_dict.get('stress_level', input_dict.get('Stress Level', 5))),
            'BMI Category': str(input_dict.get('bmi_category', input_dict.get('BMI Category', 'Normal'))),
            'Blood Pressure': str(input_dict.get('blood_pressure', input_dict.get('Blood Pressure', '120/80'))),
            'Heart Rate': int(input_dict.get('heart_rate', input_dict.get('Heart Rate', 72))),
            'Daily Steps': int(input_dict.get('daily_steps', input_dict.get('Daily Steps', 7000)))
        }

        # Transform single input
        X_trans = self.preprocessor.transform_single_dict(norm_dict)

        # Predict probabilities
        probs = self.model.predict_proba(X_trans)[0]

        # Map to class labels
        class_probs = {}
        for idx, cls_name in enumerate(TARGET_CLASSES):
            class_probs[cls_name] = round(float(probs[idx]), 4)

        pred_idx = int(np.argmax(probs))
        predicted_class = TARGET_CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        return {
            'prediction': predicted_class,
            'confidence': round(confidence, 4),
            'probabilities': class_probs,
            'class_index': pred_idx,
            'features_processed': X_trans,
            'norm_input': norm_dict
        }


def predict_sleep_disorder(data_dict):
    """
    Convenience function for inference.
    """
    predictor = ModelPredictor.get_instance()
    return predictor.predict(data_dict)
