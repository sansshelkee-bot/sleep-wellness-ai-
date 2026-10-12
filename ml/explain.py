import shap
import numpy as np
from ml.predict import ModelPredictor
from ml.preprocess import FEATURE_DISPLAY_NAMES, TARGET_CLASSES

class ShapExplainerService:
    """
    Service for calculating real SHAP (SHapley Additive exPlanations) values
    using TreeExplainer on the trained CatBoost model.
    """
    _instance = None

    def __init__(self):
        self.predictor = ModelPredictor.get_instance()
        self.explainer = None
        self._init_explainer()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ShapExplainerService()
        return cls._instance

    def _init_explainer(self):
        if self.predictor.is_ready and self.predictor.model is not None:
            try:
                self.explainer = shap.TreeExplainer(self.predictor.model)
            except Exception as e:
                print(f"[ShapExplainerService] Error initializing TreeExplainer: {e}")
                self.explainer = None

    def explain(self, input_dict, prediction_result=None):
        """
        Calculates exact SHAP values for an input assessment.
        Returns sorted list of contributing factors for the predicted class.
        """
        if prediction_result is None:
            prediction_result = self.predictor.predict(input_dict)

        if self.explainer is None:
            self._init_explainer()
            if self.explainer is None:
                raise RuntimeError("SHAP TreeExplainer could not be initialized.")

        X_trans = prediction_result['features_processed']
        pred_idx = prediction_result['class_index']
        predicted_class = prediction_result['prediction']
        raw_inputs = prediction_result['norm_input']

        # Compute SHAP values
        sv = self.explainer.shap_values(X_trans)

        # Handle shape: (1, num_features, num_classes) or (num_classes, 1, num_features)
        if isinstance(sv, np.ndarray):
            if len(sv.shape) == 3:
                if sv.shape[0] == 1:
                    class_shap = sv[0, :, pred_idx]
                else:
                    class_shap = sv[pred_idx, 0, :]
            elif len(sv.shape) == 2:
                class_shap = sv[0, :]
            else:
                class_shap = sv.ravel()
        elif isinstance(sv, list):
            class_shap = sv[pred_idx][0]
        else:
            class_shap = np.array(sv).ravel()

        feature_names = self.predictor.preprocessor.feature_names_out

        # Aggregate one-hot features back into core domain feature categories
        aggregated_factors = {}
        for feat_name, shap_val in zip(feature_names, class_shap):
            # Check if this is a one-hot category, e.g. cat__Gender_Male or Occupation_Doctor
            core_name = feat_name
            if '__' in feat_name:
                core_name = feat_name.split('__')[1]
            
            # Map back to main feature
            found_group = None
            for cat_col in ['Gender', 'Occupation', 'BMI Category']:
                if core_name.startswith(cat_col):
                    found_group = cat_col
                    break
            
            group_key = found_group if found_group else core_name

            if group_key not in aggregated_factors:
                aggregated_factors[group_key] = 0.0
            # Accumulate absolute or signed shap
            aggregated_factors[group_key] += float(shap_val)

        # Rank factors by absolute SHAP impact
        factor_items = []
        max_abs_shap = max([abs(v) for v in aggregated_factors.values()] + [1e-5])

        for key, val in aggregated_factors.items():
            abs_val = abs(val)
            normalized_importance = round((abs_val / max_abs_shap) * 100, 1)

            # Determine contribution level
            if normalized_importance >= 65:
                level = "High"
            elif normalized_importance >= 30:
                level = "Moderate"
            else:
                level = "Low"

            # Direction: positive SHAP means this feature pushed the model TOWARDS predicting this class
            direction = "Increases Risk" if val > 0 else "Decreases Risk"

            # Formulate user-friendly explanation string
            display_name = FEATURE_DISPLAY_NAMES.get(key, key)
            user_val = raw_inputs.get(key, '')

            text = self._build_factor_narrative(key, user_val, val, predicted_class)

            factor_items.append({
                'feature_key': key,
                'feature_name': display_name,
                'feature_value': str(user_val),
                'shap_value': round(val, 4),
                'abs_shap': round(abs_val, 4),
                'relative_importance': normalized_importance,
                'contribution_level': level,
                'direction': direction,
                'explanation_text': text
            })

        # Sort descending by absolute SHAP value
        factor_items.sort(key=lambda x: x['abs_shap'], reverse=True)

        return {
            'predicted_class': predicted_class,
            'top_factors': factor_items,
            'disclaimer': 'These factors influenced the model\'s prediction. They do not represent a medical diagnosis.'
        }

    def _build_factor_narrative(self, feature_key, val, shap_val, predicted_class):
        direction_phrase = "increased" if shap_val > 0 else "reduced"
        
        if feature_key == 'Stress Level':
            return f"Your reported stress level ({val}/10) {direction_phrase} the likelihood of {predicted_class}."
        elif feature_key == 'Sleep Duration':
            return f"Sleep duration of {val} hrs/night {direction_phrase} the model's score for {predicted_class}."
        elif feature_key == 'Quality of Sleep':
            return f"Reported sleep quality rating of {val}/10 {direction_phrase} the prediction."
        elif feature_key == 'Systolic_BP' or feature_key == 'Diastolic_BP':
            return f"Blood pressure reading ({val} mmHg) {direction_phrase} cardiovascular strain linked with {predicted_class}."
        elif feature_key == 'BMI Category':
            return f"BMI classification of '{val}' was identified as a {direction_phrase} factor for {predicted_class}."
        elif feature_key == 'Physical Activity Level':
            return f"Daily activity level ({val} min) {direction_phrase} sleep efficiency and recovery score."
        elif feature_key == 'Heart Rate':
            return f"Resting heart rate of {val} bpm {direction_phrase} physiological tension metrics."
        elif feature_key == 'Daily Steps':
            return f"Daily step count of {val:,} steps {direction_phrase} daytime active energy expenditure."
        else:
            return f"{feature_key} ({val}) {direction_phrase} the model's confidence for {predicted_class}."


def explain_prediction(input_dict, prediction_result=None):
    """
    Convenience function for explainability.
    """
    explainer = ShapExplainerService.get_instance()
    return explainer.explain(input_dict, prediction_result)