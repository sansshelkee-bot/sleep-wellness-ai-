import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Canonical target classes
TARGET_CLASSES = ['None', 'Insomnia', 'Sleep Apnea']
TARGET_MAP = {
    'Healthy': 'None',
    'None': 'None',
    'Insomnia': 'Insomnia',
    'Sleep Apnea': 'Sleep Apnea'
}

NUMERICAL_FEATURES = [
    'Age',
    'Sleep Duration',
    'Quality of Sleep',
    'Physical Activity Level',
    'Stress Level',
    'Heart Rate',
    'Daily Steps',
    'Systolic_BP',
    'Diastolic_BP'
]

CATEGORICAL_FEATURES = [
    'Gender',
    'Occupation',
    'BMI Category'
]

FEATURE_DISPLAY_NAMES = {
    'Age': 'Age',
    'Sleep Duration': 'Sleep Duration (hrs)',
    'Quality of Sleep': 'Sleep Quality (1-10)',
    'Physical Activity Level': 'Physical Activity (mins/day)',
    'Stress Level': 'Stress Level (1-10)',
    'Heart Rate': 'Resting Heart Rate (bpm)',
    'Daily Steps': 'Daily Steps',
    'Systolic_BP': 'Systolic Blood Pressure (mmHg)',
    'Diastolic_BP': 'Diastolic Blood Pressure (mmHg)',
    'Gender': 'Biological Gender',
    'Occupation': 'Occupation',
    'BMI Category': 'BMI Category'
}


def parse_blood_pressure(bp_val):
    """
    Parses '120/80' into (120, 80).
    """
    if pd.isna(bp_val):
        return 120, 80
    match = re.search(r'(\d+)\s*/\s*(\d+)', str(bp_val))
    if match:
        return int(match.group(1)), int(match.group(2))
    return 120, 80


def normalize_bmi_category(bmi_val):
    """
    Normalizes 'Normal Weight' to 'Normal'.
    """
    if pd.isna(bmi_val):
        return 'Normal'
    val = str(bmi_val).strip()
    if val.lower() in ['normal weight', 'normalweight']:
        return 'Normal'
    return val


def prepare_raw_dataframe(df):
    """
    Prepares raw dataframe by standardizing columns, parsing blood pressure,
    normalizing BMI category, and handling missing values.
    """
    df_clean = df.copy()

    # Normalize column names: strip spaces
    df_clean.columns = [c.strip() for c in df_clean.columns]

    # Map possible alternative column names
    col_rename = {
        'Sleep_Duration': 'Sleep Duration',
        'Quality_of_Sleep': 'Quality of Sleep',
        'Physical_Activity_Level': 'Physical Activity Level',
        'Stress_Level': 'Stress Level',
        'BMI_Category': 'BMI Category',
        'Blood_Pressure': 'Blood Pressure',
        'Heart_Rate': 'Heart Rate',
        'Daily_Steps': 'Daily Steps',
        'Sleep_Disorder': 'Sleep Disorder'
    }
    df_clean.rename(columns=col_rename, inplace=True)

    # Parse Blood Pressure into Systolic and Diastolic
    if 'Blood Pressure' in df_clean.columns:
        parsed_bp = df_clean['Blood Pressure'].apply(parse_blood_pressure)
        df_clean['Systolic_BP'] = [p[0] for p in parsed_bp]
        df_clean['Diastolic_BP'] = [p[1] for p in parsed_bp]
    elif 'Systolic_BP' not in df_clean.columns:
        df_clean['Systolic_BP'] = 120
        df_clean['Diastolic_BP'] = 80

    # Normalize BMI Category
    if 'BMI Category' in df_clean.columns:
        df_clean['BMI Category'] = df_clean['BMI Category'].apply(normalize_bmi_category)

    # Normalize Target if present
    if 'Sleep Disorder' in df_clean.columns:
        df_clean['Sleep Disorder'] = df_clean['Sleep Disorder'].apply(
            lambda x: TARGET_MAP.get(str(x).strip(), 'None') if pd.notna(x) else 'None'
        )

    # Drop identifier columns
    for id_col in ['Person ID', 'Person_ID', 'ID', 'id']:
        if id_col in df_clean.columns:
            df_clean.drop(columns=[id_col], inplace=True)

    return df_clean


class SleepDataPreprocessor:
    """
    Production-ready Preprocessing Pipeline combining Scikit-Learn ColumnTransformer
    with metadata tracking for seamless model training, inference, and SHAP explainability.
    """
    def __init__(self):
        self.numerical_features = NUMERICAL_FEATURES
        self.categorical_features = CATEGORICAL_FEATURES
        self.target_classes = TARGET_CLASSES
        self.target_to_idx = {cls_name: i for i, cls_name in enumerate(TARGET_CLASSES)}
        self.idx_to_target = {i: cls_name for i, cls_name in enumerate(TARGET_CLASSES)}
        
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_features)
            ],
            remainder='drop'
        )
        self.feature_names_out = None
        self.is_fitted = False

    def fit(self, X_df, y=None):
        X_clean = prepare_raw_dataframe(X_df)
        self.column_transformer.fit(X_clean[self.numerical_features + self.categorical_features])
        
        # Build feature names out for explainability
        cat_encoder = self.column_transformer.named_transformers_['cat']
        cat_feature_names = cat_encoder.get_feature_names_out(self.categorical_features).tolist()
        self.feature_names_out = self.numerical_features + cat_feature_names
        self.is_fitted = True
        return self

    def transform(self, X_df):
        if not self.is_fitted:
            raise ValueError("SleepDataPreprocessor has not been fitted yet.")
        X_clean = prepare_raw_dataframe(X_df)
        X_transformed = self.column_transformer.transform(X_clean[self.numerical_features + self.categorical_features])
        return X_transformed

    def fit_transform(self, X_df, y=None):
        self.fit(X_df, y)
        return self.transform(X_df)

    def encode_target(self, y_series):
        y_cleaned = y_series.apply(lambda x: TARGET_MAP.get(str(x).strip(), 'None') if pd.notna(x) else 'None')
        return np.array([self.target_to_idx[val] for val in y_cleaned])

    def decode_target(self, y_indices):
        return [self.idx_to_target[idx] for idx in y_indices]

    def transform_single_dict(self, input_dict):
        """
        Accepts a dict of single user input and returns preprocessed 2D numpy array.
        """
        df_single = pd.DataFrame([input_dict])
        return self.transform(df_single)

    def save(self, filepath):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)
