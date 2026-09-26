import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Load .env file
load_dotenv(BASE_DIR / '.env')

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'default-dev-secret-key-sleep-twin-2026')
    
    # MySQL Database credentials
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '3306')
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'admin')
    DB_NAME = os.getenv('DB_NAME', 'sleep_wellness_db')
    
    # Try MySQL URI first
    # URL encode password if it contains special characters
    from urllib.parse import quote_plus
    safe_password = quote_plus(DB_PASSWORD) if DB_PASSWORD else ''
    
    SQLALCHEMY_DATABASE_URI = (
        f"mysql+pymysql://{DB_USER}:{safe_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
    )
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_recycle': 280,
        'pool_pre_ping': True
    }
    
    # Application Paths
    DATA_DIR = BASE_DIR / 'data'
    MODELS_DIR = BASE_DIR / 'models'
    DATASET_PATH = DATA_DIR / 'Sleep_health_and_lifestyle_dataset.csv'
    
    MODEL_PATH = MODELS_DIR / 'sleep_model.cbm'
    PREPROCESSOR_PATH = MODELS_DIR / 'preprocessor.pkl'
    METRICS_PATH = MODELS_DIR / 'model_metrics.json'
    METADATA_PATH = MODELS_DIR / 'model_metadata.json'
    
    # Session Configuration
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = 86400  # 24 hours
