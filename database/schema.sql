-- ==========================================================
-- Intelligent Sleep Disorder Prediction & Digital Twin System
-- Database Schema for MySQL 8.0+
-- Database: sleep_wellness_db
-- ==========================================================

CREATE DATABASE IF NOT EXISTS sleep_wellness_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE sleep_wellness_db;

-- 1. Users Table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(120) NOT NULL,
    email VARCHAR(120) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    age INT,
    gender VARCHAR(20),
    occupation VARCHAR(100),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_user_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. Sleep Assessments Table
CREATE TABLE IF NOT EXISTS sleep_assessments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    age INT NOT NULL,
    gender VARCHAR(20) NOT NULL,
    occupation VARCHAR(100) NOT NULL,
    sleep_duration FLOAT NOT NULL,
    quality_of_sleep INT NOT NULL,
    physical_activity_level FLOAT NOT NULL,
    stress_level INT NOT NULL,
    bmi_category VARCHAR(50) NOT NULL,
    blood_pressure VARCHAR(20) NOT NULL,
    systolic_bp INT,
    diastolic_bp INT,
    heart_rate INT NOT NULL,
    daily_steps INT NOT NULL,
    is_demo BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_assessment_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_assessment_user_date (user_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 3. Predictions Table
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    assessment_id INT NOT NULL,
    user_id INT NOT NULL,
    predicted_disorder VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    prob_none FLOAT NOT NULL,
    prob_insomnia FLOAT NOT NULL,
    prob_apnea FLOAT NOT NULL,
    model_version VARCHAR(50) DEFAULT 'CatBoost v1.2',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_prediction_assessment FOREIGN KEY (assessment_id) REFERENCES sleep_assessments(id) ON DELETE CASCADE,
    CONSTRAINT fk_prediction_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_prediction_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 4. Digital Twins Table
CREATE TABLE IF NOT EXISTS digital_twins (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL UNIQUE,
    current_sleep_duration FLOAT,
    current_quality_of_sleep INT,
    current_stress_level INT,
    current_activity_level FLOAT,
    current_daily_steps INT,
    current_bmi_category VARCHAR(50),
    current_blood_pressure VARCHAR(20),
    current_heart_rate INT,
    current_predicted_disorder VARCHAR(50) DEFAULT 'Pending Assessment',
    current_confidence FLOAT DEFAULT 0.0,
    current_health_score FLOAT DEFAULT 0.0,
    current_future_risk VARCHAR(20) DEFAULT 'Unassessed',
    total_assessments INT DEFAULT 0,
    baseline_health_score FLOAT DEFAULT 0.0,
    latest_facial_score FLOAT DEFAULT NULL,
    latest_facial_level VARCHAR(20) DEFAULT NULL,
    last_facial_assessment_date DATETIME DEFAULT NULL,
    last_assessment_date DATETIME,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_digital_twin_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_digital_twin_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 5. Recommendations Table
CREATE TABLE IF NOT EXISTS recommendations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    assessment_id INT NOT NULL,
    user_id INT NOT NULL,
    category VARCHAR(50) NOT NULL,
    recommendation_text TEXT NOT NULL,
    priority VARCHAR(20) DEFAULT 'Medium',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_recommendation_assessment FOREIGN KEY (assessment_id) REFERENCES sleep_assessments(id) ON DELETE CASCADE,
    CONSTRAINT fk_recommendation_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_recommendation_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 6. SHAP Explanations Table
CREATE TABLE IF NOT EXISTS shap_explanations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    assessment_id INT NOT NULL,
    user_id INT NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    feature_value VARCHAR(100),
    shap_value FLOAT NOT NULL,
    contribution_level VARCHAR(20) NOT NULL,
    direction VARCHAR(30) NOT NULL,
    explanation_text TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_shap_assessment FOREIGN KEY (assessment_id) REFERENCES sleep_assessments(id) ON DELETE CASCADE,
    CONSTRAINT fk_shap_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_shap_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 7. Future Risk Table
CREATE TABLE IF NOT EXISTS future_risk (
    id INT AUTO_INCREMENT PRIMARY KEY,
    assessment_id INT NOT NULL,
    user_id INT NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    risk_score FLOAT NOT NULL,
    risk_trajectory VARCHAR(50) DEFAULT 'Stable',
    primary_risk_factors TEXT,
    preventive_guidance TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_risk_assessment FOREIGN KEY (assessment_id) REFERENCES sleep_assessments(id) ON DELETE CASCADE,
    CONSTRAINT fk_risk_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_risk_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 8. Progress Records Table
CREATE TABLE IF NOT EXISTS progress_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    assessment_id INT NOT NULL,
    record_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    health_score FLOAT NOT NULL,
    sleep_duration FLOAT NOT NULL,
    quality_of_sleep INT NOT NULL,
    stress_level INT NOT NULL,
    physical_activity_level FLOAT NOT NULL,
    daily_steps INT NOT NULL,
    predicted_disorder VARCHAR(50) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    CONSTRAINT fk_progress_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT fk_progress_assessment FOREIGN KEY (assessment_id) REFERENCES sleep_assessments(id) ON DELETE CASCADE,
    INDEX idx_progress_user_date (user_id, record_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 9. Facial Sleepiness Assessments Table
CREATE TABLE IF NOT EXISTS facial_assessments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    assessment_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    sleepiness_score FLOAT NOT NULL,
    sleepiness_level VARCHAR(20) NOT NULL,
    blink_count INT DEFAULT 0,
    blink_rate_per_min FLOAT DEFAULT 0.0,
    avg_ear FLOAT DEFAULT 0.0,
    perclos_percentage FLOAT DEFAULT 0.0,
    yawn_count INT DEFAULT 0,
    max_mar FLOAT DEFAULT 0.0,
    head_nod_count INT DEFAULT 0,
    head_movement_detected BOOLEAN DEFAULT FALSE,
    duration_seconds FLOAT DEFAULT 15.0,
    observed_indicators TEXT,
    contributing_factors TEXT,
    recommendation TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_facial_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_facial_user_date (user_id, assessment_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 10. Doctor Consultations Table
CREATE TABLE IF NOT EXISTS doctor_consultations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    doctor_name VARCHAR(100) NOT NULL,
    specialty VARCHAR(100) NOT NULL,
    hospital_clinic VARCHAR(150) NOT NULL,
    consultation_type VARCHAR(50) DEFAULT 'Virtual Video Teleconsultation',
    appointment_date DATETIME NOT NULL,
    time_slot VARCHAR(50) NOT NULL,
    status VARCHAR(30) DEFAULT 'Confirmed',
    patient_notes TEXT,
    digital_twin_snapshot TEXT,
    meeting_link VARCHAR(255),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_consultation_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_consultation_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
