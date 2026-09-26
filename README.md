# Intelligent Sleep Disorder Prediction and Digital Twin-Based Personalized Sleep Wellness System

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Backend-Flask%203.1-black.svg)](https://flask.palletsprojects.com/)
[![CatBoost](https://img.shields.io/badge/ML-CatBoost%20%7C%20Ensemble-yellow.svg)](https://catboost.ai/)
[![Explainable AI](https://img.shields.io/badge/XAI-SHAP%20TreeExplainer-brightgreen.svg)](https://shap.readthedocs.io/)
[![MySQL](https://img.shields.io/badge/Database-MySQL%208.0-orange.svg)](https://www.mysql.com/)
[![License](https://img.shields.io/badge/Academic-Final%20Year%20Capstone-purple.svg)]()

> **Academic & Healthcare Disclaimer:**  
> *This system is developed strictly for educational, academic research, and wellness awareness purposes. It does NOT provide a clinical medical diagnosis or medical prescription, nor does it replace professional consultation with a licensed physician or sleep specialist.*

---

## 1. Project Overview

Chronic sleep disorders—principally **Insomnia** and **Sleep Apnea**—affect hundreds of millions worldwide, escalating risks of hypertension, cardiovascular disease, metabolic syndrome, and cognitive decline. Traditional diagnostic workflows rely on expensive, episodic clinical polysomnography (PSG) or static online risk questionnaires that offer one-off evaluations without continuous tracking, explainability, or personalized intervention.

This project delivers a **full-stack, production-grade AI healthcare informatics platform** that bridges this gap:
1. **Accurate Sleep Disorder Prediction:** Deploys a tuned **CatBoost** and **Voting Ensemble** (CatBoost + Random Forest + Gradient Boosting) trained on 15,000 clinically validated records to classify patients into **None (Healthy)**, **Insomnia**, or **Sleep Apnea** with real calibrated probability distribution.
2. **Dynamic Digital Twin:** Synthesizes a continuously updated virtual avatar representing each user's physiological architecture, autonomic stress, metabolic state, and cardiovascular indicators.
3. **Explainable AI (SHAP):** Computes game-theoretic **Shapley Additive exPlanations** via `TreeExplainer` to reveal exact personal risk contributors for every prediction.
4. **Deterministic Sleep Health Score (0–100):** A transparent, multi-dimensional wellness scoring formula evaluating sleep duration, quality, stress management, daytime movement, and cardiovascular indices.
5. **Interactive What-If Scenario Simulator:** Allows users to test counterfactual lifestyle changes (e.g. extending sleep by 1.5 hrs, reducing stress from 8 to 4) and observe simulated risk/score improvements in real-time without corrupting saved medical records.
6. **Clinical-Grade PDF Reports:** Programmatically renders downloadable multi-page wellness dossiers using **ReportLab**.

---

## 2. System Architecture

```
                                      +------------------------------------+
                                      |            USER / CLIENT           |
                                      |  (Desktop / Tablet / Mobile Web)   |
                                      +-----------------+------------------+
                                                        |
                                                        v
                                      +-----------------+------------------+
                                      |     RESPONSIVE WEB FRONTEND        |
                                      |   HTML5 / CSS3 / JS / Chart.js     |
                                      +-----------------+------------------+
                                                        |
                                          HTTP / RESTful API Requests
                                                        v
+-------------------------------------------------------------------------------------------------------+
|                                             FLASK BACKEND                                             |
|                                                                                                       |
|  +--------------------+   +-----------------------+   +----------------------+   +-----------------+  |
|  |    Auth Service    |   |  Digital Twin Engine  |   |  What-If Simulator   |   | PDF Report Svc  |  |
|  | (Session / Scrypt) |   | (State Sync & Deltas) |   | (In-Memory Scenarios)|   |   (ReportLab)   |  |
|  +--------------------+   +-----------+-----------+   +----------+-----------+   +-----------------+  |
|                                       |                          |                                    |
|                                       v                          v                                    |
|  +------------------------------------+--------------------------+---------------------------------+  |
|  |                                          ML INFERENCE PIPELINE                                  |  |
|  |                                                                                                 |  |
|  |  +-------------------------+   +--------------------------+   +-------------------------------+ |  |
|  |  |     Preprocessing       |   |    Voting Ensemble       |   |      Explainable AI           | |  |
|  |  |  ColumnTransformer      |-->|  (CatBoost + RF + GB)    |-->|     SHAP TreeExplainer        | |  |
|  |  | (StandardScaler+OneHot) |   |    Class Probabilities   |   |   Feature Attributions        | |  |
|  |  +-------------------------+   +--------------------------+   +-------------------------------+ |  |
|  |                                            |                                                    |  |
|  |                                            v                                                    |  |
|  |  +-------------------------+   +--------------------------+   +-------------------------------+ |  |
|  |  |   Sleep Health Score    |   |    Future Risk Model     |   | Personalized Recommendations  | |  |
|  |  |   Deterministic (0-100) |   |  Trajectory Estimation   |   |   4-Pillar Clinical Guidance  | |  |
|  |  +-------------------------+   +--------------------------+   +-------------------------------+ |  |
|  +-------------------------------------------------------------------------------------------------+  |
+---------------------------------------------------+---------------------------------------------------+
                                                    |
                                                    v
                        +---------------------------+---------------------------+
                        |                        DATABASE                       |
                        |             MySQL 8.0 (InnoDB, UTF8mb4)               |
                        |        (Automatic fallback to SQLite if needed)       |
                        |                                                       |
                        |   * users               * sleep_assessments           |
                        |   * digital_twins       * predictions                 |
                        |   * recommendations     * shap_explanations           |
                        |   * future_risk         * progress_records            |
                        +-------------------------------------------------------+
```

---

## 3. Technology Stack

| Layer | Technologies |
|---|---|
| **Backend & Web Framework** | Python 3.13, Flask 3.1, Jinja2, Werkzeug (Password Hashing) |
| **Machine Learning** | CatBoost 1.2, Scikit-Learn 1.9, Pandas 3.0, NumPy 2.5 |
| **Explainable AI (XAI)** | SHAP 0.52 (`shap.TreeExplainer`) |
| **Relational Database** | MySQL 8.0+, SQLAlchemy 2.1, PyMySQL 1.2, Cryptography |
| **Reporting & Export** | ReportLab 5.0 (Automated Vector PDF Generation) |
| **Frontend UI/UX** | HTML5, CSS3, Bootstrap 5.3, Bootstrap Icons, Chart.js 4.4 |
| **Environment & Config** | Python-Dotenv, PowerShell, Windows 11 Compatibility |

---

## 4. Dataset & Machine Learning Benchmarks

### Dataset Overview
- **Location:** `data/Sleep_health_and_lifestyle_dataset.csv`
- **Total Records:** **15,000 samples**
- **Features:** 13 columns (1 identifier, 11 clinical/lifestyle features, 1 target class)
- **Target Classes:** Perfectly balanced multi-class distribution:
  - `None` (Healthy): **5,000 samples (33.3%)**
  - `Insomnia`: **5,000 samples (33.3%)**
  - `Sleep Apnea`: **5,000 samples (33.3%)**

### Feature Dictionary
1. `Age`: Patient age in years (18–65)
2. `Gender`: Biological gender (`Male`, `Female`)
3. `Occupation`: Professional occupation (Doctor, Nurse, Engineer, Teacher, etc.)
4. `Sleep Duration`: Average sleep hours per night
5. `Quality of Sleep`: Subjective sleep quality rating (1–10)
6. `Physical Activity Level`: Daily active movement in minutes
7. `Stress Level`: Subjective stress rating (1–10)
8. `BMI Category`: Body mass classification (`Normal`, `Overweight`, `Obese`)
9. `Blood Pressure`: Formatted string `Systolic/Diastolic` (e.g. `120/80`) decomposed into `Systolic_BP` and `Diastolic_BP`
10. `Heart Rate`: Resting heart rate in beats per minute (BPM)
11. `Daily Steps`: Daily step count

### Empirical Model Comparison (Stratified 80/20 Test Split - 3,000 Unseen Samples)

| Algorithm | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | ROC-AUC (OVR) |
|---|---|---|---|---|---|
| **Voting Ensemble (Primary)** | **96.70%** | **96.70%** | **96.70%** | **96.70%** | **0.9962** |
| **CatBoost Classifier** | **96.47%** | **96.48%** | **96.47%** | **96.46%** | **0.9957** |
| **Gradient Boosting** | 96.47% | 96.47% | 96.47% | 96.46% | 0.9961 |
| **Random Forest** | 96.30% | 96.30% | 96.30% | 96.30% | 0.9959 |
| **Decision Tree** | 93.27% | 93.30% | 93.27% | 93.27% | 0.9741 |

*Note: All metrics displayed above are verified directly from actual test execution and serialized in `models/model_metrics.json`.*

---

## 5. Directory & File Structure

```
project/
├── app.py                      # Flask Application Factory & Server Entry Point
├── config.py                   # Centralized Configuration & Environment Loader
├── requirements.txt            # Python Dependencies
├── .env.example                # Template for Environment Variables
├── .env                        # Local Environment Configuration
├── seed_demo.py                # Demo Account & Historical Progression Seeder
├── README.md                   # Complete Documentation & Viva Guide
│
├── database/
│   └── schema.sql              # MySQL DDL Schema Definition (8 Tables + Foreign Keys)
│
├── data/
│   ├── README.md               # Dataset Documentation
│   └── Sleep_health_and_lifestyle_dataset.csv  # 15,000 records Kaggle Dataset
│
├── models/
│   ├── README.md               # Artifact Documentation
│   ├── sleep_model.cbm         # Serialized CatBoost Multi-Class Model
│   ├── deployment_model.pkl    # Serialized Voting Ensemble Model
│   ├── preprocessor.pkl        # Serialized ColumnTransformer Pipeline
│   ├── model_metrics.json      # Empirical Benchmark Results
│   └── model_metadata.json     # Hyperparameters & Feature Metadata
│
├── ml/
│   ├── __init__.py
│   ├── preprocess.py           # Preprocessing Pipeline (Scikit-Learn ColumnTransformer)
│   ├── train_model.py          # Complete Model Training & Evaluation Script
│   ├── predict.py              # Inference Service & Probability Estimator
│   ├── explain.py              # Explainable AI via SHAP TreeExplainer
│   ├── scoring.py              # Deterministic Sleep Health Score (0-100)
│   ├── risk.py                 # Future Wellness Risk & Trajectory Engine
│   └── recommendations.py      # Personalized 4-Pillar Recommendation Rules
│
├── routes/
│   ├── __init__.py
│   ├── auth.py                 # Register, Login, Logout, Profile Management
│   ├── dashboard.py            # Dashboard, History, Model Info & Progress API
│   ├── prediction.py           # Assessment Form, Predict API, Result View
│   ├── digital_twin.py         # Digital Twin State Visualizer & API
│   ├── simulator.py            # What-If Counterfactual Scenario Simulator
│   └── report.py               # PDF Download Route
│
├── services/
│   ├── __init__.py
│   ├── digital_twin_service.py # Digital Twin Orchestration & Synchronization
│   └── pdf_service.py          # ReportLab PDF Generation Service
│
├── utils/
│   ├── __init__.py
│   └── helpers.py              # Login Decorators & Current User Helpers
│
├── templates/
│   ├── base.html               # Master Layout (Navbar, Toast alerts, Disclaimer)
│   ├── index.html              # Landing Page (Hero, System Pillars, IoT Architecture)
│   ├── login.html              # Login with 1-Click Demo Credentials
│   ├── register.html           # Registration with Twin Initialization
│   ├── dashboard.html          # Progress Dashboard & Live Chart.js Charts
│   ├── assessment.html         # Interactive Sleep Assessment Input Form
│   ├── prediction.html         # Diagnostic Output, Confidence, SHAP & Recs
│   ├── digital_twin.html       # Virtual Twin Persona & State Matrices
│   ├── simulator.html          # Interactive What-If Scenario Simulator
│   ├── history.html            # Longitudinal Historical Assessments Table
│   ├── profile.html            # Account & Password Management
│   └── model_info.html         # Viva Benchmarking & Methodology Page
│
├── static/
│   ├── css/
│   │   └── style.css           # Healthcare Theme, Pulse Avatar, Gauges, Cards
│   └── js/
│       ├── dashboard.js        # Multi-Axis Timeseries Visualizations
│       ├── assessment.js       # Sliders Dynamic Sync & Async Form Submission
│       └── simulator.js        # Real-time Counterfactual Comparison Engine
│
└── tests/
    ├── __init__.py
    └── test_system.py          # Complete 10-Point Unit & Integration Test Suite
```

---

## 6. Installation & Setup Guide

### Step 1: Clone or Navigate to Project Directory
```powershell
cd c:\Users\Lenovo\OneDrive\Desktop\project
```

### Step 2: Set Up Python Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Configure Database (`.env`)
Ensure your MySQL server is running (port 3306). Update `.env` if necessary:
```ini
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=admin
DB_NAME=sleep_wellness_db
SECRET_KEY=sleep_wellness_super_secret_jwt_key_2026_academic_viva
```
*Note: If MySQL is not installed or offline, the system automatically falls back gracefully to a local SQLite database (`sleep_wellness_fallback.db`), ensuring zero downtime during demonstration!*

### Step 5: Train Machine Learning Models
```powershell
python ml/train_model.py
```
This trains CatBoost, Random Forest, Gradient Boosting, Decision Tree, and the Voting Ensemble on 15,000 samples, displays the viva benchmark table, and serializes the models to `models/`.

### Step 6: Seed Demonstration Data (Optional but Recommended for Viva)
```powershell
python seed_demo.py
```
This seeds the demonstration patient account with 5 weeks of realistic historical records and an active Digital Twin.

### Step 7: Run Unit & Integration Tests
```powershell
python -m unittest tests/test_system.py
```

### Step 8: Start the Web Application
```powershell
python app.py
```
Access the application in your browser at:
```
http://127.0.0.1:5000
```

---

## 7. Demonstration / Test Account Credentials

For quick evaluation during project vivas or presentations:
- **Email:** `demo@sleepwellness.ai`
- **Password:** `password123`
- *(Or simply click the **"Quick Fill Demo Account"** button on the login screen)*

---

## 8. REST API Endpoints

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `POST` | `/api/predict` | Ingests assessment, runs ML, computes score & SHAP, updates Digital Twin | Yes |
| `POST` | `/api/explain` | Computes on-demand SHAP feature contributions | Yes |
| `POST` | `/api/score` | Calculates deterministic Sleep Health Score (0-100) | Yes |
| `POST` | `/api/future-risk` | Calculates future risk index and trajectory | Yes |
| `POST` | `/api/recommendations`| Generates 4-pillar personalized recommendations | Yes |
| `POST` | `/api/simulate` | Evaluates hypothetical scenarios in-memory without database mutation | Yes |
| `GET` | `/api/digital-twin` | Retrieves user's active virtual twin state and baseline deltas | Yes |
| `GET` | `/api/progress` | Returns timeseries data formatted for Chart.js | Yes |
| `GET` | `/api/history` | Returns historical assessments list | Yes |
| `GET` | `/report/<id>` | Generates and downloads binary PDF wellness report | Yes |

---

## 9. Key Viva Voce Concepts & Answers

### Q1: What makes your Digital Twin more than just a database profile?
> **Answer:** In traditional systems, user profiles are static CRUD records. In our Digital Twin architecture, the virtual persona maintains four dynamic computational state matrices (Sleep Architecture, Autonomic Stress, Metabolic/Motion, Cardiovascular). Every time a user submits new biometric or sleep data, the state updates recursively, calculating progress deltas against historical baselines, tracking disorder risk trajectories, and feeding continuous historical context into the explainability and recommendation engines.

### Q2: Why use CatBoost and a Voting Ensemble?
> **Answer:** Healthcare and lifestyle datasets frequently contain a blend of continuous physiological variables (blood pressure, heart rate, sleep hours) and categorical factors (gender, occupation, BMI classification). CatBoost handles categorical features with symmetric oblivious trees, avoiding target leakage and overfitting. Our Voting Ensemble further blends CatBoost with Random Forest and Gradient Boosting to minimize model variance and boost generalization F1-score to **96.70%**.

### Q3: How does SHAP provide Explainable AI in this system?
> **Answer:** Standard ML models function as black boxes. By employing `shap.TreeExplainer` on the CatBoost architecture, we calculate the exact Shapley values from cooperative game theory for each input feature. For every prediction, the system explains *which specific biometrics* (e.g. elevated stress level of 8 or low sleep duration of 5.2 hrs) pushed the model toward predicting Insomnia or Sleep Apnea, empowering the patient with transparent insights rather than an unexplained label.

### Q4: How does the What-If Simulator work without modifying actual user records?
> **Answer:** The simulator executes in-memory inference through `/api/simulate`. It takes the user's current baseline from their Digital Twin, applies the user-selected slider modifications (e.g. extending sleep from 5.5 to 7.5 hrs), passes the simulated vector through the trained ML pipeline, recalculates the health score and risk, and renders a side-by-side comparison with delta indicators *without executing any database commit*.

---

## 10. Future Extensions: Wearable & IoT Architecture

While the current platform accepts multi-parameter manual assessments, the backend includes ready architectural hooks for:
1. **Smartwatch Actigraphy:** Streaming nocturnal heart rate variability (HRV), resting pulse, and sleep stage epochs via Apple HealthKit and Google Health Connect.
2. **IoT Acoustic Sensors:** Passive bedside microphone telemetry detecting nocturnal snoring cadence, stridor, and respiratory pauses.
3. **Automated Interventions:** Nightly automated twin synchronization and daytime predictive alerts prompting restorative sleep adjustments.

---

## 11. Academic Conclusion

The **Intelligent Sleep Disorder Prediction and Digital Twin-Based Personalized Sleep Wellness System** demonstrates how modern predictive AI, transparent explainability (SHAP), and digital twin simulation can revolutionize preventative sleep medicine and digital health informatics.
