# Sleep Health and Lifestyle Dataset

This directory contains the dataset used for training the Machine Learning models for sleep disorder prediction.

## File Location
```
data/Sleep_health_and_lifestyle_dataset.csv
```

## Dataset Specifications
- **Total Records:** 15,000 samples
- **Features:** 13 columns (1 identifier, 11 predictor features, 1 target class)
- **Source:** Kaggle Sleep Health and Lifestyle Data

## Feature Dictionary
1. `Person ID`: Unique record identifier (dropped during preprocessing)
2. `Gender`: Biological gender (`Male`, `Female`)
3. `Age`: Patient age in years
4. `Occupation`: Professional occupation (`Software Engineer`, `Doctor`, `Nurse`, `Teacher`, `Engineer`, etc.)
5. `Sleep Duration`: Average hours of sleep per night
6. `Quality of Sleep`: Subjective sleep quality rating on a scale of 1 to 10
7. `Physical Activity Level`: Daily physical activity duration in minutes
8. `Stress Level`: Subjective stress rating on a scale of 1 to 10
9. `BMI Category`: Body Mass Index classification (`Normal`, `Overweight`, `Obese`)
10. `Blood Pressure`: Systolic / Diastolic reading in mmHg (e.g. `120/80`)
11. `Heart Rate`: Resting heart rate in beats per minute (BPM)
12. `Daily Steps`: Average daily step volume
13. `Sleep Disorder` (Target):
    - `None` (Healthy - 5,000 samples)
    - `Insomnia` (5,000 samples)
    - `Sleep Apnea` (5,000 samples)

## Preprocessing Note
During training and inference, the `Blood Pressure` string is automatically decomposed into `Systolic_BP` and `Diastolic_BP` for numerical gradient modeling, and `BMI Category` is standardized.
