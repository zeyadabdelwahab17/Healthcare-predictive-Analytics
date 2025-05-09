# MLflow Model Evaluation Documentation

## Overview
This documentation describes the process for training and evaluating multiple machine learning models using MLflow in the `Health_Care_Predictive_Analysis` project. The workflow includes training models directly using scikit-learn, performing hyperparameter optimization, and logging results to MLflow for comparison and monitoring.

---

## Directory Structure
```
Health_Care_Predictive_Analysis/
├── data/
│   └── cleaned_HealthCare_data.csv
├── evaluate_models_with_mlflow.py
└── ...
```

---

## Prerequisites
- Python 3.11+
- All dependencies installed:
  - `mlflow`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `xgboost`
- Test dataset available at `data/cleaned_HealthCare_data.csv`

---

## Models Implemented
1. Random Forest Classifier
2. Logistic Regression
3. Support Vector Machine (SVM)
4. XGBoost
5. Multi-layer Perceptron (MLP)

Each model is trained with hyperparameter optimization using RandomizedSearchCV.

---

## Feature and Label Columns
- **Features:**
  - `Age`, `Gender`, `Blood Type`, `Medical Condition`, `Insurance Provider`, `Billing Amount`, `Admission Type`, `Medication`, `Test Results Encoded`, `Length of Stay`
- **Label:**
  - `Test Results Encoded`

---

## Running Model Evaluation

1. **Activate your virtual environment:**
   ```bash
   venv\Scripts\activate  # or source venv/bin/activate
   ```

2. **Run the evaluation script:**
   ```bash
   python Health_Care_Predictive_Analysis/evaluate_models_with_mlflow.py
   ```
   - The script will:
     - Load and preprocess the data
     - Train each model with hyperparameter optimization
     - Calculate metrics (accuracy, precision, recall, f1)
     - Log results to MLflow
     - Print a summary for each model

3. **Start the MLflow UI:**
   ```bash
   mlflow ui
   ```
   - Open [http://localhost:5000](http://localhost:5000) in your browser to view and compare results.

---

## Hyperparameter Optimization
Each model is trained using RandomizedSearchCV with the following parameter spaces:

### Random Forest
- n_estimators: [100, 200, 300]
- max_depth: [10, 20, 30, None]
- min_samples_split: [2, 5, 10]

### Logistic Regression
- C: [0.1, 1, 10]
- solver: ['liblinear', 'saga']

### SVM
- C: [0.1, 1, 10]
- kernel: ['rbf', 'linear']

### XGBoost
- n_estimators: [100, 200]
- max_depth: [3, 5, 7]
- learning_rate: [0.01, 0.1]

### MLP
- hidden_layer_sizes: [(50,), (100,), (50, 50)]
- activation: ['relu', 'tanh']
- alpha: [0.0001, 0.001]

---

## Data Preprocessing
The script includes the following preprocessing steps:
1. Train-test split (80-20 split)
2. Feature scaling using StandardScaler
3. Handling of categorical variables (already encoded in the dataset)

---

## Model Evaluation Metrics
The following metrics are calculated and logged for each model:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)

---

## Troubleshooting
- **Memory Issues:**
  - If you encounter memory issues, reduce the number of iterations in RandomizedSearchCV or the size of parameter spaces
- **Training Time:**
  - The script may take some time to run due to hyperparameter optimization
  - Consider reducing the number of iterations or parameter combinations if needed
- **MLflow UI not starting:**
  - Ensure MLflow is installed and your virtual environment is active
- **Metrics not appearing:**
  - Check that the script completed successfully and that the experiment name matches in the MLflow UI

---

## Best Practices
- Monitor the training progress through the printed output
- Use MLflow's UI to compare models and select the best one for deployment
- Consider adjusting hyperparameter spaces based on your specific needs
- Regularly update your requirements files and document any new dependencies

---

## References
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/) 