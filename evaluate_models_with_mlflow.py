import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Model imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# Define feature and label columns
FEATURE_COLUMNS = [
    'Age', 'Gender', 'Blood Type', 'Medical Condition',
    'Insurance Provider', 'Billing Amount', 'Admission Type',
    'Medication', 'Test Results Encoded', 'Length of Stay'
]
LABEL_COLUMN = 'Test Results Encoded'

# Define categorical and numerical columns
CATEGORICAL_COLUMNS = [
    'Gender', 'Blood Type', 'Medical Condition',
    'Insurance Provider', 'Admission Type', 'Medication'
]
NUMERICAL_COLUMNS = [
    'Age', 'Billing Amount', 'Length of Stay'
]

# Load and prepare data
DATA_PATH = 'Health_Care_Predictive_Analysis/data/cleaned_HealthCare_data.csv'
df = pd.read_csv(DATA_PATH)
X = df[FEATURE_COLUMNS]
y = df[LABEL_COLUMN]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMERICAL_COLUMNS),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_COLUMNS)
    ])

# Define models and their hyperparameter spaces (reduced for faster training)
models = {
    'RandomForest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(),
        'params': {
            'C': [0.1, 1],
            'solver': ['liblinear']
        }
    },
    'SVM': {
        'model': SVC(),
        'params': {
            'C': [0.1, 1],
            'kernel': ['linear']  # Using only linear kernel for speed
        }
    },
    'XGBoost': {
        'model': XGBClassifier(),
        'params': {
            'n_estimators': [100],
            'max_depth': [3, 5],
            'learning_rate': [0.1]
        }
    },
    'MLP': {
        'model': MLPClassifier(),
        'params': {
            'hidden_layer_sizes': [(50,)],
            'activation': ['relu'],
            'alpha': [0.001]
        }
    }
}

# Set MLflow experiment
mlflow.set_experiment("Model Evaluation - Direct Training")

# Train and evaluate models
for model_name, model_config in models.items():
    with mlflow.start_run(run_name=model_name):
        # Create pipeline with preprocessing and model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model_config['model'])
        ])
        
        # Perform random search with reduced iterations and folds
        random_search = RandomizedSearchCV(
            pipeline,
            {
                f'classifier__{param}': value 
                for param, value in model_config['params'].items()
            },
            n_iter=3,  # Reduced from 5
            cv=2,      # Reduced from 3
            random_state=42
        )
        
        # Train model
        random_search.fit(X_train, y_train)
        
        # Get best model
        best_model = random_search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Log parameters and metrics
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        print(f"Evaluated {model_name}:")
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Metrics: {metrics}") 