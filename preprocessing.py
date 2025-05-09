# preprocessing.py

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define feature lists
numeric_features = ['Age', 'Billing Amount', 'Length of Stay', 'Gender']
categorical_features = ['Blood Type', 'Medical Condition', 'Admission Type', 'Insurance Provider', 'Medication']

# Create the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
