import os
import argparse
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Setup argparse for input and output data paths
parser = argparse.ArgumentParser()
parser.add_argument('--input-data', type=str, required=True)
parser.add_argument('--output-data', type=str, required=True)
args = parser.parse_args()

# Define paths
input_data_path = os.path.join(args.input_data, 'AWS_credit_score.csv')
output_data_path = os.path.join(args.output_data, 'processed-data.csv')

# Load the dataset
df = pd.read_csv(input_data_path)

# Optionally remove the 'CUST_ID' column if it exists and isn't needed for modeling
if 'CUST_ID' in df.columns:
    df.drop(['CUST_ID'], axis=1, inplace=True)

# Define numerical and categorical columns
categorical_features = [col for col in df.columns if col.startswith('CAT')]
numerical_features = [col for col in df.columns if not col.startswith('CAT')]

# Define the preprocessing steps for numerical columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Define the preprocessing steps for categorical columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create the preprocessing model
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Apply preprocessing
preprocessed_data = preprocessor.fit_transform(df)

# Generating feature names for the preprocessed data
feature_names = numerical_features.copy()
cat_transformer = preprocessor.named_transformers_['cat']['onehot']
for i, cat_feature in enumerate(categorical_features):
    categories = cat_transformer.categories_[i]
    feature_names.extend([f'{cat_feature}_{category}' for category in categories])

# Create the preprocessed DataFrame without calling toarray()
preprocessed_df = pd.DataFrame(preprocessed_data, columns=feature_names)

# Save the preprocessed DataFrame
preprocessed_df.to_csv(output_data_path, index=False)