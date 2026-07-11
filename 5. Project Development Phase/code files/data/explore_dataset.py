"""
Dataset Explorer Script
Rising Waters: A Machine Learning Approach to Flood Prediction
Reads the flood dataset and prints complete analysis for verification.
"""

import pandas as pd
import numpy as np
import os

# Load the dataset
dataset_path = os.path.join(os.path.dirname(__file__), 'flood_dataset.xlsx')
dataset = pd.read_excel(dataset_path)

print("=" * 60)
print("RISING WATERS - DATASET EXPLORATION")
print("=" * 60)

print(f"\n1. SHAPE: {dataset.shape}")
print(f"   Rows: {dataset.shape[0]}, Columns: {dataset.shape[1]}")

print(f"\n2. COLUMNS: {list(dataset.columns)}")

print(f"\n3. DATA TYPES:\n{dataset.dtypes}")

print(f"\n4. FIRST 5 ROWS:\n{dataset.head()}")

print(f"\n5. LAST 5 ROWS:\n{dataset.tail()}")

print(f"\n6. DESCRIPTIVE STATISTICS:\n{dataset.describe()}")

print(f"\n7. NULL VALUES:\n{dataset.isnull().sum()}")
print(f"\n   Any nulls? {dataset.isnull().any().any()}")

print(f"\n8. TARGET VARIABLE DISTRIBUTION:")
target_col = dataset.columns[-1]
print(f"   Target column: '{target_col}'")
print(f"{dataset[target_col].value_counts()}")

print(f"\n9. INFO:")
dataset.info()

print("\n" + "=" * 60)
print("EXPLORATION COMPLETE")
print("=" * 60)
