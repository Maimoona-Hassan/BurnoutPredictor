# src/inspect_data.py
import pandas as pd
import os

# Define path to raw data (update filename to match your dataset)
data_path = os.path.join("data", "raw", "studentMentalHealth.csv") 

print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)

print("\n--- 1. Dataset Shape (Rows, Columns) ---")
print(df.shape)

print("\n--- 2. Dataset Info & Data Types ---")
df.info()

print("\n--- 3. Numerical Summary ---")
print(df.describe())

print("\n--- 4. Missing Values Check ---")
print(df.isnull().sum())

print("\n--- 5. Duplicate Rows Count ---")
print("Duplicates found:", df.duplicated().sum())