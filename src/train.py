import pandas as pd
import os

# 1. Define the path to our raw dataset
data_path = os.path.join( "data", "RawData", "studentMentalHealth.csv")

print(f"Loading data from: {data_path}")

# 2. Load the CSV into a pandas DataFrame
df = pd.read_csv(data_path)

# 3. Quick inspection to verify everything loaded correctly
print("\n--- Dataset Info ---")
df.info()

# print("\n--- First 5 Rows ---")
print(df.head())

# 4. Check for any missing/null values in the dataset
print("\n--- Missing Values Check ---")
print(df.isnull().sum())
# 5. Fill the missing value in 'Age' with the median age of the dataset
median_age = df["Age"].median()
df["Age"] = df["Age"].fillna(median_age)

# Verify it worked (should now be 0)
print(f"\nMissing 'Age' values remaining: {df['Age'].isnull().sum()}")


# 6. View the Age column to check our work
print("\n--- Checking Age Column ---")
print(df["Age"].head(10))  # Prints the first 10 ages