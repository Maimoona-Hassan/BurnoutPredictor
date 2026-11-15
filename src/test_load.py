import pandas as pd

# Load the dataset from the data folder
file_path = "data/student_lifestyle.csv"
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("--- First 3 rows ---")
print(df.head(3))