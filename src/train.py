import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
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

df.info()
df.describe()




# Set visual style for plots
sns.set_theme(style="whitegrid")

# 1. Value Counts for Categorical Columns
print("\n--- Categorical Feature: Depression Status ---")
print(df["Do you have Depression?"].value_counts())

print("\n--- Categorical Feature: Gender Breakdown ---")
print(df["Choose your gender"].value_counts())

# 2. Numerical Summary Statistics
print("\n--- Numerical Summary Statistics ---")
print(df[["Age"]].describe())

# 3. Outlier Detection using IQR (Interquartile Range) for Age
Q1 = df["Age"].quantile(0.25)
Q3 = df["Age"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter rows that fall outside the normal range
outliers = df[(df["Age"] < lower_bound) | (df["Age"] > upper_bound)]
print(f"\n--- Outlier Flag ---")
print(f"Number of outliers detected in Age: {len(outliers)}")

# 4. Generate Visualizations (Histogram and Boxplot)
plt.figure(figsize=(12, 5))

# Histogram with Kernel Density Estimation (KDE)
plt.subplot(1, 2, 1)
sns.histplot(df["Age"], kde=True, color="teal", bins=10)
plt.title("Age Distribution (Histogram)")
plt.xlabel("Age")
plt.ylabel("Count")

# Boxplot for Outlier Visualization
plt.subplot(1, 2, 2)
sns.boxplot(x=df["Age"], color="orange")
plt.title("Age Spread (Boxplot)")
plt.xlabel("Age")

plt.tight_layout()

# Save the plot locally as an artifact of your EDA
os.makedirs("data/processed", exist_ok=True)
plt.savefig("data/processed/age_eda_plots.png")
print("\n EDA plots saved successfully to data/processed/age_eda_plots.png!")

# Show the plot window
plt.show()