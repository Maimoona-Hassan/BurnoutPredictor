import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


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




# Univariate EDA & Outlier Check

# Update filename if you chose a different CSV in data/RawData
DATA_PATH = Path("data") / "RawData" / "studentMentalHealth.csv"
FIG_DIR = Path("reports") / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("\nColumns & Dtypes:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# Inspect numeric columns: distributions, summary stats, and IQR outlier counts
num_cols = df.select_dtypes(include="number").columns.tolist()
if num_cols:
    print("\nSummary stats:\n", df[num_cols].describe().T[["min", "25%", "50%", "75%", "max", "mean", "std"]])

    for col in num_cols:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = s[(s < lower) | (s > upper)]
        print(f"[{col}] IQR outliers: {len(outliers)} (bounds: {lower:.2f} to {upper:.2f})")

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        sns.histplot(s, kde=True, ax=axes[0])
        axes[0].set_title(f"{col} — Distribution")

        sns.boxplot(x=s, ax=axes[1])
        axes[1].set_title(f"{col} — Boxplot (IQR)")

        plt.tight_layout()
        plt.savefig(FIG_DIR / f"univariate_{col}.png", dpi=150)
        plt.close()

# Inspect categorical columns (including potential target columns)
cat_cols = df.select_dtypes(exclude="number").columns.tolist()
for col in cat_cols:
    print(f"\nValue counts for '{col}' (top 10):\n", df[col].value_counts(dropna=False).head(10))



  

DATA_PATH = Path("data") / "RawData" / "studentMentalHealth.csv"
TARGET_COL = "REPLACE_WITH_TARGET_COLUMN"  # e.g., "Burnout_Risk"

df = pd.read_csv(DATA_PATH).drop_duplicates().copy()

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# 1. Split BEFORE any imputation or scaling to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_features = X_train.select_dtypes(include="number").columns.tolist()
cat_features = X_train.select_dtypes(exclude="number").columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, num_features),
    ("cat", categorical_pipe, cat_features),
])

models = {
    "Dummy (most_frequent)": DummyClassifier(strategy="most_frequent"),
    "LogReg (default)": LogisticRegression(max_iter=1000, random_state=42),
    "LogReg (class_weight=balanced)": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
}

for name, clf in models.items():
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", clf),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print(f"\n=== {name} ===")
    print("Recall:", round(recall_score(y_test, preds, pos_label=y.iloc[0], zero_division=0), 4)
          if y.nunique() == 2 else "")
    print("F1 (weighted):", round(f1_score(y_test, preds, average="weighted", zero_division=0), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))

    from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==========================================
# 1. CONFIGURATION (Update these two lines)
# ==========================================
DATA_PATH = Path("data") / "RawData" / "student_lifestyle.csv"
TARGET_COL = "REPLACE_WITH_TARGET_COLUMN"  # e.g., "Burnout_Risk"
MODEL_OUT = Path("models") / "burnout_pipeline.joblib"


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build leak-free numeric and categorical preprocessing pipelines."""
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Check DATA_PATH in src/train.py."
        )
    if TARGET_COL == "REPLACE_WITH_TARGET_COLUMN":
        raise ValueError(
            "Set TARGET_COL in src/train.py to your actual target column name first."
        )

    # Load and drop exact duplicate rows only (keep raw file untouched)
    df = pd.read_csv(DATA_PATH).drop_duplicates().dropna(subset=[TARGET_COL]).copy()
    print(f"Loaded {DATA_PATH} -> shape after exact dedup: {df.shape}")
    print("\nTarget distribution:\n", df[TARGET_COL].value_counts(dropna=False))

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Determine positive class label for binary recall/F1 if applicable
    classes = sorted(y.unique(), key=str)
    is_binary = len(classes) == 2
    pos_label = classes[1] if is_binary else None

    # 2. Stratified train/test split FIRST (prevents data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    print(f"\nTrain shape: {X_train.shape} | Test shape: {X_test.shape}")

    preprocessor = build_preprocessor(X_train)

    candidates = {
        "Dummy (most_frequent)": DummyClassifier(strategy="most_frequent"),
        "LogReg (default)": LogisticRegression(max_iter=1000, random_state=42),
        "LogReg (balanced)": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "RandomForest (default)": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "RandomForest (balanced)": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }

    # 3. Stratified K-Fold Cross-Validation on training data only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = (
        {"f1": f"f1", "recall": "recall", "precision": "precision", "acc": "accuracy"}
        if is_binary and str(pos_label) in ("1", "True", "Yes", "High")
        else {"f1_weighted": "f1_weighted", "acc": "accuracy"}
    )

    cv_rows = []
    fitted_pipes = {}

    for name, estimator in candidates.items():
        pipe = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", estimator),
            ]
        )
        cv_scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        pipe.fit(X_train, y_train)
        fitted_pipes[name] = pipe

        row = {"Model": name}
        for metric_key, values in cv_scores.items():
            if metric_key.startswith("test_"):
                clean_name = metric_key.replace("test_", "CV_")
                row[f"{clean_name}_mean"] = round(float(np.mean(values)), 4)
                row[f"{clean_name}_std"] = round(float(np.std(values)), 4)
        cv_rows.append(row)

    cv_df = pd.DataFrame(cv_rows)
    print("\n=== 5-Fold Stratified CV Results (Training Set) ===")
    print(cv_df.to_string(index=False))

    # Pick best non-dummy model by primary CV metric
    non_dummy_df = cv_df[cv_df["Model"] != "Dummy (most_frequent)"]
    primary_col = "CV_f1_mean" if "CV_f1_mean" in non_dummy_df.columns else "CV_f1_weighted_mean"
    best_row = non_dummy_df.sort_values(by=primary_col, ascending=False).iloc[0]
    best_name = best_row["Model"]
    best_pipe = fitted_pipes[best_name]

    # 4. Single final evaluation on held-out test set
    test_preds = best_pipe.predict(X_test)
    print(f"\n=== Final Test Evaluation: {best_name} ===")
    print("Accuracy :", round(accuracy_score(y_test, test_preds), 4))
    if is_binary:
        print(f"Precision ({pos_label}):", round(precision_score(y_test, test_preds, pos_label=pos_label, zero_division=0), 4))
        print(f"Recall    ({pos_label}):", round(recall_score(y_test, test_preds, pos_label=pos_label, zero_division=0), 4))
        print(f"F1        ({pos_label}):", round(f1_score(y_test, test_preds, pos_label=pos_label, zero_division=0), 4))
    else:
        print("F1 (weighted):", round(f1_score(y_test, test_preds, average="weighted", zero_division=0), 4))

    print("\nConfusion Matrix:\n", confusion_matrix(y_test, test_preds))
    print("\nClassification Report:\n", classification_report(y_test, test_preds, zero_division=0))

    # 5. Permutation feature importance on test set (model-agnostic)
    try:
        feature_names = best_pipe.named_steps["prep"].get_feature_names_out()
        perm = permutation_importance(
            best_pipe, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        imp_df = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance_mean": perm.importances_mean,
                    "importance_std": perm.importances_std,
                }
            )
            .sort_values(by="importance_mean", ascending=False)
            .head(10)
        )
        print("\n=== Top 10 Features (Permutation Importance) ===")
        print(imp_df.to_string(index=False))
    except Exception as exc:
        print(f"\nSkipping permutation importance details: {exc}")

    # 6. Save best pipeline artifact
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, MODEL_OUT)
    print(f"\nSaved best pipeline to: {MODEL_OUT}")


if __name__ == "__main__":
    main()