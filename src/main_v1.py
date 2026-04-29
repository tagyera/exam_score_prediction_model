import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


# loading dataset
df = pd.read_csv("../data/raw/student_habits_performance.csv")

df.head(10)
df.info()


sns.set(style="whitegrid")


# checking for null values
df.isna().sum()

df.duplicated().sum()

df[df["parental_education_level"].isna()]

df["parental_education_level"].value_counts()

# filling missing values
df["parental_education_level"] = df["parental_education_level"].fillna(
    df["parental_education_level"].mode()[0]
)


# statistical data

df.describe()

df.describe(include="object")


# numerical data

numeric_cols = df.select_dtypes(exclude="object").columns.to_list()

plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i + 1)  # subplot index starts at 1
    plt.hist(df[numeric_cols[i]], bins=20, edgecolor="black")
    plt.title(f"Distribution of {numeric_cols[i]}")

plt.tight_layout()
plt.savefig("../reports/v1_figs/distribution_numeric_columns.png")
plt.show()


# categorical data
categorical_cols = df.describe(include="object").columns.tolist()[1:]

for col in categorical_cols:
    col_value = df[col].value_counts().reset_index()
    print(f"Value count for {col}:\n{col_value}\n")

plt.figure(figsize=(9, 6))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.hist(df[categorical_cols[i]], color="skyblue", edgecolor="black")
    plt.title(f"Distribution of {categorical_cols[i]}")

plt.tight_layout()
plt.savefig("../reports/v1_figs/distribution_categorical_columns.png")
plt.show()


# correlation

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.savefig("../reports/v1_figs/correlation_matrix.png")
plt.show()


num_features = df.describe().columns.tolist()

for feature in num_features:
    sns.scatterplot(data=df, x=feature, y="exam_score")
    plt.title(f"{feature} vs Exam score")
    plt.savefig(f"../reports/v1_figs/{feature}_vs_examscore.png")
    plt.show()

for col in categorical_cols:
    sns.boxplot(data=df, x=col, y="exam_score")
    plt.title(f"Exam Score by {col}")
    plt.xticks(rotation=45)
    plt.savefig(f"../reports/v1_figs/{col}_vs_examscore.png")
    plt.show()


# features selection

df.columns

features = [
    "study_hours_per_day",
    "attendance_percentage",
    "mental_health_rating",
    "sleep_hours",
    "part_time_job",
]
target = "exam_score"

df_model = df[features + [target]].copy()

# encoding categorical data

le = LabelEncoder()

df_model["part_time_job"] = le.fit_transform(df_model["part_time_job"])


# train test split

X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

len(y_train)
len(y_test)

# models

models = {
    "LinearRegression": {"model": LinearRegression(), "params": {}},
    "DecisionTree": {
        "model": DecisionTreeRegressor(),
        "params": {"max_depth": [3, 5, 10], "min_samples_split": [2, 5]},
    },
    "RandomForest": {
        "model": RandomForestRegressor(),
        "params": {"n_estimators": [50, 100], "max_depth": [5, 10]},
    },
}

best_models = []

for name, config in models.items():
    print(f"Training {name}")

    grid = GridSearchCV(
        config["model"], config["params"], cv=5, scoring="neg_mean_squared_error"
    )
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    best_models.append(
        {"model": name, "best_params": grid.best_params_, "rmse": rmse, "r2": r2}
    )

results_df = pd.DataFrame(best_models)


# selecting best model

best_row = results_df.sort_values(by="rmse").iloc[0]

best_model_name = best_row["model"]

best_model_config = models[best_model_name]

# final_model = best_model_config["model"].set_params(**best_row['best_params'])
final_model = best_model_config["model"]

final_model.fit(X, y)

joblib.dump(final_model, "../models/model_v1/best_model.pkl")
