import pandas as pd
import mlflow
import json
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split

df = pd.read_csv("../data/training_data.csv")

X = df.drop("fare_amount", axis=1)
y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [50, 150, 250],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 3, 5]
}

rf = RandomForestRegressor(random_state=42)

mlflow.set_experiment("urbanride-fare-amount")

with mlflow.start_run(run_name="tuning-urbanride"):

    search = RandomizedSearchCV(
        rf,
        param_distributions=param_grid,
        n_iter=10,
        cv=5,
        scoring="neg_mean_absolute_error",
        random_state=42
    )

    search.fit(X_train, y_train)

    best_params = search.best_params_
    best_mae = -search.best_score_

    output = {
        "search_type": "random",
        "n_folds": 5,
        "total_trials": 10,
        "best_params": best_params,
        "best_mae": best_mae,
        "best_cv_mae": best_mae,
        "parent_run_name": "tuning-urbanride"
    }

    with open("../results/step2_s2.json", "w") as f:
        json.dump(output, f, indent=4)

print("Task 2 completed")