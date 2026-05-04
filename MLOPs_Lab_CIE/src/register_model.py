import pandas as pd
import mlflow
import mlflow.sklearn
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("../data/training_data.csv")

X = df.drop("fare_amount", axis=1)
y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=10,
    min_samples_split=2,
    random_state=42
)

mlflow.set_experiment("urbanride-fare-amount")

with mlflow.start_run(run_name="model-registration"):

    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="UrbanRideFareModel"
    )

output = {
    "model_name": "UrbanRideFareModel",
    "stage": "None",
    "framework": "sklearn",
    "version": 1
}

with open("../results/step3_s3.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 3 completed")