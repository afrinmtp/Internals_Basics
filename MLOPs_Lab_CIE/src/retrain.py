import pandas as pd
import mlflow
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load new data
df = pd.read_csv("../data/new_data.csv")

X = df.drop("fare_amount", axis=1)
y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=10,
    min_samples_split=2,
    random_state=42
)

mlflow.set_experiment("urbanride-fare-amount")

with mlflow.start_run(run_name="retraining"):

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    mlflow.log_metric("mae", mae)

output = {
    "retrained": True,
    "new_mae": mae,
    "model_name": "UrbanRideFareModel"
}

with open("../results/step4_s4.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 4 completed")