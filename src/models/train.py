import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import os
import mlflow
import mlflow.sklearn
import dagshub


dagshub.init(
    repo_owner="AxelMunerol",
    repo_name="examen-dvc",
    mlflow=True,
)

X_train = pd.read_csv('data/processed/X_train_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
X_test = pd.read_csv('data/processed/X_test_scaled.csv')
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
best_params = joblib.load('models/best_params.pkl')

with mlflow.start_run(run_name="random-forest-training"):
    mlflow.log_params(best_params)
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        'mse': mean_squared_error(y_test, predictions),
        'r2': r2_score(y_test, predictions),
    }
    mlflow.log_metrics(metrics)

    joblib.dump(model, 'models/model.pkl')
    os.makedirs('metrics', exist_ok=True)
    with open('metrics/scores.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    mlflow.log_artifact('metrics/scores.json', artifact_path='metrics')
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="examen-dvc-random-forest",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )