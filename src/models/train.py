import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
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
best_params = joblib.load('models/best_params.pkl')

with mlflow.start_run(run_name="random-forest-training"):
    mlflow.log_params(best_params)
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, 'models/model.pkl')
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="examen-dvc-random-forest",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )