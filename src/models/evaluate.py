import pandas as pd
import json
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

os.makedirs('metrics', exist_ok=True)

X_test = pd.read_csv('data/processed/X_test_scaled.csv')
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
model = joblib.load('models/model.pkl')

predictions = model.predict(X_test)

# Sauvegarde des prédictions
df_preds = pd.DataFrame({'actual': y_test, 'predicted': predictions})
df_preds.to_csv('data/predictions.csv', index=False)

# Calcul des métriques
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

metrics = {
    'mse': mse,
    'r2': r2
}

with open('metrics/scores.json', 'w') as f:
    json.dump(metrics, f, indent=4)