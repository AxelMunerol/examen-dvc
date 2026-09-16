import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.makedirs('data/processed', exist_ok=True)
data = pd.read_csv('data/raw/raw.csv')


numeric_data = data.select_dtypes(include=['float64', 'int64'])

# La variable cible est la dernière colonne des données numériques
X = numeric_data.iloc[:, :-1]
y = numeric_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)