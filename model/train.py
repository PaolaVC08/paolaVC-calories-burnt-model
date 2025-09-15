from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
import pandas as pd
import pathlib

# Leer dataset
df = pd.read_csv(pathlib.Path('data/calories.csv'))

# Convertir género a número
df['Gender'] = df['Gender'].map({'male': 1, 'female': 0})

# Quitar User_ID
df = df.drop(columns=['User_ID'])

# Separar target
y = df.pop('Calories')
X = df

# Split en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Training model..')
clf = RandomForestRegressor(n_estimators=100,
                            max_depth=10,
                            random_state=0)

clf.fit(X_train, y_train)

# Evaluación rápida
y_pred = clf.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

print('Saving model..')
dump(clf, pathlib.Path('model/calories-v1.joblib'))

