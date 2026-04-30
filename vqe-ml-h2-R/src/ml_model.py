import os
import pandas as pd
from sklearn.neural_network import MLPRegressor
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'data'))

FILE_PATH = os.path.join(DATA_DIR, 'vqe_dataset.csv')
if os.path.exists(FILE_PATH):
    df = pd.read_csv(FILE_PATH)
    print("File loaded successfully!")
else:
    print(f"Error: File not found at {FILE_PATH}")


X = df[["R"]]
y = df.filter(like="theta")

#model = MLPRegressor(hidden_layer_sizes=(64,64))
model = MLPRegressor(
    hidden_layer_sizes=(128, 128, 64),
    activation='relu',
    solver='adam',
    max_iter=3000,
    learning_rate_init=0.001,
    random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

model.fit(X_scaled, y_scaled)

MODEL_SAVE_PATH = os.path.join(DATA_DIR, 'sklearn_model.joblib')

# 3. Save the model
# Assuming 'model' is your trained sklearn object (e.g., RandomForest, LinearRegression)
joblib.dump(model, MODEL_SAVE_PATH)

print(f"Model successfully saved to: {MODEL_SAVE_PATH}")