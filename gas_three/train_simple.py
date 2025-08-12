
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load data
X_train = pd.read_csv("data/processed/X_train.csv").values
Y_train = pd.read_csv("data/processed/Y_train.csv").values
X_test = pd.read_csv("data/processed/X_test.csv").values
Y_test = pd.read_csv("data/processed/Y_test.csv").values

# Standardize
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
Y_train_scaled = scaler_Y.fit_transform(Y_train)

# Train model
model = PLSRegression(n_components=10)
model.fit(X_train_scaled, Y_train_scaled)

# Predict and evaluate
Y_test_pred_scaled = model.predict(X_test_scaled)
Y_test_pred = scaler_Y.inverse_transform(Y_test_pred_scaled)

rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
r2 = r2_score(Y_test, Y_test_pred)

print(f"Model Performance: RMSE={rmse:.5f}, R2={r2:.5f}")

# Save model and scalers
os.makedirs("data/models", exist_ok=True)
joblib.dump(model, "data/models/enhanced_pls_model.pkl")
joblib.dump(scaler_X, "data/models/scaler_X.pkl")
joblib.dump(scaler_Y, "data/models/scaler_Y.pkl")

print("Model saved successfully!")
