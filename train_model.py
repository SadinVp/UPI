# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
# 1. Load dataset
df = pd.read_csv("upidataset.csv")   # change name if different

print("Dataset shape:", df.shape)
print(df.head())


# 3. Preprocessing
target = "fraud"

# Identify categorical and numerical columns
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
num_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()

if target in num_cols:
    num_cols.remove(target)

# Encode categorical
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Features & labels
X = df.drop(columns=[target])
y = df[target]

# Scale numerical features
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Train XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=200, max_depth=6)
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)
y_proba = xgb.predict_proba(X_test)[:,1]

print("XGBoost Report:\n", classification_report(y_test, y_pred))
print("XGBoost ROC-AUC:", roc_auc_score(y_test, y_proba))

# 6. Train Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X_train)
iso_preds = iso.predict(X_test)
iso_preds = [1 if p == -1 else 0 for p in iso_preds]  # -1 = anomaly → fraud

print("Isolation Forest Report:\n", classification_report(y_test, iso_preds))

# 7. Save models & scaler
os.makedirs("models", exist_ok=True)
joblib.dump(xgb, "models/xgboost_model.pkl")
joblib.dump(iso, "models/isolation_forest.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Models trained and saved successfully!")
print(df['fraud'].value_counts())