from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load trained models
xgb_model = joblib.load("models/xgboost_model.pkl")
iso_model = joblib.load("models/isolation_forest.pkl")

# Define home page
@app.route("/")
def home():
    return render_template("index.html")

# API for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from form
        data = {
            "amount": float(request.form["amount"]),
            "transaction_type": request.form["transaction_type"],
            "merchant": request.form["merchant"],
            "customer_age": int(request.form["customer_age"])
        }

        # Convert to DataFrame (make sure columns match training dataset)
        df = pd.DataFrame([data])

        # One-hot encoding (if categorical vars exist)
        df = pd.get_dummies(df)

        # Align with training features
        # (important: same feature order as training)
        xgb_features = xgb_model.get_booster().feature_names
        df = df.reindex(columns=xgb_features, fill_value=0)

        # Get predictions
        xgb_pred = xgb_model.predict(df)[0]
        iso_pred = iso_model.predict(df)[0]

        result = {
            "XGBoost": "Fraud" if xgb_pred == 1 else "Legit",
            "IsolationForest": "Fraud" if iso_pred == -1 else "Legit"
        }

        return render_template("index.html", prediction=result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
