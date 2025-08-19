■ UPI Fraud Detection Using Machine Learning
This project detects fraudulent UPI (Unified Payments Interface) transactions using XGBoost and
Isolation Forest models.
It also includes a Flask web application with an aesthetic UI where users can enter transaction
details and check whether a transaction is Fraudulent or Legit.
■ Features
- Fraud detection using XGBoost (Supervised) and Isolation Forest (Unsupervised)
- Achieved ~100% accuracy with XGBoost on the dataset
- Flask Web Application with a modern UI
- Real-time prediction from user input
- Clean project structure for deployment
■ Project Structure
UPI-Fraud-Detection/
- app.py (Flask application)
- train_model.py (Script to train and save models)
- models/ (Saved ML models)
- templates/ (HTML templates)
- static/ (Optional CSS/JS/images)
- dataset.csv (Fraud detection dataset from Kaggle)
- requirements.txt (Required Python packages)
- README.md (Documentation)
■ Dataset
- 7,000 transactions with 20 features.
- Target variable: fraud (0 = Legit, 1 = Fraud).
- Source: Kaggle (Synthetic UPI Fraud Detection dataset).
■■ Installation
1. Clone the repository
git clone https://github.com/SadinVP/UPI
cd upi-fraud-detection
2. Create a virtual environment
python -m venv venv
source venv/bin/activate (Mac/Linux)
venv\Scripts\activate (Windows)
3. Install dependencies
pip install -r requirements.txt
■ Running the Project
1. Train the models
python train_model.py
2. Run the Flask app
python app.py
3. Open in browser at http://127.0.0.1:5000/
■ Requirements
- Python 3.8+
- Flask
- Pandas
- Scikit-learn
- XGBoost
- Joblib
■ Model Performance
XGBoost:
- Precision: 1.00
- Recall: 1.00
- F1-score: 1.00
- ROC-AUC: 1.0
Isolation Forest:
- Precision: 0.61 (fraud class)
- Recall: 0.13 (fraud class)
■■ Future Improvements
- Integrate with real UPI transaction logs
- Add deep learning models (LSTMs)
- Deploy on Heroku / AWS / GCP
■■■ Author
Mohammed Sadin V P
Final Year B.Tech AI & Data Science Student
GitHub: https://github.com/Sadinvp
LinkedIn: https://linkedin.com/in/mohammedsadinvp
