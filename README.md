# Corporate Customer Churn Predictor (Machine Learning)

An automated data science pipeline built to identify at-risk customers before they cancel subscriptions, calculating the direct financial impact of retention.

## Business Value & Analytics
Acquiring a new customer costs significantly more than retaining an existing one. This engine simulates a corporate environment (5,000 users) and applies Artificial Intelligence to flag high-risk accounts based on billing history, support tickets, and payment delays.
* Predictive AI: Utilizes a 'RandomForestClassifier' to map complex, non-linear customer behavior patterns.
* Financial Translation: Automatically converts precision/recall metrics into a live "Revenue Saved" corporate report.
* Feature Importance: Extracts the exact statistical weights driving the AI's decisions, allowing management to fix the root causes of churn.

## 🛠️ Tech Stack & Execution
* Language: Python 3
* Data Manipulation: NumPy, Pandas
* Machine Learning: Scikit-Learn ('train_test_split', 'RandomForestClassifier', 'confusion_matrix')

To run the localized pipeline:
'python ml_engine.py'
