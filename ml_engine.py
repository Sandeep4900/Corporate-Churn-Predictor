import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

print("\n[SYSTEM] Initializing Machine Learning Churn Engine...")

# 1. GENERATE SYNTHETIC CORPORATE DATA
# Simulating 5,000 customers for a telecom/SaaS business
np.random.seed(42)
num_customers = 5000

data = {
    'Customer_Age': np.random.randint(18, 70, num_customers),
    'Monthly_Bill': np.random.uniform(500, 5000, num_customers),
    'Support_Calls_Last_Month': np.random.randint(0, 6, num_customers),
    'Months_Subscribed': np.random.randint(1, 60, num_customers),
    'Payment_Delay_Days': np.random.randint(0, 30, num_customers)
}
df = pd.DataFrame(data)

# Logic: Customers with high bills, many support calls, and payment delays are more likely to churn (leave).
churn_probability = (
    (df['Support_Calls_Last_Month'] * 0.15) + 
    (df['Payment_Delay_Days'] * 0.02) + 
    (df['Monthly_Bill'] / 5000 * 0.1) - 
    (df['Months_Subscribed'] * 0.01)
)
# Convert probability to a 1 (Churn) or 0 (Stay)
df['Churn_Risk'] = np.where(churn_probability + np.random.normal(0, 0.1, num_customers) > 0.6, 1, 0)

print(f"[DATA] Successfully loaded dataset: {len(df)} customer records.")

# 2. TRAIN THE ARTIFICIAL INTELLIGENCE
print("[AI] Training Random Forest Classification Model...")
# X contains the behavior data, y contains the answer (Did they churn?)
X = df.drop('Churn_Risk', axis=1)
y = df['Churn_Risk']

# Split data: 80% to train the AI, 20% to test if it actually works
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Algorithm
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. TEST THE MODEL AND PREDICT
predictions = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, predictions)

# 4. GENERATE THE CORPORATE EXECUTIVE REPORT
print("\n" + "="*50)
print("📊 EXECUTIVE CHURN PREDICTION REPORT")
print("="*50)

# Calculate financial metrics
average_customer_value = df['Monthly_Bill'].mean()
customers_saved = conf_matrix[1][1] # True Positives (Correctly identified churners)
revenue_saved = customers_saved * average_customer_value

print(f"Model Accuracy Score:      {model.score(X_test, y_test) * 100:.1f}%")
print(f"High-Risk Accounts Caught: {customers_saved} customers")
print(f"Potential Revenue Saved:   ₹ {revenue_saved:,.2f} per month\n")

print("--- AI DECISION WEIGHTS (Feature Importance) ---")
importance = model.feature_importances_
for feature, weight in zip(X.columns, importance):
    print(f"• {feature.replace('_', ' ')}: {weight*100:.1f}% impact")

print("\n[SYSTEM] Run complete. Ready for deployment.")