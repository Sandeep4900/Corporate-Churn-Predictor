import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Churn Predictor", layout="wide")

st.title("📉 Corporate Customer Churn AI")
st.markdown("Automated Machine Learning pipeline predicting high-risk subscription cancellations.")
st.markdown("---")

with st.spinner("Initializing AI and generating corporate dataset..."):
    # 1. GENERATE SYNTHETIC CORPORATE DATA
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

    churn_probability = (
        (df['Support_Calls_Last_Month'] * 0.15) + 
        (df['Payment_Delay_Days'] * 0.02) + 
        (df['Monthly_Bill'] / 5000 * 0.1) - 
        (df['Months_Subscribed'] * 0.01)
    )
    df['Churn_Risk'] = np.where(churn_probability + np.random.normal(0, 0.1, num_customers) > 0.6, 1, 0)

    # 2. TRAIN THE ARTIFICIAL INTELLIGENCE
    X = df.drop('Churn_Risk', axis=1)
    y = df['Churn_Risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 3. TEST THE MODEL AND PREDICT
    predictions = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predictions)

    # 4. FINANCIAL METRICS
    average_customer_value = df['Monthly_Bill'].mean()
    customers_saved = conf_matrix[1][1] 
    revenue_saved = customers_saved * average_customer_value
    accuracy = model.score(X_test, y_test) * 100

# 5. UI RENDER
st.subheader("📊 Executive Financial Impact Report")

col1, col2, col3 = st.columns(3)
col1.metric("AI Accuracy Score", f"{accuracy:.1f}%")
col2.metric("High-Risk Accounts Caught", f"{customers_saved} Users")
col3.metric("Monthly Revenue Saved", f"₹ {revenue_saved:,.2f}", delta="Protected Capital")

st.markdown("### 🧠 AI Decision Weights (Feature Importance)")
st.write("This shows exactly which factors the algorithm weighted most heavily when deciding if a customer was going to leave.")

importance_df = pd.DataFrame({
    'Metric': [col.replace('_', ' ') for col in X.columns],
    'Impact Weight': model.feature_importances_
}).sort_values(by='Impact Weight', ascending=False)

st.bar_chart(data=importance_df, x='Metric', y='Impact Weight', color="#00ff41")
