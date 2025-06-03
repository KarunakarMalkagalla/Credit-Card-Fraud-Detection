import numpy as np
import pandas as pd
import os
import gdown
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import streamlit as st

# === Download the dataset from Google Drive ===
file_id = '1kf0xO4s8oi6rB0V61dl3zFaHf8iOzbkf'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'creditcard.csv'

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# === Load and prepare data ===
data = pd.read_csv(output)

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split features and labels
X = data.drop(columns="Class", axis=1)
y = data["Class"]

# === Feature Scaling ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=2)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy (optional display)
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# === Streamlit UI ===
st.title("💳 Credit Card Fraud Detection")
st.write("Enter all 30 feature values separated by commas to check if a transaction is **legitimate or fraudulent**.")

# Input field
input_text = st.text_input("📝 Feature Input", placeholder="Enter 30 comma-separated values like: 0.1, -1.2, ...")

# Submit button
if st.button("Submit"):
    try:
        # Parse and convert input to float array
        input_list = [float(i.strip()) for i in input_text.split(',')]
        
        if len(input_list) != X.shape[1]:
            st.error(f"⚠️ Please enter exactly {X.shape[1]} features.")
        else:
            # Apply the same scaling to the input
            input_array = np.array(input_list).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]
            result = "✅ Legitimate transaction" if prediction == 0 else "🚨 Fraudulent transaction"
            st.success(result)
    except Exception as e:
        st.error("⚠️ Invalid input. Please enter numeric values only.")
