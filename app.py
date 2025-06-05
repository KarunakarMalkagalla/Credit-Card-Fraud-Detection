import numpy as np
import pandas as pd
import os
import gdown
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import streamlit as st

# === Download the dataset from Google Drive ===
file_id = '1kf0xO4s8oi6rB0V61dl3zFaHf8iOzbkf'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'creditcard.csv'

# Check if the file exists before downloading
if not os.path.exists(output):
    st.info("Downloading creditcard.csv dataset...") # Inform user about download
    gdown.download(url, output, quiet=False)
    st.success("Download complete!")

# === Load and prepare data ===
data = pd.read_csv(output)

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to match the number of fraudulent transactions
# This helps balance the dataset for training
legit_sample = legit.sample(n=len(fraud), random_state=2)
data_balanced = pd.concat([legit_sample, fraud], axis=0) # Renamed to avoid confusion with original 'data'

# Split features (X) and labels (y) from the balanced dataset
X = data_balanced.drop(columns="Class", axis=1)
y = data_balanced["Class"]

# === Feature Scaling ===
# Standardize features to have a mean of 0 and variance of 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
# stratify=y ensures that the proportion of legitimate/fraudulent transactions
# is maintained in both training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=2)

# === Train the Logistic Regression model ===
# Using class_weight='balanced' to automatically handle the class imbalance.
# This gives more importance to the minority class (fraud) during training.
# solver='liblinear' is a good choice for smaller datasets and handles class_weight well.
# max_iter increased to ensure convergence.
model = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000)
model.fit(X_train, y_train)

# === Evaluate Model Performance ===
# Predict on training and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class (fraud)

# Calculate and display metrics
train_acc = accuracy_score(y_train_pred, y_train)
test_acc = accuracy_score(y_test_pred, y_test)

test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_proba)

# === Streamlit UI ===
st.title("💳 Credit Card Fraud Detection")
st.write("Enter all 30 feature values separated by commas to check if a transaction is **legitimate or fraudulent**.")

# Display model performance metrics in the app
st.subheader("📊 Model Performance (on Test Set)")
st.markdown(f"- **Accuracy:** `{test_acc:.4f}`")
st.markdown(f"- **Precision:** `{test_precision:.4f}`")
st.markdown(f"- **Recall (Fraud Detection Rate):** `{test_recall:.4f}`")
st.markdown(f"- **F1-Score:** `{test_f1:.4f}`")
st.markdown(f"- **ROC AUC Score:** `{test_roc_auc:.4f}`")
st.info("💡 **Recall** is crucial for fraud detection, as it measures how many actual fraud cases the model caught.")
st.write("---")

# Input field for user
input_text = st.text_input("📝 Feature Input", placeholder="Enter 30 comma-separated values like: 0.1, -1.2, ...")

# Submit button
if st.button("Submit"):
    try:
        # Parse and convert input to float array
        input_list = [float(i.strip()) for i in input_text.split(',')]

        if len(input_list) != X.shape[1]:
            st.error(f"⚠️ Please enter exactly {X.shape[1]} features (Time, V1-V28, Amount).")
        else:
            # Convert input to a numpy array and reshape for scaling
            input_array = np.array(input_list).reshape(1, -1)
            # Apply the same scaling used during training to the user's input
            input_scaled = scaler.transform(input_array)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Display result
            result = "✅ Legitimate transaction" if prediction == 0 else "🚨 Fraudulent transaction"
            st.success(result)
            
            # Optional: Show prediction probability
            prediction_proba = model.predict_proba(input_scaled)[0]
            st.write(f"Confidence (Legitimate): {prediction_proba[0]:.2f}")
            st.write(f"Confidence (Fraudulent): {prediction_proba[1]:.2f}")

    except ValueError:
        st.error("⚠️ Invalid input. Please ensure all values are numeric and separated by commas.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
