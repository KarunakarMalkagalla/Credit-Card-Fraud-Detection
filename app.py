import numpy as npAdd commentMore actions
import pandas as pd
import os
import gdown
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# load data
data = pd.read_csv('creditcard.csv')
# === Download the dataset from Google Drive ===
file_id = '1kf0xO4s8oi6rB0V61dl3zFaHf8iOzbkf'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'creditcard.csv'

# separate legitimate and fraudulent transactions
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# === Load and prepare data ===
data = pd.read_csv(output)

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# undersample legitimate transactions to balance the classes
# Undersample legitimate transactions
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# split data into training and testing sets
# Split features and labels
X = data.drop(columns="Class", axis=1)
y = data["Class"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# train logistic regression model
# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# evaluate model performance
# Accuracy (optional display)
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# create Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# create input fields for user to enter feature values
input_df = st.text_input('Input all features separated by commas')
input_df_lst = input_df.split(',')
# === Streamlit UI ===
st.title("💳 Credit Card Fraud Detection")
st.write("Enter all 30 feature values separated by commas to check if a transaction is **legitimate or fraudulent**.")

# create a button to submit input and get prediction
submit = st.button("Submit")
# Input field
input_text = st.text_input("📝 Feature Input", placeholder="Enter 30 comma-separated values like: 0.1, -1.2, ...")

if submit:
# Submit button
if st.button("Submit"):
    try:
        # get input feature values
        features = np.array(input_df_lst, dtype=np.float64)
        # make prediction
        prediction = model.predict(features.reshape(1, -1))
        # display result
        if prediction[0] == 0:
            st.write("Legitimate transaction")
        # Parse and convert input to float array
        input_list = [float(i.strip()) for i in input_text.split(',')]
        
        if len(input_list) != X.shape[1]:
            st.error(f"⚠️ Please enter exactly {X.shape[1]} features.")
        else:
            st.write("Fraudulent transaction")
    except ValueError:
        st.write("Please enter valid feature values separated by commas.")
            input_array = np.array(input_list).reshape(1, -1)
            prediction = model.predict(input_array)[0]
            result = "✅ Legitimate transaction" if prediction == 0 else "🚨 Fraudulent transaction"
            st.success(result)
    except Exception as e:
        st.error("⚠️ Invalid input. Please enter numeric values only.")
