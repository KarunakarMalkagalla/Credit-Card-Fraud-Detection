import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import streamlit as st
import joblib # For saving/loading the model

# --- Data Loading and Model Training (only run once) ---
# It's better to train your model and save it, then load it in the app.
# For demonstration, I'll keep it here, but ideally this part is separated.

@st.cache_data # Cache the data loading to avoid re-loading on every rerun
def load_and_prepare_data():
    data = pd.read_csv('creditcard.csv')
    # Using class_weight in LogisticRegression, so explicit undersampling is not strictly needed here
    # However, if you want to explicitly undersample for a balanced dataset before training, keep these lines:
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]
    # If you still want to undersample for training, use:
    # legit_sample = legit.sample(n=len(fraud), random_state=2)
    # data_balanced = pd.concat([legit_sample, fraud], axis=0)
    # return data_balanced

    return data

data = load_and_prepare_data()

X = data.drop(columns="Class", axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train Logistic Regression model with balanced class weights
# solver='liblinear' is often a good choice for smaller datasets and handles L1/L2 penalties well
model = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000) # Increased max_iter for convergence

model.fit(X_train, y_train)

# Evaluate model performance (displaying in app)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_proba_test = model.predict_proba(X_test)[:, 1] # Probability of being class 1 (fraud)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)
test_roc_auc = roc_auc_score(y_test, y_proba_test)

# --- Streamlit App ---
st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

st.subheader("Model Performance on Test Set:")
st.write(f"**Accuracy:** {test_acc:.4f}")
st.write(f"**Precision:** {test_precision:.4f}")
st.write(f"**Recall:** {test_recall:.4f}")
st.write(f"**F1-Score:** {test_f1:.4f}")
st.write(f"**ROC AUC Score:** {test_roc_auc:.4f}")
st.markdown("---")


# Create individual input fields for each feature
# Get average values to use as default inputs (optional, but helpful)
avg_values = X.mean().values

col1, col2, col3 = st.columns(3) # Use columns for better layout

input_features = {}
feature_names = X.columns.tolist() # Get actual feature names from the DataFrame

# Create input fields dynamically
for i, feature_name in enumerate(feature_names):
    if i % 3 == 0:
        with col1:
            input_features[feature_name] = st.number_input(f'{feature_name}', value=float(avg_values[i]), format="%.6f", key=feature_name)
    elif i % 3 == 1:
        with col2:
            input_features[feature_name] = st.number_input(f'{feature_name}', value=float(avg_values[i]), format="%.6f", key=feature_name)
    else:
        with col3:
            input_features[feature_name] = st.number_input(f'{feature_name}', value=float(avg_values[i]), format="%.6f", key=feature_name)


# Convert input_features dictionary to a numpy array in the correct order
features_array = np.array([input_features[name] for name in feature_names]).reshape(1, -1)

# Create a button to submit input and get prediction
submit = st.button("Predict")

if submit:
    try:
        # Make prediction
        prediction = model.predict(features_array)
        prediction_proba = model.predict_proba(features_array)[0] # Get probabilities for both classes

        # Display result
        if prediction[0] == 0:
            st.success(f"**Prediction: Legitimate transaction** (Probability of fraud: {prediction_proba[1]:.4f})")
        else:
            st.error(f"**Prediction: Fraudulent transaction** (Probability of fraud: {prediction_proba[1]:.4f})")

    except Exception as e:
        st.error(f"An error occurred: {e}. Please ensure all fields are filled correctly.")
