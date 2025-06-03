import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler # Import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import streamlit as st
import joblib # For saving/loading the model

# --- Data Loading and Model Training (run once and save) ---

# @st.cache_data is good for loading data that doesn't change,
# preventing it from reloading every time the app reruns.
@st.cache_data
def load_and_prepare_data():
    data = pd.read_csv('creditcard.csv')
    return data

data = load_and_prepare_data()

# Separate features (X) and target (y)
X = data.drop(columns="Class", axis=1)
y = data["Class"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# --- Feature Scaling ---
# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train Logistic Regression model ---
model = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000, random_state=2)
model.fit(X_train_scaled, y_train) # Train on scaled data

# --- Save the trained model and scaler ---
# It's good practice to save these so you don't retrain every time the app starts
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# --- Load the model and scaler (for actual app use) ---
# Use st.cache_resource to load models/scalers once
@st.cache_resource
def load_saved_model_and_scaler():
    loaded_model = joblib.load('logistic_regression_model.pkl')
    loaded_scaler = joblib.load('scaler.pkl')
    return loaded_model, loaded_scaler

loaded_model, loaded_scaler = load_saved_model_and_scaler()

# --- Evaluate model performance on the scaled test set ---
y_pred_test = loaded_model.predict(X_test_scaled)
y_proba_test = loaded_model.predict_proba(X_test_scaled)[:, 1]

test_acc = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)
test_roc_auc = roc_auc_score(y_test, y_proba_test)

# --- Streamlit App ---
st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features (Time, V1-V28, Amount) separated by commas:")
st.markdown("---")

# Display model performance in the sidebar or at the top for context
st.subheader("Model Performance on Test Set:")
st.write(f"**Accuracy:** {test_acc:.4f}")
st.write(f"**Precision:** {test_precision:.4f} (Ability of the model to avoid false positives)")
st.write(f"**Recall:** {test_recall:.4f} (Ability of the model to find all the positive samples)")
st.write(f"**F1-Score:** {test_f1:.4f} (Harmonic mean of precision and recall)")
st.write(f"**ROC AUC Score:** {test_roc_auc:.4f} (Overall measure of classification performance)")
st.markdown("---")

# Create a single text input field for user to enter all feature values
input_df = st.text_input('Input all 30 features separated by commas (e.g., Time,V1,...,V28,Amount)')

# Create a button to submit input and get prediction
submit = st.button("Predict")

if submit:
    if input_df: # Check if input is not empty
        try:
            # Split the input string by commas and convert to float
            input_df_lst = input_df.split(',')
            
            # Ensure exactly 30 features are provided
            if len(input_df_lst) != 30:
                st.error(f"Please enter exactly 30 feature values. You entered {len(input_df_lst)}.")
            else:
                # Convert list of strings to numpy array of floats
                features = np.array(input_df_lst, dtype=np.float64)
                
                # Reshape for single prediction (1 row, 30 columns)
                # The model expects a 2D array, even for a single sample.
                input_features_reshaped = features.reshape(1, -1)
                
                # --- Scale the user input using the loaded scaler ---
                scaled_input_features = loaded_scaler.transform(input_features_reshaped)
                
                # Make prediction using the loaded model on scaled input
                prediction = loaded_model.predict(scaled_input_features)
                prediction_proba = loaded_model.predict_proba(scaled_input_features)[0] # Get probabilities

                # Display result
                if prediction[0] == 0:
                    st.success(f"**Prediction: Legitimate transaction** (Probability of fraud: {prediction_proba[1]:.4f})")
                else:
                    st.error(f"**Prediction: Fraudulent transaction** (Probability of fraud: {prediction_proba[1]:.4f})")
        except ValueError:
            st.error("Please enter valid numerical feature values separated by commas.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}. Please check your input format.")
    else:
        st.warning("Please enter the feature values before clicking Predict.")
