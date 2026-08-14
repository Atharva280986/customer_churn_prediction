import pandas as pd
import joblib

# Load model
model = joblib.load("churn_model.pkl")

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert TotalCharges into numerical data
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Remove missing values
df = df.dropna()

# Remove customer ID
df = df.drop("customerID", axis=1)

# Remove target column
X = df.drop("Churn", axis=1)

# Convert categorical columns into numbers
X = pd.get_dummies(X)

# Make prediction data same as training data
X = pd.get_dummies(X)

# Get columns expected by trained model
expected_features = model.get_booster().feature_names

# Add missing columns and keep correct order
X = X.reindex(columns=expected_features, fill_value=0)
# Make sure columns match the training model
prediction = model.predict(X)

# Show prediction for first customer
if prediction[0] == 1:
    print("Customer is likely to churn")
else:
    print("Customer is not likely to churn")