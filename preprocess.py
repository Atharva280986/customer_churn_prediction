import pandas as pd

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display basic information
print("Dataset loaded successfully")
print("Shape:", df.shape)
print(df.head())