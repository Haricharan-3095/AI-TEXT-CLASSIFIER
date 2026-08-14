import pandas as pd

# Load the dataset
df = pd.read_csv("data/customer_support.csv")

# Display the dataset
print(df)

# Display the number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Display the column names
print("\nColumns:")
print(df.columns)

# Display how many examples belong to each category
print("\nCategory distribution:")
print(df["label"].value_counts())