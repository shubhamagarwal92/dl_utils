import pandas as pd


df = pd.read_csv("file.csv")  # sep=";", header=0 etc. if needed

# Show first/last few rows
df.head()       # first 5 rows
df.tail(10)     # last 10 rows

# Column names & data types
df.columns
df.dtypes
df.shape  # (rows, columns)

# Check missing values
df.isnull().sum()
df.dropna()         # drop rows with NaN
df.dropna(axis=1)   # drop columns with NaN

# concat
pd.concat([df1, df2], axis=0)  # rows
pd.concat([df1, df2], axis=1)  # columns

# Handle mismatched keys
pd.merge(df1, df2, on='id', how='outer')
