import pandas as pd
import numpy as np
import time


# Create a large sample DataFrame
df = pd.DataFrame({'price': np.random.randint(50, 500, 1000000)})

# with chunking for large file size 
# Define the chunk size
chunksize = 100000 
# Initialize variables to hold the sum and count
total_sum = 0
total_count = 0
# Read the CSV in chunks
for chunk in pd.read_csv(file_path, chunksize=chunksize):
    # Calculate the sum and count for the current chunk
    total_sum += chunk['sales'].sum()
    total_count += chunk['sales'].count()
# Calculate the overall mean
overall_mean = total_sum / total_count
print(f"The mean of the 'sales' column is: {overall_mean}")


# Show first/last few rows
df.head()       # first 5 rows
df.tail(10)     # last 10 rows

# Column names & data types
df.columns
df.dtypes
df.shape  # (rows, columns)

# Reset index
df.reset_index(drop=True, inplace=True)


# Check missing values
df.isnull().sum()
df.dropna()         # drop rows with NaN
df.dropna(axis=1)   # drop columns with NaN

# concat
pd.concat([df1, df2], axis=0)  # rows
pd.concat([df1, df2], axis=1)  # columns

# .copy() vs assignment
df2 = df[['col']].copy()  # avoids SettingWithCopyWarning

# apply function
df['col2'] = df['col1'].apply(lambda x: x**2)


# Handle mismatched keys
pd.merge(df1, df2, on='id', how='outer')
# Merge on key
pd.merge(df1, df2, on='id', how='inner')

# group by
df.groupby('category')['value'].mean().sort_values(ascending=False)

# EDA
df['col'].hist()

 
df = pd.DataFrame({'a':[10,20,30]}, index=['x','y','z'])
df.loc['x':'y']    # returns rows 'x' AND 'y' -- NOTE: y is included 
df.iloc[0:2]       # returns rows at positions 0 and 1 (excludes pos 2)



# Vector vs for loop
df = pd.DataFrame({'price': np.random.randint(50, 500, 1000000)})
start_time = time.time()
# Using a for loop and .iterrows()
# This is a slow, row-by-row operation
new_prices = []
for index, row in df.iterrows():
    new_prices.append(row['price'] * 0.9)
df['discounted_price_loop'] = new_prices
end_time = time.time()
print(f"Time taken with for loop: {end_time - start_time} seconds")

start_time = time.time()

# Using a vectorized operation
# Pandas applies the operation to the entire Series at once
df['discounted_price_vectorized'] = df['price'] * 0.9
end_time = time.time()
print(f"Time taken with vectorization: {end_time - start_time} seconds")


# Optimizing data type
data = {
    'user_id': np.random.randint(1, 50001, 1000000),
    'product_category': np.random.choice(['Electronics', 'Books', 'Clothing', 'Home', 'Toys'], 1000000),
    'transaction_amount': np.random.rand(1000000) * 1000
}
df = pd.DataFrame(data)
# Print initial memory usage
print("Memory usage before optimization:")
print(df.memory_usage(deep=True).sum())
# Convert 'user_id' to a more memory-efficient integer type
# 'int16' is sufficient for numbers up to 32,767, 'int32' for up to 2,147,483,647
df['user_id'] = df['user_id'].astype('int32')
# Convert 'product_category' to the 'category' data type
# This is ideal for string columns with low cardinality (few unique values)
df['product_category'] = df['product_category'].astype('category')
# Print memory usage after optimization
print("\nMemory usage after optimization:")
print(df.memory_usage(deep=True).sum())
