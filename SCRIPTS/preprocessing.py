import pandas as pd
import pyarrow.parquet as pq





# Replace with your actual file path
file_path = "DATA/train-00001-of-00002-823ac5dae71e0e87.parquet"

table = pq.read_table(file_path)
df = table.to_pandas()

print(df.head())         # Quick look at the first few rows
print(df.info())         # Column types and memory usage
print(df.describe())     # Basic stats
