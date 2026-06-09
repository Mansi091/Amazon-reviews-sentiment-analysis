import csv
import sys
import os

csv_path = r"c:\PROJECTS\AMAZON_REVIEWS_SENTIMENT\Reviews.csv"

if not os.path.exists(csv_path):
    print(f"Error: {csv_path} does not exist.")
    sys.exit(1)

print(f"CSV File Size: {os.path.getsize(csv_path) / (1024*1024):.2f} MB")

try:
    import pandas as pd
    print("Pandas is installed. Reading with Pandas...")
    df = pd.read_csv(csv_path, nrows=5)
    print("Columns:", list(df.columns))
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
except ImportError:
    print("Pandas not installed. Reading with built-in csv module...")
    with open(csv_path, mode='r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        header = next(reader)
        print("Columns:", header)
        print("\nFirst 3 rows:")
        for i in range(3):
            try:
                row = next(reader)
                print(f"Row {i+1}: {row}")
            except StopIteration:
                break
