import pandas as pd
import numpy as np

# Test reading Excel files
excel_file = "E:/generate_mixture/gcms/hanliang/22kvGCMS产物检测.xlsx"

try:
    df = pd.read_excel(excel_file)
    print("Excel file content:")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nFirst column content:")
    print(df.iloc[:, 0].values)
    
except Exception as e:
    print(f"Error reading Excel: {e}")