import pyreadstat
import os

folder_path = r"E:\Children Project\Bangladesh MICS6 SPSS Datasets"
file_path = os.path.join(folder_path, "fs.sav")

if not os.path.exists(file_path):
    print("ERROR: fs.sav not found.")
else:
    df, meta = pyreadstat.read_sav(file_path)
    print("SUCCESS: File loaded.")
    print("Total rows (children):", len(df))
    print("Total columns (variables):", len(df.columns))
    print("\nFirst 3 rows:")
    print(df.head(3))