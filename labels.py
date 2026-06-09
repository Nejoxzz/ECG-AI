import pandas as pd
import ast

df = pd.read_csv('ptbxl/ptbxl_database.csv')  # senin path'e göre ayarla

df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)
labels = df['scp_codes'].apply(lambda x: list(x.keys()))

print(labels.head())
