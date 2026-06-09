import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# CSV dosyasını yükle
data_dir = "C:/Users/NejoxZz/Desktop/ECG_PROJECT/ptbxl"
df = pd.read_csv(f"{data_dir}/ptbxl_database.csv")

# scp_codes sütununu string -> dict -> liste çevir
df['scp_codes'] = df['scp_codes'].apply(eval)  # str -> dict
labels = df['scp_codes'].apply(lambda x: list(x.keys()))  # sadece etiket isimlerini al

# One-hot encoding (Multi-label)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(labels)

# Gerekirse çıktı bak
print("Etiket sayısı:", len(mlb.classes_))
print("Etiket isimleri:", mlb.classes_)
print("y şekli:", y.shape)
