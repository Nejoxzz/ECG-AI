import os
import numpy as np
import pandas as pd
import wfdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Dosya ve klasör yollarını buraya yaz
base_dir_lr = r'C:/Users/NejoxZz/Desktop/ECG_PROJECT/ptbxl/records100'  # LR kayıtların klasörü
base_dir_hr = r'C:/Users/NejoxZz/Desktop/ECG_PROJECT/ptbxl/records500'  # HR kayıtların klasörü
csv_path = r'C:/Users/NejoxZz/Desktop/ECG_PROJECT/ptbxl/ptbxl_database.csv'  # CSV dosyası

# CSV oku
df = pd.read_csv(csv_path)

# scp_codes sütunundaki stringleri listelere çevir (etiketler)
df['scp_codes'] = df['scp_codes'].apply(lambda x: list(eval(x).keys()) if pd.notna(x) else [])

# Multi-label etiketleri binary matrise dönüştür
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['scp_codes'])

print(f"Etiket sayısı: {len(mlb.classes_)}")
print(f"Etiket isimleri: {mlb.classes_}")

def load_combined_signal(row):
    try:
        lr_path = row['filename_lr']
        hr_path = row['filename_hr']

        # Dosya yolundaki prefixleri temizle
        if lr_path.startswith('records100/'):
            lr_path = lr_path[len('records100/'):]
        if hr_path.startswith('records500/'):
            hr_path = hr_path[len('records500/'):]

        full_path_lr = os.path.join(base_dir_lr, lr_path)
        full_path_hr = os.path.join(base_dir_hr, hr_path)

        print(f"Yükleniyor LR: {full_path_lr}")
        print(f"Yükleniyor HR: {full_path_hr}")

        record_lr = wfdb.rdrecord(full_path_lr)
        record_hr = wfdb.rdrecord(full_path_hr)

        sig_lr = record_lr.p_signal
        sig_hr = record_hr.p_signal

        if sig_lr.ndim > 1:
            sig_lr = sig_lr.mean(axis=1)
        if sig_hr.ndim > 1:
            sig_hr = sig_hr.mean(axis=1)

        min_len = min(len(sig_lr), len(sig_hr))
        combined = np.vstack([sig_lr[:min_len], sig_hr[:min_len]]).T  # shape (min_len, 2)

        return combined
    except Exception as e:
        print(f"Hata: {row['ecg_id']} okunamadı: {e}")
        return None

print("ECG sinyalleri yükleniyor (combined LR + HR)...")
signals = df.apply(load_combined_signal, axis=1)

# None olanları çıkar
signals = [s for s in signals if s is not None]
X = np.array(signals)

print(f"Yüklenen sinyal sayısı: {X.shape[0]}")
print(f"Sinyal şekli (örnek): {X[0].shape}")

# Etiketleri sinyallerle eşleştir (sinyal olmayanlar çıkarıldı)
valid_indices = [i for i, s in enumerate(signals) if s is not None]
y = y[valid_indices]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model tanımı (örnek 1D CNN)
model = Sequential([
    Conv1D(32, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Modeli eğit
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    callbacks=[early_stop]
)

# Test doğruluğu
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Doğruluğu: {accuracy:.4f}')
