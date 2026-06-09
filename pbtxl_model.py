import os
import wfdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# Veri klasörü yolu (ptbxl_database.csv ve record dosyalarının olduğu yer)
data_dir = "C:/Users/NejoxZz/Desktop/ECG_PROJECT/ptbxl"
# Tüm dataset CSV dosyasını yükle
df = pd.read_csv(os.path.join(data_dir, 'ptbxl_database.csv'))

# Sinyalleri ve etiketleri hazırla
def load_ecg_signal(record_name):
    record_path = os.path.join(data_dir, record_name)
    record = wfdb.rdrecord(record_path)
    return record.p_signal

# Multi-label etiketleri hazırla
mlb = MultiLabelBinarizer()

# Örnek olarak 'diagnostic' kolonundaki etiketleri ';' ile ayırıp çoklu sınıf haline getiriyoruz
labels = df['diagnostic'].str.split(';')
y = mlb.fit_transform(labels)

# Sinyalleri numpy arrayine alıyoruz (uzun sürebilir!)
X = np.array([load_ecg_signal(rec) for rec in df['record_name']])

# Veriyi eğitim/test olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluştur
model = Sequential([
    Conv1D(32, kernel_size=5, activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_train.shape[1], activation='sigmoid')  # Çoklu etiket için sigmoid
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğit
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Modeli değerlendir
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test doğruluğu: {accuracy:.4f}')

model.save("C:/Users/NejoxZz/Desktop/ECG_PROJECT/ecg_model.h5")
