import tensorflow as tf
from tensorflow.keras import layers, models

def create_sample_ecg_model(input_shape=(1000, 1), num_classes=71):
    model = models.Sequential([
        layers.Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=5, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_sample_ecg_model()
model.save("sample_ecg_model.h5")
print("Model dosyası 'sample_ecg_model.h5' olarak oluşturuldu.")
