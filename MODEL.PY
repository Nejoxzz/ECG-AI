import torch
import numpy as np
from torch_ecg.models import ECG_CRNN  # Örnek model

# Önceden eğitilmiş bir modeli yükle
model = ECG_CRNN.from_pretrained("ptbxl_cnn")  # torch_ecg içinden indirilebilir

model.eval()

# Örnek sinyal: shape (batch_size, channels, length)
# Mesela tek bir kayıt: 12 kanal, uzunluk 5000
ecg = torch.rand(1, 12, 5000)

with torch.no_grad():
    output = model(ecg)  # Çıktı: sınıflar için olasılıklar
print(output.shape)

