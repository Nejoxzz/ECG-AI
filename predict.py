import numpy as np
import tensorflow as tf

# Örnek olarak yüklü bir model (buraya kendi .h5 model yolunu koymalısın)
model = tf.keras.models.load_model("sample_ecg_model.h5")

# Etiket isimleri (PTB-XL datasetinden 71 etiket)
labels = ['1AVB', '2AVB', '3AVB', 'ABQRS', 'AFIB', 'AFLT', 'ALMI', 'AMI', 'ANEUR', 'ASMI',
          'BIGU', 'CLBBB', 'CRBBB', 'DIG', 'EL', 'HVOLT', 'ILBBB', 'ILMI', 'IMI', 'INJAL',
          'INJAS', 'INJIL', 'INJIN', 'INJLA', 'INVT', 'IPLMI', 'IPMI', 'IRBBB', 'ISCAL',
          'ISCAN', 'ISCAS', 'ISCIL', 'ISCIN', 'ISCLA', 'ISC_', 'IVCD', 'LAFB', 'LAO/LAE',
          'LMI', 'LNGQT', 'LOWT', 'LPFB', 'LPR', 'LVH', 'LVOLT', 'NDT', 'NORM', 'NST_', 'NT_',
          'PAC', 'PACE', 'PMI', 'PRC(S)', 'PSVT', 'PVC', 'QWAVE', 'RAO/RAE', 'RVH', 'SARRH',
          'SBRAD', 'SEHYP', 'SR', 'STACH', 'STD_', 'STE_', 'SVARR', 'SVTAC', 'TAB_', 'TRIGU',
          'VCLVH', 'WPW']

# Burada kendi ECG sinyalini numpy dizisi olarak hazırlamalısın
# Örnek dummy sinyal (uzunluğunu model girişine göre ayarla)
ecg_signal = np.random.rand(1000)  # Örnek sadece

# Modelin beklediği şekilde input shape yap (örneğin: (1, 1000, 1))
input_data = ecg_signal.reshape(1, -1, 1)

# Tahmin yap
pred = model.predict(input_data)

print("Tahmin (olasılık):", pred[0])

# 0.5 üstü olanları pozitif kabul et
predicted_labels = (pred[0] >= 0.5).astype(int)

# Pozitif tahmin edilen etiketleri al
active_labels = [labels[i] for i, val in enumerate(predicted_labels) if val == 1]

print("Tahmin edilen pozitif etiketler:", active_labels)
