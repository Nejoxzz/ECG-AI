import os
import wfdb

base_dir_lr = r"C:/Users/NejoxZz/Desktop/ECG_PROJECT/ptbxl/records100"
base_dir_hr = r"C:/Users/NejoxZz/Desktop/ECG_PROJECT/ptbxl/records500"

for i, row in df.iterrows():
    lr_file = os.path.join(base_dir_lr, row['filename_lr'] + '.dat')
    hr_file = os.path.join(base_dir_hr, row['filename_hr'] + '.dat')
    
    try:
        record_lr = wfdb.rdrecord(lr_file[:-4])  # '.dat' uzantısını kaldır
        record_hr = wfdb.rdrecord(hr_file[:-4])
        
        signal_lr = record_lr.p_signal
        signal_hr = record_hr.p_signal
        
        # Burada istediğin işlemi yapabilirsin
        print(f"Okunan kayıt {row['ecg_id']} LR shape: {signal_lr.shape}, HR shape: {signal_hr.shape}")
        
    except Exception as e:
        print(f"{lr_file} veya {hr_file} okunamadı: {e}")
