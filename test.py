import tensorflow as tf
print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ {len(gpus)} GPU bulundu:")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i+1}: {gpu.name}")
else:
    print("❌ GPU bulunamadı. TensorFlow şu anda sadece CPU kullanıyor.")
