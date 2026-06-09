import serial
import time
import numpy as np
import streamlit as st
import tensorflow as tf
from scipy.signal import butter, filtfilt

# -------------------------
# MODEL
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# -------------------------
# SETTINGS
# -------------------------
COM_PORT = "COM11"
BAUD_RATE = 115200
TARGET_LENGTH = 1000

CLASS_NAMES = [
    "Normal Sinus Rhythm",
    "PVC",
    "PAC",
    "Atrial Fibrillation",
    "Atrial Flutter",
    "AV Block",
    "Bundle Branch Block",
    "Myocardial Infarction",
    "ST Elevation",
    "ST Depression",
    "T Wave Abnormality",
    "Bradycardia",
    "Tachycardia",
    "Other Arrhythmia"
]

# -------------------------
# FILTER (PTB-XL STYLE)
# -------------------------
def bandpass(signal, low=0.5, high=40, fs=100, order=2):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype="band")
    return filtfilt(b, a, signal)

# -------------------------
# PREPROCESS (CRITICAL FIX)
# -------------------------
def preprocess(signal):

    signal = np.array(signal, dtype=np.float32)

    # 1. baseline normalize
    signal = signal - np.mean(signal)

    # 2. bandpass filter
    signal = bandpass(signal)

    # 3. normalize (PTB-XL style)
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    # 4. length fix
    if len(signal) > TARGET_LENGTH:
        signal = signal[-TARGET_LENGTH:]
    else:
        signal = np.pad(signal, (0, TARGET_LENGTH - len(signal)))

    return signal.reshape(1, TARGET_LENGTH, 1)

# -------------------------
# PREDICT
# -------------------------
def predict(signal):

    x = preprocess(signal)
    preds = model.predict(x, verbose=0)[0]

    idx = np.argmax(preds)
    prob = float(preds[idx])

    # 🔥 UNKNOWN REDUCTION LOGIC
    if prob < 0.70:
        return [("Uncertain ECG (Low Quality Signal)", prob)]

    if idx >= len(CLASS_NAMES):
        return [("Unknown Pattern", prob)]

    return [(CLASS_NAMES[idx], prob)]

# -------------------------
# STREAMLIT UI
# -------------------------
def main():

    st.set_page_config(layout="wide")
    st.title("🫀 Clean PTB-XL ECG AI Monitor")

    if "running" not in st.session_state:
        st.session_state.running = False

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶ Start"):
            st.session_state.running = True

    with col2:
        if st.button("⏹ Stop"):
            st.session_state.running = False

    if not st.session_state.running:
        st.info("Stopped")
        return

    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
    except Exception as e:
        st.error(f"Serial Error: {e}")
        return

    chart = st.empty()
    result_box = st.empty()

    buffer = []
    csv = open("ecg.csv", "w")
    csv.write("value\n")

    counter = 0

    while st.session_state.running:

        line = ser.readline().decode(errors="ignore").strip()

        if not line:
            continue

        if line.lstrip("-").isdigit():

            value = int(line)

            buffer.append(value)
            csv.write(f"{value}\n")

            if len(buffer) > TARGET_LENGTH:
                buffer.pop(0)

            chart.line_chart(buffer)

            counter += 1

            if len(buffer) == TARGET_LENGTH and counter % 200 == 0:

                results = predict(buffer)
                name, prob = results[0]

                if "Uncertain" in name:
                    result_box.warning(f"⚠ {name} — %{prob*100:.1f}")
                else:
                    result_box.success(f"🫀 {name} — %{prob*100:.1f}")

    csv.close()
    ser.close()

# -------------------------
if __name__ == "__main__":
    main()