import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Cấu hình (chỉnh TARGET_SIZE nếu model của bạn khác)
TARGET_SIZE = (128, 128)
MODEL_PATH = "skin_cancer_cnn_model.h5"

CLASS_NAMES = [
  "akiec","bcc","bkl","df","mel","nv","vasc"
]
CLASS_NAMES_VN = {
  "akiec":"Ung thư biểu mô tế bào sừng/AK",
  "bcc":"Ung thư tế bào đáy (BCC)",
  "bkl":"Tổn thương keratosis lành tính",
  "df":"Dermatofibroma",
  "mel":"Melanoma",
  "nv":"Nốt ruồi (nevus)",
  "vasc":"Tổn thương mạch máu"
}

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def preprocess_image(img: Image.Image):
    img = img.convert('RGB')
    img = img.resize(TARGET_SIZE)
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape (1, h, w, 3)
    return arr

st.set_page_config(page_title="Dự đoán ung thư da (7 lớp)", layout="centered")

st.title("🔬 Dự đoán 7 loại bệnh về da")
st.write("Upload ảnh da (close-up)")

model = None
try:
    model = load_model()
except Exception as e:
    st.error(f"Không thể load model: {e}")

uploaded = st.file_uploader("Chọn ảnh", type=["jpg","jpeg","png"])
if uploaded is not None and model is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Ảnh đầu vào", use_column_width=True)
    X = preprocess_image(img)
    with st.spinner("Đang dự đoán..."):
        preds = model.predict(X)[0]
    top_idx = int(np.argmax(preds))
    top_name = CLASS_NAMES[top_idx]
    top_prob = float(preds[top_idx])
    st.subheader("Kết quả dự đoán")
    st.write(f"**Nhãn (EN):** {top_name}")
    st.write(f"**Nhãn (VN):** {CLASS_NAMES_VN.get(top_name, top_name)}")
    st.write(f"**Xác suất:** {top_prob*100:.2f}%")

    import pandas as pd
    df = pd.DataFrame({
      "class_en": CLASS_NAMES,
      "class_vn": [CLASS_NAMES_VN.get(c,c) for c in CLASS_NAMES],
      "probability": [float(p) for p in preds]
    })
    df = df.sort_values("probability", ascending=False)
    st.table(df.style.format({"probability":"{:.4f}"}))
