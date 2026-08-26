import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import av

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Penerjemah BISINDO AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling (FaiqDev Theme: Direct Button Type Targeting, High Visibility)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], .main {
        font-family: 'Inter', -apple-system, sans-serif !important;
        background-color: #f4f7f5 !important;
        color: #061d19 !important;
    }
    
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 100% !important;
    }
    
    /* Floating Header Navbar */
    .brand-header {
        background-color: #ffffff;
        border: 2px solid #cbd5e1;
        border-radius: 20px;
        padding: 1.8rem 2.6rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .brand-left {
        flex: 1;
    }
    .badge-status {
        display: inline-block;
        background-color: #e6f7f3;
        color: #008767;
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        padding: 0.4rem 1rem;
        border-radius: 50px;
        margin-bottom: 0.6rem;
        text-transform: uppercase;
        border: 1px solid #bbf7d0;
    }
    .brand-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #061d19;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .brand-title span {
        color: #00a884;
    }
    .brand-sub {
        font-size: 1.08rem;
        color: #475569;
        margin-top: 0.4rem;
        font-weight: 400;
    }
    
    /* Section Headers */
    .section-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #061d19;
        margin-top: 0.2rem;
        margin-bottom: 0.8rem;
        letter-spacing: -0.01em;
    }
    
    .subtitle-desc {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 1.2rem;
    }
    
    /* Custom Cards dengan Garis Tepi (Border) 2px Jelas */
    .custom-card {
        background-color: #ffffff;
        border: 2px solid #cbd5e1;
        border-left: 6px solid #00a884;
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        font-size: 1.05rem;
        line-height: 1.8;
        color: #334155;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.8rem;
    }
    .custom-card ol {
        margin: 0;
        padding-left: 1.3rem;
    }
    .custom-card li {
        margin-bottom: 0.55rem;
    }
    
    .info-box {
        background-color: #ffffff;
        border: 2px solid #cbd5e1;
        border-left: 6px solid #061d19;
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        font-size: 1.05rem;
        color: #334155;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.8rem;
    }
    .info-line {
        margin-bottom: 0.6rem;
    }
    .info-line:last-child {
        margin-bottom: 0;
    }
    .info-tag {
        font-weight: 800;
        color: #00a884;
    }
    
    /* STREAMLIT BUTTON DIRECT TARGETING (STRICT OVERRIDE) */
    .stButton > button[data-testid="stBaseButton-primary"],
    button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background-color: #00a884 !important;
        color: #ffffff !important;
        border: 2px solid #00a884 !important;
        border-radius: 50px !important;
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        padding: 0.85rem 1.4rem !important;
        cursor: pointer !important;
        box-shadow: 0 4px 14px rgba(0, 168, 132, 0.35) !important;
        width: 100% !important;
    }
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background-color: #008767 !important;
        border-color: #008767 !important;
        color: #ffffff !important;
        transform: translateY(-2px) !important;
    }
    
    .stButton > button[data-testid="stBaseButton-secondary"],
    button[kind="secondary"],
    button[data-testid="stBaseButton-secondary"] {
        background-color: #061d19 !important;
        color: #ffffff !important;
        border: 2px solid #061d19 !important;
        border-radius: 50px !important;
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        padding: 0.85rem 1.4rem !important;
        cursor: pointer !important;
        box-shadow: 0 4px 14px rgba(6, 29, 25, 0.25) !important;
        width: 100% !important;
    }
    .stButton > button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #0f3832 !important;
        border-color: #0f3832 !important;
        color: #ffffff !important;
        transform: translateY(-2px) !important;
    }
    
    .stButton > button *,
    button[data-testid="stBaseButton-primary"] *,
    button[data-testid="stBaseButton-secondary"] * {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
    }
    
    /* Hide Streamlit Chrome */
    #MainMenu, footer, header[data-testid="stHeader"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Header Utama Floating ala FaiqDev
st.markdown("""
<div class="brand-header">
    <div class="brand-left">
        <div class="badge-status">Real-Time AI System</div>
        <div class="brand-title">Penerjemah BISINDO <span>AI</span></div>
        <div class="brand-sub">Sistem Penerjemah Bahasa Isyarat Indonesia (A-Z) berbasis Convolutional Neural Network</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 2. Cache Pemuatan Model AI & MediaPipe
@st.cache_resource
def load_resources():
    try:
        import tf_keras as legacy_keras
        model = legacy_keras.models.load_model('cnn_bisindo.h5', compile=False)
    except Exception:
        import tensorflow as tf
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(63, 1)),
            tf.keras.layers.Conv1D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(128, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(26, activation='softmax')
        ])
        model.load_weights('cnn_bisindo.h5')
        
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    classes = np.load('classes_bisindo.npy', allow_pickle=True)
    
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
    
    return model, classes, detector

try:
    model, classes, detector = load_resources()
except Exception as e:
    st.error(f"Gagal memuat model atau detector: {e}")
    st.stop()

def normalize_landmarks(hand_landmarks):
    base_x = hand_landmarks[0].x
    base_y = hand_landmarks[0].y
    base_z = hand_landmarks[0].z
    
    temp_landmarks = []
    for lm in hand_landmarks:
        temp_landmarks.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
        
    max_value = max(max(abs(val) for val in temp_landmarks), 1e-6)
    return [val / max_value for val in temp_landmarks]

# Class Processor untuk Streamlit-WebRTC
class BISINDOProcessor(VideoProcessorBase):
    def __init__(self):
        self.current_word = ""
        self.last_prediction = None
        self.stable_frames = 0
        self.REQUIRED_FRAMES = 12

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        detection_result = detector.detect(mp_image)
        
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            h, w, _ = img.shape
            cx, cy = int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h)
            cv2.circle(img, (cx, cy), 10, (132, 168, 0), cv2.FILLED)
            
            row = normalize_landmarks(hand_landmarks)
            X_input = np.array(row, dtype=np.float32).reshape(1, 63, 1)
            
            predictions = model(X_input, training=False).numpy()
            pred_index = np.argmax(predictions)
            confidence = float(predictions[0][pred_index])
            pred_label = str(classes[pred_index])
            
            if confidence > 0.8:
                if pred_label == self.last_prediction:
                    self.stable_frames += 1
                else:
                    self.last_prediction = pred_label
                    self.stable_frames = 0
                    
                if self.stable_frames == self.REQUIRED_FRAMES:
                    self.current_word += pred_label
                    self.stable_frames = 0
                    
                cv2.putText(img, f"Mengeja: {self.last_prediction} ({int(confidence*100)}%)", 
                            (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                            
        cv2.rectangle(img, (0, img.shape[0]-65), (img.shape[1], img.shape[0]), (6, 29, 25), -1)
        cv2.putText(img, f"Hasil Kata: {self.current_word}", (15, img.shape[0]-22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (132, 168, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Layout Utama Full Width
col_cam, col_info = st.columns([65, 35], gap="large")

with col_cam:
    st.markdown('<div class="section-title">Stream Kamera Live</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-desc">Pilih Mode Kamera yang sesuai dengan jaringan internet Anda:</div>', unsafe_allow_html=True)
    
    # Mode Pilihan Kamera (WebRTC vs Kamera Native Direct HTTPS)
    cam_mode = st.radio(
        "Pilih Mode Kamera:",
        ["Kamera Native Streamlit (100% Bebas Error/Koneksi Cloud)", "Kamera WebRTC Live (Real-Time Stream)"],
        index=0,
        horizontal=True
    )
    
    if "Kamera Native" in cam_mode:
        img_buffer = st.camera_input("Ambil Foto Gestur Tangani Di Sini")
        if img_buffer is not None:
            bytes_data = img_buffer.getvalue()
            img_np = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            img_np = cv2.flip(img_np, 1)
            
            rgb_image = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            detection_result = detector.detect(mp_image)
            
            if detection_result.hand_landmarks:
                hand_landmarks = detection_result.hand_landmarks[0]
                row = normalize_landmarks(hand_landmarks)
                X_input = np.array(row, dtype=np.float32).reshape(1, 63, 1)
                
                predictions = model(X_input, training=False).numpy()
                pred_index = np.argmax(predictions)
                confidence = float(predictions[0][pred_index])
                pred_label = str(classes[pred_index])
                
                st.success(f"Terdeteksi Abjad: **{pred_label}** (Akurasi: {int(confidence*100)}%)")
            else:
                st.warning("Tangan tidak terdeteksi di kamera. Pastikan posisi tangan terlihat jelas!")
    else:
        rtc_config = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:openrelay.metered.ca:80"]},
                {"urls": ["turn:openrelay.metered.ca:80"], "username": "openrelay", "credential": "openrelay"},
                {"urls": ["turn:openrelay.metered.ca:443"], "username": "openrelay", "credential": "openrelay"},
                {"urls": ["turns:openrelay.metered.ca:443?transport=tcp"], "username": "openrelay", "credential": "openrelay"}
            ]
        })
        
        ctx = webrtc_streamer(
            key="bisindo-camera",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_processor_factory=BISINDOProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

with col_info:
    # 1. INFORMASI MODEL AI (PERTAMA)
    st.markdown('<div class="section-title">Informasi Model AI</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
        <div class="info-line"><span class="info-tag">Arsitektur Model:</span> 1D Convolutional Neural Network (CNN)</div>
        <div class="info-line"><span class="info-tag">Jumlah Kelas:</span> {len(classes)} Label Abjad (A - Z)</div>
        <div class="info-line"><span class="info-tag">Preprocessing:</span> Relative & Scale Landmark Normalization</div>
    </div>
    """, unsafe_allow_html=True)

    # 2. PANDUAN PENGGUNAAN (KEDUA)
    st.markdown('<div class="section-title">Panduan Penggunaan</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="custom-card">
        <ol>
            <li>Pilih mode kamera <b>Kamera Native Streamlit</b> untuk koneksi paling stabil di cloud.</li>
            <li>Arahkan 1 tangan membentuk gestur abjad BISINDO (A-Z) di depan kamera.</li>
            <li>Ambil foto atau gunakan stream live.</li>
            <li>Teks hasil terjemahan langsung dianalisis otomatis oleh model CNN.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # 3. KONTROL KATA (KETIGA / PALING BAWAH)
    st.markdown('<div class="section-title">Kontrol Kata</div>', unsafe_allow_html=True)
    btn_c1, btn_c2 = st.columns(2, gap="medium")
    with btn_c1:
        if st.button("Hapus Kata", type="primary", use_container_width=True):
            if 'ctx' in locals() and ctx and ctx.video_processor:
                ctx.video_processor.current_word = ""
                
    with btn_c2:
        if st.button("Hapus 1 Huruf", type="secondary", use_container_width=True):
            if 'ctx' in locals() and ctx and ctx.video_processor:
                ctx.video_processor.current_word = ctx.video_processor.current_word[:-1]
