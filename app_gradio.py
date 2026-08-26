import gradio as gr
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

# 1. Memuat Model AI & MediaPipe Detector
model = load_model('cnn_bisindo.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classes = np.load('classes_bisindo.npy', allow_pickle=True)

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

def normalize_landmarks(hand_landmarks):
    base_x = hand_landmarks[0].x
    base_y = hand_landmarks[0].y
    base_z = hand_landmarks[0].z
    
    temp_landmarks = []
    for lm in hand_landmarks:
        temp_landmarks.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
        
    max_value = max(max(abs(val) for val in temp_landmarks), 1e-6)
    return [val / max_value for val in temp_landmarks]

state = {"current_word": "", "last_pred": None, "stable_frames": 0}

def predict_bisindo(frame, current_word):
    if frame is None:
        return None, state["current_word"]
        
    img = cv2.flip(frame, 1)
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
            if pred_label == state["last_pred"]:
                state["stable_frames"] += 1
            else:
                state["last_pred"] = pred_label
                state["stable_frames"] = 0
                
            if state["stable_frames"] == 12:
                state["current_word"] += pred_label
                state["stable_frames"] = 0
                
            cv2.putText(img, f"Mengeja: {pred_label} ({int(confidence*100)}%)", 
                        (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                        
    # Baris Bawah (Hasil Terjemahan Kata Real-Time)
    cv2.rectangle(img, (0, img.shape[0]-65), (img.shape[1], img.shape[0]), (6, 29, 25), -1)
    cv2.putText(img, f"Hasil Kata: {state['current_word']}", (15, img.shape[0]-22), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (132, 168, 0), 2)
                
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), state["current_word"]

def clear_word():
    state["current_word"] = ""
    state["last_pred"] = None
    state["stable_frames"] = 0
    return ""

def delete_char():
    state["current_word"] = state["current_word"][:-1]
    return state["current_word"]

# 2. Desain Antarmuka Gradio UI
with gr.Blocks(title="Penerjemah BISINDO AI") as demo:
    gr.HTML("""
    <div style="background:#ffffff; border:2px solid #cbd5e1; border-radius:20px; padding:1.8rem 2.4rem; margin-bottom:1.5rem; box-shadow:0 10px 30px rgba(0,0,0,0.04);">
        <div style="background:#e6f7f3; color:#008767; font-size:0.8rem; font-weight:800; padding:0.4rem 1rem; border-radius:50px; display:inline-block; margin-bottom:0.5rem;">REAL-TIME AI SYSTEM</div>
        <div style="font-size:2.2rem; font-weight:800; color:#061d19; margin-bottom:0.3rem;">Penerjemah BISINDO <span style="color:#00a884;">AI</span></div>
        <div style="font-size:1rem; color:#52605d;">Sistem Penerjemah Bahasa Isyarat Indonesia (A-Z) berbasis Convolutional Neural Network</div>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=65):
            gr.Markdown("### Stream Kamera Live")
            webcam_input = gr.Image(sources=["webcam"], streaming=True, label="Webcam Live Stream")
            
        with gr.Column(scale=35):
            gr.Markdown("### Informasi Model AI")
            gr.Markdown("""
            - **Arsitektur Model:** 1D Convolutional Neural Network (CNN)
            - **Jumlah Kelas:** 26 Label Abjad (A - Z)
            - **Preprocessing:** Relative & Scale Landmark Normalization
            """)
            
            gr.Markdown("### Panduan Penggunaan")
            gr.Markdown("""
            1. Aktifkan webcam di sebelah kiri.
            2. Peragakan gestur abjad BISINDO (A-Z) di depan kamera.
            3. Tahan gestur ~0.5 detik hingga huruf tercatat.
            4. Teks hasil terjemahan langsung muncul di bagian bawah layar video.
            """)
            
            gr.Markdown("### Kontrol Kata")
            word_output = gr.Textbox(label="Hasil Terjemahan Kata", value="", interactive=False)
            
            with gr.Row():
                btn_clear = gr.Button("Hapus Kata", variant="primary")
                btn_del = gr.Button("Hapus 1 Huruf", variant="secondary")
                
    webcam_input.stream(fn=predict_bisindo, inputs=[webcam_input, word_output], outputs=[webcam_input, word_output], stream_every=0.1)
    btn_clear.click(fn=clear_word, outputs=word_output)
    btn_del.click(fn=delete_char, outputs=word_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
