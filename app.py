import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

# ==========================================
# 1. PERSIAPAN MODEL DAN MEDIAPIPE
# ==========================================
print("Memuat model CNN dan label...")
try:
    model = load_model('cnn_bisindo.h5')
    # TAMBAHKAN BARIS INI UNTUK MENGATASI WARNING COMPILE METRICS:
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    classes = np.load('classes_bisindo.npy', allow_pickle=True)
except Exception as e:
    print(f"Error detail: {e}")
    print("Error: Gagal memuat cnn_bisindo.h5 atau classes_bisindo.npy!")
    print("Pastikan kamu sudah melatih modelnya di Jupyter Notebook.")
    exit()

print("Memuat MediaPipe Hand Landmarker...")
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

# ==========================================
# 2. VARIABEL LOGIKA PENAHAN (THRESHOLDING)
# ==========================================
current_word = ""       # Untuk menyimpan kata yang sedang dirangkai
last_prediction = None  # Menyimpan tebakan huruf terakhir
stable_frames = 0       # Menghitung berapa lama tangan stabil membentuk huruf
REQUIRED_FRAMES = 15    # Butuh ~0.5 detik stabil agar huruf dicatat ke layar

# ==========================================
# 3. MENGHIDUPKAN KAMERA
# ==========================================
cap = cv2.VideoCapture(0) # 0 adalah ID default webcam
print("Kamera menyala! Tekan 'Q' untuk keluar, atau 'C' untuk menghapus kata.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Membalik gambar agar seperti cermin (mirror)
    frame = cv2.flip(frame, 1)
    
    # Pemrosesan gambar untuk MediaPipe
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    # Deteksi tangan
    detection_result = detector.detect(mp_image)
    
    if detection_result.hand_landmarks:
        # Ambil koordinat tangan pertama yang terdeteksi
        hand_landmarks = detection_result.hand_landmarks[0]
        
        # Gambar titik hijau di ujung jari telunjuk (landmark 8)
        h, w, c = frame.shape
        cx, cy = int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h)
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        
        # Ekstraksi dan normalisasi 63 titik koordinat (X, Y, Z)
        row = normalize_landmarks(hand_landmarks)
        
        # Prediksi dengan model CNN (Reshape agar bentuknya 1 baris, 63 fitur, 1 channel)
        X_input = np.array(row).reshape(1, 63, 1)
        predictions = model.predict(X_input, verbose=0) # verbose=0 agar terminal tidak penuh
        
        # Ambil hasil probabilitas tertinggi
        pred_index = np.argmax(predictions)
        confidence = predictions[0][pred_index]
        pred_label = classes[pred_index]
        
        # ==========================================
        # 4. LOGIKA PERANGKAIAN KATA (THRESHOLDING)
        # ==========================================
        # Jika AI yakin di atas 80%
        if confidence > 0.8:
            # Jika gerakannya masih sama dengan frame sebelumnya
            if pred_label == last_prediction:
                stable_frames += 1
            else:
                last_prediction = pred_label
                stable_frames = 0
                
            # Jika tangan sudah stabil selama REQUIRED_FRAMES (misal 15 frame berturut-turut)
            if stable_frames == REQUIRED_FRAMES:
                current_word += pred_label
                stable_frames = 0 # Reset hitungan agar huruf tidak ganda (A -> AA)
                
            # Tampilkan informasi target bidikan saat ini
            cv2.putText(frame, f"Mengeja: {last_prediction} ({stable_frames}/{REQUIRED_FRAMES}) - {confidence*100:.1f}%", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # ==========================================
    # 5. TAMPILAN ANTARMUKA PENGGUNA (UI)
    # ==========================================
    # Kotak latar belakang untuk teks kata
    cv2.rectangle(frame, (0, 400), (640, 480), (0, 0, 0), -1)
    
    # Menampilkan kata yang sudah dirangkai
    cv2.putText(frame, f"Hasil Kata: {current_word}", (10, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
    # Menampilkan panduan tombol
    cv2.putText(frame, "Tekan 'C' hapus kata | 'Q' keluar", (10, 475), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Tampilkan jendela
    cv2.imshow('Penerjemah BISINDO Real-Time', frame)

    # Membaca tombol keyboard
    key = cv2.waitKey(1)
    if key == ord('q'):    # Tekan q untuk keluar
        break
    elif key == ord('c'):  # Tekan c untuk mengosongkan kata
        current_word = ""

cap.release()
cv2.destroyAllWindows()