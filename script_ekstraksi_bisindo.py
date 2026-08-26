import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import urllib.request

# ==========================================
# KONFIGURASI PATH
# ==========================================
DATASET_DIR = "bisindo/images"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

# ==========================================
# UNDUH MODEL MEDIAPIPE (Hanya 1x jalan)
# ==========================================
MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print("Mengunduh model Hand Landmarker dari Google...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Unduhan model selesai!")

# 1. Inisialisasi MediaPipe Tasks API
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
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

def process_directory(directory_path, output_csv):
    print(f"Mulai memproses folder: {directory_path}")
    
    # Menyiapkan header tabel CSV
    headers = ['label']
    for i in range(21):
        headers.extend([f'x{i}', f'y{i}', f'z{i}'])
        
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        if not os.path.exists(directory_path):
            print(f"Folder {directory_path} tidak ditemukan!")
            return
            
        for label in sorted(os.listdir(directory_path)):
            label_path = os.path.join(directory_path, label)
            if not os.path.isdir(label_path):
                continue
                
            print(f"Ekstraksi huruf {label}...")
            
            for image_name in os.listdir(label_path):
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                image_path = os.path.join(label_path, image_name)
                
                # Baca gambar
                cv2_image = cv2.imread(image_path)
                if cv2_image is None:
                    continue
                
                # Ubah format gambar ke bentuk yang diminta MediaPipe Tasks
                rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                
                # Lakukan Deteksi Landmark
                detection_result = detector.detect(mp_image)
                
                if detection_result.hand_landmarks:
                    for hand_landmarks in detection_result.hand_landmarks:
                        normalized_coords = normalize_landmarks(hand_landmarks)
                        row = [label] + normalized_coords
                        writer.writerow(row)
                        
    print(f"Selesai! Data disimpan ke: {output_csv}\n")

if __name__ == "__main__":
    process_directory(TRAIN_DIR, "train_landmarks.csv")
    process_directory(VAL_DIR, "val_landmarks.csv")
    print("Proses ekstraksi selesai dengan MediaPipe Tasks API!")