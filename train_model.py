import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

print("TensorFlow version:", tf.__version__)

print("Memuat dataset landmarks yang telah dinormalisasi...")
train_df = pd.read_csv('train_landmarks.csv')
val_df = pd.read_csv('val_landmarks.csv')

X_train = train_df.drop('label', axis=1).values
y_train_labels = train_df['label'].values

X_val = val_df.drop('label', axis=1).values
y_val_labels = val_df['label'].values

encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train_labels)
y_val_encoded = encoder.transform(y_val_labels)

y_train = to_categorical(y_train_encoded)
y_val = to_categorical(y_val_encoded)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

# ==========================================
# DATA AUGMENTATION UNTUK FOTO/LANDMARK
# (Rotasi acak & noise agar M vs N dan B vs P/D lebih akurat)
# ==========================================
print("Menerapkan Data Augmentation (Rotasi & Noise)...")
def augment_landmarks(X, y, num_copies=2):
    X_aug, y_aug = [], []
    for i in range(len(X)):
        X_aug.append(X[i])
        y_aug.append(y[i])
        
        for _ in range(num_copies):
            pts = X[i].reshape(21, 3)
            # Rotasi acak antara -15 sampai +15 derajat
            angle = np.radians(np.random.uniform(-15, 15))
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a,  cos_a, 0],
                [0,          0, 1]
            ])
            pts_rot = np.dot(pts, rot_matrix)
            # Noise halus (1.5%)
            noise = np.random.normal(0, 0.015, pts_rot.shape)
            pts_aug = pts_rot + noise
            
            X_aug.append(pts_aug.reshape(63, 1))
            y_aug.append(y[i])
            
    return np.array(X_aug), np.array(y_aug)

X_train_aug, y_train_aug = augment_landmarks(X_train, y_train, num_copies=2)
print(f"Bentuk data X_train setelah Augmentasi: {X_train_aug.shape}")
print(f"Bentuk data Y_train setelah Augmentasi: {y_train_aug.shape}")

model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(26, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Memulai training model (50 epoch)...")
history = model.fit(
    X_train_aug, y_train_aug,
    epochs=50,
    batch_size=64,
    validation_data=(X_val, y_val),
    verbose=1
)

model.save('cnn_bisindo.h5')
np.save('classes_bisindo.npy', encoder.classes_)
print("Training selesai! Model 'cnn_bisindo.h5' dan label 'classes_bisindo.npy' berhasil diperbarui.")
