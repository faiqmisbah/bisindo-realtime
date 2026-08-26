FROM python:3.10-slim

WORKDIR /app

# Install dependensi C++ Linux untuk OpenCV & MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install pustaka Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh file aplikasi
COPY . .

# Expose port standar Hugging Face (7860)
EXPOSE 7860

# Jalankan Streamlit App
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=7860", "--server.address=0.0.0.0"]
