#!/usr/bin/env python3
import subprocess
import time
import os
from pyngrok import ngrok

# Đọc NGROK_TOKEN từ biến môi trường hoặc từ file .env
try:
    from dotenv import load_dotenv
    load_dotenv()
    ngrok_token = os.getenv("NGROK_TOKEN")
    if not ngrok_token:
        raise ValueError("NGROK_TOKEN không được tìm thấy trong biến môi trường hoặc file .env")
    ngrok.set_auth_token(ngrok_token)
except Exception as e:
    print(f"Lỗi khi thiết lập ngrok: {e}")
    print("Vui lòng tạo file .env với nội dung NGROK_TOKEN=your_token hoặc thiết lập biến môi trường NGROK_TOKEN")
    exit(1)

# Khởi chạy Streamlit trong tiến trình con với file từ thư mục Demo
streamlit_path = "./venv/bin/streamlit"
if os.name == "nt":  # Windows
    streamlit_path = "./venv/Scripts/streamlit"

streamlit_process = subprocess.Popen([streamlit_path, "run", "Demo/vn_real_estate_app.py"])

# Tạo tunnel HTTP đến cổng Streamlit
try:
    http_tunnel = ngrok.connect(addr="8501", proto="http", bind_tls=True)
    print("\n" + "="*60)
    print(f"🌐 URL NGROK PUBLIC: {http_tunnel.public_url}")
    print("🔗 Chia sẻ URL này để cho phép người khác truy cập ứng dụng của bạn")
    print("="*60 + "\n")

    try:
        # Giữ script chạy
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Dọn dẹp khi người dùng nhấn Ctrl+C
        print("\n🛑 Đang dừng ứng dụng...")
        ngrok.kill()
        streamlit_process.terminate()
except Exception as e:
    print(f"Lỗi khi kết nối ngrok: {e}")
    streamlit_process.terminate()
    exit(1)
