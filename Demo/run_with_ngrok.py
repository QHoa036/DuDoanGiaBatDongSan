#!/usr/bin/env python3
import subprocess
import time
import os
from pyngrok import ngrok
from dotenv import load_dotenv

# Tải biến môi trường từ các file .env và .env.local
load_dotenv()
load_dotenv('.env.local', override=True)  # .env.local ghi đè lên .env

# Đọc NGROK_TOKEN từ biến môi trường
ngrok_token = os.getenv("NGROK_TOKEN")
if not ngrok_token:
    print("❌ Không tìm thấy NGROK_TOKEN trong biến môi trường hoặc file .env.local")
    print("Vui lòng chạy script run_demo.sh để thiết lập token")
    exit(1)

# Thiết lập ngrok
ngrok.set_auth_token(ngrok_token)

# Xác định đường dẫn Streamlit dựa trên hệ điều hành
def get_streamlit_path():
    if os.name == "nt":  # Windows
        return "../venv/Scripts/streamlit"
    else:  # macOS, Linux
        return "../venv/bin/streamlit"

# Khởi chạy Streamlit trong tiến trình con
streamlit_path = get_streamlit_path()
streamlit_process = subprocess.Popen([streamlit_path, "run", "vn_real_estate_app.py"])

try:
    # Tạo tunnel HTTP đến cổng Streamlit
    http_tunnel = ngrok.connect(addr="8501", proto="http", bind_tls=True)
    print("\n" + "="*60)
    print(f"🌐 URL NGROK PUBLIC: {http_tunnel.public_url}")
    print("🔗 Chia sẻ URL này để cho phép người khác truy cập ứng dụng của bạn")
    print("="*60 + "\n")

    # Giữ script chạy
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # Dọn dẹp khi người dùng nhấn Ctrl+C
    print("\n🛑 Đang dừng ứng dụng...")
    ngrok.kill()
    streamlit_process.terminate()
except Exception as e:
    print(f"❌ Lỗi khi kết nối ngrok: {e}")
    streamlit_process.terminate()
    exit(1)
