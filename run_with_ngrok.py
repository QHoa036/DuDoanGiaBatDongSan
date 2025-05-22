import os
import subprocess
import time
import sys
import signal
from pyngrok import ngrok
from dotenv import load_dotenv

# Hàm để lấy token an toàn
def get_secure_token():
    # Thử tìm token từ biến môi trường
    token = os.environ.get('NGROK_TOKEN')
    if token:
        return token

    # Thử tìm trong file .env.local hoặc .env
    load_dotenv('.env.local')
    load_dotenv('.env.example')
    token = os.environ.get('NGROK_TOKEN')
    if token:
        return token

    # Nếu không tìm thấy, yêu cầu nhập từ người dùng
    print("[WARNING] Không tìm thấy ngrok token trong biến môi trường hoặc file .env")
    print("[INFO] Bạn có thể tạo token tại https://dashboard.ngrok.com/auth")
    token = input("Nhập ngrok token của bạn: ")

    # Lưu token vào .env.local để sử dụng sau này
    if token:
        with open('.env.local', 'a+') as f:
            # Kiểm tra xem file đã có token chưa
            f.seek(0)
            content = f.read()
            if 'NGROK_TOKEN' not in content:
                f.write(f"\nNGROK_TOKEN={token}")
        print("[INFO] Đã lưu token vào .env.local")

    return token

# Cài đặt xử lý tín hiệu Ctrl+C
def signal_handler(sig, frame):
    print("\n[STOP] Đang dừng ứng dụng...")
    try:
        ngrok.kill()
        if 'streamlit_process' in globals() and streamlit_process.poll() is None:
            streamlit_process.terminate()
    except Exception as e:
        print(f"[ERROR] {e}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Lấy token an toàn
ngrok_token = get_secure_token()
if not ngrok_token:
    print("[ERROR] Không thể tiếp tục mà không có ngrok token")
    sys.exit(1)

# Thiết lập ngrok
try:
    ngrok.set_auth_token(ngrok_token)
except Exception as e:
    print(f"[ERROR] Token không hợp lệ: {e}")
    sys.exit(1)

# Khởi chạy Streamlit trong tiến trình con với file từ thư mục Demo
# Sử dụng đường dẫn đầy đủ tới streamlit thay vì chỉ 'streamlit'
streamlit_process = subprocess.Popen(["./venv/Scripts/streamlit", "run", "Demo/vn_real_estate_app.py"])

# Tạo tunnel HTTP đến cổng Streamlit
http_tunnel = ngrok.connect(addr="8501", proto="http", bind_tls=True)
print("\n" + "="*60)
print(f"[URL] URL NGROK PUBLIC: {http_tunnel.public_url}")
print("[SHARE] Chia sẻ URL này để cho phép người khác truy cập ứng dụng của bạn")
print("="*60 + "\n")

try:
    # Giữ script chạy
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # Dọn dẹp khi người dùng nhấn Ctrl+C - đã được xử lý trong signal_handler
    pass
