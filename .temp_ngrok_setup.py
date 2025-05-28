#!/usr/bin/env python3
"""
Tập lệnh tạm thời để thiết lập đường hầm Ngrok
"""
import os
import sys
import time
import signal
import argparse
import subprocess
from pyngrok import ngrok, conf

# Thiết lập ghi nhật ký
import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ngrok-setup")

# Biến toàn cục
streamlit_process = None

def signal_handler(sig, frame):
    """Xử lý tắt an toàn khi Ctrl+C"""
    logger.info("Đang tắt ứng dụng...")
    try:
        # Kết thúc Ngrok
        ngrok.kill()
        # Kết thúc Streamlit nếu đang chạy
        if streamlit_process and streamlit_process.poll() is None:
            streamlit_process.terminate()
            streamlit_process.wait(timeout=5)
    except Exception as e:
        logger.error(f"Lỗi trong quá trình dọn dẹp: {e}")
    sys.exit(0)

def main():
    """Hàm chính để thiết lập Ngrok và chạy Streamlit"""
    # Phân tích đối số dòng lệnh
    parser = argparse.ArgumentParser(description='Chạy Streamlit với đường hầm Ngrok')
    parser.add_argument('--streamlit_path', required=True, help='Đường dẫn đến tập tin thực thi Streamlit')
    parser.add_argument('--app_path', required=True, help='Đường dẫn đến ứng dụng Streamlit')
    parser.add_argument('--ngrok_token', required=True, help='Token xác thực Ngrok')
    args = parser.parse_args()

    # Đăng ký trình xử lý tín hiệu
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Cấu hình Ngrok
    conf.get_default().auth_token = args.ngrok_token

    # Đóng các đường hầm hiện có
    existing_tunnels = ngrok.get_tunnels()
    if existing_tunnels:
        logger.info(f"Đang đóng {len(existing_tunnels)} đường hầm hiện có...")
        for tunnel in existing_tunnels:
            ngrok.disconnect(tunnel.public_url)

    # Khởi động Streamlit
    global streamlit_process
    streamlit_process = subprocess.Popen([args.streamlit_path, "run", args.app_path])

    # Chờ Streamlit khởi động
    time.sleep(3)

    # Tạo đường hầm Ngrok
    try:
        # Thử các phương pháp kết nối khác nhau với dự phòng
        try:
            tunnel = ngrok.connect(addr=8501, proto="http", bind_tls=True)
        except Exception as e:
            logger.warning(f"Phương pháp kết nối đầu tiên thất bại: {e}")
            try:
                tunnel = ngrok.connect(addr="8501", options={"bind_tls": True})
            except Exception as e:
                logger.warning(f"Phương pháp kết nối thứ hai thất bại: {e}")
                tunnel = ngrok.connect("8501")

        public_url = tunnel.public_url

        # In URL ở định dạng dễ tìm
        print("\n" + "="*60)
        print(f"URL CÔNG KHAI: {public_url}")
        print("Chia sẻ URL này để cho phép người khác truy cập ứng dụng của bạn")
        print("URL này sẽ hoạt động cho đến khi tập lệnh này bị dừng")
        print("="*60 + "\n")

        # Theo dõi tiến trình Streamlit
        logger.info("Ứng dụng đang chạy. Nhấn Ctrl+C để dừng.")
        while True:
            if streamlit_process.poll() is not None:
                logger.warning("Tiến trình Streamlit đã kết thúc bất ngờ!")
                ngrok.kill()
                break
            time.sleep(2)

    except KeyboardInterrupt:
        # Được xử lý bởi signal_handler
        pass
    except Exception as e:
        logger.error(f"Lỗi: {e}")
    finally:
        # Đảm bảo dọn dẹp
        logger.info("Đang dọn dẹp tài nguyên...")
        try:
            ngrok.kill()
            if streamlit_process and streamlit_process.poll() is None:
                streamlit_process.terminate()
                streamlit_process.wait(timeout=5)
        except Exception as e:
            logger.error(f"Lỗi dọn dẹp: {e}")

if __name__ == "__main__":
    main()
