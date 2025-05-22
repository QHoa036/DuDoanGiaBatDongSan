#!/usr/bin/env python3
"""
Trình khởi chạy Streamlit với ngrok cho ứng dụng Dự đoán giá Bất động sản Việt Nam
Script này khởi chạy Streamlit và tạo đường hầm ngrok để truy cập công khai từ internet.
"""

# Nhập các thư viện cần thiết
import subprocess
import time
import sys
import signal
import argparse
from pyngrok import ngrok

def signal_handler(sig, frame):
    """Xử lý tắt máy an toàn khi nhấn Ctrl+C"""
    print("\n[STOP] Đang dừng ứng dụng...")
    try:
        # Dừng ngrok trước
        ngrok.kill()
        # Kiểm tra và dừng tiến trình Streamlit nếu đang chạy
        if 'streamlit_process' in globals() and streamlit_process.poll() is None:
            streamlit_process.terminate()
            streamlit_process.wait(timeout=5)  # Chờ tối đa 5 giây để tiến trình kết thúc
    except Exception as e:
        print(f"Lỗi khi dọn dẹp: {e}")
    sys.exit(0)  # Thoát chương trình với mã thành công

def main():
    """Hàm chính để khởi chạy Streamlit và tạo đường hầm ngrok"""
    # Thiết lập và phân tích các tham số dòng lệnh
    parser = argparse.ArgumentParser(description='Chạy Streamlit với đường hầm ngrok')
    parser.add_argument('--streamlit_path', required=True, help='Đường dẫn đến thực thi streamlit')
    parser.add_argument('--app_path', required=True, help='Đường dẫn đến ứng dụng Streamlit')
    parser.add_argument('--ngrok_token', required=True, help='Token xác thực ngrok')
    args = parser.parse_args()  # Phân tích các tham số truyền vào

    # Đăng ký bộ xử lý tín hiệu để tắt máy an toàn
    signal.signal(signal.SIGINT, signal_handler)   # Bắt tín hiệu Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Bắt tín hiệu kết thúc từ hệ thống

    # Thiết lập ngrok với token được cung cấp
    ngrok.set_auth_token(args.ngrok_token)

    # Khởi chạy Streamlit như một tiến trình con
    global streamlit_process  # Khai báo biến toàn cục để có thể truy cập trong signal_handler
    streamlit_process = subprocess.Popen([args.streamlit_path, "run", args.app_path])

    try:
        # Tạo đường hầm HTTP đến cổng Streamlit với cấu hình đơn giản
        # Sử dụng cấu hình cơ bản nhất để đảm bảo tương thích với mọi phiên bản ngrok
        try:
            # Cách 1: Sử dụng API đơn giản nhất (tương thích với hầu hết các phiên bản)
            http_tunnel = ngrok.connect(8501)
        except Exception as e:
            print(f"[WARNING] Lỗi khi sử dụng API đơn giản: {e}")
            print("[INFO] Thử sử dụng cấu hình khác...")

            try:
                # Cách 2: Thử với tham số cụ thể hơn
                http_tunnel = ngrok.connect(addr="8501", proto="http")
            except Exception as e:
                print(f"[WARNING] Lỗi khi sử dụng cấu hình thứ hai: {e}")
                print("[INFO] Thử phương pháp cuối cùng...")

                # Cách 3: Phương pháp cuối cùng - chỉ dùng cổng
                http_tunnel = ngrok.connect("8501")

        # Hiển thị thông tin đường hầm
        print("\n" + "="*60)
        print(f"[URL] URL NGROK PUBLIC: {http_tunnel.public_url}")
        print("[SHARE] Chia sẻ URL này để cho phép người khác truy cập ứng dụng của bạn")
        print("[TIP] Tips: Đường dẫn này chỉ hoạt động khi script này đang chạy")
        print("="*60 + "\n")

        print("[RUNNING] Ứng dụng đang chạy... Nhấn Ctrl+C để dừng")

        # Giữ cho script chạy và giám sát tiến trình Streamlit
        while True:
            # Kiểm tra xem Streamlit còn đang chạy không
            if streamlit_process.poll() is not None:  # Nếu tiến trình đã kết thúc
                print("\n[WARNING] Streamlit đã dừng hoạt động không mong muốn! Đang dọn dẹp...")
                ngrok.kill()  # Đóng kết nối ngrok
                break
            time.sleep(2)  # Tạm dừng 2 giây trước khi kiểm tra lại

    except KeyboardInterrupt:
        # Đã được xử lý trong signal_handler
        pass
    except Exception as e:
        # Xử lý các lỗi khác
        print(f"\n[ERROR] Lỗi không mong muốn: {e}")
    finally:
        # Đảm bảo dọn dẹp trong mọi trường hợp
        print("\n[CLEAN] Đang dọn dẹp tài nguyên...")
        try:
            ngrok.kill()  # Đóng kết nối ngrok
            # Kiểm tra và kết thúc tiến trình Streamlit nếu còn đang chạy
            if 'streamlit_process' in globals() and streamlit_process.poll() is None:
                streamlit_process.terminate()  # Gửi tín hiệu kết thúc đến tiến trình
                streamlit_process.wait(timeout=5)  # Chờ tối đa 5 giây
        except Exception as e:
            print(f"Lỗi khi dọn dẹp: {e}")

if __name__ == "__main__":
    main()
