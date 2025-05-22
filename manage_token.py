#!/usr/bin/env python3
"""
Tiện ích quản lý token ngrok cho ứng dụng Dự đoán giá Bất động sản Việt Nam
Tiện ích này quản lý việc đọc, lưu trữ và xác thực token ngrok để tạo URL public
"""

import os
import sys
import argparse
import getpass

def read_token():
    """Thành phần đọc token từ file .env.local nếu tồn tại"""
    token = None
    if os.path.exists(".env.local"):  # Kiểm tra xem file .env.local có tồn tại hay không
        with open(".env.local", "r") as f:  # Mở file để đọc
            for line in f:  # Đọc từng dòng trong file
                if line.startswith("NGROK_TOKEN="):  # Tìm dòng bắt đầu bằng NGROK_TOKEN=
                    # Tách chuỗi theo dấu = và lấy phần giá trị token
                    token = line.strip().split("=", 1)[1].strip().strip('"\'')
                    # Kiểm tra xem token có giá trị và không phải là placeholder
                    if token and token != "<your_ngrok_token>":
                        return token  # Trả về token hợp lệ
    return None  # Trả về None nếu không tìm thấy token hợp lệ

def save_token(token):
    """Lưu token vào file .env.local"""
    if os.path.exists(".env.local"):
        # Đọc file hiện có
        with open(".env.local", "r") as f:
            lines = f.readlines()  # Đọc tất cả các dòng trong file

        # Cập nhật hoặc thêm dòng token
        token_line_found = False  # Cờ đánh dấu xem đã tìm thấy dòng token chưa
        for i, line in enumerate(lines):
            if line.startswith("NGROK_TOKEN="):  # Tìm dòng chứa token
                lines[i] = f"NGROK_TOKEN={token}\n"  # Thay thế bằng token mới
                token_line_found = True  # Đánh dấu đã tìm thấy
                break

        if not token_line_found:  # Nếu không tìm thấy dòng token
            lines.append(f"NGROK_TOKEN={token}\n")  # Thêm dòng token mới vào cuối file

        # Ghi lại vào file
        with open(".env.local", "w") as f:
            f.writelines(lines)  # Ghi tất cả các dòng vào file
    else:
        # Tạo file mới nếu chưa tồn tại
        with open(".env.local", "w") as f:
            f.write(f"NGROK_TOKEN={token}\n")  # Ghi dòng token vào file mới

def main():
    """Hàm chính xử lý tương tác token ngrok"""
    # Thiết lập parser tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Công cụ quản lý token ngrok")
    parser.add_argument("--get", action="store_true", help="Lấy token từ file .env.local")
    parser.add_argument("--prompt", action="store_true",
                    help="Nhập token từ người dùng và lưu vào file .env.local")

    # Phân tích các tham số dòng lệnh
    args = parser.parse_args()

    if args.get:  # Chế độ lấy token
        token = read_token()  # Đọc token từ file .env.local
        if token:  # Nếu tìm thấy token
            print(token)  # In token ra màn hình (stdout) để shell script có thể sử dụng
            return 0  # Trả về mã thành công
        else:  # Nếu không tìm thấy token
            return 1  # Trả về mã lỗi

    elif args.prompt:  # Chế độ nhập token từ người dùng
        token = read_token()  # Đầu tiên thử đọc token từ file .env.local
        if not token:  # Nếu không tìm thấy token hoặc token không hợp lệ
            # Hiển thị thông báo yêu cầu nhập token (ra stderr để không gây nhiễu stdout)
            print("[TOKEN] Nhập ngrok authtoken của bạn (đăng ký tại ngrok.com):", file=sys.stderr)
            token = getpass.getpass("")  # Sử dụng getpass để nhập mật khẩu an toàn không hiển thị

            # Kiểm tra tính hợp lệ của token
            if not token or token == "<your_ngrok_token>":
                print("[LOI] Bạn phải cung cấp ngrok token hợp lệ để sử dụng tính năng này!", file=sys.stderr)
                return 1  # Trả về mã lỗi

            # Lưu token vào file .env.local
            save_token(token)
            print("[OK] Đã lưu token vào file .env.local", file=sys.stderr)
        else:  # Nếu đã tìm thấy token trong file
            print("[TOKEN] Đã tìm thấy ngrok token trong file .env.local", file=sys.stderr)

        # Luôn xuất token ra stdout để shell script có thể bắt lấy
        print(token)
        return 0  # Trả về mã thành công

    else:  # Nếu không có tham số nào được chỉ định
        parser.print_help()  # Hiển thị hướng dẫn sử dụng
        return 1  # Trả về mã lỗi

if __name__ == "__main__":
    sys.exit(main())
