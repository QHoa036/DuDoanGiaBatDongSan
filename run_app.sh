#!/bin/bash

# Script khởi chạy ứng dụng demo Dự đoán giá bất động sản Việt Nam
# với Streamlit và Ngrok

echo "===== KHỞI CHẠY ỨNG DỤNG DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN VIỆT NAM ====="

# Phát hiện hệ điều hành
DETECT_OS="unknown"
case "$(uname -s)" in
Darwin*)
    DETECT_OS="macos"
    ;;
Linux*)
    DETECT_OS="linux"
    ;;
CYGWIN* | MINGW* | MSYS*)
    DETECT_OS="windows"
    ;;
esac

echo "Hệ điều hành được phát hiện: $DETECT_OS"

# Kiểm tra thư mục App
if [ ! -d "App" ]; then
    echo "Không tìm thấy thư mục App. Vui lòng kiểm tra lại cấu trúc dự án!"
    exit 1
fi

# Kiểm tra file requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo "⚠️ Cảnh báo: Không tìm thấy file requirements.txt. Sẽ sử dụng danh sách thư viện mặc định."
fi

# Kiểm tra và cài đặt các thư viện hệ thống cần thiết dựa trên OS
echo "Kiểm tra các thư viện hệ thống..."

install_system_dependencies() {
    case "$DETECT_OS" in
    macos)
        if ! command -v brew &>/dev/null; then
            echo "Homebrew không được cài đặt. Vui lòng cài đặt Homebrew trước."
            echo "Xem hướng dẫn tại: https://brew.sh"
            exit 1
        fi
        # Cài đặt python-setuptools nếu chưa có
        if ! brew list python-setuptools &>/dev/null; then
            echo "Cài đặt python-setuptools..."
            brew install python-setuptools
        fi
        ;;
    linux)
        if command -v apt-get &>/dev/null; then
            echo "Kiểm tra và cài đặt các thư viện trên hệ thống Debian/Ubuntu..."
            sudo apt-get update
            sudo apt-get install -y python3-setuptools python3-pip python3-venv
        elif command -v yum &>/dev/null; then
            echo "Kiểm tra và cài đặt các thư viện trên hệ thống CentOS/RHEL/Fedora..."
            sudo yum install -y python3-setuptools python3-pip python3-virtualenv
        else
            echo "Không thể xác định trình quản lý gói trên Linux. Vui lòng cài đặt python3-setuptools, python3-pip, python3-venv thủ công."
        fi
        ;;
    windows)
        echo "Đang chạy trên Windows thông qua MSYS/MINGW/Cygwin..."
        echo "Vui lòng đảm bảo rằng Python và pip đã được cài đặt."
        ;;
    *)
        echo "Hệ điều hành không được hỗ trợ: $DETECT_OS"
        echo "Vui lòng cài đặt Python, pip, và setuptools thủ công."
        ;;
    esac
}

install_system_dependencies

# Kích hoạt môi trường ảo dựa trên hệ điều hành
activate_venv() {
    case "$DETECT_OS" in
    macos | linux)
        source venv/bin/activate
        ;;
    windows)
        # Trong Windows với MSYS/MINGW/Cygwin, sử dụng cú pháp khác
        source venv/Scripts/activate
        ;;
    *)
        echo "Không thể kích hoạt môi trường ảo trên hệ điều hành không xác định"
        exit 1
        ;;
    esac
}

create_venv() {
    case "$DETECT_OS" in
    macos | linux)
        python3 -m venv venv
        ;;
    windows)
        python -m venv venv
        ;;
    *)
        echo "Không thể tạo môi trường ảo trên hệ điều hành không xác định"
        exit 1
        ;;
    esac
}

# Kích hoạt môi trường ảo nếu tồn tại
if [ -d "venv" ]; then
    echo "🚀 Kích hoạt môi trường ảo..."
    activate_venv
else
    echo "Tạo môi trường ảo mới..."
    create_venv
    activate_venv

    echo "Cài đặt các thư viện cần thiết..."
    pip install --upgrade pip setuptools wheel

    # Cài đặt từ requirements.txt nếu tồn tại
    if [ -f "requirements.txt" ]; then
        echo "Cài đặt các thư viện từ requirements.txt..."
        pip install -r requirements.txt
    else
        echo "Cài đặt các thư viện mặc định..."
        pip install selenium webdriver-manager pandas numpy pyspark matplotlib seaborn streamlit pyngrok ngrok folium plotly
    fi
fi

# Hàm hiển thị menu lựa chọn
show_menu() {
    echo ""
    echo "MENU CHÍNH:"
    echo "1. Thu thập dữ liệu bất động sản (1_fetch_real_estate.py)"
    echo "2. Lấy thông tin chi tiết bất động sản (2_property_details.py)"
    echo "3. Tiền xử lý dữ liệu (3_preprocess_data.py)"
    echo "4. Lưu trữ dữ liệu trên HDFS (4_HDFS_storage.py)"
    echo "5. Huấn luyện mô hình (5_model_training.py)"
    echo "6. Khởi chạy ứng dụng Streamlit (6_streamlit_app.py)"
    echo "7. Trực quan hóa dữ liệu (7_visualize_data.py)"
    echo "8. Chạy toàn bộ quy trình (1-7)"
    echo "9. Thoát"
    echo ""
    echo "Lựa chọn của bạn (1-9): "
}

# Hàm chạy file Python
run_python_file() {
    file_name=$1
    echo "===== ĐANG THỰC THI $file_name ====="

    # Đường dẫn Python dựa trên hệ điều hành
    get_python_path() {
        case "$DETECT_OS" in
        macos | linux)
            echo "./venv/bin/python"
            ;;
        windows)
            echo "./venv/Scripts/python"
            ;;
        *)
            echo "python" # Sử dụng Python hệ thống nếu không xác định được OS
            ;;
        esac
    }

    # Đảm bảo môi trường ảo được kích hoạt
    if [ -d "venv" ]; then
        # Sử dụng Python từ môi trường ảo
        $(get_python_path) "App/$file_name"
    else
        echo "Môi trường ảo chưa được tạo. Đang tạo môi trường..."
        create_venv
        activate_venv
        pip install --upgrade pip setuptools wheel

        # Cài đặt các thư viện từ requirements.txt
        if [ -f "requirements.txt" ]; then
            echo "Cài đặt thư viện từ file requirements.txt..."
            pip install -r requirements.txt
        else
            echo "Không tìm thấy file requirements.txt. Cài đặt thư viện mặc định..."
            pip install selenium webdriver-manager pandas numpy pyspark matplotlib seaborn streamlit pyngrok ngrok folium plotly
        fi

        $(get_python_path) "App/$file_name"
    fi

    # Kiểm tra lỗi
    if [ $? -ne 0 ]; then
        echo "Lỗi khi thực thi $file_name. Kiểm tra lại!"
        return 1
    else
        echo "Thực thi $file_name thành công!"
        return 0
    fi
}

# Hàm chạy Streamlit với Ngrok
run_streamlit_with_ngrok() {
    echo "Bạn có muốn sử dụng ngrok để tạo URL public không? (y/n)"
    read use_ngrok

    # Lấy đường dẫn Python và Streamlit dựa trên hệ điều hành
    get_python_path() {
        case "$DETECT_OS" in
        macos | linux)
            echo "./venv/bin/python"
            ;;
        windows)
            echo "./venv/Scripts/python"
            ;;
        *)
            echo "python" # Sử dụng Python hệ thống nếu không xác định được OS
            ;;
        esac
    }

    get_streamlit_path() {
        case "$DETECT_OS" in
        macos | linux)
            echo "./venv/bin/streamlit"
            ;;
        windows)
            echo "./venv/Scripts/streamlit"
            ;;
        *)
            echo "streamlit" # Sử dụng streamlit hệ thống nếu không xác định được OS
            ;;
        esac
    }

    # Đảm bảo môi trường ảo được kích hoạt
    if [ ! -d "venv" ]; then
        echo "Môi trường ảo chưa được tạo. Đang tạo môi trường..."
        create_venv
        activate_venv
        pip install --upgrade pip setuptools wheel

        # Cài đặt từ requirements.txt
        if [ -f "requirements.txt" ]; then
            echo "Cài đặt thư viện từ file requirements.txt..."
            pip install -r requirements.txt
        else
            echo "Không tìm thấy file requirements.txt. Cài đặt thư viện mặc định..."
            pip install selenium webdriver-manager pandas numpy pyspark matplotlib seaborn streamlit pyngrok ngrok folium plotly
        fi
    fi

    if [[ $use_ngrok == "y" || $use_ngrok == "Y" ]]; then
        echo "Nhập ngrok authtoken của bạn (đăng ký tại ngrok.com):"
        read -s ngrok_token

        echo "Cấu hình ngrok và khởi chạy Streamlit..."

        # Lấy đường dẫn đầy đủ tới streamlit trong môi trường ảo
        STREAMLIT_PATH=$(get_streamlit_path)
        # Ghi ra console để debug
        echo "Sử dụng Streamlit tại: $STREAMLIT_PATH"

        # Tạo file python tạm để khởi chạy ngrok
        cat >run_streamlit_ngrok.py <<EOF
import os
import subprocess
import time
from pyngrok import ngrok

# Cấu hình ngrok
ngrok_token = "$ngrok_token"
ngrok.set_auth_token(ngrok_token)

# Khởi chạy Streamlit trong tiến trình con
# Sử dụng đường dẫn đầy đủ tới streamlit thay vì chỉ 'streamlit'
streamlit_process = subprocess.Popen(["$STREAMLIT_PATH", "run", "App/6_streamlit_app.py"])

# Khởi tạo tunnel
http_tunnel = ngrok.connect(addr="8501", proto="http", bind_tls=True)
print("\n" + "="*50)
print(f"URL NGROK PUBLIC: {http_tunnel.public_url}")
print("Chia sẻ URL này để cho phép người khác truy cập ứng dụng của bạn")
print("="*50 + "\n")

try:
    # Giữ script chạy
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # Dọn dẹp khi người dùng nhấn Ctrl+C
    print("\nĐang dừng ứng dụng...")
    ngrok.kill()
    streamlit_process.terminate()
EOF

        # Chạy file python tạm với môi trường ảo
        $(get_python_path) run_streamlit_ngrok.py

        # Xóa file tạm sau khi chạy
        rm run_streamlit_ngrok.py
    else
        echo "Khởi chạy Streamlit trên localhost:8501..."
        $(get_streamlit_path) run App/6_streamlit_app.py
    fi
}

# Hàm chạy toàn bộ quy trình
run_full_pipeline() {
    for i in {1..5} {7..7}; do
        file_name="${i}_*.py"
        python_file=$(find App -name "$file_name" -type f)
        if [ -n "$python_file" ]; then
            base_name=$(basename "$python_file")
            run_python_file "$base_name"
            if [ $? -ne 0 ]; then
                echo "Dừng quy trình do lỗi."
                return 1
            fi
        fi
    done

    # Chạy Streamlit cuối cùng
    run_streamlit_with_ngrok
    return 0
}

# Vòng lặp chính
while true; do
    show_menu
    read choice

    case $choice in
    1)
        run_python_file "1_fetch_real_estate.py"
        ;;
    2)
        run_python_file "2_property_details.py"
        ;;
    3)
        run_python_file "3_preprocess_data.py"
        ;;
    4)
        run_python_file "4_HDFS_storage.py"
        ;;
    5)
        run_python_file "5_model_training.py"
        ;;
    6)
        run_streamlit_with_ngrok
        ;;
    7)
        run_python_file "7_visualize_data.py"
        ;;
    8)
        run_full_pipeline
        ;;
    9)
        echo "Cảm ơn bạn đã sử dụng ứng dụng. Tạm biệt!"
        exit 0
        ;;
    *)
        echo "Lựa chọn không hợp lệ. Vui lòng chọn từ 1-9."
        ;;
    esac
done
