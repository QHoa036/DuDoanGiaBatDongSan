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

# Chuyển đến thư mục chứa script này
cd "$(dirname "$0")"

# Đảm bảo thư mục Demo tồn tại
if [ ! -d "Demo" ]; then
    echo "❌ Không tìm thấy thư mục Demo. Vui lòng kiểm tra lại cấu trúc dự án."
    exit 1
fi

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

echo "🌐 Bạn muốn chạy ứng dụng với ngrok để tạo URL public không? (y/n)"
read use_ngrok

if [[ $use_ngrok == "y" || $use_ngrok == "Y" ]]; then
    # Kiểm tra xem file .env.local có tồn tại và có chứa NGROK_TOKEN
    if [ -f ".env.local" ] && grep -q "NGROK_TOKEN=" ".env.local"; then
        # Đọc token từ file .env.local
        ngrok_token=$(grep "NGROK_TOKEN=" ".env.local" | cut -d'=' -f2)

        # Kiểm tra xem token có giá trị hay không hoặc có phải là placeholder
        if [ -z "$ngrok_token" ] || [ "$ngrok_token" = "<your_ngrok_token>" ]; then
            echo "⚠️ Cần có ngrok token hợp lệ để tiếp tục."
            echo "🔑 Vui lòng nhập ngrok authtoken của bạn (đăng ký tại ngrok.com):"
            read -s ngrok_token

            # Kiểm tra nếu người dùng không nhập gì hoặc nhập placeholder
            if [ -z "$ngrok_token" ] || [ "$ngrok_token" = "<your_ngrok_token>" ]; then
                echo "❌ Bạn phải cung cấp ngrok token hợp lệ để sử dụng tính năng này!"
                echo "🛑 Đang thoát chương trình..."
                exit 1
            fi

            # Cập nhật file .env.local với token mới
            sed -i '' "s/NGROK_TOKEN=.*/NGROK_TOKEN=$ngrok_token/" .env.local
        else
            echo "🔑 Đã tìm thấy ngrok token trong file .env.local"
        fi
    else
        echo "🔑 Nhập ngrok authtoken của bạn (đăng ký tại ngrok.com):"
        read -s ngrok_token

        # Kiểm tra nếu người dùng không nhập gì hoặc nhập placeholder
        if [ -z "$ngrok_token" ] || [ "$ngrok_token" = "<your_ngrok_token>" ]; then
            echo "❌ Bạn phải cung cấp ngrok token hợp lệ để sử dụng tính năng này!"
            echo "🛑 Đang thoát chương trình..."
            exit 1
        fi

        # Lưu token vào file .env.local nếu file tồn tại
        if [ -f ".env.local" ]; then
            echo "NGROK_TOKEN=$ngrok_token" >>.env.local
        else
            echo "NGROK_TOKEN=$ngrok_token" >.env.local
        fi
    fi

    echo "⚙️ Cấu hình ngrok và khởi chạy Streamlit..."

    # Lấy đường dẫn đầy đủ tới streamlit trong môi trường ảo
    STREAMLIT_PATH=$(get_streamlit_path)
    # Ghi ra console để debug
    echo "Sử dụng Streamlit tại: $STREAMLIT_PATH"

    # Tạo file Python tạm thời để chạy ngrok
    cat >run_with_ngrok.py <<EOF
import subprocess
import time
from pyngrok import ngrok

# Thiết lập ngrok
ngrok_token = "$ngrok_token"
ngrok.set_auth_token(ngrok_token)

# Khởi chạy Streamlit trong tiến trình con với file từ thư mục Demo
# Sử dụng đường dẫn đầy đủ tới streamlit thay vì chỉ 'streamlit'
streamlit_process = subprocess.Popen(["$STREAMLIT_PATH", "run", "Demo/vn_real_estate_app.py"])

# Tạo tunnel HTTP đến cổng Streamlit
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
EOF

    # Chạy script Python với ngrok
    $(get_python_path) run_with_ngrok.py

    # Xóa file tạm thời sau khi chạy
    rm run_with_ngrok.py

else
    echo "💻 Khởi chạy Streamlit trên localhost:8501..."
    $(get_streamlit_path) run Demo/vn_real_estate_app.py
fi

# Trở về thư mục gốc và deactivate môi trường ảo khi kết thúc
cd - >/dev/null
deactivate
