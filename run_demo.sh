#!/bin/bash

# Script khởi chạy ứng dụng demo Dự đoán giá bất động sản Việt Nam
# với Streamlit và Ngrok

echo "===== KHỞI CHẠY ỨNG DỤNG DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN VIỆT NAM ====="

# Phát hiện hệ điều hành với cách xử lý tốt hơn
DETECT_OS="unknown"
if [[ "$(uname -s)" == "Darwin"* ]]; then
    DETECT_OS="macos"
elif [[ "$(uname -s)" == "Linux"* ]]; then
    DETECT_OS="linux"
elif [[ "$(uname -s)" == "CYGWIN"* || "$(uname -s)" == "MINGW"* || "$(uname -s)" == "MSYS"* ]]; then
    DETECT_OS="windows"
    # Kiểm tra xem có đang chạy trong Git Bash không
    if command -v git --version >/dev/null 2>&1; then
        echo "[OK] Đang chạy trong Git Bash (khuyến nghị cho Windows)"
    else
        echo "[WARNING] Khuyến nghị: Trên Windows, nên sử dụng Git Bash để có trải nghiệm tốt nhất"
        echo "   Tải Git Bash tại: https://git-scm.com/downloads"
    fi
fi

echo "Hệ điều hành được phát hiện: $DETECT_OS"

# Kiểm tra thư mục Demo
if [ ! -d "Demo" ]; then
    echo "Không tìm thấy thư mục Demo. Vui lòng kiểm tra lại cấu trúc dự án!"
    exit 1
fi

# Kiểm tra file requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo "[WARNING] Cảnh báo: Không tìm thấy file requirements.txt. Sẽ sử dụng danh sách thư viện mặc định."
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
        echo "Đang chạy trên Windows thông qua Git Bash/MSYS/MINGW/Cygwin..."
        echo "Kiểm tra Python và pip..."
        if ! command -v python --version >/dev/null 2>&1 && ! command -v python3 --version >/dev/null 2>&1; then
            echo "[ERROR] Python không tìm thấy. Vui lòng cài đặt Python từ https://www.python.org/downloads/"
            echo "[WARNING] Lưu ý: Đảm bảo đã chọn 'Add Python to PATH' trong quá trình cài đặt"
            exit 1
        fi
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
    # Kiểm tra sự tồn tại của thư mục venv
    if [ ! -d "venv" ]; then
        echo "[ERROR] Thư mục venv không tồn tại. Cần tạo môi trường ảo trước."
        return 1
    fi

    case "$DETECT_OS" in
    macos | linux)
        if [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
            return 0
        else
            echo "[ERROR] File kích hoạt venv không tìm thấy tại venv/bin/activate"
            return 1
        fi
        ;;
    windows)
        # Trong Windows với Git Bash/MSYS/MINGW/Cygwin
        if [ -f "venv/Scripts/activate" ]; then
            source venv/Scripts/activate
            return 0
        else
            echo "[ERROR] File kích hoạt venv không tìm thấy tại venv/Scripts/activate"
            return 1
        fi
        ;;
    *)
        echo "Không thể kích hoạt môi trường ảo trên hệ điều hành không xác định"
        return 1
        ;;
    esac
}

create_venv() {
    echo "[TAO] Tạo môi trường ảo mới..."
    case "$DETECT_OS" in
    macos | linux)
        if command -v python3 >/dev/null 2>&1; then
            python3 -m venv venv
        else
            echo "[ERROR] Python3 không tìm thấy. Vui lòng cài đặt Python3."
            exit 1
        fi
        ;;
    windows)
        # Thử python3 trước, nếu không có thì dùng python
        if command -v python3 >/dev/null 2>&1; then
            python3 -m venv venv
        elif command -v python >/dev/null 2>&1; then
            # Kiểm tra phiên bản Python
            PY_VER=$(python -c "import sys; print(sys.version_info.major)")
            if [ "$PY_VER" -lt 3 ]; then
                echo "[ERROR] Cần Python 3.x, phiên bản hiện tại là Python $PY_VER.x"
                exit 1
            fi
            python -m venv venv
        else
            echo "[ERROR] Python không tìm thấy. Vui lòng cài đặt Python 3.x."
            exit 1
        fi
        ;;
    *)
        echo "[ERROR] Không thể tạo môi trường ảo trên hệ điều hành không xác định"
        exit 1
        ;;
    esac

    # Kiểm tra xem venv đã được tạo thành công chưa
    if [ ! -d "venv" ]; then
        echo "[ERROR] Không thể tạo môi trường ảo. Vui lòng kiểm tra quyền truy cập thư mục."
        exit 1
    fi
}

# Kích hoạt môi trường ảo nếu tồn tại
if [ -d "venv" ]; then
    echo "Kích hoạt môi trường ảo..."
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
    echo "Không tìm thấy thư mục Demo. Vui lòng kiểm tra lại cấu trúc dự án."
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

echo "Bạn muốn chạy ứng dụng với ngrok để tạo URL public không? (y/n)"
read use_ngrok

if [[ $use_ngrok == "y" || $use_ngrok == "Y" ]]; then
    # Đảm bảo Python scripts có quyền thực thi
    chmod +x manage_token.py run_ngrok.py

    echo "[CONFIG] Kiểm tra và lấy ngrok token..."

    # Sử dụng utility Python để quản lý token - sẽ nhập và lưu token nếu cần
    ngrok_token=$($(get_python_path) manage_token.py --prompt)

    # Kiểm tra xem token có được trả về không
    if [ -z "$ngrok_token" ]; then
        echo "[ERROR] Không thể lấy ngrok token!"
        echo "[WARNING] Đang thoát chương trình..."
        exit 1
    fi

    echo "[CONFIG] Cấu hình ngrok và khởi chạy Streamlit..."

    # Lấy đường dẫn đầy đủ tới streamlit trong môi trường ảo
    STREAMLIT_PATH=$(get_streamlit_path)
    echo "Sử dụng Streamlit tại: $STREAMLIT_PATH"

    echo "[CONFIG] Khởi chạy ứng dụng với ngrok..."

    # Đảm bảo run_ngrok.py có quyền thực thi
    chmod +x run_ngrok.py

    # Chạy script Python độc lập với ngrok, truyền các tham số cần thiết
    $(get_python_path) run_ngrok.py \
        --streamlit_path "$STREAMLIT_PATH" \
        --app_path "Demo/vn_real_estate_app.py" \
        --ngrok_token "$ngrok_token"

else
    echo "[LOCAL] Khởi chạy Streamlit trên localhost:8501..."
    $(get_streamlit_path) run Demo/vn_real_estate_app.py
fi

# Trở về thư mục gốc và deactivate môi trường ảo khi kết thúc
cd - >/dev/null
deactivate
