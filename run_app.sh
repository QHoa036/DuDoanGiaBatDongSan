#!/bin/bash
# =============================================================================
# Dự Đoán Giá Bất Động Sản Việt Nam - Tập Lệnh Khởi Chạy Tối Ưu
# Tập lệnh hợp nhất thay thế manage_token.py, run_ngrok.py, run_with_ngrok.py
# =============================================================================

# Định nghĩa màu chữ để UX tốt hơn
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# Hàm ghi nhật ký với định dạng nhất quán
log_info() {
    echo -e "${GREEN}[THÔNG TIN]${RESET} $1"
}

log_warning() {
    echo -e "${YELLOW}[CẢNH BÁO]${RESET} $1" >&2
}

log_error() {
    echo -e "${RED}[LỖI]${RESET} $1" >&2
}

log_success() {
    echo -e "${BOLD}${GREEN}[THÀNH CÔNG]${RESET} $1"
}

log_banner() {
    echo -e "\n${BOLD}${BLUE}===== $1 =====${RESET}\n"
}

# Tạo thư mục nhật ký nếu chưa tồn tại
LOGS_DIR="./App/src/logs"
mkdir -p "$LOGS_DIR"
LOG_FILE="$LOGS_DIR/app_$(date '+%Y%m%d').log"

# Hàm ghi nhật ký vào cả console và tập tin
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Ghi vào tập tin nhật ký
    echo "[$timestamp] [$level] $message" >>"$LOG_FILE"

    # Ghi ra console tùy theo cấp độ
    case "$level" in
    "INFO") log_info "$message" ;;
    "WARNING") log_warning "$message" ;;
    "ERROR") log_error "$message" ;;
    "SUCCESS") log_success "$message" ;;
    esac
}

# =============================================================================
# PHÁT HIỆN HỆ ĐIỀU HÀNH
# =============================================================================
log_banner "ĐANG PHÁT HIỆN HỆ ĐIỀU HÀNH"

DETECT_OS="unknown"
if [[ "$(uname -s)" == "Darwin"* ]]; then
    DETECT_OS="macos"
elif [[ "$(uname -s)" == "Linux"* ]]; then
    DETECT_OS="linux"
elif [[ "$(uname -s)" == "CYGWIN"* || "$(uname -s)" == "MINGW"* || "$(uname -s)" == "MSYS"* ]]; then
    DETECT_OS="windows"
    # Kiểm tra nếu đang chạy trong Git Bash
    if command -v git --version >/dev/null 2>&1; then
        log "INFO" "Đang chạy trong Git Bash (được khuyến nghị cho Windows)"
    else
        log "WARNING" "Trên Windows, Git Bash được khuyến nghị để có trải nghiệm tốt nhất"
        log "INFO" "Tải Git Bash tại: https://git-scm.com/downloads"
    fi
fi

log "INFO" "Hệ điều hành đã phát hiện: $DETECT_OS"

# =============================================================================
# KIỂM TRA CẤU TRÚC DỰ ÁN
# =============================================================================
log_banner "ĐANG KIỂM TRA CẤU TRÚC DỰ ÁN"

# Kiểm tra các thư mục thiết yếu
if [ ! -d "App" ]; then
    log "ERROR" "Không tìm thấy thư mục App. Vui lòng kiểm tra cấu trúc dự án!"
    exit 1
fi

if [ ! -d "App/src" ]; then
    log "ERROR" "Không tìm thấy thư mục src. Vui lòng kiểm tra cấu trúc dự án!"
    exit 1
fi

log "INFO" "Cấu trúc dự án đã được xác nhận thành công"

# =============================================================================
# THƯ VIỆN HỆ THỐNG
# =============================================================================
log_banner "ĐANG KIỂM TRA THƯ VIỆN HỆ THỐNG"

install_system_dependencies() {
    case "$DETECT_OS" in
    macos)
        log "INFO" "Kiểm tra các thư viện macOS..."
        if ! command -v brew &>/dev/null; then
            log "WARNING" "Homebrew chưa được cài đặt. Vui lòng cài đặt Homebrew trước."
            log "INFO" "Xem hướng dẫn tại: https://brew.sh"
            exit 1
        fi
        # Cài đặt python-setuptools nếu cần
        if ! brew list python-setuptools &>/dev/null; then
            log "INFO" "Đang cài đặt python-setuptools..."
            brew install python-setuptools
        fi
        ;;
    linux)
        log "INFO" "Kiểm tra các thư viện Linux..."
        if command -v apt-get &>/dev/null; then
            log "INFO" "Đang cài đặt các thư viện cho Debian/Ubuntu..."
            sudo apt-get update
            sudo apt-get install -y python3-setuptools python3-pip python3-venv
        elif command -v yum &>/dev/null; then
            log "INFO" "Đang cài đặt các thư viện cho CentOS/RHEL/Fedora..."
            sudo yum install -y python3-setuptools python3-pip python3-virtualenv
        else
            log "WARNING" "Không thể xác định trình quản lý gói trên Linux. Vui lòng cài đặt thủ công python3-setuptools, python3-pip, python3-venv."
        fi
        ;;
    windows)
        log "INFO" "Kiểm tra các thư viện Windows..."
        if ! command -v python --version >/dev/null 2>&1 && ! command -v python3 --version >/dev/null 2>&1; then
            log "ERROR" "Không tìm thấy Python. Vui lòng cài đặt Python từ https://www.python.org/downloads/"
            log "WARNING" "Lưu ý: Đảm bảo đã chọn 'Add Python to PATH' trong quá trình cài đặt"
            exit 1
        fi
        ;;
    *)
        log "WARNING" "Hệ điều hành không được hỗ trợ: $DETECT_OS"
        log "INFO" "Vui lòng cài đặt thủ công Python, pip, và setuptools."
        ;;
    esac
}

install_system_dependencies

# =============================================================================
# THIẾT LẬP MÔI TRƯỜNG ẢO PYTHON
# =============================================================================
log_banner "ĐANG THIẾT LẬP MÔI TRƯỜNG PYTHON"

# Thiết lập đường dẫn môi trường ảo dựa trên hệ điều hành
VENV_PATH="venv"
PYTHON_CMD="python3"
PIP_CMD="pip3"

if [[ "$DETECT_OS" == "windows" ]]; then
    VENV_ACTIVATE="$VENV_PATH/Scripts/activate"
    PYTHON_PATH="./$VENV_PATH/Scripts/python"
    STREAMLIT_PATH="./$VENV_PATH/Scripts/streamlit"
    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    VENV_ACTIVATE="$VENV_PATH/bin/activate"
    PYTHON_PATH="./$VENV_PATH/bin/python"
    STREAMLIT_PATH="./$VENV_PATH/bin/streamlit"
fi

activate_venv() {
    # Kiểm tra nếu thư mục venv tồn tại
    if [ ! -d "$VENV_PATH" ]; then
        log "ERROR" "Thư mục môi trường ảo không tồn tại. Cần tạo trước."
        return 1
    fi

    log "INFO" "Kích hoạt môi trường ảo..."
    if [ -f "$VENV_ACTIVATE" ]; then
        source "$VENV_ACTIVATE"
        return 0
    else
        log "ERROR" "Không tìm thấy tập tin kích hoạt tại $VENV_ACTIVATE"
        return 1
    fi
}

create_venv() {
    log "INFO" "Tạo môi trường ảo mới..."

    # Chọn lệnh phù hợp dựa trên hệ điều hành
    if [[ "$DETECT_OS" == "windows" ]]; then
        if command -v python3 >/dev/null 2>&1; then
            python3 -m venv "$VENV_PATH"
        elif command -v python >/dev/null 2>&1; then
            # Kiểm tra phiên bản Python trên Windows
            PY_VER=$(python -c "import sys; print(sys.version_info.major)")
            if [ "$PY_VER" -lt 3 ]; then
                log "ERROR" "Yêu cầu Python 3.x, phiên bản hiện tại là Python $PY_VER.x"
                exit 1
            fi
            python -m venv "$VENV_PATH"
        else
            log "ERROR" "Không tìm thấy Python. Vui lòng cài đặt Python 3.x."
            exit 1
        fi
    else
        if command -v python3 >/dev/null 2>&1; then
            python3 -m venv "$VENV_PATH"
        else
            log "ERROR" "Không tìm thấy Python3. Vui lòng cài đặt Python3."
            exit 1
        fi
    fi

    # Kiểm tra xem venv đã được tạo thành công chưa
    if [ ! -d "$VENV_PATH" ]; then
        log "ERROR" "Không thể tạo môi trường ảo. Vui lòng kiểm tra quyền truy cập thư mục."
        exit 1
    fi

    log "SUCCESS" "Môi trường ảo đã được tạo thành công"
}

# Thiết lập môi trường ảo
if [ -d "$VENV_PATH" ]; then
    activate_venv
else
    create_venv
    activate_venv

    log "INFO" "Đang cài đặt các gói cần thiết..."
    $PIP_CMD install --upgrade pip setuptools wheel

    # Cài đặt gói từ requirements.txt nếu tồn tại
    if [ -f "requirements.txt" ]; then
        log "INFO" "Đang cài đặt các gói từ requirements.txt..."
        $PIP_CMD install -r requirements.txt
    else
        log "INFO" "Đang cài đặt các gói mặc định..."
        $PIP_CMD install streamlit==1.31.0 pyspark==3.5.0 pandas==2.1.1 numpy==1.26.0 plotly==5.18.0 matplotlib==3.8.0 seaborn==0.13.0 pyngrok==7.0.0 python-dotenv==1.0.0
    fi

    log "SUCCESS" "Môi trường Python đã được thiết lập thành công"
fi

# =============================================================================
# QUẢN LÝ TOKEN NGROK - Python nhúng
# =============================================================================

manage_ngrok_token() {
    log "INFO" "Kiểm tra token Ngrok..."

    # Tạo tập tin Python tạm thời để quản lý token
    cat >.temp_token_manager.py <<'EOF'
#!/usr/bin/env python3
"""
Tập lệnh tạm thời để quản lý token Ngrok
"""
import os
import sys
import getpass
from dotenv import load_dotenv

def read_token():
    """Đọc token từ tập tin .env.local hoặc .env"""
    # Thử tải từ tập tin môi trường
    for env_file in ['.env.local', '.env']:
        if os.path.exists(env_file):
            load_dotenv(env_file)
            token = os.environ.get('NGROK_TOKEN')
            if token and token != "<your_ngrok_token>":
                return token

    # Đọc trực tiếp từ tập tin như phương án dự phòng
    if os.path.exists(".env.local"):
        with open(".env.local", "r") as f:
            for line in f:
                if line.startswith("NGROK_TOKEN="):
                    token = line.strip().split("=", 1)[1].strip().strip('"\'')
                    if token and token != "<your_ngrok_token>":
                        return token

    return None

def save_token(token):
    """Lưu token vào tập tin .env.local"""
    env_content = ""
    token_line = f"NGROK_TOKEN={token}\n"

    if os.path.exists(".env.local"):
        # Đọc tập tin hiện có
        with open(".env.local", "r") as f:
            lines = f.readlines()

        # Cập nhật hoặc thêm dòng token
        token_found = False
        for i, line in enumerate(lines):
            if line.startswith("NGROK_TOKEN="):
                lines[i] = token_line
                token_found = True
                break

        if not token_found:
            lines.append(token_line)

        env_content = "".join(lines)
    else:
        # Tạo tập tin mới
        env_content = token_line

    # Ghi vào tập tin
    with open(".env.local", "w") as f:
        f.write(env_content)

    return True

def main():
    """Hàm chính"""
    # Đầu tiên thử đọc token hiện có
    token = read_token()

    # Nếu không tìm thấy token, yêu cầu người dùng nhập
    if not token:
        print("Vui lòng nhập token Ngrok của bạn (đăng ký tại ngrok.com):", file=sys.stderr)
        token = getpass.getpass("")

        # Xác thực token
        if not token or token == "<your_ngrok_token>":
            print("LỖI: Yêu cầu token Ngrok hợp lệ!", file=sys.stderr)
            return 1

        # Lưu token
        save_token(token)
        print("Token đã được lưu vào tập tin .env.local", file=sys.stderr)

    # In token ra stdout để tập lệnh shell thu thập
    print(token)
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF

    # Thiết lập quyền thực thi cho tập lệnh
    chmod +x .temp_token_manager.py

    # Chạy tập lệnh và thu thập token
    NGROK_TOKEN=$($PYTHON_PATH .temp_token_manager.py)

    # Kiểm tra xem token đã được trả về chưa
    if [ -z "$NGROK_TOKEN" ]; then
        log "ERROR" "Không thể lấy token Ngrok!"
        rm .temp_token_manager.py
        exit 1
    fi

    # Dọn dẹp tập lệnh tạm thời
    rm .temp_token_manager.py

    log "SUCCESS" "Token Ngrok đã được cấu hình thành công"
    return 0
}

# =============================================================================
# THIẾT LẬP ĐƯỜNG HẦM NGROK - Python nhúng
# =============================================================================

setup_ngrok_tunnel() {
    log "INFO" "Thiết lập đường hầm Ngrok..."

    # Tạo tập lệnh Python tạm thời để thiết lập Ngrok
    cat >.temp_ngrok_setup.py <<'EOF'
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
EOF

    # Thiết lập quyền thực thi cho tập lệnh
    chmod +x .temp_ngrok_setup.py

    # Thiết lập biến môi trường để ẩn thông báo nhật ký không cần thiết từ Ngrok
    export NGROK_LOG_LEVEL=warn
    export PYNGROK_LOG_LEVEL=warning

    # Chạy tập lệnh thiết lập Ngrok
    log "INFO" "Đang khởi động Streamlit với đường hầm Ngrok..."
    $PYTHON_PATH .temp_ngrok_setup.py --streamlit_path "$STREAMLIT_PATH" --app_path "App/app.py" --ngrok_token "$NGROK_TOKEN"

    # Dọn dẹp tập lệnh tạm thời
    rm .temp_ngrok_setup.py

    return 0
}

# =============================================================================
# THỰC THI CHÍNH
# =============================================================================
log_banner "DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN VIỆT NAM"

# Chuyển đến thư mục tập lệnh
cd "$(dirname "$0")"

# Hỏi người dùng có muốn sử dụng Ngrok
echo -e "\n${BOLD}${BLUE}Bạn có muốn chạy ứng dụng với Ngrok để tạo URL công khai không? (y/n)${RESET}"
read -r use_ngrok

if [[ $use_ngrok == "y" || $use_ngrok == "Y" ]]; then
    # Lấy token Ngrok và thiết lập đường hầm
    manage_ngrok_token
    setup_ngrok_tunnel
else
    # Chạy cục bộ không cần Ngrok
    log "INFO" "Đang khởi động Streamlit trên localhost:8501..."
    $STREAMLIT_PATH run App/app.py
fi

# Dọn dẹp
log "INFO" "Ứng dụng đã thoát. Hủy kích hoạt môi trường ảo..."
deactivate
log "SUCCESS" "Phiên ứng dụng đã kết thúc"

exit 0
