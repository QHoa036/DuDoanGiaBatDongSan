# MARK: - Thư viện

import os
import logging
from datetime import datetime

# MARK: - Set Logger

def setup_logger(name='vn_real_estate', log_level=logging.INFO, log_to_file=True):
    """
    Thiết lập và cấu hình logger cho ứng dụng

    Args:
        name (str): Tên của logger
        log_level (int): Mức độ log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file (bool): Có ghi log ra file hay không

    Returns:
        logger: Logger đã được cấu hình
    """
    # Tạo logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Xóa các handler hiện có để tránh log trùng lặp
    if logger.hasHandlers():
        logger.handlers.clear()

    # Định dạng log
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Thiết lập handler ghi ra console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Thiết lập handler ghi ra file nếu được yêu cầu
    if log_to_file:
        # Đảm bảo thư mục logs tồn tại
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logs_dir = os.path.join(current_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)

        # Tạo tên file log với timestamp
        log_file = os.path.join(
            logs_dir,
            f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        # Thiết lập file handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# MARK: - Get logger

def get_logger(name='vn_real_estate'):
    """
    Lấy logger đã được thiết lập

    Args:
        name (str): Tên của logger

    Returns:
        logger: Logger đã được cấu hình
    """
    logger = logging.getLogger(name)

    # Nếu logger chưa được cấu hình, thiết lập nó
    if not logger.hasHandlers():
        logger = setup_logger(name)

    return logger
