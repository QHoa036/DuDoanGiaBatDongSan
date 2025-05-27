#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hệ thống ghi log nâng cao cho ứng dụng Dự đoán Giá Bất Động Sản Việt Nam

Mô-đun này cung cấp một hệ thống ghi log toàn diện với:
- Nhiều cấp độ log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Ghi log ra file và console với định dạng tùy chỉnh
- Tích hợp với Streamlit để hiển thị log trong giao diện người dùng
- Hỗ trợ đa nền tảng
- Theo dõi metrics và hiệu suất của mô hình
"""

import os
import logging
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
import streamlit as st
import functools
import inspect
import threading
import json

# MARK: - Màu sắc cho logger

class LogColors:
    """
    Màu sắc cho log trên console
    """
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    # Màu chữ
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Màu nền
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

# MARK: - Ánh xạ log

LEVEL_COLORS = {
    logging.DEBUG: LogColors.CYAN,
    logging.INFO: LogColors.GREEN,
    logging.WARNING: LogColors.YELLOW,
    logging.ERROR: LogColors.RED,
    logging.CRITICAL: f"{LogColors.BOLD}{LogColors.RED}"
}

# MARK: - Định dạng log

class CustomFormatter(logging.Formatter):
    """
    Lớp định dạng log tùy chỉnh với màu sắc cho console
    """
    def __init__(self, use_colors=True, fmt=None, datefmt=None):
        if fmt is None:
            fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
        if datefmt is None:
            datefmt = "%Y-%m-%d %H:%M:%S"

        self.use_colors = use_colors
        self.default_fmt = fmt
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record):
        # Tạo bản sao của record để tránh thay đổi bản gốc
        formatted_record = logging.makeLogRecord(record.__dict__)

        # Xử lý các trường hợp lỗi đặc biệt
        if formatted_record.exc_info:
            # Định dạng lại traceback
            formatted_record.exc_text = self.formatException(formatted_record.exc_info)

        levelname = formatted_record.levelname

        # Thêm màu sắc cho log nếu được bật
        try:
            # Kiểm tra an toàn trước khi sử dụng isatty()
            use_colors = self.use_colors and hasattr(sys, 'stderr') and sys.stderr and hasattr(sys.stderr, 'isatty') and sys.stderr.isatty()
            if use_colors:
                color = LEVEL_COLORS.get(formatted_record.levelno, LogColors.RESET)
                formatted_record.levelname = f"{color}{levelname}{LogColors.RESET}"
        except (AttributeError, TypeError):
            # An toàn nếu có lỗi
            pass

        return super().format(formatted_record)

# MARK: - Xử lý log cho Streamlit

class StreamlitHandler(logging.Handler):
    """
    Handler ghi log vào giao diện Streamlit
    """
    def __init__(self, level=logging.INFO):
        super().__init__(level)
        self.logs = []
        self.max_logs = 100  # Giới hạn số lượng log hiển thị

    def emit(self, record):
        try:
            msg = self.format(record)
            self.logs.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'level': record.levelname,
                'message': record.getMessage(),
                'formatted': msg
            })

            # Giới hạn số lượng log
            if len(self.logs) > self.max_logs:
                self.logs.pop(0)

            # Cập nhật UI nếu có thể
            self._update_ui()
        except Exception:
            self.handleError(record)

    def _update_ui(self):
        """
        Cập nhật giao diện Streamlit nếu có thể
        """
        if 'st_log_container' in st.session_state:
            with st.session_state.st_log_container:
                st.empty()
                for log in self.logs[-10:]:  # Chỉ hiển thị 10 log gần nhất
                    level = log['level']
                    if level == 'DEBUG':
                        st.text(f"🔍 {log['time']} - {log['message']}")
                    elif level == 'INFO':
                        st.info(f"{log['time']} - {log['message']}")
                    elif level == 'WARNING':
                        st.warning(f"{log['time']} - {log['message']}")
                    elif level == 'ERROR' or level == 'CRITICAL':
                        st.error(f"{log['time']} - {log['message']}")
                    else:
                        st.text(f"{log['time']} - {log['message']}")

# MARK: - Lưu trữ và quản lý logger
# Thread-local storage để lưu trữ thông tin logger
_thread_local = threading.local()

def get_logger(name=None, level=None, log_file=None, enable_streamlit=False) -> logging.Logger:
    """
    Tạo và cấu hình logger
    """
    # Xác định tên logger
    if name is None:
        # Lấy tên module gọi
        frame = inspect.currentframe().f_back
        module = inspect.getmodule(frame)
        name = module.__name__ if module else 'root'

    # Kiểm tra xem logger đã tồn tại chưa
    if hasattr(_thread_local, 'loggers') and name in _thread_local.loggers:
        return _thread_local.loggers[name]

    # Tạo logger mới
    logger = logging.getLogger(name)

    # Thiết lập cấp độ log
    if level is None:
        level = logging.ERROR  # Tăng cấp độ log mặc định lên ERROR để giảm log tối thiểu
    logger.setLevel(level)

    # Tạo bộ lọc để loại bỏ các tin nhắn log không cần thiết
    class ComprehensiveLogFilter(logging.Filter):
        def filter(self, record):
            # Lọc ra các log không cần thiết
            message = record.getMessage().lower()
            filtered_patterns = [
                'starting streamlit',
                'ngrok tunnel',
                'starting up',
                'running on',
                'java_tool_options',
                'picked up',
                'listening on',
                'started session',
                'connected to',
                'initiating',
                'loading',
                'loading settings',
                'retrieving',
                'connection established',
                'binding to',
                'set current project',
                'http-8501',
                'waiting',
                'downloading',
                't=',
                'lvl=info',
                'obj=tunnels',
                'ivy',
                'hadoop',
                'spark',
                'jvm',
                'jar',
                'default cache',
                'artifacts',
                'modules',
                'confs',
                'resolution',
                'apache',
                'native',
                'found',
                'resolver',
                'opening'
            ]

            # Kiểm tra xem thông điệp có chứa bất kỳ mẫu nào không
            for pattern in filtered_patterns:
                if pattern in message:
                    return False
            return True

    # Ngăn chặn việc truyền log lên logger cha
    # Đây là nguyên nhân chính gây ra các log trùng lặp
    logger.propagate = False

    # Xóa tất cả handler hiện có
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Thêm handler cho console với bộ lọc
    console_handler = logging.StreamHandler(sys.stdout)  # Sử dụng stdout thay vì stderr để có màu sắc
    console_handler.setFormatter(CustomFormatter(use_colors=True))

    # Áp dụng bộ lọc cho console handler
    console_handler.addFilter(ComprehensiveLogFilter())
    logger.addHandler(console_handler)

    # Thêm handler cho file nếu được yêu cầu
    if log_file is not None:
        # Đảm bảo thư mục logs tồn tại
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Thêm handler cho Streamlit nếu được yêu cầu
    if enable_streamlit:
        streamlit_handler = StreamlitHandler()
        streamlit_formatter = logging.Formatter(
            "%(message)s",
            "%H:%M:%S"
        )
        streamlit_handler.setFormatter(streamlit_formatter)
        logger.addHandler(streamlit_handler)

    # Lưu logger vào thread-local storage
    if not hasattr(_thread_local, 'loggers'):
        _thread_local.loggers = {}
    _thread_local.loggers[name] = logger

    return logger

# MARK: - Decorator và tiện ích

def log_execution_time(func=None, logger=None, level=logging.INFO):
    """
    Decorator để ghi log thời gian thực thi của hàm
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Xác định logger
            nonlocal logger
            if logger is None:
                module = inspect.getmodule(func)
                logger_name = module.__name__ if module else func.__module__
                logger = get_logger(logger_name)

            # Ghi log bắt đầu
            logger.log(level, f"Bắt đầu thực thi {func.__name__}")
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time

                # Ghi log kết thúc
                logger.log(level, f"Hoàn thành {func.__name__} trong {execution_time:.4f} giây")
                return result
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time

                # Ghi log lỗi
                logger.exception(f"Lỗi trong {func.__name__} sau {execution_time:.4f} giây: {str(e)}")
                raise

        return wrapper

    # Hỗ trợ sử dụng decorator với hoặc không có tham số
    if func is not None:
        return decorator(func)
    return decorator

def setup_streamlit_logging_container():
    """
    Thiết lập container để hiển thị log trong Streamlit
    Gọi hàm này trong view trước khi sử dụng logger với enable_streamlit=True
    """
    if 'st_log_container' not in st.session_state:
        st.session_state.st_log_container = st.container()

def get_all_logs(max_count=None) -> List[Dict]:
    """
    Lấy tất cả log đã được ghi nhận bởi StreamlitHandler
    """
    logs = []

    # Thu thập log từ tất cả logger
    if hasattr(_thread_local, 'loggers'):
        for logger in _thread_local.loggers.values():
            for handler in logger.handlers:
                if isinstance(handler, StreamlitHandler):
                    logs.extend(handler.logs)

    # Sắp xếp theo thời gian
    logs.sort(key=lambda x: x['time'])

    # Giới hạn số lượng nếu cần
    if max_count is not None and max_count > 0:
        logs = logs[-max_count:]

    return logs

def configure_root_logger(level=logging.INFO, log_file=None, enable_streamlit=False):
    """
    Cấu hình logger gốc của ứng dụng
    """
    # Tạo thư mục logs trong thư mục src
    if log_file is None:
        # Lấy đường dẫn đến thư mục src (chỉ đi lên 1 cấp từ utils)
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logs_dir = os.path.join(src_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(logs_dir, f"app_{timestamp}.log")

    # Cấu hình logger gốc
    root_logger = get_logger("root", level=level, log_file=log_file, enable_streamlit=enable_streamlit)

    # Đặt logger gốc cho logging module
    logging.root = root_logger

    return root_logger

# MARK: - Theo dõi metrics và hiệu suất

class MetricsLogger:
    """
    Lớp tiện ích để theo dõi metrics và hiệu suất của mô hình
    """
    def __init__(self, name: str, metrics_service=None):
        """
        Khởi tạo logger cho metrics
        """
        self.logger = get_logger(f"metrics.{name}")
        self.metrics_service = metrics_service
        self.name = name
        self._history = {}

    def log_training_metrics(self, metrics: Dict[str, float], model_name: str):
        """
        Ghi lại các metrics trong quá trình huấn luyện mô hình
        """
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Mô hình {model_name} - Metrics: {metrics_str}")

        # Lưu vào metrics service nếu có
        if self.metrics_service:
            self.metrics_service.add_metrics(model_name, metrics)

        # Lưu vào lịch sử nội bộ
        if model_name not in self._history:
            self._history[model_name] = {}

        for metric_name, value in metrics.items():
            if metric_name not in self._history[model_name]:
                self._history[model_name][metric_name] = []
            self._history[model_name][metric_name].append(value)

    def log_prediction(self, input_data: Dict[str, Any], prediction: float, confidence: Optional[float] = None):
        """
        Ghi lại thông tin về một dự đoán
        """
        msg = f"Dự đoán: {prediction:.2f}"
        if confidence:
            msg += f", Độ tin cậy: {confidence:.2f}"

        # Ghi log chi tiết ở cấp độ debug
        self.logger.debug(f"Input: {input_data}, {msg}")

        # Ghi log tổng quan ở cấp độ info
        self.logger.info(msg)

    def log_performance(self, operation: str, duration_ms: float):
        """
        Ghi lại hiệu suất của một thao tác
        """
        self.logger.info(f"Hiệu suất - {operation}: {duration_ms:.2f}ms")

        # Lưu vào metrics service nếu có
        if self.metrics_service:
            perf_metrics = {f"perf_{operation}": duration_ms}
            self.metrics_service.add_metrics(f"{self.name}_performance", perf_metrics)

    def timing_decorator(self, func):
        """
        Decorator để đo thời gian thực thi của một hàm

        Usage:
            @metrics_logger.timing_decorator
            def my_function():
                pass
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            duration_ms = (end_time - start_time) * 1000
            self.log_performance(func.__name__, duration_ms)

            return result
        return wrapper


# Tiện ích để lấy timestamp định dạng chuẩn
def get_timestamp() -> str:
    """
    Lấy timestamp hiện tại theo định dạng chuẩn
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# Tiện ích để format thời gian
def format_duration(milliseconds: float) -> str:
    """
    Format thời gian thực thi thành chuỗi dễ đọc
    """
    if milliseconds < 1000:
        return f"{milliseconds:.2f}ms"

    seconds = milliseconds / 1000
    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes = seconds / 60
    seconds_remainder = seconds % 60
    return f"{int(minutes)}m {seconds_remainder:.2f}s"
