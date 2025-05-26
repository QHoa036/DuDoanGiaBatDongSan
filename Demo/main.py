#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dự đoán Giá Bất Động Sản Việt Nam
Thực hiện theo Kiến trúc MVVM (Model-View-ViewModel)

Cấu trúc:
- src/models: Chứa các lớp dữ liệu và dịch vụ dữ liệu
- src/viewmodels: Chứa các lớp xử lý logic nghiệp vụ, kết nối giữa View và Model
- src/views: Chứa các thành phần giao diện người dùng
- src/utils: Chứa các tiện ích, công cụ hỗ trợ
- src/styles: Chứa các file CSS định dạng giao diện
- src/data: Chứa dữ liệu đầu vào cho ứng dụng
- logs: Chứa các tập tin ghi log của ứng dụng
"""

# MARK: - Thư viện

import os
import sys
import logging
from dotenv import load_dotenv
import streamlit as st

from src.utils.logger_utils import get_logger, configure_root_logger, log_execution_time
from src.viewmodels.app_viewmodel import AppViewModel
from src.views.app_view import AppView

# MARK: - Cấu hình

# Streamlit
st.set_page_config(
    page_title="Dự Đoán Giá Bất Động Sản Việt Nam",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="auto",
)

# Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# MARK: - Hàm chính

@log_execution_time
def main():
    # Logger
    logger = configure_root_logger(
        level=logging.INFO,
        enable_streamlit=True
    )

    try:
        # Tải các biến môi trường từ file .env
        load_dotenv()

        # Tạo đối tượng ViewModel chính của ứng dụng
        app_viewmodel = AppViewModel()

        # Khởi tạo ứng dụng và chuẩn bị dữ liệu cần thiết
        app_viewmodel.initialize_app()

        # Tạo đối tượng View chính, truyền ViewModel vào View
        app_view = AppView(app_viewmodel)

        # Hiển thị giao diện người dùng với Streamlit
        app_view.render()

    except Exception as e:
        logger.error(f"Lỗi khởi chạy ứng dụng: {str(e)}", exc_info=True)
        raise

# MARK: - Main

if __name__ == "__main__":
    try:
        # Khởi chạy ứng dụng
        main()

    except Exception as e:
        logger = get_logger("main")
        logger.critical(f"Lỗi không thể khắt phục khi chạy ứng dụng: {str(e)}", exc_info=True)

        try:
            import streamlit as st
            st.error(f"***Lỗi nghiêm trọng:*** {str(e)}")
            st.error("Vui lòng kiểm tra file log để biết thêm chi tiết.")

        except:
            pass
