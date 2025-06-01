#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vietnam Real Estate Price Prediction
"""

# MARK: - Thư viện

import streamlit as st
from typing import Dict, Any
import os
import sys
import time
import traceback

# Import các module của ứng dụng
from src.config.app_config import AppConfig
from src.utils.logger_util import get_logger
from src.utils.spark_utils import SparkUtils

# Import services
from src.services.data_service import DataService
from src.services.model_service import ModelService

# Import viewmodels
from src.viewmodels.app_viewmodel import AppViewModel
from src.viewmodels.prediction_viewmodel import PredictionViewModel
from src.viewmodels.analytics_viewmodel import AnalyticsViewModel

# Import views
from src.views.shared.sidebar import render_sidebar
from src.views.prediction_view import render_prediction_view
from src.views.analytics_view import render_analytics_view
from src.views.about_view import render_about_view

# MARK: - Cấu hình

# Khởi tạo logger
logger = get_logger(__name__)

# Thiết lập biến môi trường cho PySpark
os.environ['SPARK_HOME'] = AppConfig.SPARK_HOME # Đường dẫn đến thư mục PySpark

# Cập nhật PYTHONPATH để tìm thấy PySpark
pyspark_python = os.path.join(os.environ['SPARK_HOME'], 'python')
py4j = os.path.join(pyspark_python, 'lib', 'py4j-0.10.9-src.zip')
if pyspark_python not in sys.path:
    sys.path.insert(0, pyspark_python)
    sys.path.insert(0, py4j)

# Thêm đường dẫn gốc vào sys.path
app_path = os.path.dirname(os.path.abspath(__file__))
if app_path not in sys.path:
    sys.path.append(app_path)

# Cấu hình trang Streamlit
st.set_page_config(
    page_title=AppConfig.APP_TITLE,
    page_icon=AppConfig.APP_ICON,
    layout=AppConfig.APP_LAYOUT,
    initial_sidebar_state=AppConfig.APP_INITIAL_SIDEBAR_STATE
)

# MARK: - Hàm chính

def main():
    """Hàm chính khởi chạy ứng dụng"""
    startup_time = time.time()

    # Cấu hình logging cho Spark
    SparkUtils.configure_spark_logging()
    logger.info("Khởi động ứng dụng dự đoán giá bất động sản")

    # Khởi tạo session state nếu chưa có
    initialize_session_state()

    # Tải CSS tùy chỉnh
    load_custom_css()

    # Khởi tạo services
    services = initialize_services()

    # Khởi tạo viewmodels
    viewmodels = initialize_viewmodels(services)

    # Khởi tạo app_viewmodel
    try:
        app_viewmodel = AppViewModel(
            _data_service=services["data_service"],
            _model_service=services["model_service"]
        )

    except Exception as e:
        logger.error(f"Lỗi khởi tạo AppViewModel: {str(e)}")
        st.error(f"Lỗi khởi tạo ViewModel chính: {str(e)}")
        return

    # Đảm bảo dữ liệu được tải và mô hình được huấn luyện
    logger.info("Đảm bảo dữ liệu được tải và mô hình được huấn luyện")
    if app_viewmodel.load_data():
        logger.info("Dữ liệu đã được tải thành công, huấn luyện mô hình nếu cần")
        model_trained = app_viewmodel.train_model_if_needed()
        logger.info(f"Trạng thái huấn luyện mô hình: {model_trained}")
        logger.info(f"Các chỉ số hiện tại trong session state sau khi huấn luyện: R2={st.session_state.get('model_r2_score', 0.0)}, RMSE={st.session_state.get('model_rmse', 0.0)}")

    # Hiển thị sidebar và lấy chế độ ứng dụng hiện tại
    app_modes = AppConfig.APP_MODES
    render_sidebar(app_modes, app_viewmodel.handle_mode_change)
    current_mode = st.session_state.get("app_mode", AppConfig.DEFAULT_APP_MODE)

    # Dựa vào chế độ để hiển thị view tương ứng
    try:
        if current_mode == AppConfig.APP_MODES[0]:
            render_prediction_view(viewmodels["prediction_viewmodel"])

        elif current_mode == AppConfig.APP_MODES[1]:
            render_analytics_view(viewmodels["analytics_viewmodel"])

        elif current_mode == AppConfig.APP_MODES[2]:
            render_about_view()

        else:
            st.error(f"Không tìm thấy chế độ {current_mode}")

    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi hiển thị giao diện: {str(e)}")
        if st.checkbox("Hiện chi tiết lỗi"):
            st.code(traceback.format_exc())

    # Log thời gian khởi động
    end_time = time.time()
    logger.info(f"Khởi động ứng dụng hoàn tất trong {end_time - startup_time:.2f} giây")

# MARK: - Khởi tạo

def initialize_session_state():
    """Khởi tạo session state cho ứng dụng"""
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = AppConfig.DEFAULT_APP_MODE

    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False

    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False

    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    if "using_spark" not in st.session_state:
        st.session_state.using_spark = False

    if "using_fallback" not in st.session_state:
        st.session_state.using_fallback = False

    logger.info("Đã khởi tạo Session State")

# MARK: - Tải tài nguyên

def load_custom_css():
    """Tải CSS tùy chỉnh để định dạng giao diện ứng dụng"""
    # Tải CSS từ file nếu có
    css_path = AppConfig.get_css_path()
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.error("Không tìm thấy file CSS. Vui lòng kiểm tra lại.")

    logger.info("Đã tải CSS tùy chỉnh")


# MARK: - Dịch vụ

def initialize_services() -> Dict[str, Any]:
    """Khởi tạo các services của ứng dụng"""
    services = {}

    # Khởi tạo DataService
    try:
        data_service = DataService(
            data_path=AppConfig.get_data_path(),
            cache_dir=os.path.join(AppConfig.get_base_dir(), 'cache')
        )
        services["data_service"] = data_service

    except Exception as e:
        logger.error(f"Lỗi khởi tạo DataService: {str(e)}")
        st.error(f"Có lỗi xảy ra!")

    # Khởi tạo ModelService
    try:
        # Kiểm tra xem DataService đã được khởi tạo thành công chưa
        if "data_service" not in services:
            raise ValueError("DataService chưa được khởi tạo thành công")

        # Kiểm tra xem Spark có khả dụng không
        spark_available = SparkUtils.is_spark_available()
        st.session_state.using_spark = spark_available

        model_service = ModelService(
            _data_service=services["data_service"],
            model_dir=os.path.join(AppConfig.get_base_dir(), 'src', 'models'),
            using_spark=spark_available
        )
        services["model_service"] = model_service

    except Exception as e:
        logger.error(f"Lỗi khởi tạo ModelService: {str(e)}")
        st.error(f"Có lỗi xảy ra!")

    return services


# MARK: - View models

def initialize_viewmodels(services: Dict[str, Any]) -> Dict[str, Any]:
    """Khởi tạo các viewmodels của ứng dụng"""
    viewmodels = {}

    # Kiểm tra xem các dịch vụ cần thiết đã được khởi tạo chưa
    required_services = ["data_service", "model_service"]
    for service_name in required_services:
        if service_name not in services:
            logger.error(f"Dịch vụ {service_name} chưa được khởi tạo")
            st.error(f"Thiếu dịch vụ {service_name}. Không thể khởi tạo các viewmodel.")
            return viewmodels

    # Khởi tạo PredictionViewModel
    try:
        prediction_vm = PredictionViewModel(
            _data_service=services["data_service"],
            _model_service=services["model_service"]
        )
        viewmodels["prediction_viewmodel"] = prediction_vm

    except Exception as e:
        logger.error(f"Có lỗi xảy ra!")

    # Khởi tạo AnalyticsViewModel
    try:
        analytics_vm = AnalyticsViewModel(
            _data_service=services["data_service"]
        )
        viewmodels["analytics_viewmodel"] = analytics_vm

    except Exception as e:
        logger.error(f"Có lỗi xảy ra!")

    return viewmodels


# MARK: - Chạy ứng dụng

if __name__ == "__main__":
    main()
