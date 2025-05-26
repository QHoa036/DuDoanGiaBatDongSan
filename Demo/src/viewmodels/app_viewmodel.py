#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
App ViewModel - Điều phối chính cho ứng dụng

Lớp ViewModel chính điều phối các hoạt động của ứng dụng, bao gồm:
- Quản lý điều hướng giữa các màn hình
- Khởi tạo các dịch vụ và mô hình dữ liệu
- Điều phối giữa các module khác nhau
- Quản lý trạng thái chung của ứng dụng
- Thiết lập kết nối Ngrok (nếu có)
"""

# MARK: - Thư viện

import os
import streamlit as st
from typing import Dict, List, Optional

from .prediction_viewmodel import PredictionViewModel
from .analytics_viewmodel import AnalyticsViewModel

from ..utils.ngrok_utils import check_ngrok_status, run_ngrok
from ..utils.logger_utils import get_logger, log_execution_time
from ..utils.session_utils import initialize_session
from ..services.data_service import DataService

# MARK: - Lớp chính

class AppViewModel:
    """
    ViewModel chính điều phối ứng dụng
    Quản lý điều hướng, khởi tạo và điều phối các module
    """
    def __init__(self):
        """
        Cài đặt logger và dịch vụ
        """
        # Logger
        self.logger = get_logger(__name__)

        # Khởi tạo dịch vụ dữ liệu
        self._data_service = DataService()

        # Khởi tạo các ViewModel con
        self._prediction_vm = PredictionViewModel(self._data_service)
        self._analytics_vm = AnalyticsViewModel(self._data_service)

        # Khởi tạo trạng thái ứng dụng
        if 'app_mode' not in st.session_state:
            st.session_state.app_mode = "Dự đoán giá"

        # Khởi tạo các biến session state cho metrics
        initialize_session()

        # Quản lý các trang
        self._app_modes = ["Dự đoán giá", "Trực quan hóa", "Về dự án"]

    # MARK: - Thuộc tính

    @property
    def prediction_viewmodel(self) -> PredictionViewModel:
        """
        Get the prediction view model
        """
        return self._prediction_vm

    @property
    def analytics_viewmodel(self) -> AnalyticsViewModel:
        """
        ViewModel phân tích
        """
        return self._analytics_vm

    @property
    def data_service(self) -> DataService:
        """
        Dịch vụ dữ liệu
        """
        return self._data_service

    @property
    def app_modes(self) -> List[str]:
        """
        Các chế độ ứng dụng có sẵn
        """
        return self._app_modes

    @property
    def current_mode(self) -> str:
        """
        Lấy chế độ ứng dụng hiện tại
        """
        return st.session_state.app_mode

    # MARK: - Điều hướng

    def set_app_mode(self, mode: str) -> None:
        """
        Thiết lập chế độ ứng dụng hiện tại
        """
        st.session_state.app_mode = mode

    # MARK: - Khởi tạo

    @log_execution_time
    def initialize_app(self) -> None:
        """
        Khởi tạo ứng dụng bằng cách tải dữ liệu và chuẩn bị các mô hình
        """
        try:
            # Tải dữ liệu
            data = self._data_service.load_data()

            # Tiền xử lý dữ liệu
            preprocessed_data = self._data_service.preprocess_data(data)

            # Chuyển đổi sang Spark DataFrame
            spark_df = self._data_service.convert_to_spark(preprocessed_data)

            # Huấn luyện mô hình
            if spark_df is not None:
                # Nếu Spark khả dụng, sử dụng Spark DataFrame
                self._data_service.train_model(spark_df)
                self.logger.info("Đã huấn luyện mô hình với Spark")
            else:
                # Fallback: Sử dụng scikit-learn nếu Spark không khả dụng
                self.logger.warning("Không thể khởi tạo Spark DataFrame, sử dụng scikit-learn fallback")
                self._data_service.train_model(None)  # Truyền None sẽ kích hoạt fallback

            # Hiển thị thông tin metrics
            metrics = self._data_service.model_metrics
            if metrics:
                for name, value in metrics.items():
                    self.logger.info(f"Chỉ số {name}: {value:.4f}")

            self.logger.info("Khởi tạo ứng dụng hoàn tất")

        except Exception as e:
            self.logger.error(f"Lỗi trong quá trình khởi tạo ứng dụng: {str(e)}", exc_info=True)
            raise

    def get_model_metrics(self) -> Dict[str, float]:
        """
        Lấy các chỉ số của mô hình hiện tại

        Trả về:
            Dict[str, float]: Các chỉ số mô hình
        """
        return self._data_service.model_metrics

    # MARK: - Kết nối Ngrok

    @log_execution_time
    def setup_ngrok(self) -> Optional[str]:
        """
        Thiết lập Ngrok để tạo URL công khai cho ứng dụng

        Trả về:
            Optional[str]: URL công khai nếu thành công, None nếu thất bại
        """
        # Lấy path thư mục utils
        utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils')

        # Add utils to path if needed
        if utils_path not in sys.path:
            sys.path.append(utils_path)

        try:
            # Set up Ngrok
            public_url = run_ngrok()

            if public_url:
                return public_url
            else:
                return None

        except ImportError:
            self.logger.error("Lỗi khi import module")
            return None
        except Exception:
            self.logger.error(f"Lỗi khi thiết lập Ngrok", exc_info=True)
            return None

    @log_execution_time
    def check_ngrok_status(self) -> List[Dict[str, str]]:
        """
        Kiểm tra trạng thái kết nối Ngrok

        Returns:
            List[Dict[str, str]]: Danh sách các tunnel đang hoạt động
        """
        # Lấy path thư mục utils
        utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils')

        # Add utils to path if needed
        if utils_path not in sys.path:
            sys.path.append(utils_path)

        try:
            # Check Ngrok status
            tunnels = check_ngrok_status()

            if tunnels:
                for i, tunnel in enumerate(tunnels):
                    self.logger.debug(f"Tunnel {i+1}: {tunnel.get('public_url', 'Unknown')}")
            else:
                self.logger.warning("Không tìm thấy tunnel nào đang hoạt động")

            return tunnels

        except ImportError:
            self.logger.error(f"Lỗi import module")
            return []
        except Exception:
            self.logger.error(f"Lỗi khi kiểm tra trạng thái Ngrok", exc_info=True)
            return []
