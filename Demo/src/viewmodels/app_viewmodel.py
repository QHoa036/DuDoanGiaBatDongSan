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
import sys
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional

from .prediction_viewmodel import PredictionViewModel
from .analytics_viewmodel import AnalyticsViewModel

from ..utils.ngrok_utils import check_ngrok_status, run_ngrok
from ..utils.logger_utils import get_logger, log_execution_time
from ..utils.session_utils import initialize_session
from ..services.core.services_factory import ServicesFactory
from ..services.interfaces import IProgressDataService, ITrainModelService, IPredictionService

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

        # Khởi tạo các dịch vụ sử dụng Factory Pattern
        self._data_service = ServicesFactory.get_progress_data_service()
        self._model_service = ServicesFactory.get_train_model_service()
        self._prediction_service = ServicesFactory.get_prediction_service()

        # Khởi tạo các ViewModel con
        self._prediction_vm = PredictionViewModel(_data_service=self._data_service, _model_service=self._model_service, prediction_service=self._prediction_service)
        self._analytics_vm = AnalyticsViewModel(_data_service=self._data_service, _model_service=self._model_service)

        # Khởi tạo trạng thái ứng dụng
        if 'app_mode' not in st.session_state:
            st.session_state.app_mode = "Dự đoán"

        # Khởi tạo các biến session state cho metrics
        initialize_session()

        # Quản lý các trang
        self._app_modes = ["Dự đoán", "Trực quan hóa", "Về dự án"]

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
    def data_service(self) -> IProgressDataService:
        """
        Dịch vụ dữ liệu
        """
        return self._data_service

    @property
    def model_service(self) -> ITrainModelService:
        """
        Dịch vụ huấn luyện mô hình
        """
        return self._model_service

    @property
    def prediction_service(self) -> IPredictionService:
        """
        Dịch vụ dự đoán giá bất động sản
        """
        return self._prediction_service

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
            # Tải dữ liệu sử dụng dịch vụ dữ liệu
            self.logger.info("Bắt đầu tải dữ liệu...")
            data = self._data_service.load_data()

            # Tiền xử lý dữ liệu
            self.logger.info("Tiền xử lý dữ liệu...")
            preprocessed_data = self._data_service.preprocess_data(data)

            # Kiểm tra dữ liệu đã tải
            data_info = self._data_service.data_info
            self.logger.info(f"Tải dữ liệu hoàn tất: {data_info.get('rows', 0)} dòng, {len(data_info.get('columns', []))} cột")

            # Chuyển đổi sang Spark DataFrame
            self.logger.info("Chuyển đổi dữ liệu sang Spark DataFrame...")
            spark_df = self._data_service.convert_to_spark(preprocessed_data)

            # Huấn luyện mô hình
            self.logger.info("Bắt đầu huấn luyện mô hình...")

            # Huấn luyện mô hình với Spark hoặc fallback tùy theo tình hình
            self._model_service.train_model(spark_df if spark_df is not None else None)

            # Factory sẽ tự động cập nhật mô hình cho dịch vụ dự đoán khi tạo lại instance
            # Cập nhật dịch vụ dự đoán với instance mới nếu cần
            self._prediction_service = ServicesFactory.get_prediction_service()

            # Hiển thị thông tin metrics
            metrics = self._model_service.model_metrics
            if metrics:
                for name, value in metrics.items():
                    if isinstance(value, (float, int)):
                        self.logger.info(f"Chỉ số {name}: {value:.4f}")
                    else:
                        self.logger.info(f"Chỉ số {name}: {value}")

            self.logger.info("Khởi tạo ứng dụng hoàn tất")

        except Exception as e:
            self.logger.error(f"Lỗi trong quá trình khởi tạo ứng dụng: {str(e)}", exc_info=True)
            raise

    def get_model_metrics(self) -> Dict[str, float]:
        """
        Lấy các chỉ số của mô hình hiện tại
        """
        return self._model_service.model_metrics

    @log_execution_time
    def process_raw_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Xử lý dữ liệu thô từ file CSV
        """
        try:
            self.logger.info(f"Bắt đầu xử lý dữ liệu thô từ: {file_path}")

            # Gọi phương thức process_raw_data của ProgressDataService
            processed_data = self._data_service.process_raw_data(file_path)

            if processed_data is not None:
                self.logger.info(f"Xử lý dữ liệu thô thành công: {len(processed_data)} dòng")

                # Sau khi xử lý xong, khởi tạo lại ứng dụng để sử dụng dữ liệu mới
                self.initialize_app()

                return processed_data
            else:
                self.logger.error("Xử lý dữ liệu thô thất bại")
                return None

        except Exception as e:
            self.logger.error(f"Lỗi khi xử lý dữ liệu thô: {str(e)}", exc_info=True)
            return None

    # MARK: - Kết nối Ngrok

    @log_execution_time
    def setup_ngrok(self) -> Optional[str]:
        """
        Thiết lập Ngrok để tạo URL công khai cho ứng dụng
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
