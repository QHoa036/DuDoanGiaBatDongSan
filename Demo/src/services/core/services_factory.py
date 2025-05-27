#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Factory cho các dịch vụ - Quản lý việc tạo và cung cấp các instance dịch vụ
"""

from ..interfaces import IMetricsService, IProgressDataService, ITrainModelService, IPredictionService
from ..prediction_service import PredictionService
from ..train_model_service import TrainModelService
from ..metrics_service import MetricsService
from ..progress_data_service import ProgressDataService

class ServicesFactory:
    """
    Factory quản lý việc khởi tạo và cung cấp các instance dịch vụ
    Áp dụng Singleton pattern và Dependency Injection
    """

    # Các instance singleton được cache ở cấp lớp
    _metrics_service = None
    _progress_data_service = None
    _train_model_service = None
    _prediction_service = None

    @staticmethod
    def get_metrics_service() -> IMetricsService:
        """
        Trả về instance của MetricsService
        Cached qua các phiên làm việc
        """
        if ServicesFactory._metrics_service is None:
            ServicesFactory._metrics_service = MetricsService()
        return ServicesFactory._metrics_service

    @staticmethod
    def get_progress_data_service() -> IProgressDataService:
        """
        Trả về instance của ProgressDataService
        Cached qua các phiên làm việc
        """
        if ServicesFactory._progress_data_service is None:
            metrics_service = ServicesFactory.get_metrics_service()
            ServicesFactory._progress_data_service = ProgressDataService(metrics_service=metrics_service)
        return ServicesFactory._progress_data_service

    @staticmethod
    def get_train_model_service() -> ITrainModelService:
        """
        Trả về instance của TrainModelService
        """
        if ServicesFactory._train_model_service is None:
            data_service = ServicesFactory.get_progress_data_service()
            metrics_service = ServicesFactory.get_metrics_service()
            ServicesFactory._train_model_service = TrainModelService(
                _data_service=data_service,
                _metrics_service=metrics_service
            )
        return ServicesFactory._train_model_service

    @staticmethod
    def get_prediction_service() -> IPredictionService:
        """
        Trả về instance của PredictionService
        """
        if ServicesFactory._prediction_service is None:
            # Sử dụng các service đã có
            data_service = ServicesFactory.get_progress_data_service()
            model_service = ServicesFactory.get_train_model_service()
            metrics_service = ServicesFactory.get_metrics_service()

            # Lấy mô hình từ dịch vụ huấn luyện (truy cập trực tiếp thuộc tính)
            model = getattr(model_service, '_model', None)
            fallback_model = getattr(model_service, '_fallback_model', None)

            # Khởi tạo dịch vụ dự đoán
            ServicesFactory._prediction_service = PredictionService(
                data_service=data_service,
                model=model,
                fallback_model=fallback_model,
                metrics_service=metrics_service
            )
        return ServicesFactory._prediction_service
