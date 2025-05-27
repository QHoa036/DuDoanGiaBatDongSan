#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface cho dịch vụ quản lý metrics của mô hình
"""

from abc import abstractmethod
from typing import Dict, Any, List, Optional

from .base_interface import IBaseService

class IMetricsService(IBaseService):
    """
    Interface định nghĩa các thuộc tính và phương thức cho dịch vụ quản lý metrics
    """

    @property
    @abstractmethod
    def all_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Lấy tất cả các metrics đã lưu trữ
        """
        pass

    @abstractmethod
    def add_metrics(self, model_name: str, metrics: Dict[str, float]) -> bool:
        """
        Thêm metrics mới cho một mô hình
        """
        pass

    @abstractmethod
    def get_metrics(self, model_name: str) -> Optional[Dict[str, float]]:
        """
        Lấy metrics của một mô hình cụ thể
        """
        pass

    @abstractmethod
    def update_metric(self, model_name: str, metric_name: str, value: float) -> bool:
        """
        Cập nhật giá trị của một metric cụ thể
        """
        pass

    @abstractmethod
    def get_metric_history(self, model_name: str, metric_name: str) -> List[float]:
        """
        Lấy lịch sử giá trị của một metric
        """
        pass

    @abstractmethod
    def get_best_model(self, metric_name: str, higher_is_better: bool = True) -> str:
        """
        Lấy tên của mô hình tốt nhất dựa trên một metric
        """
        pass
