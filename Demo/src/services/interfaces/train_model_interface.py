#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface cho dịch vụ huấn luyện mô hình
"""

from abc import abstractmethod
from typing import Dict, Any, Tuple

from .base_interface import IBaseService

class ITrainModelService(IBaseService):
    """
    Interface định nghĩa các thuộc tính và phương thức cho dịch vụ huấn luyện mô hình
    """

    @property
    @abstractmethod
    def model_metrics(self) -> Dict[str, float]:
        """
        Lấy các chỉ số của mô hình hiện tại
        """
        pass

    @abstractmethod
    def train_model(self, spark_df=None, force_retrain=False) -> Tuple[Any, float, float]:
        """
        Huấn luyện mô hình học máy để dự đoán giá bất động sản
        """
        pass
