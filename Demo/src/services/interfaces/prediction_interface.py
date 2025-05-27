#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface cho dịch vụ dự đoán giá bất động sản
"""

from abc import abstractmethod
from typing import Dict

from ...models.property_model import Property, PredictionResult
from .base_interface import IBaseService

class IPredictionService(IBaseService):
    """
    Interface định nghĩa các thuộc tính và phương thức cho dịch vụ dự đoán
    """

    @property
    @abstractmethod
    def model_metrics(self) -> Dict[str, float]:
        """
        Lấy các chỉ số của mô hình dự đoán
        """
        pass

    @abstractmethod
    def predict_property_price(self, property_data: Property) -> PredictionResult:
        """
        Dự đoán giá của một bất động sản sử dụng mô hình đã được huấn luyện
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Lấy mức độ quan trọng của các đặc trưng trong mô hình đã được huấn luyện
        """
        pass
