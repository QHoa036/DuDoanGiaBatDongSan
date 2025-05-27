#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface cơ sở cho tất cả các dịch vụ
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class IBaseService(ABC):
    """
    Interface cơ sở xác định các phương thức chung cho tất cả các dịch vụ
    """

    @property
    @abstractmethod
    def service_name(self) -> str:
        """Tên dịch vụ"""
        pass

    @abstractmethod
    def save_state(self, state_name: str, state_data: Any) -> bool:
        """
        Lưu trạng thái dịch vụ
        """
        pass

    @abstractmethod
    def load_state(self, state_name: str, default: Any = None) -> Any:
        """
        Tải trạng thái dịch vụ
        """
        pass

    @abstractmethod
    def save_model(self, model: Any, model_name: str) -> str:
        """
        Lưu mô hình
        """
        pass

    @abstractmethod
    def load_model(self, model_name: str, specific_timestamp: Optional[str] = None) -> Any:
        """
        Tải mô hình
        """
        pass

    @abstractmethod
    def get_service_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin về dịch vụ
        """
        pass
