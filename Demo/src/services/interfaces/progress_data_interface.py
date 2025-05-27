#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface cho dịch vụ xử lý dữ liệu
"""

from abc import abstractmethod
from typing import Dict, Any, Optional
import pandas as pd

from pyspark.sql import SparkSession, DataFrame
from .base_interface import IBaseService

class IProgressDataService(IBaseService):
    """
    Interface định nghĩa các thuộc tính và phương thức cho dịch vụ xử lý dữ liệu
    """

    @property
    @abstractmethod
    def data_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin về dữ liệu đã được tải
        """
        pass

    @abstractmethod
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Tải dữ liệu từ file CSV
        """
        pass

    @abstractmethod
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Làm sạch và tiền xử lý dữ liệu
        """
        pass

    @abstractmethod
    def get_spark_session_cached(self) -> Optional[SparkSession]:
        """
        Lấy SparkSession đã được cache
        """
        pass

    @abstractmethod
    def get_spark_dataframe(self, pandas_df: pd.DataFrame) -> Optional[DataFrame]:
        """
        Chuyển đổi pandas DataFrame sang Spark DataFrame
        """
        pass

    @abstractmethod
    def get_area_comparison(self, location: str) -> Dict[str, Any]:
        """
        Lấy dữ liệu so sánh cho một khu vực
        """
        pass
