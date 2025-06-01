# MARK: - Thư viện

import os
import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any, List, Union
import numpy as np

from src.config.app_config import AppConfig
from src.utils.logger_util import get_logger

# Khởi tạo logger
logger = get_logger()

# MARK: - Dịch vụ dữ liệu

class DataService:
    """
    Service xử lý dữ liệu bất động sản
    """

    # MARK: - Khởi tạo

    def __init__(self, data_path=None, cache_dir=None):
        """
        Khởi tạo DataService

        Args:
            data_path (str, optional): Đường dẫn đến file dữ liệu
            cache_dir (str, optional): Thư mục lưu cache
        """
        self.data_path = data_path if data_path else AppConfig.get_data_path()
        self.cache_dir = cache_dir
        logger.info(f"DataService khởi tạo với data_path: {self.data_path}")

    # MARK: - Tải dữ liệu

    @st.cache_data
    def load_data(_self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Đọc dữ liệu từ file CSV

        Args:
            file_path (str, optional): Đường dẫn đến file dữ liệu. Nếu None, sử dụng đường dẫn mặc định.

        Returns:
            pd.DataFrame: DataFrame chứa dữ liệu bất động sản
        """
        try:
            # Xác định đường dẫn tuyệt đối đến file dữ liệu
            if file_path is None:
                file_path = _self.data_path

            # Kiểm tra xem file có tồn tại không
            if not os.path.exists(file_path):
                logger.error(f"Không tìm thấy file dữ liệu: {file_path}")
                return pd.DataFrame()

            # Đọc dữ liệu từ file CSV
            data = pd.read_csv(file_path)
            logger.info(f"Đã đọc thành công {len(data)} dòng dữ liệu từ {file_path}")

            return data
        except Exception as e:
            logger.error(f"Lỗi khi đọc dữ liệu: {e}")
            return pd.DataFrame()

    # MARK: - Tiền xử lý dữ liệu

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Tiền xử lý dữ liệu cho phân tích và mô hình hóa

        Args:
            data (pd.DataFrame): DataFrame gốc chứa dữ liệu bất động sản

        Returns:
            pd.DataFrame: DataFrame đã được tiền xử lý
        """
        try:
            # Tạo bản sao để tránh thay đổi dữ liệu gốc
            df = data.copy()

            # Xử lý dữ liệu thiếu cho các cột số
            numeric_columns = AppConfig.NUMERIC_COLUMNS
            for col in numeric_columns:
                if col in df.columns:
                    # Xác định giá trị mặc định cho mỗi loại cột
                    default_val = -1 if col in ["bedroom_num", "floor_num", "toilet_num", "livingroom_num"] else 0

                    # Áp dụng hàm xử lý thiếu
                    df = self._handle_missing_numeric(df, [col], default_val)

            # Chuyển đổi kiểu dữ liệu
            for col in df.columns:
                if col in ["bedroom_num", "floor_num", "toilet_num", "livingroom_num", "built_year"]:
                    df[col] = df[col].astype(np.int32, errors='ignore')
                elif any(num_col in col for num_col in numeric_columns):
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    df[col] = df[col].astype(np.float64, errors='ignore')

            logger.info(f"Đã tiền xử lý dữ liệu thành công: {len(df)} dòng")
            return df

        except Exception as e:
            logger.error(f"Lỗi khi tiền xử lý dữ liệu: {e}")
            return data

    # MARK: - Phương thức hỗ trợ

    def _handle_missing_numeric(self, df: pd.DataFrame, columns: List[str], default_val: Union[float, int] = 0) -> pd.DataFrame:
        """
        Xử lý dữ liệu thiếu cho các cột số

        Args:
            df (pd.DataFrame): DataFrame gốc
            columns (List[str]): Danh sách các cột cần xử lý
            default_val (Union[float, int], optional): Giá trị mặc định cho dữ liệu thiếu.
                                                        Mặc định là 0

        Returns:
            pd.DataFrame: DataFrame đã được xử lý
        """
        result_df = df.copy()
        for col in columns:
            if col in result_df.columns:
                # Thêm cột cờ để đánh dấu dữ liệu thiếu
                missing_flag_col = f"{col}_missing"
                result_df[missing_flag_col] = result_df[col].isna() | (result_df[col] == '') | (result_df[col] == 'NaN')

                # Điền giá trị thiếu bằng giá trị mặc định
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(default_val)

        return result_df

    # MARK: - Lọc dữ liệu

    def filter_data(self, data: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Lọc dữ liệu theo các điều kiện

        Args:
            data (pd.DataFrame): DataFrame gốc
            filters (Dict[str, Any]): Từ điển chứa các điều kiện lọc

        Returns:
            pd.DataFrame: DataFrame đã được lọc
        """
        try:
            # Tạo bản sao để tránh thay đổi dữ liệu gốc
            filtered_data = data.copy()

            # Áp dụng các bộ lọc
            for column, value in filters.items():
                if column in filtered_data.columns and value:
                    if isinstance(value, (list, tuple)):
                        # Lọc theo danh sách giá trị
                        filtered_data = filtered_data[filtered_data[column].isin(value)]
                    else:
                        # Lọc theo giá trị đơn
                        filtered_data = filtered_data[filtered_data[column] == value]

            return filtered_data
        except Exception as e:
            logger.error(f"Lỗi khi lọc dữ liệu: {e}")
            return data

    # MARK: - Phân tích dữ liệu

    def get_unique_values(self, data: pd.DataFrame, column: str) -> List[Any]:
        """
        Lấy danh sách các giá trị duy nhất của một cột

        Args:
            data (pd.DataFrame): DataFrame gốc
            column (str): Tên cột cần lấy giá trị duy nhất

        Returns:
            List[Any]: Danh sách các giá trị duy nhất
        """
        if column in data.columns:
            return sorted(data[column].unique().tolist())
        return []

    # MARK: - Thống kê

    def get_stats(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Tính toán các chỉ số thống kê từ dữ liệu

        Args:
            data (pd.DataFrame): DataFrame gốc

        Returns:
            Dict[str, float]: Từ điển chứa các chỉ số thống kê
        """
        stats = {}

        # Đảm bảo DataFrame không rỗng
        if data.empty:
            return stats

        try:
            # Giá trung bình của bất động sản
            stats['avg_price'] = data['price'].mean()

            # Giá trung bình trên m²
            stats['avg_price_per_m2'] = data['price_per_m2'].mean()

            # Tổng số bất động sản
            stats['total_properties'] = len(data)

            # Thêm các thống kê khác nếu cần
            stats['min_price'] = data['price'].min()
            stats['max_price'] = data['price'].max()
            stats['median_price'] = data['price'].median()

            return stats
        except Exception as e:
            logger.error(f"Lỗi khi tính toán chỉ số thống kê: {e}")
            return stats
