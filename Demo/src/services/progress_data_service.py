#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dịch vụ Xử lý Dữ liệu - Chịu trách nhiệm tải, xử lý và chuẩn bị dữ liệu cho huấn luyện mô hình
"""

# MARK: - Thư viện

import os
import sys
import time
import json
import glob
import hashlib
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Dict, Any

from pyspark.sql import SparkSession, DataFrame
from datetime import datetime, timedelta

from ..utils.spark_utils import get_spark_session, configure_spark_logging
from ..utils.logger_utils import get_logger, MetricsLogger
from .interfaces.progress_data_interface import IProgressDataService
from .interfaces.metrics_interface import IMetricsService
from .core.base_service import BaseService

# MARK: - Cấu hình

utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils')
if utils_path not in sys.path:
    sys.path.append(utils_path)

# MARK: - Lớp xử lý

class ProgressDataService(BaseService, IProgressDataService):
    """
    Lớp dịch vụ chịu trách nhiệm tải, tiền xử lý và làm sạch dữ liệu
    Triển khai interface IProgressDataService
    Kế thừa từ BaseService để sử dụng các tính năng chung
    """

    # MARK: - Khởi tạo

    def __init__(self, data_dir: str = None, metrics_service: IMetricsService = None):
        """
        Khởi tạo dịch vụ xử lý dữ liệu
        """
        # Gọi khởi tạo lớp cơ sở
        super().__init__(service_name="ProgressDataService")

        # Khởi tạo logger tiêu chuẩn
        self._logger = get_logger("progress_data_service")
        self._logger.info("Khởi tạo ProgressDataService")

        # Cấu hình đường dẫn
        self._data_dir = data_dir if data_dir else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
        os.makedirs(self._data_dir, exist_ok=True)
        self._cache_dir = os.path.join(self._data_dir, 'cache')
        os.makedirs(self._cache_dir, exist_ok=True)

        # Khởi tạo metrics logger
        if metrics_service is None:
            # Import động để tránh circular import
            from .core.services_factory import ServicesFactory
            self._metrics_service = ServicesFactory.get_metrics_service()
        else:
            self._metrics_service = metrics_service

        self._metrics_logger = MetricsLogger("data_processing", self._metrics_service)

        # Cấu hình logging cho Spark
        configure_spark_logging()

        # Khởi tạo phiên Spark (lazy loading)
        self._spark = None
        self._data = None
        self._spark_df = None
        self._cache_info = {}
        self._last_cache_check = datetime.now()

        # Định nghĩa tên các cột đặc trưng
        self._feature_columns = {
            'area': 'area (m2)',
            'street': 'street (m)',
            'bedrooms': 'bedroom_num',
            'toilets': 'toilet_num',
            'floors': 'floor_num'
        }

        # Kiểm tra và tải dữ liệu đã lưu nếu có
        self._try_load_cached_data()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Làm sạch và tiền xử lý dữ liệu
        """
        self._logger.info("Tiến hành làm sạch dữ liệu...")
        start_time = time.time()

        try:
            # Tạo bản sao để tránh cảnh báo SettingWithCopyWarning
            cleaned_df = df.copy()

            # Loại bỏ các cột không cần thiết (nếu có)
            # columns_to_drop = ['column1', 'column2']
            # if all(col in cleaned_df.columns for col in columns_to_drop):
            #    cleaned_df = cleaned_df.drop(columns=columns_to_drop)

            # Xử lý giá trị thiếu
            numeric_columns = cleaned_df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_columns:
                # Thay thế giá trị thiếu bằng giá trị trung bình của cột
                if cleaned_df[col].isnull().sum() > 0:
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())

            # Xử lý các cột không phải số
            categorical_columns = cleaned_df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                # Thay thế giá trị thiếu bằng giá trị phổ biến nhất
                if cleaned_df[col].isnull().sum() > 0:
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])

            # Loại bỏ các outlier nếu cần
            # Ví dụ: loại bỏ các hàng có giá trị nằm ngoài khoảng 3 độ lệch chuẩn
            for col in numeric_columns:
                mean_val = cleaned_df[col].mean()
                std_val = cleaned_df[col].std()
                cleaned_df = cleaned_df[(cleaned_df[col] <= mean_val + 3*std_val) &
                                        (cleaned_df[col] >= mean_val - 3*std_val)]

            # Ghi log thời gian xử lý
            duration_ms = (time.time() - start_time) * 1000
            self._logger.info("Hoàn thành làm sạch dữ liệu (mất %0.2f ms)", duration_ms)

            # Ghi metrics nếu có
            if hasattr(self, '_metrics_logger'):
                self._metrics_logger.log_performance("clean_data", duration_ms)

            return cleaned_df

        except Exception as e:
            self._logger.error("Lỗi trong quá trình làm sạch dữ liệu: %s", str(e))
            # Trả về DataFrame gốc nếu có lỗi
            return df

    def get_spark_dataframe(self, pandas_df: pd.DataFrame) -> Optional[DataFrame]:
        """
        Chuyển đổi pandas DataFrame sang Spark DataFrame
        """
        self._logger.info("Chuyển đổi sang Spark DataFrame...")
        start_time = time.time()

        try:
            # Lấy hoặc tạo phiên Spark
            spark = self.get_spark_session_cached()
            if spark is None:
                self._logger.warning("Không thể tạo phiên Spark")
                return None

            # Chuyển đổi pandas DataFrame sang Spark DataFrame
            spark_df = spark.createDataFrame(pandas_df)

            # Ghi log thời gian xử lý
            duration_ms = (time.time() - start_time) * 1000
            self._logger.info("Hoàn thành chuyển đổi sang Spark DataFrame (mất %0.2f ms)", duration_ms)

            # Ghi metrics nếu có
            if hasattr(self, '_metrics_logger'):
                self._metrics_logger.log_performance("get_spark_dataframe", duration_ms)

            # Lưu lại để sử dụng sau
            self._spark_df = spark_df

            return spark_df

        except Exception as e:
            self._logger.error("Lỗi trong quá trình chuyển đổi sang Spark DataFrame: %s", str(e))
            return None

    @property
    def data_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin về dữ liệu đã được tải
        """
        if self._data is None or not isinstance(self._data, pd.DataFrame) or self._data.empty:
            return {
                "status": "not_loaded",
                "message": "Dữ liệu chưa được tải",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        return {
            "status": "loaded",
            "row_count": len(self._data),
            "column_count": len(self._data.columns),
            "columns": list(self._data.columns),
            "memory_usage": self._data.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
            "dtypes": {col: str(dtype) for col, dtype in self._data.dtypes.items()},
            "missing_values": {col: int(self._data[col].isnull().sum()) for col in self._data.columns},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_updated": getattr(self, '_last_updated', datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        }

    def get_area_comparison(self, location: str) -> Dict[str, Any]:
        """
        Lấy dữ liệu so sánh cho một khu vực
        """
        self._logger.info(f"Lấy dữ liệu so sánh cho khu vực: {location}")
        start_time = time.time()

        try:
            if self._data is None or not isinstance(self._data, pd.DataFrame) or self._data.empty:
                return {
                    "status": "error",
                    "message": "Dữ liệu chưa được tải"
                }

            # Tìm cột chứa thông tin về khu vực/địa điểm
            location_columns = [col for col in self._data.columns if any(keyword in col.lower() for keyword in [
                'location', 'area', 'district', 'city', 'province', 'region', 'address', 'ward',
                'vị trí', 'khu vực', 'quận', 'huyện', 'tỉnh', 'thành phố', 'địa chỉ', 'phường', 'xã'
            ])]

            price_columns = [col for col in self._data.columns if any(keyword in col.lower() for keyword in [
                'price', 'cost', 'value', 'giá', 'total_price', 'unit_price'
            ])]

            # Nếu không tìm thấy cột phù hợp
            if not location_columns or not price_columns:
                return {
                    "status": "error",
                    "message": "Không tìm thấy cột khu vực hoặc giá"
                }

            # Chọn cột đầu tiên tìm được
            location_col = location_columns[0]
            price_col = price_columns[0]

            # Lọc dữ liệu theo khu vực
            area_data = self._data[self._data[location_col].str.contains(location, case=False, na=False)]

            if area_data.empty:
                return {
                    "status": "error",
                    "message": f"Không tìm thấy dữ liệu cho khu vực: {location}"
                }

            # Tính toán các thống kê
            avg_price = area_data[price_col].mean()
            median_price = area_data[price_col].median()
            min_price = area_data[price_col].min()
            max_price = area_data[price_col].max()
            std_price = area_data[price_col].std()
            count = len(area_data)

            # So sánh với toàn bộ dữ liệu
            overall_avg = self._data[price_col].mean()
            price_diff = avg_price - overall_avg
            price_diff_pct = (price_diff / overall_avg) * 100 if overall_avg != 0 else 0

            # Xử lý thời gian
            duration_ms = (time.time() - start_time) * 1000
            self._logger.info(f"Hoàn thành lấy dữ liệu so sánh cho khu vực {location} (mất {duration_ms:.2f} ms)")

            # Ghi metrics nếu có
            if hasattr(self, '_metrics_logger'):
                self._metrics_logger.log_performance("get_area_comparison", duration_ms)

            return {
                "status": "success",
                "location": location,
                "property_count": count,
                "avg_price": avg_price,
                "median_price": median_price,
                "min_price": min_price,
                "max_price": max_price,
                "std_price": std_price,
                "overall_avg_price": overall_avg,
                "price_difference": price_diff,
                "price_difference_percent": price_diff_pct,
                "is_above_average": price_diff > 0
            }

        except Exception as e:
            self._logger.error(f"Lỗi khi lấy dữ liệu so sánh cho khu vực {location}: {str(e)}")
            return {
                "status": "error",
                "message": f"Lỗi khi xử lý: {str(e)}"
            }

    def _try_load_cached_data(self) -> bool:
        """
        Thử tải dữ liệu từ cache hoặc session state
        """
        start_time = time.time()

        # Thử tải từ session state trước
        cached_data = self.load_state("processed_data")
        if cached_data is not None and isinstance(cached_data, pd.DataFrame) and not cached_data.empty:
            self._logger.info("Tải dữ liệu từ session state (mất %0.2f ms)", (time.time() - start_time) * 1000)
            self._data = cached_data
            return True

        # Thử tải từ file cache
        cache_file = os.path.join(self._cache_dir, "processed_data_cache.pkl")
        if os.path.exists(cache_file):
            try:
                # Kiểm tra tuổi của cache
                file_mtime = os.path.getmtime(cache_file)
                file_datetime = datetime.fromtimestamp(file_mtime)
                current_time = datetime.now()

                # Nếu cache cũ hơn 24 giờ, bỏ qua
                if (current_time - file_datetime) > timedelta(hours=24):
                    self._logger.info("Cache quá cũ (%s), bỏ qua", file_datetime.strftime("%Y-%m-%d %H:%M:%S"))
                    return False

                # Nếu cache còn mới, tải lên
                self._data = pd.read_pickle(cache_file)

                if self._data is not None and isinstance(self._data, pd.DataFrame) and not self._data.empty:
                    duration_ms = (time.time() - start_time) * 1000
                    self._logger.info("Tải dữ liệu từ file cache (mất %0.2f ms)", duration_ms)

                    # Lưu vào session state
                    self.save_state("processed_data", self._data)

                    # Ghi metrics nếu có
                    if hasattr(self, '_metrics_logger'):
                        self._metrics_logger.log_performance("load_from_cache", duration_ms)

                    return True
            except Exception as e:
                self._logger.warning("Lỗi khi tải cache: %s", str(e))

        return False

    def _generate_cache_key(self, data: pd.DataFrame, prefix: str = "") -> str:
        """
        Tạo khóa cache dựa trên dữ liệu input

        Args:
            data: DataFrame cần tạo khóa
            prefix: Tiền tố cho khóa

        Returns:
            str: Khóa cache duy nhất cho dữ liệu
        """
        # Tạo chuỗi mô tả dữ liệu
        data_desc = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in zip(data.dtypes.index, data.dtypes.values)},
            'sum_values': float(data.select_dtypes(include=[np.number]).sum().sum()),
            'timestamp': datetime.now().strftime("%Y%m%d")
        }

        # Chuyển thành chuỗi và tạo hash
        data_str = json.dumps(data_desc, sort_keys=True)
        hash_obj = hashlib.md5(data_str.encode())
        cache_key = f"{prefix}_{hash_obj.hexdigest()}"

        return cache_key

    def _save_to_cache(self, data: pd.DataFrame, cache_key: str) -> bool:
        """
        Lưu dữ liệu vào cache
        """
        try:
            # Lưu vào file
            cache_file = os.path.join(self._cache_dir, f"{cache_key}.pkl")
            data.to_pickle(cache_file)

            # Lưu thông tin vào cache_info
            self._cache_info[cache_key] = {
                'file': cache_file,
                'timestamp': datetime.now().isoformat(),
                'rows': len(data),
                'columns': list(data.columns)
            }

            # Lưu cache_info vào session
            self.save_state("cache_info", self._cache_info)

            self._logger.info("Lưu dữ liệu vào cache: %s (rows=%d)", cache_key, len(data))
            return True

        except Exception as e:
            self._logger.error("Lỗi khi lưu cache: %s", str(e))
            return False

    def _cleanup_old_cache(self, max_age_hours: int = 48):
        """
        Dọn dẹp các file cache cũ
        """
        # Kiểm tra xem có cần dọn dẹp không
        current_time = datetime.now()
        if (current_time - self._last_cache_check).total_seconds() < 3600:  # Chỉ kiểm tra một lần mỗi giờ
            return

        self._last_cache_check = current_time
        self._logger.info("Bắt đầu dọn dẹp cache")

        # Liệt kê tất cả các file cache
        if not os.path.exists(self._cache_dir):
            return

        for filename in os.listdir(self._cache_dir):
            if filename.endswith('.pkl'):
                file_path = os.path.join(self._cache_dir, filename)
                try:
                    # Kiểm tra thời gian sửa đổi
                    file_mtime = os.path.getmtime(file_path)
                    file_datetime = datetime.fromtimestamp(file_mtime)

                    # Nếu file quá cũ, xóa nó
                    if (current_time - file_datetime) > timedelta(hours=max_age_hours):
                        os.remove(file_path)
                        self._logger.info("Xóa file cache cũ: %s", filename)
                except Exception as e:
                    self._logger.error("Lỗi khi dọn dẹp cache: %s", str(e))

    @property
    def data_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin về dữ liệu đã được tải
        """
        if self._data is None:
            return {"loaded": False}

        return {
            "loaded": True,
            "rows": len(self._data),
            "columns": list(self._data.columns),
            "missing_values": self._data.isna().sum().sum(),
            "memory_usage": self._data.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
            "numerical_columns": list(self._data.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(self._data.select_dtypes(include=['object', 'category']).columns),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    @property
    def spark(self):
        """
        Lấy phiên Spark (khởi tạo khi truy cập lần đầu)
        """
        if self._spark is None:
            self._spark = get_spark_session(app_name="VNRealEstatePricePrediction")
        return self._spark

    @property
    def data(self):
        """
        Trả về dữ liệu đã xử lý
        """
        return self._data

    # MARK: - Tải và xử lý dữ liệu

    @st.cache_data
    def load_data(_self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Đọc dữ liệu từ file CSV và lưu trữ vào bộ nhớ cache
        Nếu tìm thấy dữ liệu thô (raw_dataset.csv), sẽ tự động xử lý nó trước
        """
        # Kiểm tra nếu đã tải dữ liệu trước đó và lưu trong cache
        if _self._data is not None:
            _self._logger.info("Sử dụng dữ liệu đã lưu trong bộ nhớ")
            return _self._data

        # Nếu đã lưu vào session state, sử dụng lại
        if st.session_state and 'data' in st.session_state and st.session_state.data is not None:
            _self._logger.info("Lấy dữ liệu từ session state")
            _self._data = st.session_state.data
            return _self._data

        try:
            # Kiểm tra xem có dữ liệu thô cần xử lý không (raw_dataset.csv)
            raw_data_path = os.path.join(_self._data_dir, "raw_dataset.csv")
            if os.path.exists(raw_data_path):
                _self._logger.info(f"Tìm thấy file dữ liệu thô: {raw_data_path}")
                _self._logger.info("Bắt đầu xử lý dữ liệu thô...")

                # Hiển thị thông báo cho người dùng
                with st.status("Đang xử lý dữ liệu thô...", expanded=True) as status:
                    try:
                        # Gọi phương thức xử lý dữ liệu thô
                        processed_data = _self.process_raw_data(raw_data_path)

                        if processed_data is not None:
                            status.update(label="Xử lý dữ liệu thô thành công!", state="complete")
                            st.success(f"Xử lý thành công {len(processed_data)} dòng dữ liệu thô.")

                            # Đổi tên file raw_dataset.csv thành raw_dataset.csv.processed để tránh xử lý lại
                            processed_mark = raw_data_path + ".processed"
                            os.rename(raw_data_path, processed_mark)
                            _self._logger.info(f"Đã đổi tên file dữ liệu thô thành: {processed_mark}")

                            # Sử dụng dữ liệu đã xử lý
                            _self._data = processed_data
                            _self._file_path = os.path.join(_self._data_dir, 'processed_data.csv')

                            # Lưu vào session state để có thể sử dụng lại giữa các phiên
                            if st.session_state is not None:
                                st.session_state.data = processed_data

                            return processed_data
                        else:
                            status.update(label="Xử lý dữ liệu thô thất bại!", state="error")
                            st.error("Không thể xử lý dữ liệu thô. Tiếp tục tìm kiếm dữ liệu đã xử lý...")
                    except Exception as e:
                        status.update(label="Lỗi khi xử lý dữ liệu thô!", state="error")
                        st.error(f"Lỗi: {str(e)}")

            # Tìm kiếm file dữ liệu
            if file_path is None:
                # Tìm kiếm file dữ liệu mặc định trong thư mục data
                files = glob.glob(os.path.join(_self._data_dir, "*.csv"))
                _self._logger.info(f"Tìm thấy {len(files)} file CSV trong thư mục data")

                if not files:
                    _self._logger.error("Không tìm thấy file dữ liệu CSV nào trong thư mục data")
                    return pd.DataFrame()

                # Uu tiên file có tên chứa 'processed'
                processed_files = [f for f in files if 'processed' in os.path.basename(f).lower()]

                if processed_files:
                    file_path = processed_files[0]
                    _self._logger.info(f"Sử dụng file dữ liệu đã xử lý: {os.path.basename(file_path)}")
                else:
                    # Sắp xếp theo thời gian sửa đổi mới nhất
                    files.sort(key=os.path.getmtime, reverse=True)
                    file_path = files[0]
                    _self._logger.info(f"Sử dụng file dữ liệu mới nhất: {os.path.basename(file_path)}")

            # Đọc dữ liệu từ file CSV
            _self._logger.info(f"Bắt đầu đọc dữ liệu từ file: {file_path}")
            data = pd.read_csv(file_path)
            _self._logger.info(f"Đã đọc dữ liệu: {len(data)} dòng, {len(data.columns)} cột")

            # Lưu vào cache và session state
            if data is None:
                raise ValueError("Không thể đọc dữ liệu từ file CSV")

            # Lưu trữ dữ liệu vào bộ nhớ
            _self._data = data

            # Lưu dữ liệu vào cache địa phương
            _self.save_state("processed_data", data)

            _self._logger.info(f"Đã tải dữ liệu: {len(data)} dòng, {len(data.columns)} cột")
            return data

        except Exception as e:
            _self._logger.error(f"Lỗi khi tải dữ liệu: {str(e)}")
            # Tạo dữ liệu giả lập nếu không thể tải
            _self._logger.warning("Tạo dữ liệu mẫu để phục vụ cho việc thử nghiệm")

            # Tạo DataFrame mẫu với các cột được yêu cầu
            sample_data = pd.DataFrame({
                'price': np.random.uniform(1000, 10000, 100),
                'area (m2)': np.random.uniform(30, 500, 100),
                'street (m)': np.random.uniform(2, 20, 100),
                'bedrooms': np.random.randint(1, 6, 100),
                'bathrooms': np.random.randint(1, 5, 100),
                'location': np.random.choice(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], 100),
                'year_built': np.random.randint(1990, 2023, 100),
            })

            _self._data = sample_data
            return sample_data

    @st.cache_data
    def preprocess_data(_self, data: pd.DataFrame):
        """
        Tiền xử lý dữ liệu cho phân tích và mô hình hóa
        """
        if data is None or data.empty:
            _self.logger.warning("Không có dữ liệu để tiền xử lý")
            return None

        try:
            _self.logger.info("Bắt đầu tiền xử lý dữ liệu...")

            # Tạo bản sao để tránh warning SettingWithCopyWarning
            df = data.copy()

            # Chuyển đổi cột ngày tháng nếu có
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

            # Chuẩn hóa dữ liệu
            _self.logger.info("Hoàn thành tiền xử lý dữ liệu")
            _self.save_state("preprocessed_data", df)

            _self._logger.info("Tiền xử lý dữ liệu hoàn tất, số cột: " + str(len(df.columns)))
            return df

        except Exception as e:
            _self._logger.error(f"Lỗi trong quá trình tiền xử lý dữ liệu: {e}")
            return data  # Trả về dữ liệu gốc nếu có lỗi

    @st.cache_resource
    def convert_to_spark(_self, _data: pd.DataFrame):
        """
        Chuyển đổi DataFrame pandas sang DataFrame Spark
        """
        if _self._spark_df is not None:
            return _self._spark_df

        try:
            # Kiểm tra xem Spark có khả dụng không
            spark = _self.get_spark_session_cached()
            if spark is None:
                _self.logger.warning("Không thể khởi tạo Spark, sử dụng phương pháp dự phòng")
                return None

            # Đảm bảo dữ liệu có tất cả các cột cần thiết (cả tên cũ và mới)
            if 'area (m2)' in _data.columns and 'area_m2' not in _data.columns:
                _data['area_m2'] = _data['area (m2)'].copy()
            if 'street (m)' in _data.columns and 'street_width_m' not in _data.columns:
                _data['street_width_m'] = _data['street (m)'].copy()

            # Chuyển đổi DataFrame pandas sang DataFrame Spark
            spark_df = spark.createDataFrame(_data)
            _self._spark_df = spark_df
            return spark_df

        except Exception as e:
            _self.logger.error(f"Lỗi khi chuyển đổi dữ liệu sang Spark: {e}")
            return None

    def get_spark_session_cached(_self) -> Optional[SparkSession]:
        """
        Lấy SparkSession đã được cache
        """
        # Kiểm tra nếu Spark đã được khởi tạo trước đó
        if _self._spark is not None:
            return _self._spark

        # Kiểm tra trạng thái Spark đã lưu
        spark_status = _self.load_state("spark_status")
        if spark_status and spark_status.get("initialized", False) is False:
            # Nếu đã có ghi nhận trước đó về việc không thể khởi tạo, không cần thử lại
            _self._logger.warning("Bỏ qua khởi tạo Spark do lần trước đã thất bại")
            return None

        try:
            # Sử dụng safe_execute để gọi hàm khởi tạo Spark
            spark = _self.safe_execute(
                get_spark_session,
                app_name="VNRealEstatePricePrediction",
                enable_hive=False,
                default=None
            )

            if spark is None:
                raise ValueError("Không thể khởi tạo Spark session")

            # Kiểm tra kết nối để đảm bảo Spark hoạt động
            _self.safe_execute(
                lambda: spark.sparkContext.parallelize([1]).collect(),
                default=None
            )

            # Cập nhật trạng thái và lưu trữ
            _self._spark = spark
            _self._last_updated = datetime.now()
            _self.save_state("spark_status", {"initialized": True, "timestamp": str(_self._last_updated)})

            # Ghi log thành công
            _self._logger.info("Khởi tạo SparkSession thành công")
            return spark

        except Exception as e:
            _self._logger.error(f"Không thể khởi tạo Spark: {str(e)}")
            # Lưu trạng thái khởi tạo thất bại để tránh các lần thử lại không cần thiết
            _self.save_state("spark_status", {"initialized": False, "error": str(e), "timestamp": str(_self._last_updated)})
        # Trả về dữ liệu so sánh
        return area_comparison

    def process_raw_data(self, raw_file_path: str) -> pd.DataFrame:
        """
        Xử lý dữ liệu thô từ file raw_dataset.csv
        """
        if not os.path.exists(raw_file_path):
            self.logger.error(f"File dữ liệu thô không tồn tại: {raw_file_path}")
            return None

        try:
            self.logger.info(f"Bắt đầu xử lý dữ liệu thô từ file: {raw_file_path}")
            start_time = time.time()

            # Đọc dữ liệu thô
            raw_df = pd.read_csv(raw_file_path)
            self.logger.info(f"Đã đọc file dữ liệu thô với {len(raw_df)} dòng")

            # Khởi tạo Spark Session
            spark = self.get_spark_session_cached()
            if spark is None:
                self.logger.warning("Không thể khởi tạo Spark, chuyển sang xử lý với pandas")
                return self._process_raw_with_pandas(raw_df)

            # Chuyển đổi pandas DataFrame sang Spark DataFrame
            self.logger.info("Chuyển đổi dữ liệu sang Spark DataFrame")
            df = spark.createDataFrame(raw_df)

            # --- XỬ LÝ DỮ LIỆU THEO LOGIC FINAL_NHOM05_PROGRESS_DATA.PY ---

            # 1. Xóa các cột không cần thiết
            if 'link' in df.columns and 'title' in df.columns:
                df = df.drop('link', 'title')

            # 2. Xóa các dòng có giá trị null trong các cột 'price','area'
            df = df.dropna(subset=['price', 'area'])

            # 3. Xử lý các giá trị null trong các cột số
            from pyspark.sql.functions import col, when
            numeric_columns = ['floor_num', 'toilet_num', 'livingroom_num', 'bedroom_num']
            for column_name in numeric_columns:
                if column_name in df.columns:
                    df = df.withColumn(
                        column_name,
                        when(col(column_name).isNull(), -1).otherwise(col(column_name))
                    )

            # 4. Xử lý các giá trị null trong các cột chữ
            string_columns = ['category', 'direction', 'liability', 'location']
            for column_name in string_columns:
                if column_name in df.columns:
                    df = df.withColumn(
                        column_name,
                        when(col(column_name).isNull(), 'Không xác định').otherwise(col(column_name))
                    )

            # 5. Xử lý cột Location thành district và city_province
            from pyspark.sql.functions import regexp_replace, split, trim, element_at
            if 'location' in df.columns:
                df = df.withColumn(
                    'district',
                    trim(element_at(split(regexp_replace(col('location'), r'\s*Lưu tin\s*$', ''), ','), -2))
                ).withColumn(
                    'city_province',
                    trim(element_at(split(regexp_replace(col('location'), r'\s*Lưu tin\s*$', ''), ','), -1))
                )

            # 6. Xử lý cột price
            from pyspark.sql.functions import regexp_replace, when, lit
            import pyspark.sql.functions as F

            if 'price' in df.columns:
                # Chuyển đổi cột price sang float
                df = df.withColumn('price',
                    when(
                        F.col("price").contains("tỷ"),
                        F.regexp_replace(F.col("price"), "tỷ", "").cast("float") * 1000000000
                    ).when(
                        F.col("price").contains("triệu"),
                        F.regexp_replace(F.col("price"), "triệu", "").cast("float") * 1000000
                    ).otherwise(
                        F.regexp_replace(F.col("price"), ",", "").cast("float")
                    )
                )

                # Loại bỏ các dòng "Thương lượng"
                df = df.filter(~F.col("price").like("%Thương lượng%"))

            # 7. Xử lý cột area (m2)
            if 'area' in df.columns:
                df = df.withColumn('area',
                    regexp_replace(col('area'), 'm2', '').cast('float')
                )

                # Đổi tên cột
                df = df.withColumnRenamed("area", "area (m2)")

            # 8. Xử lý cột floor_num
            if 'floor_num' in df.columns:
                df = df.withColumn('floor_num',
                    regexp_replace(col('floor_num').cast('string'), r'\s*tầng$', '').cast('int')
                )

            # 9. Xử lý cột bedroom_num
            if 'bedroom_num' in df.columns:
                df = df.withColumn('bedroom_num',
                    regexp_replace(col('bedroom_num').cast('string'), r'\s*phòng$', '').cast('int')
                )

            # 10. Xử lý cột street
            if 'street' in df.columns:
                # Bỏ chữ "m"
                df = df.withColumn("street_clean", regexp_replace(col("street"), "m", ""))

                # Xử lý khoảng cách "a - b"
                df = df.withColumn("split", split(col("street_clean"), "-"))

                # Tính giá trị trung bình và thay thế giá trị null
                df = df.withColumn(
                    "street_float",
                    when(
                        col("street_clean").isNull(), lit(-1)  # Nếu cột street_clean là null, gán -1
                    ).when(
                        (col("split").getItem(1).isNotNull()),  # Nếu có dấu "-"
                        (col("split").getItem(0).cast("float") + col("split").getItem(1).cast("float")) / 2
                    ).otherwise(
                        col("split").getItem(0).cast("float")  # Chỉ có 1 số
                    )
                )

                # Thay thế giá trị null trong cột street_float bằng -1
                df = df.withColumn(
                    "street_float",
                    when(col("street_float").isNull(), lit(-1)).otherwise(col("street_float"))
                )

                # Xóa các cột không cần thiết
                df = df.drop("street", "street_clean", "split")

                # Đổi tên cột
                df = df.withColumnRenamed("street_float", "street (m)")

            # 11. Chuyển đổi kiểu dữ liệu cuối cùng
            # Danh sách các cột cần chuyển đổi và kiểu dữ liệu mới
            column_types = {
                "area (m2)": "float",
                "floor_num": "int",
                "toilet_num": "int",
                "livingroom_num": "int",
                "bedroom_num": "int",
                "street (m)": "float"
            }

            # Chuyển đổi kiểu dữ liệu cho từng cột
            for column_name, data_type in column_types.items():
                if column_name in df.columns:
                    df = df.withColumn(column_name, col(column_name).cast(data_type))

            # Tính toán cột price_per_m2 nếu chưa có
            if 'price_per_m2' not in df.columns and 'price' in df.columns and 'area (m2)' in df.columns:
                df = df.withColumn(
                    "price_per_m2",
                    when(col("area (m2)") > 0, col("price") / col("area (m2)")).otherwise(0)
                )

            # Chuyển đổi từ Spark DataFrame về pandas DataFrame
            self.logger.info("Chuyển đổi kết quả từ Spark DataFrame về pandas DataFrame")
            result_df = df.toPandas()

            # Lưu kết quả vào thư mục data
            output_file = os.path.join(self._data_dir, 'processed_data.csv')
            result_df.to_csv(output_file, index=False)
            self.logger.info(f"Đã lưu dữ liệu đã xử lý vào: {output_file}")

            # Ghi log thời gian xử lý
            processing_time = time.time() - start_time
            self.logger.info(f"Xử lý dữ liệu thô hoàn tất trong {processing_time:.2f} giây")

            # Ghi metrics
            self._metrics_service.add_metrics("data_processing", {
                "raw_rows": len(raw_df),
                "processed_rows": len(result_df),
                "processing_time": processing_time
            })

            return result_df

        except Exception as e:
            self.logger.error(f"Lỗi trong quá trình xử lý dữ liệu thô: {str(e)}", exc_info=True)
            return None

    def _process_raw_with_pandas(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Xử lý dữ liệu thô bằng pandas khi Spark không khả dụng
        """
        try:
            self.logger.info("Xử lý dữ liệu thô bằng pandas...")
            df = raw_df.copy()

            # 1. Xóa các cột không cần thiết
            if 'link' in df.columns and 'title' in df.columns:
                df = df.drop(columns=['link', 'title'])

            # 2. Xóa các dòng có giá trị null trong các cột 'price','area'
            df = df.dropna(subset=['price', 'area'])

            # 3. Xử lý các giá trị null trong các cột số
            numeric_columns = ['floor_num', 'toilet_num', 'livingroom_num', 'bedroom_num']
            for column in numeric_columns:
                if column in df.columns:
                    df[column] = df[column].fillna(-1)

            # 4. Xử lý các giá trị null trong các cột chữ
            string_columns = ['category', 'direction', 'liability', 'location']
            for column in string_columns:
                if column in df.columns:
                    df[column] = df[column].fillna('Không xác định')

            # 5. Xử lý cột Location thành district và city_province
            if 'location' in df.columns:
                # Loại bỏ 'Lưu tin' ở cuối chuỗi location
                df['location_clean'] = df['location'].str.replace(r'\s*Lưu tin\s*$', '', regex=True)

                # Tách location thành district và city_province
                df['location_split'] = df['location_clean'].str.split(',')

                # Lấy phần tử áp chót và phần tử cuối cùng
                df['district'] = df['location_split'].apply(lambda x: x[-2].strip() if isinstance(x, list) and len(x) >= 2 else 'Không xác định')
                df['city_province'] = df['location_split'].apply(lambda x: x[-1].strip() if isinstance(x, list) and len(x) >= 1 else 'Không xác định')

                # Xóa các cột tạm
                df = df.drop(columns=['location_clean', 'location_split'])

            # 6. Xử lý cột price
            if 'price' in df.columns:
                # Tạo một bản sao của cột price để xử lý
                df['price_processed'] = df['price']

                # Xử lý các trường hợp có chứa 'tỷ'
                mask_ty = df['price_processed'].str.contains('tỷ', na=False)
                df.loc[mask_ty, 'price_processed'] = df.loc[mask_ty, 'price_processed'].str.replace('tỷ', '').astype(float) * 1000000000

                # Xử lý các trường hợp có chứa 'triệu'
                mask_trieu = df['price_processed'].str.contains('triệu', na=False)
                df.loc[mask_trieu, 'price_processed'] = df.loc[mask_trieu, 'price_processed'].str.replace('triệu', '').astype(float) * 1000000

                # Xử lý các trường hợp còn lại
                mask_other = ~(mask_ty | mask_trieu)
                df.loc[mask_other, 'price_processed'] = df.loc[mask_other, 'price_processed'].str.replace(',', '').astype(float)

                # Loại bỏ các dòng "Thương lượng"
                df = df[~df['price_processed'].astype(str).str.contains('Thương lượng')]

                # Cập nhật lại cột price
                df['price'] = df['price_processed']
                df = df.drop(columns=['price_processed'])

            # 7. Xử lý cột area (m2)
            if 'area' in df.columns:
                df['area'] = df['area'].astype(str).str.replace('m2', '').astype(float)
                df = df.rename(columns={'area': 'area (m2)'})

            # 8. Xử lý cột floor_num
            if 'floor_num' in df.columns:
                df['floor_num'] = df['floor_num'].astype(str).str.replace(r'\s*tầng$', '', regex=True).astype(float).astype('Int64')

            # 9. Xử lý cột bedroom_num
            if 'bedroom_num' in df.columns:
                df['bedroom_num'] = df['bedroom_num'].astype(str).str.replace(r'\s*phòng$', '', regex=True).astype(float).astype('Int64')

            # 10. Xử lý cột street
            if 'street' in df.columns:
                # Bỏ chữ "m"
                df['street_clean'] = df['street'].astype(str).str.replace('m', '')

                # Xử lý khoảng cách "a - b"
                df['street_float'] = df['street_clean'].apply(lambda x:
                    -1 if pd.isna(x) else
                    (float(x.split('-')[0]) + float(x.split('-')[1])) / 2 if '-' in str(x) else
                    float(x) if str(x).replace('.', '', 1).isdigit() else -1
                )

                # Xóa cột tạm và đổi tên
                df = df.drop(columns=['street', 'street_clean'])
                df = df.rename(columns={'street_float': 'street (m)'})

            # 11. Chuyển đổi kiểu dữ liệu cuối cùng
            # Danh sách các cột cần chuyển đổi và kiểu dữ liệu mới
            column_types = {
                "area (m2)": float,
                "floor_num": 'Int64',
                "toilet_num": 'Int64',
                "livingroom_num": 'Int64',
                "bedroom_num": 'Int64',
                "street (m)": float
            }

            # Chuyển đổi kiểu dữ liệu cho từng cột
            for column_name, data_type in column_types.items():
                if column_name in df.columns:
                    df[column_name] = df[column_name].astype(data_type)

            # Tính toán cột price_per_m2 nếu chưa có
            if 'price_per_m2' not in df.columns and 'price' in df.columns and 'area (m2)' in df.columns:
                df['price_per_m2'] = df.apply(lambda row: row['price'] / row['area (m2)'] if row['area (m2)'] > 0 else 0, axis=1)

            # Lưu kết quả vào thư mục data
            output_file = os.path.join(self._data_dir, 'processed_data.csv')
            df.to_csv(output_file, index=False)
            self.logger.info(f"Đã lưu dữ liệu đã xử lý vào: {output_file}")

            return df

        except Exception as e:
            self.logger.error(f"Lỗi trong quá trình xử lý dữ liệu thô với pandas: {str(e)}", exc_info=True)
            return None

    def get_area_comparison(self, location: str) -> Dict[str, Any]:
        """
        Lấy dữ liệu so sánh giá cho một vị trí
        """
        data = self._data
        if data is None or data.empty:
            return {}

        # Kiểm tra xem location có giá trị hợp lệ không
        if location is None or not isinstance(location, str) or location.strip() == '':
            self.logger.warning("Vị trí không hợp lệ hoặc trống, sử dụng toàn bộ dữ liệu")
            location_data = data
        else:
            try:
                # Filter by location if possible
                location_filter = location.lower()
                # Kiểm tra xem cột 'location' có tồn tại trong dữ liệu không
                if 'location' in data.columns:
                    location_data = data[data['location'].str.lower().str.contains(location_filter)]
                else:
                    self.logger.warning("Không tìm thấy cột 'location' trong dữ liệu")
                    location_data = data
            except Exception as e:
                self.logger.error(f"Lỗi khi lọc dữ liệu theo vị trí: {e}")
                location_data = data

        # If no data for location, use all data
        if location_data.empty:
            location_data = data

        # Tính toán thống kê
        avg_price = location_data['price'].mean()
        min_price = location_data['price'].min()
        max_price = location_data['price'].max()

        avg_price_per_sqm = location_data['price_per_sqm'].mean()
        min_price_per_sqm = location_data['price_per_sqm'].min()
        max_price_per_sqm = location_data['price_per_sqm'].max()

        return {
            'location': location,
            'avg_price': avg_price,
            'min_price': min_price,
            'max_price': max_price,
            'avg_price_per_sqm': avg_price_per_sqm,
            'min_price_per_sqm': min_price_per_sqm,
            'max_price_per_sqm': max_price_per_sqm
        }