#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dịch vụ Huấn luyện Mô hình - Chịu trách nhiệm huấn luyện mô hình máy học để dự đoán giá bất động sản
"""

# MARK: - Thư viện

import os
import sys
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
import streamlit as st

from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

from ..utils.session_utils import save_model_metrics, get_model_metrics, metrics_exist
from .interfaces.train_model_interface import ITrainModelService
from .interfaces.metrics_interface import IMetricsService
from .interfaces.progress_data_interface import IProgressDataService
from .core.base_service import BaseService

# Trả về True nếu scikit-learn đã được cài đặt
sklearn_available = False
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    sklearn_available = True
except ImportError:
    st.warning("Thư viện scikit-learn không có sẵn. Cài đặt bằng cách chạy: pip install scikit-learn")
    st.info("Sẽ sử dụng chế độ dự phòng đơn giản hơn.")

# MARK: - Cấu hình

utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils')
if utils_path not in sys.path:
    sys.path.append(utils_path)

# MARK: - Huấn luyện mô hình

class TrainModelService(BaseService, ITrainModelService):
    """
    Lớp dịch vụ chịu trách nhiệm huấn luyện mô hình máy học để dự đoán giá bất động sản
    Triển khai interface ITrainModelService
    Kế thừa từ BaseService để sử dụng các tính năng chung
    """

    # MARK: - Khởi tạo

    def __init__(self, _data_service: Optional[IProgressDataService] = None, _metrics_service: Optional[IMetricsService] = None,
                models_dir: str = None, data_dir: str = None):
        """
        Khởi tạo dịch vụ huấn luyện mô hình
        """
        # Gọi khởi tạo lớp cơ sở
        super().__init__(service_name="TrainModelService")

        # Đường dẫn thư mục
        self._models_dir = models_dir if models_dir else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
        self._data_dir = data_dir if data_dir else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')

        # Import động để tránh circular import
        from .core.services_factory import ServicesFactory

        # Dịch vụ dữ liệu (nếu được cung cấp)
        self._data_service = _data_service if _data_service else ServicesFactory.get_progress_data_service()

        # Dịch vụ quản lý metrics
        self._metrics_service = _metrics_service if _metrics_service else ServicesFactory.get_metrics_service()

        # Các thuộc tính mô hình
        self._model = self.load_model("gbt_model")
        self._using_fallback = False
        self._fallback_model = self.load_model("fallback_model")
        self._fallback_features = None

        # Define feature column names
        self._feature_columns = {
            'area': 'area (m2)',
            'street': 'street (m)',
            'bedrooms': 'bedroom_num',
            'toilets': 'toilet_num',
            'floors': 'floor_num'
        }

        # Nếu có metrics đã lưu, tải lên
        stored_metrics = self.load_state("model_metrics")
        if stored_metrics:
            self._logger.info("Tải metrics đã lưu của mô hình")
            self._metrics_service.add_metrics("gbt_model", stored_metrics)

    # MARK: - Properties

    @property
    def model_metrics(self) -> Dict[str, float]:
        """
        Lấy các chỉ số của mô hình hiện tại
        Sử dụng MetricsService để quản lý metrics
        """
        # Lấy mô hình hiện tại đang sử dụng
        model_name = "fallback_model" if self._using_fallback else "gbt_model"

        # Ưu tiên lấy metrics từ MetricsService
        metrics = self._metrics_service.get_metrics(model_name)
        if metrics:
            # Đảm bảo có trường 'accuracy' cho tương thích ngược
            if 'r2' in metrics and 'accuracy' not in metrics:
                metrics['accuracy'] = metrics['r2']
            return {k: v for k, v in metrics.items() if not k.startswith('_')}

        # Nếu không có trong MetricsService, kiểm tra session state
        if metrics_exist():
            session_metrics = get_model_metrics()
            # Cập nhật vào MetricsService để sử dụng sau này
            self._metrics_service.add_metrics(model_name, session_metrics)

            # Trả về metrics từ session state, đảm bảo trường 'accuracy' cho tương thích ngược
            metrics = {
                'r2': session_metrics.get('r2', 0.0),
                'accuracy': session_metrics.get('r2', 0.0),
                'rmse': session_metrics.get('rmse', 0.0)
            }
            return metrics

        # Nếu không có metrics nào khả dụng, trả về giá trị mặc định
        self._logger.warning(f"Không tìm thấy metrics cho mô hình {model_name}, sử dụng giá trị mặc định")
        default_metrics = {'r2': 0.0, 'accuracy': 0.0, 'rmse': 0.0}

        # Lưu trữ metrics mặc định
        self.save_state("model_metrics", default_metrics)
        self._metrics_service.add_metrics(model_name, default_metrics)

        return default_metrics

    # MARK: - Huấn luyện mô hình

    @st.cache_resource
    def train_model(_self, _spark_df=None, _force_retrain=False) -> Tuple[Any, float, float]:
        """
        Huấn luyện mô hình học máy để dự đoán giá bất động sản
        """
        # Kiểm tra xem metrics có tồn tại trong session state không
        if metrics_exist() and not _force_retrain:
            # Lấy metrics từ session state
            session_metrics = get_model_metrics()
            _self.logger.info("Sử dụng metrics từ session state: R² = {:.4f}, RMSE = {:.4f}".format(
                session_metrics.get('r2', 0.0), session_metrics.get('rmse', 0.0)
            ))
            # Cập nhật các metrics của TrainModelService từ session state
            _self._r2 = session_metrics.get('r2', 0.0)
            _self._accuracy = _self._r2  # Đảm bảo tương thích ngược
            _self._rmse = session_metrics.get('rmse', 0.0)
            return _self._model, _self._accuracy, _self._rmse

        # Nếu đã có mô hình và không bắt buộc huấn luyện lại, trả về mô hình và các chỉ số hiện tại
        if _self._model is not None and not _force_retrain:
            return _self._model, _self._accuracy, _self._rmse

        # Lấy dữ liệu từ data_service
        if _self._data_service.data is None:
            _self._data_service.load_data()

        # Nếu không có dữ liệu, trả về None
        if _self._data_service.data is None or _self._data_service.data.empty:
            _self.logger.error("Không có dữ liệu để huấn luyện mô hình")
            return None, 0, 0

        # Tiền xử lý dữ liệu
        processed_data = _self._data_service.preprocess_data(_self._data_service.data)

        # Nếu spark_df không được cung cấp, tạo mới
        if _spark_df is None:
            # Chuyển đổi dữ liệu sang Spark DataFrame
            _spark_df = _self._data_service.convert_to_spark(processed_data)

        # Kiểm tra nếu data_spark là None (khi Spark không khả dụng)
        if _spark_df is None:
            # Thiết lập giá trị metrics mặc định
            _self._rmse = 0.0
            _self._r2 = 0.0

            # Sử dụng fallback mode với scikit-learn
            return _self._train_with_sklearn(processed_data)

        # Nếu có Spark DataFrame, huấn luyện với Spark ML
        return _self._train_with_spark(_spark_df)

    # MARK: - Huấn luyện với Scikit-learn

    def _train_with_sklearn(self, processed_data: pd.DataFrame) -> Tuple[Any, float, float]:
        """
        Huấn luyện mô hình bằng scikit-learn khi Spark không khả dụng
        """
        try:
            # Sử dụng biến toàn cục để kiểm tra scikit-learn
            # sklearn_available đã được khai báo ở đầu tập tin

            if sklearn_available:
                # Chuẩn bị dữ liệu cho scikit-learn
                X = processed_data.drop(['price_per_sqm', 'price_million_vnd'], axis=1, errors='ignore')

                # Chọn label (giá) - ưu tiên price_per_sqm nếu có
                if 'price_per_sqm' in processed_data.columns:
                    y = processed_data['price_per_sqm']
                elif 'price' in processed_data.columns:
                    y = processed_data['price']
                else:
                    raise ValueError("Không tìm thấy cột giá (price hoặc price_per_sqm) trong dữ liệu")

                # Tạo bộ lọc cho các cột số (loại bỏ cột object/categorical)
                numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                X = X[numeric_cols]  # Chỉ sử dụng các cột số

                # Chia dữ liệu train/test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Huấn luyện mô hình
                fallback_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                fallback_model.fit(X_train, y_train)

                # Đánh giá mô hình
                y_pred = fallback_model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)

                # Lưu metrics
                self._model = fallback_model
                self._accuracy = r2
                self._r2 = r2
                self._rmse = rmse
                self._using_fallback = True
                self._fallback_model = fallback_model
                self._fallback_features = numeric_cols

                # Lưu metrics vào session state
                save_model_metrics(r2=r2, rmse=rmse)

                self.logger.info(f"Huấn luyện mô hình dự phòng với scikit-learn, R²: {r2:.4f}, RMSE: {rmse:.4f}")
                return fallback_model, r2, rmse

            else:
                # Sử dụng chế độ dự phòng rất đơn giản khi không có scikit-learn
                self._using_fallback = True
                self.logger.warning("❗ Không thể huấn luyện mô hình nâng cao. Sử dụng phương pháp tính trung bình đơn giản.")
                return None, 0.0, 0.0

        except Exception as e:
            st.error(f"Lỗi khi huấn luyện mô hình dự phòng: {e}")
            # Đã thiết lập giá trị mặc định cho metrics ở trên
            return None, 0.0, 0.0

    # MARK: - Huấn luyện với Spark ML

    def _train_with_spark(self, spark_df) -> Tuple[Any, float, float]:
        """
        Huấn luyện mô hình bằng Spark ML
        """
        try:
            # Định nghĩa các cột để sử dụng trong mô hình
            area_column = self._feature_columns['area']  # 'area (m2)'
            street_column = self._feature_columns['street']  # 'street (m)'

            # Đặc trưng số
            numeric_features = [area_column, "bedroom_num", "floor_num", "toilet_num", "livingroom_num", street_column]

            # Chỉ sử dụng các cột tồn tại trong dữ liệu
            numeric_features = [col for col in numeric_features if col in spark_df.columns]

            # Đặc trưng phân loại
            categorical_features = ["category", "direction", "liability", "district", "city_province"]

            # Loại trừ các đặc trưng không tồn tại trong dữ liệu
            categorical_features = [col for col in categorical_features if col in spark_df.columns]

            # Tạo onehot encoding cho các biến phân loại
            indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="keep")
                        for col in categorical_features]

            encoders = [OneHotEncoder(inputCol=col+"_index", outputCol=col+"_encoded")
                        for col in categorical_features]

            # Gộp tất cả các đặc trưng đã xử lý vào một vector
            assembler_inputs = numeric_features + [col+"_encoded" for col in categorical_features]

            assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="skip")

            # Tạo chuẩn hóa dữ liệu
            scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

            # Khởi tạo mô hình GBT
            gbt = GBTRegressor(featuresCol="scaled_features", labelCol="price_per_m2", maxIter=10)

            # Tạo pipeline
            pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, gbt])

            # Chia dữ liệu thành tập huấn luyện và kiểm tra
            train_data, test_data = spark_df.randomSplit([0.8, 0.2], seed=42)

            # Huấn luyện mô hình
            with st.spinner('Đang huấn luyện mô hình...'):
                model = pipeline.fit(train_data)

            # Đánh giá mô hình
            predictions = model.transform(test_data)

            # Tính toán các chỉ số đánh giá
            evaluator = RegressionEvaluator(labelCol="price_per_m2", predictionCol="prediction", metricName="rmse")
            rmse = evaluator.evaluate(predictions)

            evaluator.setMetricName("r2")
            r2 = evaluator.evaluate(predictions)

            # Hiển thị kết quả đánh giá
            self._rmse = rmse
            self._r2 = r2
            self._accuracy = r2  # For backwards compatibility

            # Đánh dấu đang sử dụng Spark
            self._using_fallback = False
            self._model = model

            # Lưu metrics vào session state để duy trì giữa các views
            save_model_metrics(r2=r2, rmse=rmse)

            # Log model metrics
            self.logger.info(f"Đã huấn luyện mô hình với Spark, R²: {r2:.4f}, RMSE: {rmse:.4f}")

            return model, r2, rmse

        except Exception as e:
            self.logger.error(f"Lỗi khi huấn luyện mô hình Spark: {e}")
            self._rmse = 0.0
            self._r2 = 0.0
            return None, 0.0, 0.0