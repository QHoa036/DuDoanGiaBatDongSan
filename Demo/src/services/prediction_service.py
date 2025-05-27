#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dịch vụ Dự đoán - Chịu trách nhiệm dự đoán giá bất động sản
"""

# MARK: - Thư viện

import os
import sys
import pandas as pd
import numpy as np
import time
import streamlit as st

from datetime import datetime
from typing import Optional, Dict

from ..models.property_model import Property, PredictionResult
from ..utils.session_utils import save_model_metrics, get_model_metrics, metrics_exist
from ..utils.logger_utils import get_logger, MetricsLogger
from .interfaces.prediction_interface import IPredictionService
from .interfaces.progress_data_interface import IProgressDataService
from .interfaces.metrics_interface import IMetricsService
from .core.base_service import BaseService

# MARK: - Cấu hình

utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils')
if utils_path not in sys.path:
    sys.path.append(utils_path)

# MARK: - Lớp dịch vụ dự đoán

class PredictionService(BaseService, IPredictionService):
    """
    Lớp dịch vụ chịu trách nhiệm dự đoán giá bất động sản
    Triển khai interface IPredictionService
    Kế thừa từ BaseService để sử dụng các tính năng chung
    """

    # MARK: - Khởi tạo

    def __init__(self, data_service: Optional[IProgressDataService] = None, model=None, fallback_model=None,
                models_dir: str = None, data_dir: str = None, metrics_service: IMetricsService = None):
        """
        Khởi tạo dịch vụ dự đoán
        """
        # Gọi khởi tạo lớp cơ sở
        super().__init__(service_name="PredictionService")

        # Khởi tạo logger tiêu chuẩn
        self._logger = get_logger("prediction_service")
        self._logger.info("Khởi tạo PredictionService")

        # Đường dẫn thư mục
        self._models_dir = models_dir if models_dir else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
        self._data_dir = data_dir if data_dir else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')

        # Import động để tránh circular import
        from .core.services_factory import ServicesFactory

        # Khởi tạo metrics logger
        self._metrics_service = metrics_service if metrics_service else ServicesFactory.get_metrics_service()
        self._metrics_logger = MetricsLogger("prediction", self._metrics_service)

        # Dịch vụ dữ liệu (nếu được cung cấp)
        self._data_service = data_service if data_service else ServicesFactory.get_progress_data_service()

        # Các thuộc tính mô hình
        self._model = model if model else self.load_model("gbt_model")
        self._using_fallback = fallback_model is not None
        self._fallback_model = fallback_model if fallback_model else self.load_model("fallback_model")
        self._fallback_features = None

        # Khởi tạo các biến theo dõi dự đoán
        self._prediction_count = 0
        self._prediction_accuracy = 0.0
        self._last_predictions = []

        # Tải thông tin dự đoán đã lưu (nếu có)
        stored_predictions = self.load_state("prediction_history")
        if stored_predictions:
            self._logger.info("Tải lịch sử dự đoán đã lưu")
            self._last_predictions = stored_predictions

        # Define feature column names
        self._feature_columns = {
            'area': 'area (m2)',
            'street': 'street (m)'
        }

    @property
    def model_metrics(self) -> Dict[str, float]:
        """
        Lấy metrics của mô hình dự đoán

        Returns:
            Dict[str, float]: Dictionary chứa các metrics như accuracy và số lần dự đoán
        """
        # Lấy mô hình hiện tại đang sử dụng
        model_name = "fallback_model" if self._using_fallback else "gbt_model"

        # Ưu tiên lấy metrics từ MetricsService
        prediction_metrics = self._metrics_service.get_metrics(f"{model_name}_predictions")
        if prediction_metrics:
            return {k: v for k, v in prediction_metrics.items() if not k.startswith('_')}

        # Nếu không có trong MetricsService, kiểm tra session state
        if metrics_exist():
            session_metrics = get_model_metrics()

            # Cập nhật vào MetricsService
            self._metrics_service.add_metrics(f"{model_name}_predictions", session_metrics)
            return session_metrics

        # Nếu không có dữ liệu nào, trả về metrics mặc định và lưu vào MetricsService
        default_metrics = {
            'prediction_count': self._prediction_count,
            'accuracy': self._prediction_accuracy,
            'last_prediction_time': datetime.now().isoformat()
        }

        # Lưu trữ metrics mặc định
        self.save_state("prediction_metrics", default_metrics)
        self._metrics_service.add_metrics(f"{model_name}_predictions", default_metrics)

        return default_metrics

    # MARK: - Dự đoán giá

    @st.cache_data
    def predict_property_price(_self, property_data: Property) -> PredictionResult:
        """
        Dự đoán giá của một bất động sản sử dụng mô hình đã được huấn luyện
        """
        _self._logger.info(f"Bắt đầu dự đoán giá cho bất động sản: {property_data.location} - {property_data.property_type}")

        try:
            # Kiểm tra nếu mô hình dự phòng scikit-learn được sử dụng
            if _self._using_fallback and _self._fallback_model is not None:
                _self._logger.info("Sử dụng mô hình dự phòng (fallback model)")
                return _self._predict_price_fallback(property_data)

            # Kiểm tra nếu mô hình Spark không khả dụng
            spark = _self._data_service.get_spark_session_cached()
            if _self._model is None or spark is None:
                _self._logger.warning("Mô hình chưa được huấn luyện hoặc Spark không khả dụng, sử dụng phương pháp dự phòng")
                return _self._predict_price_fallback(property_data)

            # Chuẩn bị dữ liệu đầu vào cho dự đoán
            _self._logger.debug("Chuẩn bị dữ liệu đầu vào cho dự đoán")
            property_dict = property_data.to_dict()
            _self._logger.debug(f"Thuộc tính bất động sản: {property_dict}")
            data_copy = {k: [v] for k, v in property_dict.items()}

            # Tạo pandas DataFrame
            input_df = pd.DataFrame(data_copy)

            # Sao chép dữ liệu để không ảnh hưởng đến dữ liệu gốc
            data_copy = input_df.copy()

            # Xử lý các giá trị không tồn tại
            for col in data_copy.columns:
                if col in ["bedroom_num", "floor_num", "toilet_num", "livingroom_num"]:
                    data_copy[col] = data_copy[col].fillna(-1).astype(int)

            # Đảm bảo chúng ta có các cột đúng tên chính xác
            # Đảm bảo không sử dụng area_m2 mà sử dụng 'area (m2)'
            if 'area_m2' in data_copy.columns and 'area (m2)' not in data_copy.columns:
                data_copy['area (m2)'] = data_copy['area_m2'].copy()
                del data_copy['area_m2']
            elif 'area' in data_copy.columns and 'area (m2)' not in data_copy.columns and 'area_m2' not in data_copy.columns:
                data_copy['area (m2)'] = data_copy['area'].copy()

            # Đảm bảo không sử dụng street_width_m mà sử dụng 'street (m)'
            if 'street_width_m' in data_copy.columns and 'street (m)' not in data_copy.columns:
                data_copy['street (m)'] = data_copy['street_width_m'].copy()
                del data_copy['street_width_m']

            # Chuyển đổi dữ liệu sang Spark DataFrame
            try:
                # Chuyển đổi dữ liệu sang Spark DataFrame
                spark_df = spark.createDataFrame(data_copy)

                # Dự đoán giá với hiệu ứng hiển thị
                with st.spinner('Đang dự đoán giá bất động sản...'):
                    predictions = _self._model.transform(spark_df)

                    # Lấy kết quả dự đoán
                    prediction_value = predictions.select("prediction").collect()[0][0]

                # Nếu kết quả hợp lệ
                if prediction_value is not None:
                    # Tính giá trên mét vuông
                    area = property_data.area
                    predicted_price_per_sqm = prediction_value
                    predicted_price = prediction_value * area

                    # Xác thực và lấy dữ liệu so sánh khu vực
                    # Đảm bảo rằng location là hợp lệ trước khi sử dụng
                    if hasattr(property_data, 'location') and property_data.location is not None:
                        location = property_data.location
                    else:
                        location = ""
                        _self.logger.warning("Vị trí không hợp lệ hoặc trống, sử dụng chuỗi rỗng")

                    comparison_data = self._data_service.get_area_comparison(location)

                    # Tạo kết quả dự đoán
                    result = PredictionResult.create(
                        predicted_price=predicted_price,
                        predicted_price_per_sqm=predicted_price_per_sqm,
                        property_details=property_data.to_dict(),
                        comparison_data=comparison_data
                    )

                    # Cập nhật metrics và lưu trữ
                    self._prediction_count += 1
                    accuracy = 0.95  # Giá trị mẫu, có thể được cập nhật dựa trên dữ liệu thực tế

                    # Thêm dự đoán vào lịch sử
                    prediction_record = {
                        'price': float(predicted_price),
                        'price_per_sqm': float(predicted_price_per_sqm),
                        'area': float(area),
                        'property_type': property_data.property_type,
                        'location': location,
                        'timestamp': datetime.now().isoformat()
                    }

                    # Giới hạn số lượng dự đoán được lưu trữ (giữ 20 dự đoán gần nhất)
                    self._last_predictions.append(prediction_record)
                    if len(self._last_predictions) > 20:
                        self._last_predictions = self._last_predictions[-20:]

                    # Lưu lịch sử dự đoán vào cache
                    self.save_state("prediction_history", self._last_predictions)

                    # Cập nhật metrics trong MetricsService
                    model_name = "fallback_model" if self._using_fallback else "gbt_model"
                    current_metrics = self._metrics_service.get_metrics(f"{model_name}_predictions") or {}

                    updated_metrics = {
                        'prediction_count': self._prediction_count,
                        'accuracy': accuracy,
                        'last_prediction_time': datetime.now().isoformat(),
                        'total_predictions': current_metrics.get('total_predictions', 0) + 1
                    }

                    self._metrics_service.add_metrics(f"{model_name}_predictions", updated_metrics)

                    # Lưu kết quả dự đoán vào session cho tương thích ngược
                    save_model_metrics(
                        prediction_count=self._prediction_count,
                        accuracy=accuracy
                    )

                    return result
                else:
                    # Sử dụng phương pháp dự phòng nếu giá trị dự đoán là None
                    st.warning("Kết quả dự đoán không hợp lệ, sử dụng phương pháp dự phòng.")
                    return _self._predict_price_fallback(property_data)
            except Exception as e:
                st.warning(f"Lỗi khi dự đoán với Spark: {e}. Sử dụng phương pháp dự phòng.")
                return _self._predict_price_fallback(property_data)
        except Exception as e:
            _self.logger.error(f"Lỗi khi dự đoán giá: {e}")
            # Fallback to traditional calculation
            return _self._predict_price_fallback(property_data)

    # MARK: - Đặc trưng quan trọng

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Lấy mức độ quan trọng của các đặc trưng trong mô hình đã được huấn luyện
        """
        try:
            if self._model is None:
                return {}

            # Nếu đang sử dụng scikit-learn
            if self._using_fallback and self._fallback_model is not None:
                # Lấy feature_importances_ từ scikit-learn model
                importances = self._fallback_model.feature_importances_
                feature_cols = self._fallback_features
                importance_dict = {
                    feature: importance
                    for feature, importance in zip(feature_cols, importances)
                }
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

            # Sử dụng PySpark model
            # Trích xuất mô hình GBT từ pipeline
            gbt_model = self._model.stages[-1]

            # Lấy mức độ quan trọng của đặc trưng
            feature_importance = gbt_model.featureImportances

            # Lấy tên các đặc trưng
            feature_cols = [
                'area',
                'num_rooms',
                'year_built',
                'location_encoded',
                'legal_status_encoded',
                'house_direction_encoded'
            ]

            # Tạo từ điển mức độ quan trọng của đặc trưng
            importance_dict = {
                feature: importance
                for feature, importance in zip(feature_cols, feature_importance.toArray())
            }

            # Sắp xếp theo mức độ quan trọng (giảm dần)
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        except Exception as e:
            self.logger.error(f"Lỗi khi trích xuất tầm quan trọng của đặc trưng: {e}")
            return {}

    def _predict_price_fallback(self, property_data: Property) -> PredictionResult:
        """
        Phương pháp dự phòng đơn giản cho việc dự đoán giá khi mô hình Spark không khả dụng
        """
        try:
            # Nếu có mô hình scikit-learn, sử dụng nó để dự đoán
            if self._fallback_model is not None and self._fallback_features is not None:
                # Chuẩn bị dữ liệu cho dự đoán
                property_dict = property_data.to_dict()

                # Tạo một dictionary cho dữ liệu đầu vào
                input_data = {}

                # Ánh xạ các đặc trưng của bất động sản đến các đặc trưng của mô hình
                for feature in self._fallback_features:
                    if feature in property_dict:
                        input_data[feature] = property_dict[feature]
                    elif feature == 'area (m2)' and 'area' in property_dict:
                        input_data[feature] = property_dict['area']
                    elif feature == 'street (m)' and 'street_width_m' in property_dict:
                        input_data[feature] = property_dict['street_width_m']
                    else:
                        # Giá trị mặc định cho các đặc trưng không có trong dữ liệu đầu vào
                        input_data[feature] = 0

                # Tạo mảng đặc trưng đầu vào
                feature_array = np.array([[input_data[feature] for feature in self._fallback_features]])

                # Dự đoán giá
                with st.spinner('Đang dự đoán giá bất động sản (phương pháp scikit-learn)...'):
                    predicted_price_per_sqm = self._fallback_model.predict(feature_array)[0]

                # Tính toán giá tổng
                area = property_data.area
                predicted_price = predicted_price_per_sqm * area

                # Tạo dữ liệu so sánh
                location = getattr(property_data, 'location', "")
                comparison_data = self._data_service.get_area_comparison(location) if self._data_service else {}

                # Tạo kết quả dự đoán
                result = PredictionResult.create(
                    predicted_price=predicted_price,
                    predicted_price_per_sqm=predicted_price_per_sqm,
                    property_details=property_data.to_dict(),
                    comparison_data=comparison_data
                )

                return result

            # Nếu không có mô hình nào, sử dụng phương pháp ước lượng đơn giản
            self.logger.info("Sử dụng phương pháp ước lượng đơn giản cho dự đoán giá")

            # Lấy các thông tin từ property_data
            area = getattr(property_data, 'area', 0)
            if area <= 0:
                area = 50  # Giá trị mặc định

            num_rooms = getattr(property_data, 'num_rooms', 0)
            if num_rooms <= 0:
                num_rooms = 2  # Giá trị mặc định

            location = getattr(property_data, 'location', "")
            house_direction = getattr(property_data, 'house_direction', "")
            legal_status = getattr(property_data, 'legal_status', "")
            year_built = getattr(property_data, 'year_built', 0)

            # Tính toán giá dựa trên các yếu tố
            # Giá cơ bản: 30 triệu/m2
            base_price_per_sqm = 30000000

            # Điều chỉnh theo vị trí
            location_factor = 1.0
            if location:
                # Điều chỉnh theo khu vực
                premium_districts = ['quận 1', 'quận 3', 'quận 7', 'thủ đức']
                mid_districts = ['quận 2', 'quận 4', 'quận 10', 'quận phú nhuận', 'quận bình thạnh']

                if any(district in location.lower() for district in premium_districts):
                    location_factor = 1.5  # Khu vực cao cấp
                elif any(district in location.lower() for district in mid_districts):
                    location_factor = 1.2  # Khu vực trung bình

            # Điều chỉnh theo số phòng
            room_factor = 1.0
            if num_rooms >= 3:
                room_factor = 1.1  # Nhiều phòng

            # Điều chỉnh theo năm xây dựng
            age_factor = 1.0
            if year_built > 0:
                current_year = 2023
                age = current_year - year_built

                if age < 5:
                    age_factor = 1.2  # Rất mới
                elif age < 10:
                    age_factor = 1.1  # Mới
                elif age > 20:
                    age_factor = 0.9   # Cũ

            # Điều chỉnh theo tình trạng pháp lý
            legal_factor = 1.0
            if legal_status and any(good_status in legal_status.lower() for good_status in ['sổ đỏ', 'sổ hồng', 'chính chủ']):
                legal_factor = 1.1

            # Tính giá cuối cùng
            adjusted_price_per_sqm = base_price_per_sqm * location_factor * room_factor * age_factor * legal_factor
            total_price = adjusted_price_per_sqm * area

            # Tạo dữ liệu so sánh (hoặc để trống nếu không cần thiết)
            comparison_data = {
                'area_avg_price': adjusted_price_per_sqm * 0.9,  # Giả lập giá trung bình khu vực
                'location': location
            }

            # Tạo và trả về kết quả
            result = PredictionResult.create(
                predicted_price=total_price,
                predicted_price_per_sqm=adjusted_price_per_sqm,
                property_details=property_data.to_dict(),
                comparison_data=comparison_data
            )

            return result

        except Exception as e:
            # Xử lý lỗi và trả về giá trị mặc định an toàn
            st.error(f"Lỗi khi dự đoán giá (phương pháp đơn giản): {str(e)}")

            # Tính toán giá mặc định dựa trên diện tích
            default_area = getattr(property_data, 'area', 50)
            if default_area <= 0:
                default_area = 50

            default_price_per_sqm = 30000000  # 30 triệu VND/m2
            default_price = default_price_per_sqm * default_area

            # Trả về kết quả mặc định
            return PredictionResult.create(
                predicted_price=default_price,
                predicted_price_per_sqm=default_price_per_sqm,
                property_details=property_data.to_dict(),
                comparison_data={}
            )

    # MARK: - Đặc trưng quan trọng

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Lấy mức độ quan trọng của các đặc trưng trong mô hình đã được huấn luyện
        """
        try:
            if self._model is None:
                return {}

            # Nếu đang sử dụng scikit-learn
            if self._using_fallback and self._fallback_model is not None:
                # Lấy feature_importances_ từ scikit-learn model
                importances = self._fallback_model.feature_importances_
                feature_cols = self._fallback_features
                importance_dict = {
                    feature: importance
                    for feature, importance in zip(feature_cols, importances)
                }
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

            # Sử dụng PySpark model
            # Trích xuất mô hình GBT từ pipeline
            gbt_model = self._model.stages[-1]

            # Lấy mức độ quan trọng của đặc trưng
            feature_importance = gbt_model.featureImportances

            # Lấy tên các đặc trưng
            feature_cols = [
                'area',
                'num_rooms',
                'year_built',
                'location_encoded',
                'legal_status_encoded',
                'house_direction_encoded'
            ]

            # Tạo từ điển mức độ quan trọng của đặc trưng
            importance_dict = {
                feature: importance
                for feature, importance in zip(feature_cols, feature_importance.toArray())
            }

            # Sắp xếp theo mức độ quan trọng (giảm dần)
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        except Exception as e:
            self.logger.error(f"Lỗi khi trích xuất tầm quan trọng của đặc trưng: {e}")
            return {}
