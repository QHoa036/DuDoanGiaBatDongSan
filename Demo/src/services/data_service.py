#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dịch vụ Dữ liệu - Xử lý việc tải dữ liệu, xử lý và các hoạt động mô hình
"""

# MARK: - Thư viện

import os
import sys
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import streamlit as st
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from ..models.property_model import Property, PredictionResult
from ..utils.logger_utils import get_logger
from ..utils.spark_utils import get_spark_session, configure_spark_logging
from ..utils.session_utils import save_model_metrics, get_model_metrics, metrics_exist

# Trả về True nếu scikit-learn đã được cài đặt
sklearn_available = False
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    sklearn_available = True
except ImportError:
    st.warning("🔔 Thư viện scikit-learn không có sẵn. Cài đặt bằng cách chạy: pip install scikit-learn")
    st.info("📚 Sẽ sử dụng chế độ dự phòng đơn giản hơn.")

# MARK: - Cấu hình

utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils')
if utils_path not in sys.path:
    sys.path.append(utils_path)

# MARK: - Lớp dịch vụ dữ liệu

class DataService:
    """
    Lớp dịch vụ chịu trách nhiệm cho các hoạt động dữ liệu và chức năng ML
    Cung cấp các phương thức để tải, xử lý dữ liệu và các hoạt động mô hình
    """

    # MARK: - Khởi tạo

    def __init__(self):
        """
        Khởi tạo dịch vụ dữ liệu
        """
        configure_spark_logging()

        # Khởi tạo logger
        self.logger = get_logger(__name__)

        # Khởi tạo phiên Spark (lazy loading)
        self._spark = None
        self._model = None
        self._accuracy = 0.0
        self._rmse = 0.0
        self._r2 = 0.0
        self._data = None
        self._spark_df = None
        self._using_fallback = False
        self._fallback_model = None
        self._fallback_features = None

        # Define feature column names
        self._feature_columns = {
            'area': 'area (m2)',
            'street': 'street (m)'
        }

    # MARK: - Thuộc tính

    @property
    def spark(self):
        """
        Lấy phiên Spark (khởi tạo khi truy cập lần đầu)
        """
        if self._spark is None:
            self._spark = get_spark_session(app_name="VNRealEstatePricePrediction")
        return self._spark

    @property
    def model_metrics(self) -> Dict[str, float]:
        """
        Lấy các chỉ số của mô hình hiện tại
        """
        # Ưu tiên sử dụng metrics từ session state nếu có
        if metrics_exist():
            session_metrics = get_model_metrics()
            # Trả về metrics từ session state, đảm bảo trường 'accuracy' cho tương thích ngược
            metrics = {
                'r2': session_metrics.get('r2', 0.0),
                'accuracy': session_metrics.get('r2', 0.0),  # Đảm bảo tương thích ngược
                'rmse': session_metrics.get('rmse', 0.0)
            }
            return metrics

        # Nếu không có trong session state, sử dụng các giá trị trong đối tượng
        return {
            'r2': self._r2,
            'accuracy': self._accuracy,
            'rmse': self._rmse
        }

    # MARK: - Tải và xử lý dữ liệu

    @st.cache_data
    def load_data(_self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Tải dữ liệu từ file CSV
        """
        if _self._data is not None:
            return _self._data

        try:
            # Xác định đường dẫn file nếu không được cung cấp
            if file_path is None:
                # Tìm thư mục gốc của dự án
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                # Đường dẫn đến file dữ liệu trong thư mục src/data
                file_path = os.path.join(base_dir, 'data', 'final_data_cleaned.csv')

                # Kiểm tra xem file có tồn tại không
                if not os.path.exists(file_path):
                    # Thử tìm file ở vị trí khác
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    alternate_paths = [
                        os.path.join(project_root, 'Demo', 'data', 'final_data_cleaned.csv')
                    ]

                    for alt_path in alternate_paths:
                        if os.path.exists(alt_path):
                            file_path = alt_path
                            _self.logger.info(f"Tìm thấy file dữ liệu tại: {file_path}")
                            break
                    else:
                        # Nếu không tìm thấy file ở bất kỳ vị trí nào
                        raise FileNotFoundError(
                            f"❌ Không tìm thấy file dữ liệu tại: {file_path}\n"
                            "Vui lòng đảm bảo rằng:\n"
                            "1. Bạn đã tải dữ liệu và đặt trong thư mục Demo/data/\n"
                            "2. File được đặt tên chính xác là 'final_data_cleaned.csv'\n"
                            "3. Bạn đã chạy toàn bộ quy trình từ đầu bằng run_demo.sh"
                        )

            # Tải dữ liệu từ CSV
            data = pd.read_csv(file_path)

            # Kiểm tra nếu dữ liệu trống
            if data.empty:
                st.error(f"Không có dữ liệu trong file {file_path}")
                return pd.DataFrame()

            # Lưu dữ liệu để tái sử dụng
            _self._data = data

            # Sử dụng logger thay vì print
            _self.logger.info(f"Đã tải dữ liệu: {data.shape[0]} dòng và {data.shape[1]} cột")
            return data

        except FileNotFoundError as e:
            st.error(str(e))
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Lỗi khi tải dữ liệu từ {file_path}: {e}")
            return pd.DataFrame()

    @st.cache_data
    def preprocess_data(_self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Tiền xử lý dữ liệu cho phân tích và mô hình hóa
        """
        try:
            # Tạo bản sao để tránh cảnh báo của Pandas
            df = data.copy()

            # Đổi tên cột để dễ sử dụng (nếu chưa có)
            column_mapping = {
                'area (m2)': 'area_m2',
                'street (m)': 'street_width_m'
            }

            # Đảm bảo chúng ta có cả các cột cũ và mới
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    # Nếu cột cũ tồn tại, tạo cột mới dựa trên nó
                    df[new_name] = df[old_name]
                elif new_name not in df.columns and old_name not in df.columns:
                    # Nếu cả hai cột đều không tồn tại, hiển thị lỗi
                    _self.logger.warning(f"Không tìm thấy cột {old_name} hoặc {new_name} trong dữ liệu")

            # Xử lý giá trị thiếu
            numeric_cols = ["area (m2)", "bedroom_num", "floor_num", "toilet_num", "livingroom_num", "street (m)"]
            for col in numeric_cols:
                if col in df:
                    # Thay thế -1 (giá trị thiếu) bằng giá trị trung vị
                    median_val = df[df[col] != -1][col].median()
                    df[col] = df[col].replace(-1, median_val)

            # Chuyển đổi logarithm cho giá (nếu có cột giá)
            if 'price_per_m2' in df.columns:
                df['price_log'] = np.log1p(df['price_per_m2'])

            # Tính giá trên mét vuông nếu chưa có
            if 'price_per_sqm' not in df.columns and 'price' in df.columns and _self._feature_columns['area'] in df.columns:
                df['price_per_sqm'] = df['price'] / df[_self._feature_columns['area']]

            # Lọc các dòng với giá trị lỗi
            if 'price' in df.columns:
                df = df[df['price'] > 0]

            # Đảm bảo các cột số có kiểu dữ liệu đúng
            numeric_cols = [_self._feature_columns['area'], 'price'] if 'price' in df.columns else [_self._feature_columns['area']]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            _self.logger.info(f"Tiền xử lý dữ liệu hoàn tất: {df.shape[0]} dòng, {df.shape[1]} cột")
            return df

        except Exception as e:
            _self.logger.error(f"Lỗi khi tiền xử lý dữ liệu: {e}")
            return data  # Trả về dữ liệu gốc nếu có lỗi

    # MARK: - Chuyển đổi Spark

    @st.cache_resource
    def convert_to_spark(_self, data: pd.DataFrame):
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

            # For debugging - commented out
            # print(f"Các cột trong dữ liệu gốc trước khi chuyển đổi: {data.columns.tolist()}")

            # Đảm bảo dữ liệu có tất cả các cột cần thiết (cả tên cũ và mới)
            if 'area (m2)' in data.columns and 'area_m2' not in data.columns:
                data['area_m2'] = data['area (m2)'].copy()
            if 'street (m)' in data.columns and 'street_width_m' not in data.columns:
                data['street_width_m'] = data['street (m)'].copy()

            # Chuyển đổi DataFrame pandas sang DataFrame Spark
            spark_df = spark.createDataFrame(data)
            _self._spark_df = spark_df
            return spark_df

        except Exception as e:
            _self.logger.error(f"Lỗi khi chuyển đổi dữ liệu sang Spark: {e}")
            return None

    # MARK: - Huấn luyện mô hình

    @st.cache_resource
    def get_spark_session_cached(_self):
        """
        Phiên bản có cache của hàm khởi tạo Spark với cấu hình tối ưu và xử lý lỗi.
        """
        try:
            # Sử dụng tiện ích Spark đã cấu hình để giảm thiểu cảnh báo
            spark = get_spark_session(
                app_name="VNRealEstatePricePrediction",
                enable_hive=False
            )
            # Kiểm tra kết nối để đảm bảo Spark hoạt động
            spark.sparkContext.parallelize([1]).collect()
            _self._spark = spark
            return spark
        except Exception as e:
            _self.logger.error(f"Không thể khởi tạo Spark: {e}")
            return None

    @st.cache_resource
    def train_model(_self, _spark_df=None, force_retrain=False):
        """
        Huấn luyện mô hình học máy để dự đoán giá bất động sản
        """
        # Kiểm tra xem metrics có tồn tại trong session state không
        if metrics_exist() and not force_retrain:
            # Lấy metrics từ session state
            session_metrics = get_model_metrics()
            _self.logger.info("Sử dụng metrics từ session state: R² = {:.4f}, RMSE = {:.4f}".format(
                session_metrics.get('r2', 0.0), session_metrics.get('rmse', 0.0)
            ))
            # Cập nhật các metrics của DataService từ session state
            _self._r2 = session_metrics.get('r2', 0.0)
            _self._accuracy = _self._r2  # Đảm bảo tương thích ngược
            _self._rmse = session_metrics.get('rmse', 0.0)
            return _self._model, _self._accuracy, _self._rmse

        # Nếu đã có mô hình và không bắt buộc huấn luyện lại, trả về mô hình và các chỉ số hiện tại
        if _self._model is not None and not force_retrain:
            return _self._model, _self._accuracy, _self._rmse

        # Nếu chưa có dữ liệu, tải dữ liệu
        if _self._data is None:
            _self._data = _self.load_data()

        # Nếu không có dữ liệu, trả về None
        if _self._data is None or _self._data.empty:
            _self.logger.error("Không có dữ liệu để huấn luyện mô hình")
            return None, 0, 0

        # Tiền xử lý dữ liệu
        processed_data = _self.preprocess_data(_self._data)

        # Khởi tạo SparkSession
        spark = _self.get_spark_session_cached()

        # Nếu spark_df không được cung cấp, tạo mới
        if _spark_df is None:
            # Chuyển đổi dữ liệu sang Spark DataFrame
            _spark_df = _self.convert_to_spark(processed_data)

        # Kiểm tra nếu data_spark là None (khi Spark không khả dụng)
        if _spark_df is None:
            # Thiết lập giá trị metrics mặc định
            _self._rmse = 0.0
            _self._r2 = 0.0

            # Sử dụng fallback mode với scikit-learn
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
                    _self._model = fallback_model
                    _self._accuracy = r2
                    _self._r2 = r2
                    _self._rmse = rmse
                    _self._using_fallback = True
                    _self._feature_importance = dict(zip(X.columns, fallback_model.feature_importances_))

                    # Lưu metrics vào session state
                    save_model_metrics(r2=r2, rmse=rmse)

                    _self.logger.info(f"Huấn luyện mô hình dự phòng với scikit-learn, R²: {r2:.4f}, RMSE: {rmse:.4f}")
                    return fallback_model, r2, rmse
                else:
                    # Sử dụng chế độ dự phòng rất đơn giản khi không có scikit-learn
                    _self._using_fallback = True
                    _self.logger.warning("❗ Không thể huấn luyện mô hình nâng cao. Sử dụng phương pháp tính trung bình đơn giản.")
                    return None, 0.0, 0.0
            except Exception as e:
                st.error(f"Lỗi khi huấn luyện mô hình dự phòng: {e}")
                # Đã thiết lập giá trị mặc định cho metrics ở trên
                return None, 0.0, 0.0

        # Nếu có Spark DataFrame, huấn luyện với Spark ML
        try:
            # Định nghĩa các cột để sử dụng trong mô hình
            area_column = _self._feature_columns['area']  # 'area (m2)'
            street_column = _self._feature_columns['street']  # 'street (m)'

            # Đặc trưng số
            numeric_features = [area_column, "bedroom_num", "floor_num", "toilet_num", "livingroom_num", street_column]

            # Chỉ sử dụng các cột tồn tại trong dữ liệu
            numeric_features = [col for col in numeric_features if col in _spark_df.columns]

            # Đặc trưng phân loại
            categorical_features = ["category", "direction", "liability", "district", "city_province"]

            # Loại trừ các đặc trưng không tồn tại trong dữ liệu
            categorical_features = [col for col in categorical_features if col in processed_data.columns]

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
            train_data, test_data = _spark_df.randomSplit([0.8, 0.2], seed=42)

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
            _self._rmse = rmse
            _self._r2 = r2
            _self._accuracy = r2  # For backwards compatibility

            # Đánh dấu đang sử dụng Spark
            _self._using_fallback = False
            _self._model = model

            # Lưu metrics vào session state để duy trì giữa các views
            save_model_metrics(r2=r2, rmse=rmse)

            # Log model metrics
            _self.logger.info(f"Đã huấn luyện mô hình với Spark, R²: {r2:.4f}, RMSE: {rmse:.4f}")

            return model, r2, rmse

        except Exception as e:
            _self.logger.error(f"Lỗi khi huấn luyện mô hình Spark: {e}")
            # Thiết lập giá trị mặc định cho metrics
            _self._rmse = 0.0
            _self._r2 = 0.0
            return None, 0.0, 0.0

    # MARK: - Dự đoán giá

    @st.cache_data
    def predict_property_price(_self, property_data: Property) -> PredictionResult:
        """
        Dự đoán giá của một bất động sản sử dụng mô hình đã được huấn luyện
        """
        try:
            # Kiểm tra nếu mô hình dự phòng scikit-learn được sử dụng
            if _self._using_fallback and _self._fallback_model is not None:
                return _self._predict_price_fallback(property_data)

            # Kiểm tra nếu mô hình Spark không khả dụng
            spark = _self.get_spark_session_cached()
            if _self._model is None or spark is None:
                _self.logger.warning("Mô hình chưa được huấn luyện hoặc Spark không khả dụng, sử dụng phương pháp dự phòng")
                return _self._predict_price_fallback(property_data)

            # Chuẩn bị dữ liệu đầu vào cho dự đoán
            property_dict = property_data.to_dict()
            data_copy = {k: [v] for k, v in property_dict.items()}

            # Tạo pandas DataFrame
            import pandas as pd
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

                    comparison_data = _self._get_area_comparison(location)

                    # Tạo kết quả dự đoán
                    result = PredictionResult.create(
                        predicted_price=predicted_price,
                        predicted_price_per_sqm=predicted_price_per_sqm,
                        property_details=property_data.to_dict(),
                        comparison_data=comparison_data
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

    # MARK: - Phương pháp dự phòng

    def _predict_price_fallback(_self, property_data: Property) -> PredictionResult:
        """
        Phương pháp dự phòng đơn giản cho việc dự đoán giá khi mô hình Spark không khả dụng
        """
        try:
            import time

            # Lấy các thuộc tính cơ bản, với kiểm tra an toàn
            area = getattr(property_data, 'area', 0)
            if area <= 0:
                area = 50  # Diện tích mặc định nếu không có hoặc không hợp lệ

            location = getattr(property_data, 'location', '')
            num_rooms = getattr(property_data, 'num_rooms', 2)
            year_built = getattr(property_data, 'year_built', 2010)
            legal_status = getattr(property_data, 'legal_status', '')

            # Hiệu ứng loading để tạo trải nghiệm tốt hơn
            with st.spinner('Đang tính toán giá bất động sản...'):
                # Tạo chút delay cho hiệu ứng
                time.sleep(0.8)

                # Giá cơ bản dựa trên vị trí
                # Giá mặc định cho mỗi m2 là 30 triệu VND
                base_price_per_sqm = 30000000

                # Điều chỉnh theo vị trí
                location_factor = 1.0
                if location and any(premium_area in location.lower() for premium_area in ['quận 1', 'quận 3', 'thủ đức', 'bình thạnh', 'phú nhuận']):
                    location_factor = 1.5  # Khu vực cao cấp
                elif location and any(mid_area in location.lower() for mid_area in ['quận 2', 'quận 4', 'quận 7', 'quận 10', 'tân bình']):
                    location_factor = 1.2  # Khu vực trung bình
                elif location and any(low_area in location.lower() for low_area in ['quận 8', 'quận 9', 'quận 12', 'bình tân', 'tân phú']):
                    location_factor = 0.9  # Khu vực thấp hơn

                # Điều chỉnh theo số phòng ngủ
                room_factor = 1.0
                if num_rooms <= 1:
                    room_factor = 0.9
                elif num_rooms == 3:
                    room_factor = 1.1
                elif num_rooms >= 4:
                    room_factor = 1.2

                # Điều chỉnh theo năm xây dựng
                age_factor = 1.0
                current_year = 2025
                age = current_year - year_built if year_built > 0 else 10
                if age < 5:
                    age_factor = 1.15  # Mới xây
                elif age < 10:
                    age_factor = 1.05  # Khá mới
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

    # MARK: - Phân tích so sánh

    def _get_area_comparison(self, location: str) -> Dict[str, Any]:
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

    # MARK: - Đặc trưng quan trọng

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Lấy mức độ quan trọng của các đặc trưng trong mô hình đã được huấn luyện
        """
        try:
            if self._model is None:
                return {}

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
            print(f"Lỗi khi trích xuất tầm quan trọng của đặc trưng: {e}")
            return {}
