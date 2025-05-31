# MARK: - Imports và khởi tạo

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

# Spark imports
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

from src.config.app_config import AppConfig
from src.utils.spark_utils import get_spark_session
from src.utils.logger_util import get_logger
from src.models.real_estate_model import ModelMetrics, PredictionResult

# Initialize logger
logger = get_logger()

# MARK: - Dịch vụ mô hình

class ModelService:
    """
    Service xử lý huấn luyện và dự đoán mô hình
    """
    # MARK: - Khởi tạo

    def __init__(self, _data_service=None, model_dir=None, using_spark=True):
        """
        Khởi tạo ModelService

        Args:
            _data_service: Data service (với dấu gạch dưới để Streamlit không cache)
            model_dir: Thư mục chứa mô hình đã huấn luyện
            using_spark: Có sử dụng Spark hay không
        """
        # Tham số có dấu gạch dưới (_) để Streamlit không hash nó
        self._data_service = _data_service
        self.model_dir = model_dir
        self.using_spark = using_spark

    # MARK: - Quản lý Spark

    @st.cache_resource
    def get_spark_session_cached(self):
        """
        Phiên bản có cache của hàm khởi tạo Spark với cấu hình tối ưu và xử lý lỗi
        """
        try:
            # Sử dụng tiện ích Spark đã cấu hình để giảm thiểu cảnh báo
            spark = get_spark_session(
                app_name=AppConfig.SPARK_APP_NAME,
                enable_hive=AppConfig.SPARK_ENABLE_HIVE
            )
            # Kiểm tra kết nối để đảm bảo Spark hoạt động
            spark.sparkContext.parallelize([1]).collect()
            return spark
        except Exception as e:
            logger.error(f"Không thể khởi tạo Spark session: {e}")
            return None

    def convert_to_spark(self, data: pd.DataFrame):
        """
        Chuyển đổi DataFrame pandas sang DataFrame Spark

        Args:
            data (pd.DataFrame): DataFrame pandas cần chuyển đổi

        Returns:
            pyspark.sql.DataFrame: DataFrame Spark tương ứng, hoặc None nếu có lỗi
        """
        try:
            spark = self.get_spark_session_cached()
            if spark is None:
                logger.warning("Không thể khởi tạo Spark session, trả về None")
                return None

            # Chuyển đổi sang Spark DataFrame
            spark_df = spark.createDataFrame(data)
            logger.info(f"Đã chuyển đổi thành công DataFrame sang Spark DataFrame")
            return spark_df
        except Exception as e:
            logger.error(f"Lỗi khi chuyển đổi DataFrame sang Spark DataFrame: {e}")
            return None

    # MARK: - Huấn luyện mô hình

    def train_model(self, data: pd.DataFrame) -> Tuple[Any, ModelMetrics]:
        """
        Huấn luyện mô hình dự đoán giá bất động sản

        Args:
            data (pd.DataFrame): DataFrame đã được tiền xử lý

        Returns:
            Tuple[Any, ModelMetrics]: Mô hình đã huấn luyện và các chỉ số đánh giá
        """
        try:
            # Chuyển đổi DataFrame pandas sang DataFrame Spark
            spark_df = self.convert_to_spark(data)

            if spark_df is None:
                # Sử dụng fallback (dự phòng) nếu không thể khởi tạo Spark
                logger.warning("Sử dụng fallback model do không thể khởi tạo Spark")
                return self._train_fallback_model(data)

            # Tách dữ liệu thành tập huấn luyện và tập kiểm tra
            splits = spark_df.randomSplit([0.8, 0.2], seed=42)
            train_data = splits[0]
            test_data = splits[1]

            # Chuẩn bị dữ liệu
            # Lựa chọn các cột đặc trưng cho mô hình
            feature_cols = [
                "area", "bedroom_num", "floor_num", "toilet_num",
                "livingroom_num", "street", "built_year"
            ]

            # Thêm mã hóa one-hot cho các đặc trưng phân loại
            categorical_cols = ["category", "district", "direction", "legal_status"]
            for col in categorical_cols:
                # Mã hóa one-hot cho từng cột phân loại
                indexer = StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep")
                encoder = OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_vec")
                feature_cols.extend([f"{col}_vec"])

            # Tạo vector đặc trưng
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

            # Chuẩn hóa đặc trưng
            scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

            # Tạo mô hình Gradient Boosted Trees
            gbt = GBTRegressor(
                featuresCol="scaled_features",
                labelCol="price_per_m2",
                maxIter=100,
                maxDepth=5,
                stepSize=0.1
            )

            # Tạo pipeline
            pipeline = Pipeline(stages=[assembler, scaler, gbt])

            # Huấn luyện mô hình
            logger.info("Bắt đầu huấn luyện mô hình Spark GBT")
            model = pipeline.fit(train_data)
            logger.info("Đã huấn luyện xong mô hình Spark GBT")

            # Đánh giá mô hình
            predictions = model.transform(test_data)

            # Tính toán các chỉ số
            evaluator_r2 = RegressionEvaluator(
                labelCol="price_per_m2", predictionCol="prediction", metricName="r2"
            )
            evaluator_rmse = RegressionEvaluator(
                labelCol="price_per_m2", predictionCol="prediction", metricName="rmse"
            )
            evaluator_mae = RegressionEvaluator(
                labelCol="price_per_m2", predictionCol="prediction", metricName="mae"
            )

            r2 = evaluator_r2.evaluate(predictions)
            rmse = evaluator_rmse.evaluate(predictions)
            mae = evaluator_mae.evaluate(predictions)

            # Lưu các chỉ số vào ModelMetrics
            metrics = ModelMetrics(
                r2=r2,
                rmse=rmse,
                mae=mae,
                mape=0.0  # MAPE cần tính riêng
            )

            logger.info(f"Đánh giá mô hình - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

            return model, metrics

        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện mô hình: {e}")
            # Sử dụng fallback (dự phòng) nếu có lỗi
            return self._train_fallback_model(data)

    def _train_fallback_model(self, data: pd.DataFrame) -> Tuple[Any, ModelMetrics]:
        """
        Huấn luyện mô hình dự phòng khi không thể sử dụng Spark

        Args:
            data (pd.DataFrame): DataFrame đã được tiền xử lý

        Returns:
            Tuple[Any, ModelMetrics]: Mô hình dự phòng đã huấn luyện và các chỉ số đánh giá
        """
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

            # Khởi tạo session state cho mô hình fallback
            st.session_state.using_fallback = True

            # Xác định các cột đặc trưng
            numeric_features = [
                "area", "bedroom_num", "floor_num", "toilet_num",
                "livingroom_num", "street", "built_year"
            ]
            categorical_features = ["category", "district", "direction", "legal_status"]

            # Lưu danh sách đặc trưng vào session state
            st.session_state.fallback_features = numeric_features + categorical_features

            # Tiền xử lý
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ])

            # Tạo pipeline với GradientBoostingRegressor
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                ))
            ])

            # Chuẩn bị dữ liệu
            X = data[numeric_features + categorical_features]
            y = data["price_per_m2"]

            # Kiểm tra xem có nên sử dụng log transform không
            use_log = False
            if y.min() > 0 and y.skew() > 1:
                y = np.log1p(y)
                use_log = True

            # Lưu thông tin về việc sử dụng log transform
            st.session_state.fallback_uses_log = use_log

            # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Huấn luyện mô hình
            model.fit(X_train, y_train)

            # Lưu mô hình vào session state
            st.session_state.model = model

            # Đánh giá mô hình
            y_pred = model.predict(X_test)

            # Chuyển đổi từ log nếu cần
            if use_log:
                y_test_original = np.expm1(y_test)
                y_pred_original = np.expm1(y_pred)

                # Tính toán các chỉ số trên giá trị gốc
                r2 = r2_score(y_test_original, y_pred_original)
                rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
                mae = mean_absolute_error(y_test_original, y_pred_original)
            else:
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)

            # Tính MAPE
            if use_log:
                # Tránh chia cho 0
                mask = y_test_original != 0
                mape = np.mean(np.abs((y_test_original[mask] - y_pred_original[mask]) / y_test_original[mask])) * 100
            else:
                mask = y_test != 0
                mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100

            # Lưu các chỉ số vào ModelMetrics
            metrics = ModelMetrics(
                r2=r2,
                rmse=rmse,
                mae=mae,
                mape=mape
            )

            logger.info(f"Đánh giá mô hình fallback - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}")

            return model, metrics

        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện mô hình fallback: {e}")
            return None, ModelMetrics()

    # MARK: - Dự đoán

    def predict(self, input_data: Dict[str, Any], data: pd.DataFrame) -> PredictionResult:
        """
        Dự đoán giá bất động sản dựa trên đầu vào của người dùng

        Args:
            input_data (Dict[str, Any]): Dictionary chứa thông tin bất động sản
            data (pd.DataFrame): DataFrame gốc dùng cho phương pháp thống kê nếu cần

        Returns:
            PredictionResult: Kết quả dự đoán giá
        """
        try:
            # Kiểm tra xem có thể sử dụng Spark không
            spark = self.get_spark_session_cached()

            if spark is not None:
                return self._predict_with_spark(input_data)
            else:
                return self._predict_fallback(input_data, data)

        except Exception as e:
            logger.error(f"Lỗi khi dự đoán giá bất động sản: {e}")
            return self._predict_statistical(input_data, data)

    # MARK: - Phương thức dự đoán Spark

    def _predict_with_spark(self, input_data: Dict[str, Any]) -> PredictionResult:
        """
        Dự đoán giá bất động sản sử dụng mô hình Spark

        Args:
            input_data (Dict[str, Any]): Dictionary chứa thông tin bất động sản

        Returns:
            PredictionResult: Kết quả dự đoán giá
        """
        try:
            # Kiểm tra xem có mô hình trong session state không
            if 'spark_model' not in st.session_state:
                raise ValueError("Chưa có mô hình Spark được huấn luyện")

            model = st.session_state.spark_model

            # Chuyển đổi input_data thành Spark DataFrame
            spark = self.get_spark_session_cached()
            input_df = spark.createDataFrame([input_data])

            # Dự đoán giá
            prediction = model.transform(input_df)
            price_per_m2 = prediction.select("prediction").collect()[0][0]

            # Tính giá tổng
            area = float(input_data.get("area", 0))
            predicted_price = price_per_m2 * area if area > 0 else 0

            # Tạo kết quả dự đoán
            return PredictionResult(
                predicted_price=predicted_price,
                confidence_level=0.8,  # Mặc định độ tin cậy 80%
                price_range_low=predicted_price * 0.9,
                price_range_high=predicted_price * 1.1
            )

        except Exception as e:
            logger.error(f"Lỗi khi dự đoán với Spark: {e}")
            return None

    # MARK: - Phương thức dự đoán dự phòng

    def _predict_fallback(self, input_data: Dict[str, Any], data: pd.DataFrame) -> PredictionResult:
        """
        Dự đoán giá sử dụng mô hình dự phòng (fallback) khi không có Spark

        Args:
            input_data (Dict[str, Any]): Dictionary chứa thông tin bất động sản
            data (pd.DataFrame): DataFrame gốc dùng cho phương pháp thống kê nếu cần

        Returns:
            PredictionResult: Kết quả dự đoán giá
        """
        try:
            # Kiểm tra xem có sẵn mô hình dự phòng trong session_state không
            if ('model' in st.session_state and
                st.session_state.using_fallback and
                'fallback_features' in st.session_state and
                'fallback_uses_log' in st.session_state):

                # Chuẩn bị dữ liệu đầu vào
                data_copy = {k: [v] for k, v in input_data.items()}
                input_df = pd.DataFrame(data_copy)

                # Đảm bảo tên cột đúng định dạng
                if 'area' in input_df.columns and 'area (m2)' not in input_df.columns:
                    # Xử lý chuỗi rỗng cho trường area
                    input_df['area'] = input_df['area'].apply(lambda x: 0 if x == '' else x)
                    input_df['area (m2)'] = input_df['area']

                if 'street' in input_df.columns and 'street (m)' not in input_df.columns:
                    # Xử lý chuỗi rỗng cho trường street
                    input_df['street'] = input_df['street'].apply(lambda x: 0 if x == '' else x)
                    input_df['street (m)'] = input_df['street']

                # Xử lý các giá trị số
                for col in input_df.columns:
                    if col in ["bedroom_num", "floor_num", "toilet_num", "livingroom_num"]:
                        # Xử lý trường hợp chuỗi rỗng trước khi chuyển đổi sang kiểu số
                        input_df[col] = input_df[col].apply(lambda x: -1 if x == '' else x)
                        input_df[col] = input_df[col].fillna(-1).astype(int)

                # Đảm bảo tất cả các cột cần thiết đều có
                all_features = st.session_state.fallback_features
                for col in all_features:
                    if col not in input_df.columns:
                        # Nếu là cột số, điền giá trị -1
                        if col in ["bedroom_num", "floor_num", "toilet_num", "livingroom_num", "area (m2)", "street (m)"]:
                            input_df[col] = -1
                        else:  # Nếu là cột phân loại, điền giá trị rỗng
                            input_df[col] = ''

                # Xử lý toàn bộ các cột có thể chứa giá trị số
                numeric_columns = AppConfig.NUMERIC_COLUMNS

                # Xử lý chuỗi rỗng và chuyển đổi kiểu dữ liệu cho mỗi cột
                for col in input_df.columns:
                    # Đối với các cột số, thay thế chuỗi rỗng bằng giá trị mặc định
                    if any(num_col in col for num_col in numeric_columns):
                        # Thay thế chuỗi rỗng bằng 0 hoặc -1 tùy vào loại cột
                        default_value = -1 if col in ["bedroom_num", "floor_num", "toilet_num", "livingroom_num"] else 0
                        input_df[col] = input_df[col].apply(lambda x: default_value if x == '' else x)

                        # Đảm bảo được chuyển đổi kiểu số đúng
                        if col in ["bedroom_num", "floor_num", "toilet_num", "livingroom_num"]:
                            input_df[col] = input_df[col].astype(int, errors='ignore')
                        else:
                            input_df[col] = input_df[col].astype(float, errors='ignore')

                # Lấy mô hình từ session state
                model = st.session_state.model

                # Nếu model là một pipeline, sử dụng predict trực tiếp
                if hasattr(model, 'predict'):
                    # Xử lý lần cuối và CHUẨN HÓA KIỂU DỮ LIỆU một cách nghiêm ngặt
                    # Các cột được xác định rõ ràng về kiểu dữ liệu
                    numeric_columns = AppConfig.NUMERIC_COLUMNS
                    categorical_columns = AppConfig.CATEGORICAL_COLUMNS

                    # Chuẩn hóa tất cả các cột số sang float64 hoặc int64
                    for col in input_df.columns:
                        if any(num_col in col for num_col in numeric_columns):
                            # Chuyển đổi mọi chuỗi rỗng thành NaN, sau đó điền giá trị 0
                            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

                            # Kiểu int cho các trường đếm
                            if col in ["bedroom_num", "floor_num", "toilet_num", "livingroom_num", "built_year"]:
                                input_df[col] = input_df[col].astype(np.int32)
                            else:
                                # Đảm bảo kiểu float64 cho các cột khác
                                input_df[col] = input_df[col].astype(np.float64)

                        # Đảm bảo các trường phân loại được lưu dưới dạng chuỗi
                        elif any(cat_col in col for cat_col in categorical_columns):
                            input_df[col] = input_df[col].astype(str)
                            # Thay thế 'nan' và 'None' bằng chuỗi rỗng
                            input_df[col] = input_df[col].replace(['nan', 'None', 'NaN'], '')

                    # Dự đoán giá trong log scale
                    try:
                        # Sử dụng mô hình để dự đoán
                        log_prediction = model.predict(input_df)

                        # Chuyển đổi từ log về giá thực tế nếu cần
                        if st.session_state.fallback_uses_log:
                            prediction_value = np.expm1(log_prediction[0])
                        else:
                            prediction_value = log_prediction[0]

                        # Tính giá tổng dựa trên diện tích
                        area = float(input_data.get("area", 0))
                        if area > 0:
                            total_price = prediction_value * area
                        else:
                            total_price = prediction_value

                        # Tạo kết quả dự đoán
                        return PredictionResult(
                            predicted_price=total_price,
                            confidence_level=0.8,  # Mặc định độ tin cậy 80%
                            price_range_low=total_price * 0.9,
                            price_range_high=total_price * 1.1
                        )
                    except Exception as e:
                        logger.error(f"Lỗi khi dự đoán với mô hình fallback: {e}")
                        # Trở về phương pháp thống kê nếu có lỗi
                        return self._predict_statistical(input_data, data)
                else:
                    # Không có mô hình, sử dụng phương pháp thống kê
                    return self._predict_statistical(input_data, data)
            else:
                # Không có mô hình, sử dụng phương pháp thống kê
                return self._predict_statistical(input_data, data)

        except Exception as e:
            logger.error(f"Lỗi trong fallback_prediction: {e}")
            # Khi có lỗi, sử dụng phương pháp thống kê
            return self._predict_statistical(input_data, data)

    # MARK: - Phương thức dự đoán thống kê

    def _predict_statistical(self, input_data: Dict[str, Any], data: pd.DataFrame) -> PredictionResult:
        """
        Dự đoán giá sử dụng phương pháp thống kê khi không có sẵn mô hình

        Args:
            input_data (Dict[str, Any]): Dictionary chứa thông tin bất động sản
            data (pd.DataFrame): DataFrame gốc

        Returns:
            PredictionResult: Kết quả dự đoán giá
        """
        try:
            # Log dữ liệu đầu vào để debug
            logger.info(f"Input data for statistical prediction: {input_data}")
            logger.info(f"Data shape: {data.shape}")
            logger.info(f"Data columns: {data.columns.tolist()}")
            # Kiểm tra xem có cột price_per_m2 trong dữ liệu không
            if 'price_per_m2' not in data.columns:
                logger.error("Cột 'price_per_m2' không tồn tại trong dữ liệu!")
                return PredictionResult(predicted_price=0)
            # Tạo bản sao của dữ liệu đầu vào và xử lý các trường hợp trống hoặc None
            cleaned_input = {}
            for key, value in input_data.items():
                if value == '' or value is None:
                    # Các trường số sẽ được điền với giá trị mặc định
                    if key in ["bedroom_num", "floor_num", "toilet_num", "livingroom_num"]:
                        cleaned_input[key] = -1
                    elif key in ["area", "street", "longitude", "latitude", "built_year"]:
                        cleaned_input[key] = 0
                    else:
                        cleaned_input[key] = ''
                else:
                    cleaned_input[key] = value

            # Chuyển đổi dữ liệu đầu vào
            category = cleaned_input.get('category', '')
            district = cleaned_input.get('district', '')
            logger.info(f"Category: {category}, District: {district}")

            # Kiểm tra các trường area trong cleaned_input
            logger.info(f"Checking area fields in input: 'area' = {cleaned_input.get('area')}, 'area (m2)' = {cleaned_input.get('area (m2)')}")

            # Tìm area trong cả 'area' và 'area (m2)'
            area_value = cleaned_input.get('area')
            if area_value is None or area_value == '':
                area_value = cleaned_input.get('area (m2)')
                logger.info(f"Using 'area (m2)' instead: {area_value}")

            # Đảm bảo area luôn là số hợp lệ
            try:
                area = float(area_value) if area_value is not None else 0
                logger.info(f"Converted area to float: {area}")
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting area to float: {e}")
                area = 0

            # Nếu dữ liệu rỗng hoặc area ≤ 0, trả về 0
            if len(data) == 0 or area <= 0:
                logger.warning(f"Data is empty or area ≤ 0: data length = {len(data)}, area = {area}")
                return PredictionResult(predicted_price=0)

            # Lọc dữ liệu theo loại bất động sản và quận/huyện (nếu có)
            filtered_data = data.copy()
            logger.info(f"Original data size: {len(filtered_data)}")

            if category and 'category' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['category'] == category]
                logger.info(f"After filtering by category '{category}': {len(filtered_data)} rows")

            if district and 'district' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['district'] == district]
                logger.info(f"After filtering by district '{district}': {len(filtered_data)} rows")

            # Nếu không còn dữ liệu sau khi lọc, sử dụng toàn bộ dữ liệu
            if len(filtered_data) == 0:
                logger.warning("No data left after filtering, using all data")
                filtered_data = data

            # Kiểm tra có NaN trong 'price_per_m2' không
            if filtered_data['price_per_m2'].isna().any():
                logger.warning(f"Found {filtered_data['price_per_m2'].isna().sum()} NaN values in price_per_m2")
                # Lọc bỏ các dòng có price_per_m2 là NaN
                filtered_data = filtered_data[~filtered_data['price_per_m2'].isna()]
                logger.info(f"After removing NaN values: {len(filtered_data)} rows")

            # Tính giá trung bình trên m²
            avg_price_per_m2 = filtered_data['price_per_m2'].mean()
            logger.info(f"Average price per m2: {avg_price_per_m2}")

            # Kiểm tra xem giá trung bình có phải NaN không
            if pd.isna(avg_price_per_m2):
                logger.error("Average price per m2 is NaN!")
                return PredictionResult(predicted_price=0)

            # Điều chỉnh giá dựa trên các yếu tố khác
            # Yếu tố 1: Số phòng ngủ
            bedroom_factor = 1.0
            if 'bedroom_num' in cleaned_input:
                try:
                    bedroom_num = int(cleaned_input['bedroom_num'])
                    if bedroom_num >= 3:
                        bedroom_factor = 1.1  # Tăng 10% nếu có từ 3 phòng ngủ trở lên
                    elif bedroom_num <= 1 and bedroom_num > 0:
                        bedroom_factor = 0.9  # Giảm 10% nếu chỉ có 1 phòng ngủ
                    logger.info(f"Bedroom number: {bedroom_num}, factor: {bedroom_factor}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error converting bedroom_num: {e}")
                    # Nếu không chuyển đổi được, giữ nguyên hệ số
                    pass

            # Yếu tố 2: Hướng nhà
            direction_factor = 1.0
            good_directions = ['Đông', 'Nam', 'Đông Nam']
            if 'direction' in cleaned_input and cleaned_input['direction'] in good_directions:
                direction_factor = 1.05  # Tăng 5% nếu hướng tốt
            logger.info(f"Direction: {cleaned_input.get('direction', 'Unknown')}, factor: {direction_factor}")

            # Yếu tố 3: Diện tích (nhà nhỏ thường có giá trên m² cao hơn)
            area_factor = 1.0
            # Đảm bảo area là số hợp lệ
            if isinstance(area, (int, float)):
                if area < 50 and area > 0:
                    area_factor = 1.1  # Tăng 10% cho nhà diện tích nhỏ
                elif area > 100:
                    area_factor = 0.95  # Giảm 5% cho nhà diện tích lớn
            logger.info(f"Area: {area}, factor: {area_factor}")

            # Tính giá cuối cùng
            base_price = avg_price_per_m2 * area * bedroom_factor * direction_factor * area_factor
            logger.info(f"Calculation: {avg_price_per_m2} (avg price/m2) * {area} (area) * {bedroom_factor} (bedroom) * {direction_factor} (direction) * {area_factor} (area factor) = {base_price} (final price)")

            # Tạo kết quả dự đoán
            result = PredictionResult(
                predicted_price=base_price,
                confidence_level=0.7,  # Mặc định độ tin cậy 70% cho phương pháp thống kê
                price_range_low=base_price * 0.85,
                price_range_high=base_price * 1.15
            )
            logger.info(f"Final prediction result: {result.predicted_price}")
            return result

        except Exception as e:
            logger.error(f"Lỗi trong statistical_fallback: {e}")
            # Trả về giá 0 nếu có lỗi
            return PredictionResult(predicted_price=0)
