import os

class AppConfig:
    """
    Cấu hình ứng dụng dự đoán giá bất động sản Việt Nam
    """
    # Cấu hình ứng dụng
    APP_TITLE = "Vietnam Real Estate Price Prediction"
    APP_ICON = "🏠"
    APP_LAYOUT = "wide"
    APP_INITIAL_SIDEBAR_STATE = "expanded"

    # Các chế độ ứng dụng
    APP_MODES = ["Dự đoán", "Trực quan hóa", "Về dự án"]
    DEFAULT_APP_MODE = "Dự đoán"

    # PySpark Home
    SPARK_HOME =  "/Users/admin/Development/spark"

    # Đường dẫn đến các tài nguyên
    @classmethod
    def get_base_dir(cls):
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return current_dir

    @classmethod
    def get_data_dir(cls):
        return os.path.join(cls.get_base_dir(), 'src', 'data')

    @classmethod
    def get_styles_dir(cls):
        return os.path.join(cls.get_base_dir(), 'src', 'styles')

    @classmethod
    def get_logs_dir(cls):
        return os.path.join(cls.get_base_dir(), 'src', 'logs')

    @classmethod
    def get_css_path(cls):
        return os.path.join(cls.get_styles_dir(), 'main.css')

    @classmethod
    def get_data_path(cls):
        return os.path.join(cls.get_data_dir(), 'final_data_cleaned.csv')

    # Cấu hình Spark
    SPARK_APP_NAME = "VNRealEstatePricePrediction"
    SPARK_ENABLE_HIVE = True

    # Tên các cột
    FEATURE_COLUMNS = {
        'area': 'area (m2)',
        'street': 'street (m)'
    }

    # Danh sách các cột số
    NUMERIC_COLUMNS = [
        "area", "area (m2)", "street", "street (m)",
        "bedroom_num", "floor_num", "toilet_num", "livingroom_num",
        "longitude", "latitude", "built_year", "price_per_m2"
    ]

    # Danh sách các cột phân loại
    CATEGORICAL_COLUMNS = [
        "category", "district", "direction", "legal_status"
    ]
