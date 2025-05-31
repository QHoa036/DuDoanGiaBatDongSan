# MARK: - Thư viện

import streamlit as st
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.services.data_service import DataService
from src.services.model_service import ModelService
from src.models.real_estate_model import PredictionResult
from src.utils.logger_util import get_logger

# Khởi tạo logger
logger = get_logger()

# MARK: - View Model Dự đoán

class PredictionViewModel:
    """
    ViewModel xử lý logic cho chức năng dự đoán giá bất động sản
    """

    # MARK: - Khởi tạo

    def __init__(self, _data_service=None, _model_service=None):
        """
        Khởi tạo PredictionViewModel

        Args:
            _data_service: Data service (tham số có dấu _ để tránh cache)
            _model_service: Model service (tham số có dấu _ để tránh cache)
        """
        # Tham số có dấu gạch dưới (_) để Streamlit không hash
        if _data_service is None or _model_service is None:
            raise ValueError("Cần cung cấp cả _data_service và _model_service")

        self._data_service = _data_service
        self._model_service = _model_service

        # Khởi tạo session state nếu cần
        self._initialize_session_state()

    # MARK: - Cài đặt session state

    def _initialize_session_state(self):
        """
        Khởi tạo các giá trị trong session state nếu chưa tồn tại
        """
        if 'prediction_input' not in st.session_state:
            st.session_state.prediction_input = {}

        if 'prediction_result' not in st.session_state:
            st.session_state.prediction_result = None

        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []

    # MARK: - Lấy dữ liệu danh mục

    def get_categories(self) -> List[str]:
        """
        Lấy danh sách các loại bất động sản

        Returns:
            List[str]: Danh sách các loại bất động sản
        """
        data = self._get_data()
        if data is None:
            return []

        return self._data_service.get_unique_values(data, 'category')

    def get_districts(self) -> List[str]:
        """
        Lấy danh sách các quận/huyện

        Returns:
            List[str]: Danh sách các quận/huyện
        """
        data = self._get_data()
        if data is None:
            return []

        return self._data_service.get_unique_values(data, 'district')

    def get_directions(self) -> List[str]:
        """
        Lấy danh sách các hướng nhà

        Returns:
            List[str]: Danh sách các hướng nhà
        """
        data = self._get_data()
        if data is None:
            return []

        return self._data_service.get_unique_values(data, 'direction')

    def get_legal_statuses(self) -> List[str]:
        """
        Lấy danh sách các tình trạng pháp lý

        Returns:
            List[str]: Danh sách các tình trạng pháp lý
        """
        data = self._get_data()
        if data is None:
            return []

        return self._data_service.get_unique_values(data, 'legal_status')

    # MARK: - Lấy dữ liệu

    def _get_data(self) -> Optional[pd.DataFrame]:
        """
        Lấy dữ liệu từ session state

        Returns:
            Optional[pd.DataFrame]: Dữ liệu nếu có, None nếu không có
        """
        if 'data' in st.session_state and st.session_state.data is not None:
            return st.session_state.data

        # Tải dữ liệu nếu chưa có
        data = self._data_service.load_data()
        if not data.empty:
            st.session_state.data = data
            return data

        return None

    # MARK: - Cập nhật dữ liệu đầu vào

    def update_input(self, key: str, value: Any):
        """
        Cập nhật giá trị đầu vào

        Args:
            key (str): Tên trường đầu vào
            value (Any): Giá trị đầu vào
        """
        st.session_state.prediction_input[key] = value

    # MARK: - Dự đoán

    def predict(self) -> Optional[PredictionResult]:
        """
        Dự đoán giá bất động sản dựa trên đầu vào của người dùng

        Returns:
            Optional[PredictionResult]: Kết quả dự đoán, None nếu có lỗi
        """
        try:
            # Kiểm tra xem có đầu vào không
            if not st.session_state.prediction_input:
                logger.warning("Không có đầu vào cho dự đoán")
                return None

            # Lấy dữ liệu
            data = self._get_data()
            if data is None:
                logger.error("Không có dữ liệu để dự đoán")
                return None

            # Dự đoán giá
            prediction_result = self._model_service.predict(
                st.session_state.prediction_input,
                data
            )

            # Lưu kết quả dự đoán vào session state
            st.session_state.prediction_result = prediction_result

            # Thêm vào lịch sử dự đoán
            if prediction_result.predicted_price > 0:
                self._add_to_history(
                    input_data=st.session_state.prediction_input.copy(),
                    result=prediction_result
                )

            return prediction_result

        except Exception as e:
            logger.error(f"Lỗi khi dự đoán giá bất động sản: {e}")
            return None

    # MARK: - Quản lý lịch sử

    def _add_to_history(self, input_data: Dict[str, Any], result: PredictionResult):
        """
        Thêm kết quả dự đoán vào lịch sử

        Args:
            input_data (Dict[str, Any]): Dữ liệu đầu vào
            result (PredictionResult): Kết quả dự đoán
        """
        # Giới hạn lịch sử lưu trữ 10 bản ghi
        if len(st.session_state.prediction_history) >= 10:
            # Xóa bản ghi cũ nhất
            st.session_state.prediction_history.pop(0)

        # Thêm bản ghi mới
        history_item = {
            "input": input_data,
            "price": result.predicted_price,
            "price_range": (result.price_range_low, result.price_range_high)
        }

        # Thêm vào cuối danh sách
        st.session_state.prediction_history.append(history_item)

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Lấy lịch sử dự đoán

        Returns:
            List[Dict[str, Any]]: Danh sách các bản ghi lịch sử
        """
        return st.session_state.prediction_history

    def clear_history(self):
        """
        Xóa lịch sử dự đoán
        """
        st.session_state.prediction_history = []

    # MARK: - Định dạng số

    def format_currency(self, amount: float) -> str:
        """
        Định dạng số tiền thành chuỗi có dấu phân cách

        Args:
            amount (float): Số tiền cần định dạng

        Returns:
            str: Chuỗi đã được định dạng
        """
        # Làm tròn đến 2 chữ số thập phân
        rounded_amount = round(amount, 2)

        # Định dạng số với dấu phân cách hàng nghìn
        # Sử dụng dấu phẩy làm dấu phân cách hàng nghìn
        formatted_amount = "{:,.2f}".format(rounded_amount)

        # Loại bỏ phần thập phân nếu là số nguyên
        if formatted_amount.endswith('.00'):
            formatted_amount = formatted_amount[:-3]

        return formatted_amount

    def format_area(self, area: float) -> str:
        """
        Định dạng diện tích thành chuỗi

        Args:
            area (float): Diện tích cần định dạng

        Returns:
            str: Chuỗi đã được định dạng
        """
        # Làm tròn đến 2 chữ số thập phân
        rounded_area = round(area, 2)

        # Định dạng số
        formatted_area = "{:.2f}".format(rounded_area)

        # Loại bỏ phần thập phân nếu là số nguyên
        if formatted_area.endswith('.00'):
            formatted_area = formatted_area[:-3]

        return formatted_area + " m²"

    # MARK: - Tìm kiếm bất động sản tương tự

    def get_similar_properties(self, area: float) -> pd.DataFrame:
        """
        Lấy danh sách các bất động sản tương tự dựa trên diện tích với chiến lược lọc theo thứ tự ưu tiên

        Args:
            area (float): Diện tích tham chiếu

        Returns:
            pd.DataFrame: DataFrame chứa các bất động sản tương tự (hoặc DataFrame rỗng nếu không có dữ liệu)
        """
        try:
            # Lấy dữ liệu
            data = self._get_data()
            if data is None or data.empty:
                logger.warning("Không có dữ liệu để tìm kiếm bất động sản tương tự")
                return pd.DataFrame()  # Trả về DataFrame rỗng thay vì None

            # Lấy thông tin đầu vào hiện tại
            input_data = st.session_state.prediction_input
            district = input_data.get('district', '')
            city_province = input_data.get('city_province', '')

            # Kiểm tra các cột cần thiết
            area_column = 'area'
            if area_column not in data.columns and 'area_m2' in data.columns:
                area_column = 'area_m2'

            logger.info(f"Tìm kiếm BĐS tương tự với diện tích {area}, quận {district}, thành phố {city_province}")

            # Tiếp cận 1: Lọc chặt theo cả diện tích, quận và thành phố
            similar_properties = pd.DataFrame()
            if area_column in data.columns and district and city_province:
                strict_filter = (
                    (data[area_column] > area * 0.7) &
                    (data[area_column] < area * 1.3) &
                    (data['district'] == district) &
                    (data['city_province'] == city_province)
                )
                similar_properties = data[strict_filter]
                logger.info(f"Tìm thấy {len(similar_properties)} BĐS với lọc chặt")

            # Nếu có kết quả với lọc chặt, trả về tất cả
            if not similar_properties.empty:
                return similar_properties

            # Tiếp cận 2: Nếu không có kết quả, mở rộng lọc, bỏ qua quận, chỉ giữ thành phố và mở rộng diện tích (±40%)
            by_city_filter = pd.DataFrame()
            if area_column in data.columns and city_province:
                city_filter = (
                    (data[area_column] > area * 0.6) &
                    (data[area_column] < area * 1.4) &
                    (data['city_province'] == city_province)
                )
                by_city_filter = data[city_filter]
                logger.info(f"Tìm thấy {len(by_city_filter)} BĐS với lọc theo thành phố")

            # Nếu có kết quả với lọc theo thành phố, trả về tất cả
            if not by_city_filter.empty:
                return by_city_filter

            # Tiếp cận 3: Mở rộng lọc lần nữa, chỉ lọc theo diện tích (±50%)
            by_area_filter = pd.DataFrame()
            if area_column in data.columns:
                area_filter = (
                    (data[area_column] > area * 0.5) &
                    (data[area_column] < area * 1.5)
                )
                by_area_filter = data[area_filter]
                logger.info(f"Tìm thấy {len(by_area_filter)} BĐS với lọc chỉ theo diện tích")

            # Nếu có kết quả với lọc theo diện tích, trả về tất cả
            if not by_area_filter.empty:
                return by_area_filter

            # Tiếp cận 4: Nếu không có kết quả nào, lấy ngẫu nhiên từ toàn bộ dữ liệu (lấy tất cả dữ liệu)
            logger.info(f"Không tìm thấy BĐS tương tự, trả về toàn bộ dữ liệu")
            return data

        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm bất động sản tương tự: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()  # Trả về DataFrame rỗng thay vì None
