# MARK: - Thư viện

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

from src.services.data_service import DataService
from src.utils.logger_util import get_logger

# Khởi tạo logger
logger = get_logger()

# MARK: - View Model Phân tích

class AnalyticsViewModel:
    """
    ViewModel xử lý logic cho chức năng trực quan hóa dữ liệu và phân tích thống kê
    """

    # MARK: - Khởi tạo

    def __init__(self, _data_service=None):
        """
        Khởi tạo AnalyticsViewModel

        Args:
            _data_service: Data service (tham số có dấu _ để tránh cache)
        """
        # Tham số có dấu gạch dưới (_) để Streamlit không hash
        if _data_service is None:
            raise ValueError("Cần cung cấp _data_service")

        self._data_service = _data_service

        # Khởi tạo session state nếu cần
        self._initialize_session_state()

    # MARK: - Cài đặt session state

    def _initialize_session_state(self):
        """
        Khởi tạo các giá trị trong session state nếu chưa tồn tại
        """
        if 'analytics_filters' not in st.session_state:
            st.session_state.analytics_filters = {}

        if 'analytics_metric' not in st.session_state:
            st.session_state.analytics_metric = 'price'

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

    # MARK: - Quản lý bộ lọc

    def update_filter(self, key: str, value: Any):
        """
        Cập nhật bộ lọc phân tích

        Args:
            key (str): Tên trường lọc
            value (Any): Giá trị lọc
        """
        st.session_state.analytics_filters[key] = value

    def clear_filters(self):
        """
        Xóa tất cả các bộ lọc
        """
        st.session_state.analytics_filters = {}

    # MARK: - Xử lý dữ liệu đã lọc

    def get_filtered_data(self) -> pd.DataFrame:
        """
        Lấy dữ liệu đã được lọc theo bộ lọc hiện tại

        Returns:
            pd.DataFrame: Dữ liệu đã được lọc
        """
        data = self._get_data()
        if data is None or data.empty:
            return pd.DataFrame()

        # Nếu không có bộ lọc, trả về toàn bộ dữ liệu
        if not st.session_state.analytics_filters:
            return data

        # Lọc dữ liệu theo các bộ lọc
        return self._data_service.filter_data(data, st.session_state.analytics_filters)

    # MARK: - Phân tích tương quan

    def get_correlation_data(self, numeric_only: bool = True) -> pd.DataFrame:
        """
        Tính toán ma trận tương quan giữa các biến

        Args:
            numeric_only (bool): Chỉ xem xét các biến số

        Returns:
            pd.DataFrame: Ma trận tương quan
        """
        data = self.get_filtered_data()
        if data.empty:
            return pd.DataFrame()

        try:
            if numeric_only:
                # Lọc ra các cột số
                numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
                return data[numeric_cols].corr()
            else:
                # Sử dụng one-hot encoding cho các biến phân loại
                return data.corr()
        except Exception as e:
            logger.error(f"Lỗi khi tính toán ma trận tương quan: {e}")
            return pd.DataFrame()

    # MARK: - Thống kê giá

    def get_price_stats(self) -> Dict[str, float]:
        """
        Lấy thống kê về giá

        Returns:
            Dict[str, float]: Thống kê về giá
        """
        data = self.get_filtered_data()
        if data.empty:
            return {}

        stats = self._data_service.get_stats(data)

        # Thêm phần trăm thay đổi giá so với năm trước
        try:
            if 'built_year' in data.columns and 'price' in data.columns:
                # Tạo DataFrame mới với giá trung bình theo năm
                yearly_prices = data.groupby('built_year')['price'].mean().reset_index()
                yearly_prices = yearly_prices.sort_values('built_year')

                if len(yearly_prices) > 1:
                    # Tính phần trăm thay đổi
                    current_year = yearly_prices.iloc[-1]['built_year']
                    current_price = yearly_prices.iloc[-1]['price']

                    prev_year = yearly_prices.iloc[-2]['built_year']
                    prev_price = yearly_prices.iloc[-2]['price']

                    price_change = (current_price - prev_price) / prev_price * 100

                    stats['price_change_percent'] = price_change
                    stats['current_year'] = current_year
                    stats['previous_year'] = prev_year
        except Exception as e:
            logger.error(f"Lỗi khi tính toán phần trăm thay đổi giá: {e}")

        return stats

    # MARK: - Phân tích giá theo đặc điểm

    def get_price_by_feature(self, feature: str, limit: int = 15) -> pd.DataFrame:
        """
        Phân tích giá theo một đặc điểm

        Args:
            feature (str): Tên đặc điểm cần phân tích
            limit (int): Số lượng tối đa kết quả trả về

        Returns:
            pd.DataFrame: DataFrame chứa giá trung bình theo đặc điểm
        """
        data = self.get_filtered_data()
        if data.empty or feature not in data.columns:
            return pd.DataFrame()

        try:
            # Xử lý trường hợp đặc biệt cho năm xây dựng
            if feature == 'built_year':
                # Nhóm theo thập kỷ thay vì năm cụ thể
                data['decade'] = (data['built_year'] // 10) * 10
                result = data.groupby('decade')['price'].agg(['mean', 'count']).reset_index()
                result = result.rename(columns={'decade': feature, 'mean': 'average_price'})
            else:
                # Nhóm theo đặc điểm và tính giá trung bình
                result = data.groupby(feature)['price'].agg(['mean', 'count']).reset_index()
                result = result.rename(columns={'mean': 'average_price'})

            # Sắp xếp theo giá trung bình giảm dần
            result = result.sort_values('average_price', ascending=False)

            # Giới hạn số lượng kết quả
            if limit > 0 and len(result) > limit:
                result = result.head(limit)

            return result
        except Exception as e:
            logger.error(f"Lỗi khi phân tích giá theo đặc điểm {feature}: {e}")
            return pd.DataFrame()

    # MARK: - Phân tích xu hướng giá

    def get_price_trend_by_time(self) -> pd.DataFrame:
        """
        Phân tích xu hướng giá theo thời gian

        Returns:
            pd.DataFrame: DataFrame chứa giá trung bình theo thời gian
        """
        data = self.get_filtered_data()
        if data.empty:
            return pd.DataFrame()

        try:
            # Kiểm tra xem có cột post_date không
            if 'post_date' in data.columns:
                # Chuyển đổi cột post_date sang kiểu datetime
                data['post_date'] = pd.to_datetime(data['post_date'], errors='coerce')

                # Tạo cột year_month từ post_date
                data['year_month'] = data['post_date'].dt.strftime('%Y-%m')

                # Nhóm theo tháng và tính giá trung bình
                result = data.groupby('year_month')['price'].mean().reset_index()
                result = result.rename(columns={'price': 'average_price'})

                # Thêm cột price_per_m2
                price_per_m2 = data.groupby('year_month')['price_per_m2'].mean().reset_index()
                result = result.merge(price_per_m2, on='year_month')

                # Thêm cột count
                count = data.groupby('year_month').size().reset_index(name='count')
                result = result.merge(count, on='year_month')

                # Sắp xếp theo thời gian
                result = result.sort_values('year_month')

                return result
            elif 'built_year' in data.columns:
                # Sử dụng built_year nếu không có post_date
                result = data.groupby('built_year').agg({
                    'price': 'mean',
                    'price_per_m2': 'mean'
                }).reset_index()

                # Đổi tên cột
                result = result.rename(columns={
                    'price': 'average_price',
                    'built_year': 'year'
                })

                # Thêm cột count
                count = data.groupby('built_year').size().reset_index(name='count')
                result = result.merge(count, on='built_year')

                # Sắp xếp theo năm
                result = result.sort_values('year')

                return result
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Lỗi khi phân tích xu hướng giá theo thời gian: {e}")
            return pd.DataFrame()

    # MARK: - Xếp hạng quận/huyện

    def get_district_ranking(self, limit: int = 5) -> pd.DataFrame:
        """
        Xếp hạng quận/huyện theo giá trên m²

        Args:
            limit (int): Số lượng quận/huyện hiển thị

        Returns:
            pd.DataFrame: DataFrame chứa xếp hạng quận/huyện
        """
        data = self.get_filtered_data()
        if data.empty or 'district' not in data.columns:
            return pd.DataFrame()

        try:
            # Nhóm theo quận/huyện và tính giá trung bình trên m²
            result = data.groupby('district')['price_per_m2'].agg(['mean', 'count']).reset_index()
            result = result.rename(columns={'mean': 'average_price_per_m2'})

            # Sắp xếp theo giá trung bình trên m² giảm dần
            result = result.sort_values('average_price_per_m2', ascending=False)

            # Giới hạn số lượng kết quả
            if limit > 0:
                result = result.head(limit)

            return result
        except Exception as e:
            logger.error(f"Lỗi khi xếp hạng quận/huyện: {e}")
            return pd.DataFrame()

    # MARK: - Phân tích phân phối

    def get_property_distribution(self) -> pd.DataFrame:
        """
        Phân tích phân phối loại bất động sản

        Returns:
            pd.DataFrame: DataFrame chứa phân phối loại bất động sản
        """
        data = self.get_filtered_data()
        if data.empty or 'category' not in data.columns:
            return pd.DataFrame()

        try:
            # Nhóm theo loại bất động sản và đếm số lượng
            result = data.groupby('category').size().reset_index(name='count')

            # Tính phần trăm
            result['percentage'] = result['count'] / result['count'].sum() * 100

            # Sắp xếp theo số lượng giảm dần
            result = result.sort_values('count', ascending=False)

            return result
        except Exception as e:
            logger.error(f"Lỗi khi phân tích phân phối loại bất động sản: {e}")
            return pd.DataFrame()

    # MARK: - Tên đặc điểm

    def get_feature_name_mapping(self) -> Dict[str, str]:
        """
        Ánh xạ tên cột kỹ thuật sang tên hiển thị thân thiện người dùng

        Returns:
            Dict[str, str]: Từ điển ánh xạ
        """
        return {
            "category": "Loại hình BĐS",
            "district": "Quận/Huyện",
            "direction": "Hướng nhà",
            "legal_status": "Tình trạng pháp lý",
            "bedroom_num": "Số phòng ngủ",
            "floor_num": "Số tầng",
            "toilet_num": "Số phòng tắm",
            "livingroom_num": "Số phòng khách",
            "built_year": "Năm xây dựng",
            "area": "Diện tích (m²)",
            "price": "Giá (triệu VNĐ)",
            "price_per_m2": "Giá/m² (triệu VNĐ/m²)",
            "street": "Mặt tiền đường (m)"
        }

    # MARK: - Danh sách đặc điểm phân tích

    def get_feature_list_for_analysis(self) -> List[Tuple[str, str]]:
        """
        Lấy danh sách các đặc điểm có thể dùng cho phân tích

        Returns:
            List[Tuple[str, str]]: Danh sách (key, display_name)
        """
        mapping = self.get_feature_name_mapping()

        # Danh sách các đặc điểm quan trọng để phân tích
        features = [
            "category", "district", "direction", "legal_status",
            "bedroom_num", "floor_num", "built_year", "area"
        ]

        # Chuyển đổi thành danh sách các tuple (key, display_name)
        return [(f, mapping.get(f, f)) for f in features if f in mapping]

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
        formatted_amount = "{:,.2f}".format(rounded_amount)

        # Loại bỏ phần thập phân nếu là số nguyên
        if formatted_amount.endswith('.00'):
            formatted_amount = formatted_amount[:-3]

        return formatted_amount
