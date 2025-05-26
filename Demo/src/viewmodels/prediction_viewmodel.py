#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ViewModel Dự đoán - Xử lý logic cho giao diện dự đoán giá bất động sản
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Tuple

from ..services.data_service import DataService
from ..models.property_model import Property, PredictionResult
from ..utils.session_utils import save_model_metrics, get_model_metrics

# MARK: - Lớp ViewModel Dự đoán

class PredictionViewModel:
    """
    ViewModel cho chức năng dự đoán giá bất động sản
    Xử lý tương tác giữa View và Model
    """
    def __init__(self, data_service: DataService):
        """
        Khởi tạo ViewModel với dịch vụ dữ liệu
        """
        self._data_service = data_service

        # Khởi tạo các biến trạng thái
        if 'prediction_result' not in st.session_state:
            st.session_state.prediction_result = None

        if 'prediction_inputs' not in st.session_state:
            st.session_state.prediction_inputs = None

    # MARK: - Phương thức dự đoán

    def predict_price(self, input_data: Dict[str, Any]) -> PredictionResult:
        """
        Dự đoán giá bất động sản dựa trên dữ liệu người dùng nhập vào
        """
        # Chuyển đổi dữ liệu đầu vào thành đối tượng Property
        property_data = Property.from_dict(input_data)

        # Thực hiện dự đoán sử dụng dịch vụ dữ liệu
        result = self._data_service.predict_property_price(property_data)

        # Lưu kết quả trong session state để truy cập sau này
        st.session_state.prediction_result = result
        st.session_state.prediction_inputs = input_data

        return result

    def get_last_prediction(self) -> Tuple[PredictionResult, Dict[str, Any]]:
        """
        Lấy kết quả dự đoán và dữ liệu đầu vào mới nhất
        """
        return (
            st.session_state.prediction_result,
            st.session_state.prediction_inputs
        )

    def format_price(self, price: float) -> str:
        """
        Định dạng giá để hiển thị
        """
        return f"{price:,.0f} VNĐ"

    def format_price_per_sqm(self, price_per_sqm: float) -> str:
        """
        Định dạng giá trên mét vuông để hiển thị
        """
        return f"{price_per_sqm:,.0f} VNĐ/m²"

    def get_comparison_data(self, result: PredictionResult) -> Dict[str, Any]:
        """
        Lấy dữ liệu so sánh để hiển thị biểu đồ
        """
        # Lấy dữ liệu so sánh từ kết quả
        comparison = result.comparison_data

        if not comparison:
            # Tạo dữ liệu so sánh mặc định nếu không có
            return {
                'chart_data': {
                    'labels': ["Dự đoán", "Trung bình khu vực"],
                    'values': [result.predicted_price_per_sqm, result.predicted_price_per_sqm * 0.9]
                },
                'stats': {
                    'avg_price': result.predicted_price_per_sqm * 0.9,
                    'min_price': result.predicted_price_per_sqm * 0.7,
                    'max_price': result.predicted_price_per_sqm * 1.1
                }
            }

        # Chuẩn bị dữ liệu biểu đồ từ dữ liệu so sánh
        chart_data = {
            'labels': ["Dự đoán", "Trung bình khu vực", "Cao nhất khu vực", "Thấp nhất khu vực"],
            'values': [
                result.predicted_price_per_sqm,
                comparison.get('avg_price_per_sqm', 0),
                comparison.get('max_price_per_sqm', 0),
                comparison.get('min_price_per_sqm', 0)
            ]
        }

        # Chuẩn bị thống kê từ dữ liệu so sánh
        stats = {
            'avg_price': comparison.get('avg_price_per_sqm', 0),
            'min_price': comparison.get('min_price_per_sqm', 0),
            'max_price': comparison.get('max_price_per_sqm', 0)
        }

        return {
            'chart_data': chart_data,
            'stats': stats
        }

    def get_model_metrics(self) -> Dict[str, float]:
        """
        Lấy các chỉ số đánh giá của mô hình (R² và RMSE), lưu trữ giữa các view
        """
        # Kiểm tra nếu metrics tồn tại trong session state
        session_metrics = get_model_metrics()

        # Nếu metrics tồn tại trong session state, sử dụng nó
        if session_metrics and session_metrics.get('r2', 0) > 0:
            return {
                'r2': session_metrics.get('r2', 0.0),
                'rmse': session_metrics.get('rmse', 0.0)
            }

        # Nếu không, lấy từ data service và lưu vào session state
        service_metrics = self._data_service.model_metrics

        if service_metrics:
            # Lưu metrics vào session state để duy trì giữa các views
            save_model_metrics(
                r2=service_metrics.get('r2', 0.0),
                rmse=service_metrics.get('rmse', 0.0)
            )

        return service_metrics

    def get_similar_properties(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Lấy danh sách các bất động sản tương tự dựa trên dữ liệu đầu vào
        """
        try:
            # Lấy dữ liệu từ service
            data = self._data_service.load_data()

            if data is None or data.empty:
                return None

            # Kiểm tra các tên cột diện tích có thể có (area column variants)
            area_columns = ['area (m2)', 'area_m2', 'area', 'diện tích']
            area_col = next((col for col in area_columns if col in data.columns), None)

            if not area_col:
                st.warning("Không tìm thấy cột diện tích trong dữ liệu. Không thể tìm bất động sản tương tự.")
                return None

            # Lọc dữ liệu dựa trên vị trí (với validation)
            location = input_data.get('location', '')
            area = input_data.get('area', 0)

            # Tạo phạm vi diện tích (±30%)
            min_area = area * 0.7 if area > 0 else 0
            max_area = area * 1.3 if area > 0 else float('inf')

            # Lọc dữ liệu
            try:
                # Lọc dữ liệu với xử lý các ngoại lệ
                filtered_data = data.copy()

                # Lọc theo diện tích
                filtered_data = filtered_data[
                    (filtered_data[area_col] >= min_area) &
                    (filtered_data[area_col] <= max_area)
                ]

                # Lọc theo vị trí nếu có thể
                if location and 'location' in data.columns:
                    try:
                        filtered_data = filtered_data[
                            filtered_data['location'].str.contains(location, case=False, na=False)
                        ]
                    except Exception as loc_error:
                        st.warning(f"Lỗi khi lọc theo vị trí: {loc_error}. Chỉ lọc theo diện tích.")

            except Exception as e:
                st.warning(f"Lỗi khi lọc dữ liệu: {e}. Thử với các tham số khác.")
                # Tạo DataFrame trống với các cột cần thiết
                filtered_data = pd.DataFrame(columns=[area_col, 'price', 'location'])

            # Nếu không có dữ liệu tương tự, trả về None và hiển thị thông báo
            if filtered_data.empty:
                st.info("Không tìm thấy bất động sản tương tự trong dữ liệu với tiêu chí tìm kiếm hiện tại.")
                return None

            # Lấy top 5 dữ liệu tương tự
            similar_properties = filtered_data.head(5).copy()

            # Chuẩn bị dữ liệu để hiển thị
            # Kiểm tra giá (price column variants)
            price_columns = ['price_per_sqm', 'price_per_m2', 'price', 'giá']
            price_col = next((col for col in price_columns if col in similar_properties.columns), None)

            # Kiểm tra số phòng ngủ (bedroom column variants)
            bedroom_columns = ['bedroom_num', 'bedrooms', 'so_phong_ngu', 'phòng ngủ']
            bedroom_col = next((col for col in bedroom_columns if col in similar_properties.columns), None)

            # Xác định các cột để hiển thị
            display_cols = []
            if area_col: display_cols.append(area_col)
            if price_col: display_cols.append(price_col)
            if bedroom_col: display_cols.append(bedroom_col)
            if 'location' in similar_properties.columns: display_cols.append('location')

            # Nếu không có cột nào, trả về None
            if not display_cols:
                st.warning("Không tìm thấy các cột cần thiết trong dữ liệu.")
                return None

            # Lấy các cột tồn tại trong DataFrame
            result_cols = [col for col in display_cols if col in similar_properties.columns]
            result_df = similar_properties[result_cols].copy() if result_cols else similar_properties.copy()

            # Đổi tên cột cho dễ đọc
            rename_dict = {}
            if area_col: rename_dict[area_col] = 'area'
            if price_col: rename_dict[price_col] = 'price'
            if bedroom_col: rename_dict[bedroom_col] = 'bedrooms'
            if 'location' in result_df.columns: rename_dict['location'] = 'location'

            # Chỉ đổi tên các cột có trong DataFrame
            rename_cols = {k: v for k, v in rename_dict.items() if k in result_df.columns}
            if rename_cols:
                result_df.rename(columns=rename_cols, inplace=True)

            return result_df

        except Exception as e:
            st.error(f"Lỗi khi tìm bất động sản tương tự: {e}")
            # Tạo dữ liệu giả để tránh lỗi hiển thị
            dummy_data = pd.DataFrame({
                'area': [0],
                'price': [0],
                'location': ['Không có dữ liệu']
            })
            return None
