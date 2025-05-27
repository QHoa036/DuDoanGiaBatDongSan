#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ViewModel Phân tích - Xử lý logic cho giao diện phân tích dữ liệu
"""

import pandas as pd
from typing import Dict, Any

from ..services.progress_data_service import ProgressDataService
from ..services.train_model_service import TrainModelService
from ..utils.session_utils import get_model_metrics, save_model_metrics

# MARK: - Lớp ViewModel Phân tích

class AnalyticsViewModel:
    """
    ViewModel cho chức năng phân tích dữ liệu
    Xử lý tương tác giữa View Phân tích Dữ liệu và Model
    """

    def __init__(self, _data_service: ProgressDataService, _model_service: TrainModelService):
        """
        Khởi tạo ViewModel với dịch vụ dữ liệu và huấn luyện mô hình
        """
        self._data_service = _data_service
        self._model_service = _model_service

    # MARK: - Phương thức tải dữ liệu

    def get_property_data(self) -> pd.DataFrame:
        """
        Lấy dữ liệu bất động sản đã được tiền xử lý cho phân tích
        """
        # Tải dữ liệu
        data = self._data_service.load_data()

        # Tiền xử lý dữ liệu
        preprocessed_data = self._data_service.preprocess_data(data)

        return preprocessed_data

    def get_model_metrics(self) -> Dict[str, float]:
        """
        Lấy các chỉ số của mô hình hiện tại từ session state
        """
        # Kiểm tra nếu metrics tồn tại trong session state
        session_metrics = get_model_metrics()

        # Nếu metrics tồn tại trong session state, sử dụng nó
        if session_metrics and session_metrics.get('r2', 0) > 0:
            return {
                'r2': session_metrics.get('r2', 0.0),
                'rmse': session_metrics.get('rmse', 0.0),
                'accuracy': session_metrics.get('r2', 0.0)  # For backward compatibility
            }

        # Nếu không, lấy từ model service và lưu vào session state
        service_metrics = self._model_service.model_metrics

        if service_metrics:
            # Lưu metrics vào session state để duy trì giữa các views
            save_model_metrics(
                r2=service_metrics.get('r2', 0.0),
                rmse=service_metrics.get('rmse', 0.0)
            )

        # Thêm trường 'accuracy' cho tương thích ngược với mã hiện tại
        if service_metrics and 'r2' in service_metrics and 'accuracy' not in service_metrics:
            service_metrics['accuracy'] = service_metrics['r2']

        return service_metrics

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Lấy mức độ quan trọng của các đặc trưng từ mô hình
        """
        return self._model_service.get_feature_importance()

    # MARK: - Helper Functions

    def _find_column(self, data: pd.DataFrame, possible_names: list) -> str:
        """
        Tìm cột phù hợp trong DataFrame dựa trên danh sách tên có thể
        """
        for name in possible_names:
            if name in data.columns:
                return name
        return None

    # MARK: - Price Distribution

    def get_price_distribution_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Lấy dữ liệu cho phân tích phân phối giá
        """
        if data is None or data.empty:
            return {}

        # =================================================
        # 1. Xác định các cột phù hợp cho phân tích
        # =================================================

        # Tìm cột địa điểm (location) phù hợp
        location_column = None
        possible_location_columns = ['province', 'city', 'location', 'tinh_thanh', 'thanh_pho', 'district']
        location_column = self._find_column(data, possible_location_columns)

        # Tìm cột giá phù hợp
        price_column = None
        possible_price_columns = ['price', 'gia', 'price_per_m2', 'price_per_sqm', 'price_m2', 'gia_tien', 'total_price', 'value']
        for col in possible_price_columns:
            if col in data.columns:
                price_column = col
                break

        # Nếu không tìm thấy cột giá, thử tìm bất kỳ cột số nào khác
        if price_column is None:
            for col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    price_column = col
                    break

            # Nếu vẫn không tìm thấy, sử dụng cột đầu tiên (nếu có)
            if price_column is None and len(data.columns) > 0:
                price_column = data.columns[0]

        # Nếu vẫn không tìm thấy cột giá phù hợp, trả về kết quả trống
        if price_column is None:
            return {}

        # =================================================
        # 2. Tính toán thống kê tổng quan
        # =================================================

        # Thống kê tổng thể
        overall_stats = {
            'avg_price': data[price_column].mean(),
            'median_price': data[price_column].median(),
            'min_price': data[price_column].min(),
            'max_price': data[price_column].max(),
            'count': len(data)
        }

        # =================================================
        # 3. Tính toán thống kê theo khu vực
        # =================================================

        # Mặc định nếu không tìm thấy cột địa điểm
        location_stats = []
        locations = []

        # Xử lý trường hợp có hoặc không có cột location
        try:
            if location_column is not None:
                # Sử dụng cột location đã tìm thấy
                unique_locations = data[location_column].unique()
                locations = sorted(unique_locations)

                # Tính toán thống kê cho từng khu vực
                for loc in locations:
                    loc_data = data[data[location_column] == loc]
                    if not loc_data.empty:
                        location_stats.append({
                            'location': loc,
                            'avg_price': loc_data[price_column].mean(),
                            'median_price': loc_data[price_column].median(),
                            'min_price': loc_data[price_column].min(),
                            'max_price': loc_data[price_column].max(),
                            'count': len(loc_data)
                        })

                # Sắp xếp theo giá trung bình (cao đến thấp)
                location_stats.sort(key=lambda x: x['avg_price'], reverse=True)
            else:
                # Nếu không có cột location, sử dụng phân tích tổng thể
                locations = ['Tất cả']
                location_stats = [{
                    'location': 'Tất cả',
                    'avg_price': overall_stats['avg_price'],
                    'median_price': overall_stats['median_price'],
                    'min_price': overall_stats['min_price'],
                    'max_price': overall_stats['max_price'],
                    'count': overall_stats['count']
                }]
        except Exception as e:
            import streamlit as st
            st.error(f"Lỗi khi phân tích giá theo khu vực: {e}")

        # =================================================
        # 4. Kết hợp và trả về tất cả kết quả
        # =================================================

        return {
            'overall_stats': overall_stats,
            'location_stats': location_stats,
            'locations': locations,
            'price_column': price_column,
            'location_column': location_column,
            'price_stats_by_location': location_stats
        }

    # MARK: - Correlation Analysis

    def get_correlation_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Lấy dữ liệu cho phân tích tương quan
        """
        if data is None or data.empty:
            return {}

        # Tìm các cột liên quan cho phân tích
        # Cột diện tích
        area_col = self._find_column(data, ['area', 'area_m2', 'dien_tich', 'square_meters', 'dientich', 'dt'])
        # Cột giá
        price_col = self._find_column(data, ['price', 'gia', 'total_price', 'gia_tien', 'value'])
        # Cột giá trên mét vuông
        price_per_sqm_col = self._find_column(data, ['price_per_sqm', 'price_per_m2', 'price_m2', 'gia_m2'])
        # Cột số phòng ngủ
        bedroom_col = self._find_column(data, ['bedroom_num', 'bedrooms', 'phong_ngu', 'num_rooms'])
        # Cột số phòng tắm
        bathroom_col = self._find_column(data, ['toilet_num', 'bathroom_num', 'bathrooms', 'phong_tam'])
        # Cột số tầng
        floor_col = self._find_column(data, ['floor_num', 'floor', 'floors', 'tang'])
        # Cột số phòng khách
        livingroom_col = self._find_column(data, ['livingroom_num', 'living_rooms', 'phong_khach'])
        # Cột độ rộng đường
        street_width_col = self._find_column(data, ['street_width_m', 'street_width', 'do_rong_duong'])
        # Cột tỉnh/thành phố
        province_col = self._find_column(data, ['city_province', 'province', 'city', 'location', 'tinh_thanh', 'thanh_pho'])
        # Cột quận/huyện
        district_col = self._find_column(data, ['district', 'quan_huyen', 'quan'])
        # Cột loại bất động sản
        category_col = self._find_column(data, ['category', 'property_type', 'loai_bds', 'type'])

        # Thống kê tổng quan về đặc điểm bất động sản
        avg_area = None
        if area_col:
            avg_area = data[area_col].mean()

        avg_bedroom = None
        if bedroom_col:
            avg_bedroom = data[bedroom_col].mean()

        price_area_corr = None
        if price_per_sqm_col and area_col:
            try:
                price_area_corr = data[[price_per_sqm_col, area_col]].corr().iloc[0, 1]
            except:
                pass
        elif price_col and area_col:
            try:
                price_area_corr = data[[price_col, area_col]].corr().iloc[0, 1]
            except:
                pass

        # Đếm số đặc trưng số
        numeric_features_count = len([col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])])

        # Chọn các cột số có sẵn cho phân tích tương quan
        potential_numeric_cols = [col for col in [
            area_col, price_col, price_per_sqm_col, bedroom_col,
            bathroom_col, floor_col, livingroom_col, street_width_col
        ] if col is not None]

        # Thêm các cột số tiềm năng khác
        default_numeric_cols = ['area', 'price', 'price_per_sqm', 'num_rooms', 'year_built']
        for col in default_numeric_cols:
            if col in data.columns and col not in potential_numeric_cols:
                potential_numeric_cols.append(col)

        # Tìm thêm các cột số khác
        for col in data.columns:
            if col not in potential_numeric_cols and pd.api.types.is_numeric_dtype(data[col]):
                potential_numeric_cols.append(col)

        # Lọc các cột có sẵn trong dữ liệu
        available_numeric_cols = [col for col in potential_numeric_cols if col in data.columns]

        # Kiểm tra nếu không có cột số nào được tìm thấy
        if not available_numeric_cols:
            import streamlit as st
            st.warning("Không tìm thấy các cột số cần thiết cho phân tích tương quan")
            return {}

        # Chỉ sử dụng các cột có sẵn trong dữ liệu và giới hạn số lượng cột cho ma trận tương quan
        numeric_data = data[available_numeric_cols[:10]].copy()  # Giới hạn 10 cột để ma trận dễ đọc

        # Tính toán ma trận tương quan
        corr_matrix = numeric_data.corr().round(2)

        # Tạo dictionary ánh xạ tên cột sang tên hiển thị đẹp hơn
        column_display_names = {}
        if area_col:
            column_display_names[area_col] = "Diện tích (m²)"
        if price_col:
            column_display_names[price_col] = "Giá"
        if price_per_sqm_col:
            column_display_names[price_per_sqm_col] = "Giá/m²"
        if bedroom_col:
            column_display_names[bedroom_col] = "Số phòng ngủ"
        if bathroom_col:
            column_display_names[bathroom_col] = "Số phòng tắm"
        if floor_col:
            column_display_names[floor_col] = "Số tầng"
        if livingroom_col:
            column_display_names[livingroom_col] = "Số phòng khách"
        if street_width_col:
            column_display_names[street_width_col] = "Chiều rộng đường"

        # Đổi tên cột trong ma trận tương quan nếu có
        if len(column_display_names) > 0 and corr_matrix is not None:
            # Tạo bản sao để tránh lỗi khi thay đổi
            corr_matrix_display = corr_matrix.copy()
            # Đổi tên cột và chỉ mục
            corr_matrix_display.columns = [column_display_names.get(col, col) for col in corr_matrix.columns]
            corr_matrix_display.index = [column_display_names.get(idx, idx) for idx in corr_matrix.index]
        else:
            corr_matrix_display = corr_matrix

        return {
            'correlation_matrix': corr_matrix,
            'correlation_matrix_display': corr_matrix_display,
            'column_display_names': column_display_names,
            'numeric_columns': available_numeric_cols,
            'avg_area': avg_area,
            'avg_bedroom': avg_bedroom,
            'price_area_corr': price_area_corr,
            'numeric_features_count': numeric_features_count,
            'area_col': area_col,
            'price_col': price_col,
            'price_per_sqm_col': price_per_sqm_col,
            'bedroom_col': bedroom_col,
            'bathroom_col': bathroom_col,
            'floor_col': floor_col,
            'livingroom_col': livingroom_col,
            'street_width_col': street_width_col,
            'province_col': province_col,
            'district_col': district_col,
            'category_col': category_col
        }