#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analytics View - UI components for data analysis
"""

import streamlit as st
import plotly.express as px
import pandas as pd

from ..viewmodels.analytics_viewmodel import AnalyticsViewModel

class AnalyticsView:
    """
    View for data analysis
    Handles UI rendering and user interaction for data analytics
    """

    def __init__(self, viewmodel: AnalyticsViewModel):
        """
        Initialize the analytics view

        Args:
            viewmodel: ViewModel for analytics operations
        """
        self._viewmodel = viewmodel

    def render(self) -> None:
        """Render the analytics view"""
        statistics_header = """
        <div class="modern-header">
            <div class="header-title">
                <div class="header-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="18" y1="20" x2="18" y2="10"></line>
                        <line x1="12" y1="20" x2="12" y2="4"></line>
                        <line x1="6" y1="20" x2="6" y2="14"></line>
                        <line x1="2" y1="20" x2="22" y2="20"></line>
                    </svg>
                </div>
                <div class="header-text">Thống kê dữ liệu</div>
            </div>
            <div class="header-desc">
                Thông tin thống kê về dữ liệu bất động sản tại Việt Nam
            </div>
        </div>
        """
        st.markdown(statistics_header, unsafe_allow_html=True)

        # Get property data
        data = self._viewmodel.get_property_data()

        if data is None or data.empty:
            st.error("Không có dữ liệu để phân tích")
            return

        # Kiểm tra tên cột liên quan đến tỉnh/thành phố trong dữ liệu
        location_column = None
        possible_location_columns = ['province', 'city', 'location', 'tinh_thanh', 'thanh_pho', 'district']

        for col in possible_location_columns:
            if col in data.columns:
                location_column = col
                break

        if location_column is None:
            st.warning("Không tìm thấy cột chứa thông tin về tỉnh/thành phố trong dữ liệu.")
            # Nếu không tìm thấy, sử dụng cột đầu tiên làm ví dụ
            if len(data.columns) > 0:
                location_column = data.columns[0]
                st.info(f"Đang sử dụng cột '{location_column}' làm dữ liệu thay thế cho mục đích hiển thị.")
            else:
                st.stop()

        # Kiểm tra tên cột liên quan đến giá trong dữ liệu
        price_column = None
        possible_price_columns = ['price', 'gia', 'price_per_m2', 'price_m2', 'gia_tien', 'total_price', 'value', 'price_per_sqm']

        for col in possible_price_columns:
            if col in data.columns:
                price_column = col
                break

        if price_column is None:
            st.warning("Không tìm thấy cột chứa thông tin về giá trong dữ liệu.")
            # Tìm cột chứa dữ liệu số để sử dụng làm giá
            for col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    price_column = col
                    st.info(f"Đang sử dụng cột số '{price_column}' làm dữ liệu thay thế cho giá.")
                    break

            if price_column is None and len(data.columns) > 0:
                price_column = data.columns[0]  # Sử dụng cột đầu tiên nếu không có cột số
                st.info(f"Đang sử dụng cột '{price_column}' làm dữ liệu thay thế cho giá.")

            if price_column is None:
                st.stop()

        # Nếu location_column khác 'location', tạo cột 'location' mới để đảm bảo tương thích
        if location_column != 'location':
            data['location'] = data[location_column]

        # Nếu price_column khác 'price', tạo cột 'price' mới để đảm bảo tương thích
        if price_column != 'price':
            data['price'] = data[price_column]

        # Create tabs for different analyses
        tab1, tab2 = st.tabs(["Giá BĐS", "Khu Vực"])

        with tab1:
            self._render_price_distribution(data)

        with tab2:
            self._render_correlation_analysis(data)

    # MARK: - Phân bố giá

    def _render_price_distribution(self, data: pd.DataFrame) -> None:
        """
        Render price distribution analysis

        Args:
            data: Property data
        """
        # Kiểm tra dữ liệu đầu vào
        if data is None or data.empty:
            st.warning("Không có dữ liệu để hiển thị phân bố giá")
            return

        # Kiểm tra xem có các cột cần thiết không
        required_columns = ['location', 'price_per_sqm']
        missing_columns = [col for col in required_columns if col not in data.columns]

        # Nếu thiếu cột, hiển thị thông báo và thử tìm các cột thay thế
        if missing_columns:
            alternative_price_cols = ['price', 'price_per_m2', 'price_per_sqm']
            price_col = next((col for col in alternative_price_cols if col in data.columns), None)

            # Nếu không có location, hiển thị phân tích tổng quan
            if 'location' not in data.columns:
                st.info("Không có thông tin về khu vực trong dữ liệu. Hiển thị phân tích tổng thể.")

                # Tạo dữ liệu thay thế với vị trí mặc định
                if price_col:
                    tmp_data = data.copy()
                    tmp_data['location'] = 'Tất cả'
                    data = tmp_data
                else:
                    st.warning("Không thể tìm thấy cột giá trong dữ liệu")
                    return

            # Nếu không có cột giá, hiển thị thông báo
            if price_col is None:
                st.warning("Không tìm thấy thông tin giá trong dữ liệu")
                return

        # Get location metrics
        distribution_data = self._viewmodel.get_price_distribution_data(data)

        if not distribution_data:
            st.warning("Không có dữ liệu để hiển thị phân bố giá")
            return

        avg_price = distribution_data['overall_stats']['avg_price']
        median_price = distribution_data['overall_stats']['median_price']
        max_price = distribution_data['overall_stats']['max_price']

        # Hiển thị thống kê tổng quan trong grid
        st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="data-grid">
            <div class="stat-card">
                <div class="stat-value">{:,.0f}</div>
                <div class="stat-label">Giá trung bình (triệu VNĐ)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:,.0f}</div>
                <div class="stat-label">Giá trung vị (triệu VNĐ)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:,.0f}</div>
                <div class="stat-label">Giá cao nhất (triệu VNĐ)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:,d}</div>
                <div class="stat-label">Tổng số BĐS</div>
            </div>
        </div>
        """.format(avg_price, median_price, max_price, len(distribution_data['locations'])), unsafe_allow_html=True)

        if not distribution_data:
            st.warning("Không có dữ liệu để hiển thị phân bố giá")
            return

        # Xác định cột giá và khu vực để vẽ biểu đồ
        price_col = distribution_data.get('price_column')
        location_col = distribution_data.get('location_column')

        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="2" y="7" width="3" height="10"></rect>
                        <rect x="8" y="5" width="3" height="12"></rect>
                        <rect x="14" y="3" width="3" height="14"></rect>
                        <rect x="20" y="9" width="3" height="8"></rect>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Giá trung bình theo khu vực</div>
                    <div class="chart-desc">Top 10 khu vực có giá BĐS trung bình cao nhất</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tính giá trung bình theo tỉnh/thành phố
        avg_price_by_location = data.groupby(location_col)[price_col].mean().reset_index()
        avg_price_by_location = avg_price_by_location.sort_values(price_col, ascending=False)

        # Vẽ biểu đồ cột cho giá trung bình
        fig1 = px.bar(avg_price_by_location.head(10), x=location_col, y=price_col,
                    labels={price_col:'Giá trung bình (triệu VNĐ)', location_col:location_col.replace('_', ' ').title()},
                    color=price_col,
                    color_continuous_scale='Viridis')

        # Cập nhật layout của biểu đồ
        fig1.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(t=0, b=0, l=0, r=0),
            coloraxis_colorbar=dict(tickfont=dict(color='white'))
        )
        fig1.update_xaxes(tickfont=dict(color='white'))
        fig1.update_yaxes(tickfont=dict(color='white'))

        with st.container():
            st.plotly_chart(fig1, use_container_width=True)

        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M2 9h20v12H2z"></path>
                        <path d="M14 4h4v5h-4z"></path>
                        <path d="M8 4h4v5H8z"></path>
                        <path d="M4 9V5a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v4"></path>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Phân bố giá theo khu vực</div>
                    <div class="chart-desc">Biểu đồ hộp hiển thị sự phân bố giá ở các khu vực</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tạo biểu đồ hộp về phân bố giá theo khu vực
        fig = px.box(data, x=location_col, y=price_col,
                    labels={location_col: 'Khu vực', price_col: f"Giá{' (VNĐ/m²)' if 'sqm' in price_col or 'm2' in price_col else ' (VNĐ)'}"}
                )
        # Cập nhật layout để loại bỏ khoảng trống dùng cho tiêu đề
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

        # Hiển thị biểu đồ cột giá trung bình theo khu vực
        # Chuyển đổi thành DataFrame để dễ vẽ
        stats_df = pd.DataFrame(distribution_data['location_stats'])

        # Sắp xếp theo giá trung bình giảm dần
        stats_df = stats_df.sort_values('avg_price', ascending=False)

        # Đặt tiêu đề và nhãn cho biểu đồ
        price_col = distribution_data.get('price_column')
        unit_suffix = '/m²' if 'sqm' in price_col or 'm2' in price_col else ''

        # Tiêu đề cho biểu đồ giá trung bình
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="18" y1="20" x2="18" y2="10"></line>
                        <line x1="12" y1="20" x2="12" y2="4"></line>
                        <line x1="6" y1="20" x2="6" y2="14"></line>
                        <line x1="2" y1="20" x2="22" y2="20"></line>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Chi tiết giá theo khu vực</div>
                    <div class="chart-desc">Phân tích chi tiết giá bất động sản theo khu vực</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Vẽ biểu đồ giá trung bình theo khu vực
        fig2 = px.bar(stats_df,
                    x='location',
                    y='avg_price',
                    labels={'avg_price': f'Giá trung bình (VNĐ{unit_suffix})', 'location': 'Khu vực'})
        # Cập nhật layout để loại bỏ khoảng trống dùng cho tiêu đề
        fig2.update_layout(margin=dict(t=10))
        st.plotly_chart(fig2, use_container_width=True)

        # Tiêu đề cho biểu đồ số lượng
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 10V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l2-1.14"></path>
                        <path d="M16.5 9.4 7.55 4.24"></path>
                        <polyline points="3.29 7 12 12 20.71 7"></polyline>
                        <line x1="12" y1="22" x2="12" y2="12"></line>
                        <circle cx="18.5" cy="15.5" r="2.5"></circle>
                        <path d="M20.27 17.27 22 19"></path>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Số lượng BĐS theo khu vực</div>
                    <div class="chart-desc">Phân bố số lượng bất động sản theo từng khu vực</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Vẽ biểu đồ số lượng bất động sản theo khu vực
        fig3 = px.bar(stats_df,
                    x='location',
                    y='count',
                    labels={'count': 'Số lượng', 'location': 'Khu vực'})
        # Cập nhật layout để loại bỏ khoảng trống dùng cho tiêu đề
        fig3.update_layout(margin=dict(t=10))
        st.plotly_chart(fig3, use_container_width=True)

    # MARK: - Tương quan

    def _render_correlation_analysis(self, data: pd.DataFrame) -> None:
        """
        Render correlation analysis

        Args:
            data: Property data
        """
        # Get correlation data
        correlation_data = self._viewmodel.get_correlation_data(data)

        if not correlation_data:
            st.warning("Không có dữ liệu để phân tích tương quan")
            return

        # Lấy các cột đã được tìm thấy
        area_col = correlation_data.get('area_col')
        price_col = correlation_data.get('price_col')
        price_per_sqm_col = correlation_data.get('price_per_sqm_col')
        bedroom_col = correlation_data.get('bedroom_col')
        province_col = correlation_data.get('province_col')
        district_col = correlation_data.get('district_col')
        category_col = correlation_data.get('category_col')

        # Thống kê tổng quan về đặc điểm bất động sản
        avg_area = correlation_data.get('avg_area')
        avg_bedroom = correlation_data.get('avg_bedroom')
        price_area_corr = correlation_data.get('price_area_corr')
        numeric_features_count = correlation_data.get('numeric_features_count', 0)

        # Hiển thị thống kê tổng quan trong grid
        st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)

        # Tạo các giá trị mặc định nếu không có dữ liệu
        avg_area_display = f"{avg_area:.1f}" if avg_area is not None else "N/A"
        avg_bedroom_display = f"{avg_bedroom:.1f}" if avg_bedroom is not None else "N/A"
        price_area_corr_display = f"{price_area_corr:.2f}" if price_area_corr is not None else "N/A"

        st.markdown(f"""
        <div class="data-grid">
            <div class="stat-card">
                <div class="stat-value">{avg_area_display}</div>
                <div class="stat-label">Diện tích trung bình (m²)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_bedroom_display}</div>
                <div class="stat-label">Số phòng ngủ trung bình</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{price_area_corr_display}</div>
                <div class="stat-label">Tương quan giá-diện tích</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{numeric_features_count}</div>
                <div class="stat-label">Số đặc trưng số</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Card 1: Tương quan giữa diện tích và giá
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="6" cy="6" r="3"></circle>
                        <circle cx="18" cy="18" r="3"></circle>
                        <line x1="9" y1="9" x2="15" y2="15"></line>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Mối quan hệ giữa diện tích và giá</div>
                    <div class="chart-desc">Phân tích sự tương quan giữa diện tích và giá theo khu vực và số phòng ngủ</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Kiểm tra các cột cần thiết cho biểu đồ phân tán
        if area_col and (price_col or price_per_sqm_col):
            try:
                # Tạo mẫu nhỏ hơn nếu có quá nhiều dữ liệu
                sample_size = min(1000, len(data))
                sampled_data = data.sample(n=sample_size, random_state=42) if len(data) > sample_size else data.copy()

                # Tạo bản sao và chuẩn bị dữ liệu
                plot_data = sampled_data.copy()

                # Quyết định trục y cho biểu đồ
                y_axis = price_per_sqm_col if price_per_sqm_col else price_col
                y_label = "Giá/m² (VND)" if price_per_sqm_col else "Giá (VND)"

                # Xử lý cột số phòng ngủ cho biểu đồ kích thước
                if bedroom_col:
                    # Chuyển đổi giá trị âm thành 1 và đảm bảo tất cả các giá trị đều > 0
                    plot_data['size_value'] = plot_data[bedroom_col].apply(lambda x: max(1, x) if pd.notna(x) else 1)

                # Lọc dữ liệu trong khoảng hợp lý để biểu đồ đẹp hơn
                filtered_data = plot_data[
                    (plot_data[y_axis] < plot_data[y_axis].quantile(0.99)) &
                    (plot_data[area_col] < plot_data[area_col].quantile(0.99))
                ]

                # Tạo các tham số cho biểu đồ
                scatter_args = {
                    'x': area_col,
                    'y': y_axis,
                    'labels': {
                        area_col: "Diện tích (m²)",
                        y_axis: y_label
                    }
                }

                # Thêm tham số màu nếu có cột vị trí
                if province_col:
                    scatter_args['color'] = province_col
                    scatter_args['labels'][province_col] = "Tỉnh/Thành phố"

                # Thêm tham số kích thước nếu có cột số phòng ngủ
                if bedroom_col:
                    scatter_args['size'] = 'size_value'
                    scatter_args['labels'][bedroom_col] = "Số phòng ngủ"

                # Thêm dữ liệu hover nếu có các cột liên quan
                hover_data = []
                if district_col:
                    hover_data.append(district_col)
                if category_col:
                    hover_data.append(category_col)
                if bedroom_col:
                    hover_data.append(bedroom_col)
                if hover_data:
                    scatter_args['hover_data'] = hover_data

                # Vẽ biểu đồ phân tán
                fig = px.scatter(filtered_data, **scatter_args)

                # Cập nhật layout của biểu đồ
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    margin=dict(t=0, b=0, l=0, r=0),
                    legend=dict(font=dict(color='white')),
                    coloraxis_colorbar=dict(tickfont=dict(color='white'))
                )
                fig.update_xaxes(tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)')
                fig.update_yaxes(tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)')

                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            except Exception as e:
                st.error(f"Lỗi khi tạo biểu đồ phân tán: {e}")
        else:
            st.info("Không đủ dữ liệu để tạo biểu đồ phân tán giữa giá và diện tích.")

        # Card 2: Ma trận tương quan
        st.markdown('<div style="height: 30px;"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="3" y="3" width="7" height="7"></rect>
                        <rect x="14" y="3" width="7" height="7"></rect>
                        <rect x="14" y="14" width="7" height="7"></rect>
                        <rect x="3" y="14" width="7" height="7"></rect>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Ma trận tương quan giữa các đặc điểm</div>
                    <div class="chart-desc">Phân tích mối tương quan giữa các đặc trưng số trong dữ liệu</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Get correlation matrix
        if 'correlation_matrix_display' in correlation_data:
            # Plot correlation matrix with better display names
            corr_matrix_display = correlation_data['correlation_matrix_display']

            try:
                # Create heatmap using Plotly with customized appearance
                fig2 = px.imshow(corr_matrix_display,
                                labels=dict(x="Thuộc tính", y="Thuộc tính", color="Hệ số tương quan"),
                                x=corr_matrix_display.columns,
                                y=corr_matrix_display.index,
                                text_auto=True,
                                color_continuous_scale='RdBu_r',  # Đổi palette màu để dễ nhìn hơn
                                zmin=-1, zmax=1)  # Cố định thang màu từ -1 đến 1 cho hệ số tương quan

                # Cập nhật layout để thêm đường viền và các điều chỉnh khác
                fig2.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    margin=dict(t=10, b=50, l=50, r=50),
                    coloraxis_colorbar=dict(tickfont=dict(color='white'))
                )

                # Thêm đường viền cho các ô trong ma trận
                fig2.update_traces(showscale=True, hoverongaps=False,
                                hovertemplate='<b>%{y}</b> - <b>%{x}</b><br>Tương quan: %{z:.2f}<extra></extra>')

                # Cập nhật trục để dễ đọc hơn
                fig2.update_xaxes(tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)')
                fig2.update_yaxes(tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)')

                st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
            except Exception as e:
                st.error(f"Lỗi khi tạo biểu đồ ma trận tương quan: {e}")
        else:
            st.warning("Không có dữ liệu để tạo ma trận tương quan")

        # Phân tích giá theo đặc điểm
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                        <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                        <line x1="12" y1="22.08" x2="12" y2="12"></line>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Phân tích giá theo đặc điểm</div>
                    <div class="chart-desc">So sánh giá trung bình theo các đặc điểm khác nhau của BĐS</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Chọn đặc điểm để phân tích
        feature_options = {
            "Loại hình BĐS": "category",
            "Hướng nhà": "direction",
            "Tình trạng pháp lý": "liability",
            "Số phòng ngủ": "bedroom_num",
            "Số tầng": "floor_num",
            "Diện tích sàn": "area",
            "Khu vực": "location"
        }

        selected_feature_name = st.selectbox(
            "Chọn đặc điểm",
            list(feature_options.keys()),
            key="correlation_feature_select"
        )

        # Thêm khoảng cách sau selectbox
        st.markdown('<div style="height: 30px;"></div>', unsafe_allow_html=True)

        # Lấy tên thuộc tính tương ứng với tên hiển thị
        feature = feature_options[selected_feature_name]

        # Tính giá trung bình theo đặc điểm đã chọn
        # Kiểm tra xem feature có trong dữ liệu không
        if feature not in data.columns:
            # Tìm các cột tương tự dựa trên tên đặc điểm
            similar_cols = []
            feature_lower = feature.lower()

            # Phân loại tìm kiếm dựa trên loại thuộc tính
            search_terms = []
            if 'bed' in feature_lower or 'phong_ngu' in feature_lower:
                search_terms = ['bed', 'phong_ngu', 'phongngu', 'bedroom']
            elif 'bath' in feature_lower or 'phong_tam' in feature_lower:
                search_terms = ['bath', 'phong_tam', 'phongtam', 'bathroom']
            elif 'floor' in feature_lower or 'tang' in feature_lower:
                search_terms = ['floor', 'tang', 'levels']
            elif 'area' in feature_lower or 'dien_tich' in feature_lower:
                search_terms = ['area', 'dien_tich', 'dt', 'square']
            elif 'year' in feature_lower or 'nam' in feature_lower:
                search_terms = ['year', 'nam', 'built']
            elif 'category' in feature_lower or 'loai' in feature_lower:
                search_terms = ['category', 'loai', 'type']
            elif 'direction' in feature_lower or 'huong' in feature_lower:
                search_terms = ['direction', 'huong']
            elif 'location' in feature_lower or 'khu_vuc' in feature_lower:
                search_terms = ['location', 'khu_vuc', 'district', 'ward', 'city']

            # Tìm các cột tương tự trong dữ liệu
            for col in data.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in search_terms):
                    similar_cols.append(col)

            if similar_cols:
                feature = similar_cols[0]  # Sử dụng cột đầu tiên tìm thấy
            else:
                st.warning(f"Thuộc tính '{selected_feature_name}' ({feature}) không có trong dữ liệu và không tìm thấy thuộc tính tương tự")
                return

        # Kiểm tra xem có cột giá/m2 không
        price_per_sqm_col = None
        possible_price_sqm_cols = ['price_per_sqm', 'price_per_m2', 'price_m2', 'gia_tren_m2']
        for col in possible_price_sqm_cols:
            if col in data.columns:
                price_per_sqm_col = col
                break

        if price_per_sqm_col is None and 'price' in data.columns:
            st.warning("Không có thông tin giá/m². Sử dụng giá thông thường thay thế.")
            price_per_sqm_col = 'price'

        # Xử lý dữ liệu cho phân tích
        if price_per_sqm_col is None:
            st.warning("Không có thông tin giá để phân tích")
            return

        # Xử lý các loại đặc điểm khác nhau
        numeric_features = ["bedroom_num", "bathroom_num", "floor_num", "area", "year_built"]
        date_features = ["year_built"]

        if feature in numeric_features or any(term in feature.lower() for term in ['bed', 'bath', 'floor', 'tang', 'phong', 'area', 'dien_tich', 'nam', 'year']):
            # Đối với đặc điểm số, chuyển đổi thành chuỗi để nhóm
            data = data.copy()
            # Lọc các giá trị null hoặc không hợp lệ
            valid_data = data.dropna(subset=[feature, price_per_sqm_col])

            # Đối với các đặc điểm năm, có thể làm tròn để nhóm theo thập kỷ
            if feature in date_features or 'year' in feature.lower() or 'nam' in feature.lower():
                # Làm tròn năm về thập kỷ gần nhất (nhóm các năm lại)
                valid_data['decade'] = (valid_data[feature] // 10 * 10).astype(int)
                feature_price = valid_data.groupby('decade')[price_per_sqm_col].mean().reset_index()
                feature_price.columns = [feature, "Giá trung bình/m²"]
                feature_price[feature] = feature_price[feature].astype(str) + 's'  # Thêm 's' để chỉ thập kỷ
            else:
                valid_data["feature_str"] = valid_data[feature].astype(str)
                feature_price = valid_data.groupby("feature_str")[price_per_sqm_col].mean().reset_index()
                feature_price.columns = [feature, "Giá trung bình/m²"]

                # Sắp xếp theo thứ tự số
                try:
                    feature_price[feature] = feature_price[feature].astype(float)
                    feature_price = feature_price.sort_values(by=feature)
                    feature_price[feature] = feature_price[feature].astype(str)
                except:
                    # Nếu không thể chuyển đổi thành số, giữ nguyên chuỗi
                    pass
        else:
            # Đối với đặc điểm phân loại
            # Lọc các giá trị null hoặc không hợp lệ
            valid_data = data.dropna(subset=[feature, price_per_sqm_col])
            feature_price = valid_data.groupby(feature)[price_per_sqm_col].mean().sort_values(ascending=False).reset_index()
            feature_price.columns = [feature, "Giá trung bình/m²"]

        # Vẽ biểu đồ
        if not feature_price.empty:
            # Thay đổi tên cột cho biểu đồ
            feature_price = feature_price.rename(columns={feature: selected_feature_name})

            fig = px.bar(
                feature_price,
                x=selected_feature_name,
                y="Giá trung bình/m²",
                color="Giá trung bình/m²",
                color_continuous_scale='Viridis',
                template="plotly_white",
                title=f"Giá trung bình theo {selected_feature_name}"
            )

            # Cập nhật layout của biểu đồ để phù hợp với giao diện tối
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(t=30, b=0, l=0, r=0),
                coloraxis_colorbar=dict(tickfont=dict(color='white'))
            )

            # Cập nhật trục để dễ đọc hơn
            fig.update_xaxes(tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)')

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # Chỉ hiển thị 15 giá trị đầu tiên để tránh quá nhiều dữ liệu
            if len(feature_price) > 15:
                st.info(f"Hiển thị 15/{len(feature_price)} giá trị đầu tiên")
                feature_price = feature_price.head(15)
        else:
            st.warning(f"Không có dữ liệu hợp lệ để phân tích {selected_feature_name}")