#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction View - Các thành phần giao diện cho dự đoán giá bất động sản
"""

import streamlit as st
import plotly.graph_objects as go
import time
from typing import Dict, Any, Tuple
from ..viewmodels.prediction_viewmodel import PredictionViewModel

# MARK: - View Dự đoán

class PredictionView:
    """
    View cho dự đoán giá bất động sản
    Xử lý hiển thị giao diện và tương tác với người dùng cho tính năng dự đoán
    """

    def __init__(self, viewmodel: PredictionViewModel):
        """Khởi tạo view"""

        self._viewmodel = viewmodel

    def render(self) -> None:
        """Hiển thị giao diện dự đoán giá"""

        # Hiển thị header hiện đại
        st.markdown("""
        <div class="modern-header">
            <div class="header-title">
                <div class="header-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                        <polyline points="9 22 9 12 15 12 15 22"></polyline>
                    </svg>
                </div>
                <div class="header-text">Dự đoán giá bất động sản Việt Nam</div>
            </div>
            <div class="header-desc">
                Hãy nhập thông tin về bất động sản mà bạn quan tâm và chúng tôi sẽ dự đoán giá trị thị trường dựa trên mô hình học máy tiên tiến của chúng tôi.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""<div class="spacer-24"></div>""", unsafe_allow_html=True)

        # Render input form
        submit_button, input_data = self._render_prediction_form()

        # Handle form submission
        if submit_button:
            # Hiệu ứng chờ với thanh tiến trình
            with st.spinner("Đang dự đoán giá..."):
                # Hiệu ứng progress bar để tăng trải nghiệm người dùng
                progress_bar = st.progress(0)
                for percent_complete in range(0, 101, 20):
                    time.sleep(0.1)  # Tạo độ trễ giả để hiệu ứng đẹp hơn
                    progress_bar.progress(percent_complete)
                progress_bar.empty()  # Xóa thanh tiến trình sau khi hoàn thành

                # Predict price
                result = self._viewmodel.predict_price(input_data)

                # Display prediction result
                self._render_prediction_result(result, input_data)

        # Check if there's a previous prediction to show
        elif 'prediction_result' in st.session_state and st.session_state.prediction_result is not None:
            result, inputs = self._viewmodel.get_last_prediction()
            if result is not None:
                self._render_prediction_result(result, inputs)

    # MARK: - Biểu mẫu nhập liệu

    def _render_prediction_form(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Hiển thị form nhập liệu dự đoán

        Returns:
            Tuple[bool, Dict[str, Any]]: (submit_button_pressed, input_data)
        """
        # Tạo layout với 2 cột
        col1, col2 = st.columns([1, 1])

        # Variables to store input
        location = ""
        area = 0.0
        num_bedrooms = 0
        num_bathrooms = 0
        num_floors = 0
        legal_status = ""
        house_direction = ""
        year_built = 0
        street_width = 0.0

        with col1:
            # Card vị trí
            st.markdown("""
            <div class="input-card">
                <div class="card-header">
                    <div class="icon">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                            <path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"></path>
                            <circle cx="12" cy="10" r="3"></circle>
                        </svg>
                    </div>
                    <div class="title">Vị trí</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Chọn tỉnh/thành phố
            city = st.selectbox("Tỉnh/Thành phố", ["Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Cần Thơ", "Hải Phòng"])

            # Chọn quận/huyện
            district_options = ["Quận 1", "Quận 2", "Quận 7", "Quận 8"] if city == "Hồ Chí Minh" else ["Quận Ba Đình", "Quận Hoàn Kiếm", "Quận Tây Hồ"]
            location = st.selectbox("Quận/Huyện", district_options)

            st.markdown("""<div class="spacer-24"></div>""", unsafe_allow_html=True)

            # Card thông tin cơ bản
            st.markdown("""
            <div class="input-card">
                <div class="card-header">
                    <div class="icon">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                            <polyline points="9 22 9 12 15 12 15 22"></polyline>
                        </svg>
                    </div>
                    <div class="title">Thông tin cơ bản</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Một hàng 2 cột cho thông tin cơ bản
            bc1, bc2 = st.columns(2)
            with bc1:
                area = st.number_input("Diện tích (m²)", min_value=10.0, max_value=1000.0, value=80.0, step=10.0)
            with bc2:
                property_type = st.selectbox("Loại BĐS", ["Căn hộ", "Nhà riêng", "Biệt thự", "Đất nền"])

            # Hàng tiếp theo
            bc3, bc4 = st.columns(2)
            with bc3:
                house_direction = st.selectbox("Hướng nhà", ["Đông", "Tây", "Nam", "Bắc", "Đông-Nam", "Tây-Nam", "Đông-Bắc", "Tây-Bắc"])
            with bc4:
                legal_status = st.selectbox("Tình trạng pháp lý", ["Sổ hồng/Sổ đỏ", "HĐMB", "Đang chờ sổ", "Khác"])

            st.markdown("""<div class="spacer-24"></div>""", unsafe_allow_html=True)

        with col2:
            # Card thông tin phòng ốc
            st.markdown("""
            <div class="input-card">
                <div class="card-header">
                    <div class="icon">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                            <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-7-7z"></path>
                            <polyline points="13 2 13 9 20 9"></polyline>
                        </svg>
                    </div>
                    <div class="title">Thông tin phòng ốc</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Hàng 1
            rc1, rc2 = st.columns(2)
            with rc1:
                num_bedrooms = st.number_input("Số phòng ngủ", min_value=0, max_value=10, value=2, step=1)
            with rc2:
                num_bathrooms = st.number_input("Số phòng vệ sinh", min_value=0, max_value=10, value=2, step=1)

            # Hàng 2
            rc3, rc4 = st.columns(2)
            with rc3:
                livingroom = st.number_input("Số phòng khách", min_value=0, max_value=10, value=1, step=1)
            with rc4:
                num_floors = st.number_input("Số tầng", min_value=0, max_value=50, value=2, step=1)

            st.markdown("""<div class="spacer-24"></div>""", unsafe_allow_html=True)

            # Card thông tin khu vực
            st.markdown("""
            <div class="input-card">
                <div class="card-header">
                    <div class="icon">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                            <circle cx="12" cy="12" r="10"></circle>
                            <line x1="2" y1="12" x2="22" y2="12"></line>
                            <line x1="12" y1="2" x2="12" y2="22"></line>
                        </svg>
                    </div>
                    <div class="title">Thông tin khu vực</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Thông tin chiều rộng đường
            street_width = st.number_input("Chiều rộng đường (m)",
                                        min_value=0.0, max_value=50.0, value=8.0, step=0.5)

            # Năm xây dựng
            year_built = st.number_input("Năm xây dựng", min_value=1900, max_value=2025, value=2015, step=1)

            st.markdown("""<div class="spacer-24"></div>""", unsafe_allow_html=True)

        # Sử dụng cách tiếp cận khác cho nút dự đoán
        st.markdown('<div style="padding: 10px 0 20px 0;"></div>', unsafe_allow_html=True)

        # Nút dự đoán
        submit_button = st.button("Dự đoán giá", use_container_width=True, type="primary")

        # Collect input data
        input_data = {
            "location": location,
            "area": area,
            "num_rooms": num_bedrooms,
            "num_bathrooms": num_bathrooms,
            "num_floors": num_floors,
            "livingroom": livingroom,
            "legal_status": legal_status,
            "house_direction": house_direction,
            "year_built": year_built,
            "property_type": property_type,
            "street_width": street_width
        }

        return submit_button, input_data

    # MARK: - Hiển thị kết quả dự đoán

    def _render_prediction_result(self, result, input_data) -> None:
        """
        Hiển thị kết quả dự đoán

        Args:
            result: Kết quả dự đoán
            input_data: Dữ liệu đầu vào được sử dụng cho dự đoán
        """
        # Thêm hiệu ứng chờ để cải thiện UX
        progress_bar = st.progress(0)
        for percent_complete in range(0, 101, 20):
            time.sleep(0.1)  # Tạo độ trễ giả để hiệu ứng đẹp hơn
            progress_bar.progress(percent_complete)
        progress_bar.empty()  # Xóa thanh tiến trình sau khi hoàn thành

        # Hàm định dạng giá thông minh theo đơn vị
        def format_price(price):
            if price >= 1_000_000_000:  # Giá >= 1 tỷ
                billions = price // 1_000_000_000
                remaining = price % 1_000_000_000

                if remaining == 0:
                    return f"{billions:,.0f} tỷ VND"

                millions = remaining // 1_000_000
                if millions == 0:
                    return f"{billions:,.0f} tỷ VND"
                else:
                    return f"{billions:,.0f} tỷ {millions:,.0f} triệu VND"
            elif price >= 1_000_000:  # Giá >= 1 triệu
                millions = price // 1_000_000
                remaining = price % 1_000_000

                if remaining == 0:
                    return f"{millions:,.0f} triệu VND"

                thousands = remaining // 1_000
                if thousands == 0:
                    return f"{millions:,.0f} triệu VND"
                else:
                    return f"{millions:,.0f} triệu {thousands:,.0f} nghìn VND"
            elif price >= 1_000:  # Giá >= 1 nghìn
                return f"{price//1_000:,.0f} nghìn VND"
            else:
                return f"{price:,.0f} VND"

        # Tính toán tổng giá dự đoán
        total_price = int(round(result.predicted_price))
        # Định dạng giá tổng với hàm mới
        formatted_total_price = format_price(total_price)

        # Hiển thị kết quả trong container đẹp với giao diện hiện đại
        st.markdown(f'''
        <div class="result-container">
            <div class="result-header">
                <svg class="result-header-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M5 12H2V20H5V12Z" fill="currentColor"/>
                    <path d="M19 3H16V20H19V3Z" fill="currentColor"/>
                    <path d="M12 7H9V20H12V7Z" fill="currentColor"/>
                </svg>
                <div class="result-header-text">Kết quả dự đoán giá</div>
            </div>
            <div class="result-body">
                <div class="price-grid">
                    <div class="price-card">
                        <div class="price-label">Giá dự đoán / m²</div>
                        <div class="price-value">{self._viewmodel.format_price_per_sqm(result.predicted_price_per_sqm)}</div>
                    </div>
                    <div class="price-card">
                        <div class="price-label">Tổng giá dự đoán</div>
                        <div class="price-value">{formatted_total_price}</div>
                    </div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # So sánh với giá trung bình khu vực
        comparison = self._viewmodel.get_comparison_data(result)

        # Tạo layout 2 cột cho kết quả chi tiết
        col1, col2 = st.columns(2)

        with col1:
            # Hiển thị biểu đồ so sánh
            st.markdown('''
            <div class="similar-container">
                <div class="similar-header">
                    <svg class="similar-header-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M5 17.5L12 6.5L19 17.5H5Z" fill="currentColor"/>
                    </svg>
                    <div class="similar-header-text">So sánh giá trung bình</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

            # Create bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=comparison['chart_data']['labels'],
                y=comparison['chart_data']['values'],
                marker_color=["#4C63B6", "#637381", "#2E7D32", "#C62828"],
                text=comparison['chart_data']['values'],
                textposition='auto'
            ))
            fig.update_layout(
                title="",
                xaxis_title="",
                yaxis_title="Giá (VNĐ/m²)",
                template="plotly_white",
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white")
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Hiển thị bất động sản tương tự
            st.markdown('''
            <div class="similar-container">
                <div class="similar-header">
                    <svg class="similar-header-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M14 2H6C4.9 2 4.01 2.9 4.01 4L4 20C4 21.1 4.89 22 5.99 22H18C19.1 22 20 21.1 20 20V8L14 2ZM18 20H6V4H13V9H18V20Z" fill="currentColor"/>
                        <path d="M11.5 14.5C11.5 15.33 10.83 16 10 16C9.17 16 8.5 15.33 8.5 14.5C8.5 13.67 9.17 13 10 13C10.83 13 11.5 13.67 11.5 14.5Z" fill="currentColor"/>
                        <path d="M14 14.5C14 13.12 12.88 12 11.5 12H8.5C7.12 12 6 13.12 6 14.5V16H14V14.5Z" fill="currentColor"/>
                    </svg>
                    <div class="similar-header-text">Bất động sản tương tự</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

            # Tạo bảng dữ liệu với các bất động sản tương tự
            similar_properties = self._viewmodel.get_similar_properties(input_data)

            if similar_properties is not None and len(similar_properties) > 0:
                st.dataframe(
                    similar_properties,
                    use_container_width=True,
                    column_config={
                        "area": st.column_config.NumberColumn(
                            "Diện tích",
                            format="%d m²"
                        ),
                        "price": st.column_config.NumberColumn(
                            "Giá bán",
                            format="%d VNĐ"
                        )
                    },
                    hide_index=True
                )
            else:
                st.info("Không tìm thấy bất động sản tương tự trong dữ liệu.")

        # Hiển thị thông tin chi tiết đầu vào
        with st.expander("Xem chi tiết thông tin đầu vào"):
            # Tạo các cặp key-value để hiển thị dữ liệu đầu vào
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### Thông tin vị trí")
                st.write(f"**Vị trí**: {input_data['location']}")

            with col2:
                st.markdown("#### Thông tin cơ bản")
                st.write(f"**Diện tích**: {input_data['area']} m²")
                st.write(f"**Hướng nhà**: {input_data['house_direction']}")
                st.write(f"**Tình trạng pháp lý**: {input_data['legal_status']}")

            with col3:
                st.markdown("#### Thông tin phòng ốc")
                st.write(f"**Số phòng ngủ**: {input_data['num_rooms']}")
                if 'num_bathrooms' in input_data:
                    st.write(f"**Số phòng vệ sinh**: {input_data['num_bathrooms']}")
                if 'num_floors' in input_data:
                    st.write(f"**Số tầng**: {input_data['num_floors']}")
                if 'year_built' in input_data:
                    st.write(f"**Năm xây dựng**: {input_data['year_built']}")
