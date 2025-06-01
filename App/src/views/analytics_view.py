#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vietnam Real Estate Price Prediction
Analytics view for data visualization and model performance analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import sys

# Import thư viện Spark
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

from src.viewmodels.analytics_viewmodel import AnalyticsViewModel


def render_analytics_view(viewmodel: AnalyticsViewModel):
    """
    Hiển thị giao diện phân tích dữ liệu và hiệu suất mô hình

    Args:
        viewmodel (AnalyticsViewModel): ViewModel xử lý logic phân tích
    """
    # Lấy dữ liệu từ viewmodel
    data = viewmodel.get_filtered_data()

    # Tiêu đề trang
    statistics_header = """
    <div class="modern-header">
        <div class="header-title">
            <div class="header-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" y1="20" x2="18" y2="10"></line>
                    <line x1="12" y1="20" x2="12" y2="4"></line>
                    <line x1="6" y1="20" x2="6" y2="14"></line>
                    <line x1="2" y1="20" x2="22" y2="20"></line>
                </svg>
            </div>
            <div class="header-text">Phân tích dữ liệu</div>
        </div>
        <div class="header-desc">
            Phân tích dữ liệu bất động sản tại Việt Nam
        </div>
    </div>
    """
    st.markdown(statistics_header, unsafe_allow_html=True)

    # Tạo tabs để phân chia nội dung
    tab1, tab2 = st.tabs(["Giá BĐS", "Khu vực"])

    with tab1:
        # Thông tin thống kê tổng quan
        avg_price = data["price_per_m2"].mean()
        max_price = data["price_per_m2"].max()
        median_price = data["price_per_m2"].median()

        # Hiển thị thống kê tổng quan trong grid
        st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="data-grid">
            <div class="stat-card">
                <div class="stat-value">{:,.0f}</div>
                <div class="stat-label">Giá trung bình/m²</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:,.0f}</div>
                <div class="stat-label">Giá trung vị/m²</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:,.0f}</div>
                <div class="stat-label">Giá cao nhất/m²</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:,d}</div>
                <div class="stat-label">Tổng số BĐS</div>
            </div>
        </div>
        """.format(avg_price, median_price, max_price, len(data)), unsafe_allow_html=True)

        # Card 2: Lọc theo khoảng giá
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
                    <div class="chart-title">Lọc dữ liệu theo khoảng giá</div>
                    <div class="chart-desc">Tìm kiếm bất động sản trong khoảng giá mong muốn</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        price_range = st.slider(
            "Chọn khoảng giá (VND/m²)",
            min_value=float(data["price_per_m2"].min()),
            max_value=float(data["price_per_m2"].max()),
            value=(float(data["price_per_m2"].quantile(0.25)), float(data["price_per_m2"].quantile(0.75)))
        )

        # Thêm khoảng trống sau slider
        st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)

        # Lọc dữ liệu theo khoảng giá đã chọn
        filtered_data = data[(data["price_per_m2"] >= price_range[0]) & (data["price_per_m2"] <= price_range[1])]

        # Tính toán phần trăm
        total_count = len(data)
        filtered_count = len(filtered_data)
        percentage = (filtered_count / total_count) * 100 if total_count > 0 else 0

        # Thêm khoảng trống trước thông báo tìm kiếm
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)

        # Hiển thị kết quả tìm kiếm
        st.markdown(f"""
        <div style="display: flex; align-items: center; background-color: #1E293B; border-radius: 12px; padding: 15px; margin-bottom: 25px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); border-left: 4px solid #4C9AFF;">
            <div style="background-color: rgba(76, 154, 255, 0.15); width: 42px; height: 42px; border-radius: 50%; display: flex; justify-content: center; align-items: center; margin-right: 16px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#4C9AFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="11" cy="11" r="8"></circle>
                    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                </svg>
            </div>
            <div>
                <div style="font-size: 16px; font-weight: 500; color: #E2E8F0; margin-bottom: 5px;">
                    Đã tìm thấy <span style="font-weight: 700; color: #4C9AFF;">{filtered_count:,}</span> bất động sản
                </div>
                <div style="font-size: 13px; color: #94A3B8;">
                    Trong khoảng giá đã chọn • Chiếm <span style="font-weight: 600; color: #A5B4FC;">{int(percentage)}%</span> tổng số dữ liệu
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Thêm khoảng trống sau thông báo
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)

        # Hiển thị thông tin về dữ liệu đã lọc trong một dòng
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="stat-card" style="margin: 0 0 30px 0; width: 100%;">
                <div class="stat-value">{len(filtered_data):,}</div>
                <div class="stat-label">Số lượng BĐS</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card" style="margin: 0 0 30px 0; width: 100%;">
                <div class="stat-value">{filtered_data['price_per_m2'].mean():,.0f}</div>
                <div class="stat-label">Giá trung bình/m² (VND)</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="stat-card" style="margin: 0 0 30px 0; width: 100%;">
                <div class="stat-value">{filtered_data['area_m2'].mean():.1f}</div>
                <div class="stat-label">Diện tích trung bình (m²)</div>
            </div>
            """, unsafe_allow_html=True)

        # Hiển thị dữ liệu đã lọc với card
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="8" y1="6" x2="21" y2="6"></line>
                        <line x1="8" y1="12" x2="21" y2="12"></line>
                        <line x1="8" y1="18" x2="21" y2="18"></line>
                        <line x1="3" y1="6" x2="3.01" y2="6"></line>
                        <line x1="3" y1="12" x2="3.01" y2="12"></line>
                        <line x1="3" y1="18" x2="3.01" y2="18"></line>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Dữ liệu bất động sản đã lọc</div>
                    <div class="chart-desc">Danh sách 10 bất động sản đầu tiên trong khoảng giá đã chọn</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(filtered_data[["city_province", "district", "area_m2", "price_per_m2", "category"]].head(10), use_container_width=True)

    with tab2:
        # Tổng hợp thông tin theo khu vực
        total_provinces = data["city_province"].nunique()
        total_districts = data["district"].nunique()
        top_province = data["city_province"].value_counts().index[0]
        top_district = data["district"].value_counts().index[0]

        # Hiển thị thống kê tổng quan trong grid
        st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="data-grid">
            <div class="stat-card">
                <div class="stat-value">{:,d}</div>
                <div class="stat-label">Tổng số tỉnh/TP</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:,d}</div>
                <div class="stat-label">Tổng số quận/huyện</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{}</div>
                <div class="stat-label">Khu vực phổ biến nhất</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{}</div>
                <div class="stat-label">Quận/huyện phổ biến nhất</div>
            </div>
        </div>
        """.format(total_provinces, total_districts, top_province, top_district), unsafe_allow_html=True)

        # Card 1: Giá trung bình theo tỉnh/thành phố
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="2" y="7" width="3" height="10"></rect>
                        <rect x="8" y="5" width="3" height="12"></rect>
                        <rect x="14" y="3" width="3" height="14"></rect>
                        <rect x="20" y="9" width="3" height="8"></rect>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Giá trung bình theo tỉnh/thành phố</div>
                    <div class="chart-desc">Top 10 tỉnh/thành phố có giá bất động sản cao nhất</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tính giá trung bình theo tỉnh/thành phố
        city_price = data.groupby("city_province")["price_per_m2"].mean().sort_values(ascending=False).reset_index()
        city_price.columns = ["Tỉnh/Thành phố", "Giá trung bình/m²"]

        # Vẽ biểu đồ
        fig = px.bar(
            city_price.head(10),
            x="Tỉnh/Thành phố",
            y="Giá trung bình/m²",
            color="Giá trung bình/m²",
            color_continuous_scale='Viridis',
            template="plotly_white"
        )

        # Cập nhật layout của biểu đồ
        fig.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),  # Loại bỏ toàn bộ margin (lề) xung quanh biểu đồ: top, bottom, left, right đều bằng 0
            coloraxis_colorbar=dict(         # Tùy chỉnh thanh màu (colorbar) dùng trong các biểu đồ heatmap hoặc scatter có màu gradient
                tickfont=dict(color='#333333')  # Đặt màu chữ cho các giá trị trên colorbar (mã màu xám đậm)
            )
        )

        # Hiển thị biểu đồ bằng Streamlit, tự động điều chỉnh kích thước theo khung chứa
        st.plotly_chart(fig, use_container_width=True)


        # Thêm khoảng trống
        st.markdown('<div style="height: 30px;"></div>', unsafe_allow_html=True)

        # Card 2: Giá trung bình theo quận/huyện
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path>
                        <circle cx="12" cy="12" r="4"></circle>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Giá trung bình theo quận/huyện</div>
                    <div class="chart-desc">Phân tích chi tiết theo khu vực đã chọn</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Chọn tỉnh/thành phố để xem chi tiết
        selected_city = st.selectbox("Chọn tỉnh/thành phố", sorted(data["city_province"].unique()))

        # Lọc dữ liệu theo tỉnh/thành phố đã chọn
        city_data = data[data["city_province"] == selected_city]

        # Tính giá trung bình theo quận/huyện
        district_price = city_data.groupby("district")["price_per_m2"].mean().sort_values(ascending=False).reset_index()
        district_price.columns = ["Quận/Huyện", "Giá trung bình/m²"]

        # Vẽ biểu đồ
        fig = px.bar(
            district_price,
            x="Quận/Huyện",
            y="Giá trung bình/m²",
            color="Giá trung bình/m²",
            color_continuous_scale='Viridis',
            template="plotly_white"
        )

        # Cập nhật layout của biểu đồ
        fig.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),  # Loại bỏ toàn bộ margin (lề) xung quanh biểu đồ: top, bottom, left, right đều bằng 0
            coloraxis_colorbar=dict(         # Tùy chỉnh thanh màu (colorbar) dùng trong các biểu đồ heatmap hoặc scatter có màu gradient
                tickfont=dict(color='#333333')  # Đặt màu chữ cho các giá trị trên colorbar (mã màu xám đậm)
            )
        )

        # Hiển thị biểu đồ bằng Streamlit, tự động điều chỉnh kích thước theo khung chứa
        st.plotly_chart(fig, use_container_width=True)
