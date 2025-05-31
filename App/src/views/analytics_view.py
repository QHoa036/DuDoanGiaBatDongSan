#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vietnam Real Estate Price Prediction
Analytics view for data visualization and model performance analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List

from src.viewmodels.analytics_viewmodel import AnalyticsViewModel


def render_analytics_view(viewmodel: AnalyticsViewModel):
    """
    Hiển thị giao diện phân tích dữ liệu và hiệu suất mô hình

    Args:
        viewmodel (AnalyticsViewModel): ViewModel xử lý logic phân tích
    """
    st.markdown("<h3 class='section-header'>Phân tích dữ liệu bất động sản</h3>", unsafe_allow_html=True)

    # Chọn tab phân tích
    tabs = ["Tổng quan dữ liệu", "Phân phối giá", "So sánh theo khu vực", "Hiệu suất mô hình"]
    selected_tab = st.radio("Chọn loại phân tích:", tabs, horizontal=True)

    # Kiểm tra dữ liệu đã được tải hay chưa
    if not viewmodel.is_data_loaded():
        with st.spinner('Đang tải dữ liệu...'):
            # Load dữ liệu khi truy cập vào tab phân tích
            success = viewmodel.load_data()
        
        if not success:
            st.error("Không thể tải dữ liệu để phân tích. Vui lòng kiểm tra lại nguồn dữ liệu.")
            return

    # Hiển thị tab tương ứng
    if selected_tab == "Tổng quan dữ liệu":
        render_data_overview(viewmodel)
    elif selected_tab == "Phân phối giá":
        render_price_distribution(viewmodel)
    elif selected_tab == "So sánh theo khu vực":
        render_area_comparison(viewmodel)
    elif selected_tab == "Hiệu suất mô hình":
        render_model_performance(viewmodel)


def render_data_overview(viewmodel: AnalyticsViewModel):
    """
    Hiển thị tổng quan về dữ liệu
    
    Args:
        viewmodel (AnalyticsViewModel): ViewModel xử lý logic phân tích
    """
    stats = viewmodel.get_data_stats()
    df_sample = viewmodel.get_sample_data()
    
    # Hiển thị thông tin thống kê cơ bản
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Thống kê cơ bản")
        st.write(f"Tổng số bản ghi: {stats['total_records']:,}")
        st.write(f"Số lượng khu vực: {stats['num_districts']}")
        st.write(f"Số loại bất động sản: {stats['num_categories']}")
        st.write(f"Giá trung bình: {stats['avg_price']:,.0f} VND")
        st.write(f"Diện tích trung bình: {stats['avg_area']:.1f} m²")
    
    with col2:
        # Biểu đồ phân phối loại bất động sản
        fig = px.pie(
            viewmodel.get_category_distribution(), 
            values='count', 
            names='category', 
            title='Phân bố loại bất động sản'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Hiển thị mẫu dữ liệu
    st.subheader("Mẫu dữ liệu")
    st.dataframe(df_sample)
    
    # Hiển thị thông tin về missing values
    st.subheader("Thông tin về giá trị thiếu")
    missing_data = viewmodel.get_missing_data_info()
    
    if missing_data.empty:
        st.success("Không có giá trị thiếu trong dữ liệu!")
    else:
        st.dataframe(missing_data)


def render_price_distribution(viewmodel: AnalyticsViewModel):
    """
    Hiển thị phân phối giá bất động sản
    
    Args:
        viewmodel (AnalyticsViewModel): ViewModel xử lý logic phân tích
    """
    st.subheader("Phân phối giá bất động sản")
    
    # Lựa chọn loại bất động sản 
    categories = ["Tất cả"] + viewmodel.get_categories()
    selected_category = st.selectbox("Chọn loại bất động sản:", categories)
    
    # Lấy dữ liệu phân phối giá
    price_data = viewmodel.get_price_distribution(selected_category if selected_category != "Tất cả" else None)
    
    # Hiển thị histogram
    fig = px.histogram(
        price_data, 
        x="price", 
        nbins=50,
        labels={"price": "Giá (VND)"},
        title=f"Phân phối giá bất động sản {selected_category}"
    )
    
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)
    
    # Hiển thị một số thống kê về giá
    col1, col2, col3, col4 = st.columns(4)
    price_stats = viewmodel.get_price_stats(selected_category if selected_category != "Tất cả" else None)
    
    with col1:
        st.metric("Giá trung bình", f"{price_stats['mean']:,.0f} VND")
    
    with col2:
        st.metric("Giá trung vị", f"{price_stats['median']:,.0f} VND")
    
    with col3:
        st.metric("Giá thấp nhất", f"{price_stats['min']:,.0f} VND")
    
    with col4:
        st.metric("Giá cao nhất", f"{price_stats['max']:,.0f} VND")
    
    # Hiển thị box plot theo khu vực
    st.subheader("So sánh giá theo khu vực")
    price_by_district = viewmodel.get_price_by_district(selected_category if selected_category != "Tất cả" else None)
    
    fig2 = px.box(
        price_by_district, 
        x="district", 
        y="price",
        labels={"district": "Quận/Huyện", "price": "Giá (VND)"},
        title="Phân phối giá theo khu vực"
    )
    
    fig2.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig2, use_container_width=True)


def render_area_comparison(viewmodel: AnalyticsViewModel):
    """
    Hiển thị so sánh dữ liệu theo khu vực
    
    Args:
        viewmodel (AnalyticsViewModel): ViewModel xử lý logic phân tích
    """
    st.subheader("So sánh dữ liệu theo khu vực")
    
    # Chọn khu vực để so sánh
    districts = viewmodel.get_districts()
    selected_districts = st.multiselect(
        "Chọn khu vực để so sánh:", 
        districts,
        default=districts[:3] if len(districts) >= 3 else districts
    )
    
    if not selected_districts:
        st.warning("Vui lòng chọn ít nhất một khu vực để so sánh.")
        return
    
    # Lựa chọn chỉ số so sánh
    metrics = ["Giá trung bình", "Diện tích trung bình", "Số phòng ngủ trung bình", "Số lượng BĐS"]
    selected_metric = st.selectbox("Chọn chỉ số để so sánh:", metrics)
    
    # Lấy dữ liệu so sánh
    comparison_data = viewmodel.compare_districts(selected_districts, selected_metric)
    
    # Tạo biểu đồ cột
    fig = px.bar(
        comparison_data, 
        x='district', 
        y='value',
        labels={"district": "Quận/Huyện", "value": selected_metric},
        title=f"{selected_metric} theo khu vực",
        color='district'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Hiển thị bảng so sánh chi tiết
    st.subheader("Bảng so sánh chi tiết")
    detailed_comparison = viewmodel.get_detailed_district_comparison(selected_districts)
    st.dataframe(detailed_comparison)
    
    # Biểu đồ scatter plot giữa diện tích và giá
    st.subheader("Mối quan hệ giữa diện tích và giá theo khu vực")
    
    scatter_data = viewmodel.get_area_price_data(selected_districts)
    fig2 = px.scatter(
        scatter_data, 
        x="area", 
        y="price", 
        color="district",
        labels={"area": "Diện tích (m²)", "price": "Giá (VND)", "district": "Quận/Huyện"},
        title="Mối quan hệ giữa diện tích và giá",
        opacity=0.7,
        size_max=10
    )
    
    fig2.update_layout(legend_title_text='Quận/Huyện')
    st.plotly_chart(fig2, use_container_width=True)


def render_model_performance(viewmodel: AnalyticsViewModel):
    """
    Hiển thị thông tin về hiệu suất mô hình
    
    Args:
        viewmodel (AnalyticsViewModel): ViewModel xử lý logic phân tích
    """
    st.subheader("Hiệu suất mô hình dự đoán")
    
    # Kiểm tra xem mô hình đã được huấn luyện chưa
    if not viewmodel.is_model_trained():
        # Nút để huấn luyện mô hình
        if st.button("Huấn luyện mô hình"):
            with st.spinner("Đang huấn luyện mô hình..."):
                success = viewmodel.train_model()
                
                if success:
                    st.success("Mô hình đã được huấn luyện thành công!")
                else:
                    st.error("Không thể huấn luyện mô hình. Vui lòng thử lại.")
                    return
        else:
            st.warning("Mô hình chưa được huấn luyện. Vui lòng huấn luyện mô hình để xem hiệu suất.")
            return
    
    # Hiển thị thông tin về mô hình
    model_info = viewmodel.get_model_info()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Thông tin mô hình")
        st.write(f"Loại mô hình: {model_info['model_type']}")
        st.write(f"Thời gian huấn luyện: {model_info['training_time']:.2f} giây")
        st.write(f"Số lượng mẫu huấn luyện: {model_info['training_samples']:,}")
        st.write(f"Số lượng mẫu kiểm thử: {model_info['test_samples']:,}")
    
    with col2:
        st.subheader("Hiệu suất mô hình")
        st.write(f"R² Score: {model_info['r2_score']:.4f}")
        st.write(f"MSE: {model_info['mse']:.4f}")
        st.write(f"RMSE: {model_info['rmse']:.4f}")
        st.write(f"MAE: {model_info['mae']:.4f}")
    
    # Hiển thị biểu đồ so sánh giá thực tế và dự đoán
    st.subheader("So sánh giá thực tế và dự đoán")
    
    prediction_comparison = viewmodel.get_prediction_comparison()
    
    if prediction_comparison is not None:
        fig = px.scatter(
            prediction_comparison,
            x="actual",
            y="predicted",
            labels={"actual": "Giá thực tế", "predicted": "Giá dự đoán"},
            title="So sánh giá thực tế và dự đoán"
        )
        
        # Thêm đường tham chiếu y=x
        fig.add_trace(
            go.Scatter(
                x=[prediction_comparison["actual"].min(), prediction_comparison["actual"].max()],
                y=[prediction_comparison["actual"].min(), prediction_comparison["actual"].max()],
                mode="lines",
                name="Perfect Prediction",
                line=dict(color="red", dash="dash")
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Hiển thị phân phối của residuals
        residuals = prediction_comparison["actual"] - prediction_comparison["predicted"]
        
        fig2 = px.histogram(
            x=residuals,
            nbins=50,
            labels={"x": "Residual (Actual - Predicted)"},
            title="Phân phối Residuals"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Không có dữ liệu so sánh dự đoán.")
    
    # Hiển thị feature importance
    st.subheader("Tầm quan trọng của các đặc trưng")
    
    feature_importance = viewmodel.get_feature_importance()
    
    if feature_importance is not None:
        fig3 = px.bar(
            feature_importance,
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "Tầm quan trọng", "feature": "Đặc trưng"},
            title="Tầm quan trọng của các đặc trưng trong mô hình"
        )
        
        fig3.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Không có thông tin về tầm quan trọng của đặc trưng.")
