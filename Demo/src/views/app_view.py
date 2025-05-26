#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
App View - Giao diện chính của ứng dụng
"""

import streamlit as st
from .prediction_view import PredictionView
from .analytics_view import AnalyticsView
from .about_view import AboutView
from ..viewmodels.app_viewmodel import AppViewModel
from ..utils.ui_utils import initialize_styles

class AppView:
    """
    Giao diện chính của ứng dụng
    Xử lý cấu trúc tổng thể của UI, điều hướng và điều phối các giao diện con
    """

    def __init__(self, viewmodel: AppViewModel):
        """
        Khởi tạo giao diện ứng dụng

        Tham số:
            viewmodel: ViewModel để điều phối ứng dụng
        """

        self._viewmodel = viewmodel
        self._prediction_view = PredictionView(viewmodel.prediction_viewmodel)
        self._analytics_view = AnalyticsView(viewmodel.analytics_viewmodel)
        self._about_view = AboutView()

    def render(self) -> None:
        """Hiển thị giao diện chính của ứng dụng"""

        # Khởi tạo styles và cấu hình trang
        initialize_styles()

        # Tạo menu điều hướng
        self._render_navigation()

        # Tạo thanh bên với các thông số
        self._render_sidebar()

        # Hiển thị giao diện hiện tại dựa trên chế độ ứng dụng
        current_mode = self._viewmodel.current_mode

        if current_mode == "Dự đoán":
            self._prediction_view.render()
        elif current_mode == "Thống kê":
            self._analytics_view.render()
        elif current_mode == "Về dự án":
            self._about_view.render()

    # MARK: - Sidebar

    def _render_sidebar(self) -> None:
        """Hiển thị thanh bên với các thông số mô hình và hành động"""

        # Lấy thông số mô hình
        metrics = self._viewmodel.get_model_metrics()

        # Hiển thị thông số mô hình
        st.sidebar.markdown('<div class="model-stats-container"><div class="metric-header"><div class="metric-icon"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M21 16V8.00002C20.9996 7.6493 20.9071 7.30483 20.7315 7.00119C20.556 6.69754 20.3037 6.44539 20 6.27002L13 2.27002C12.696 2.09449 12.3511 2.00208 12 2.00208C11.6489 2.00208 11.304 2.09449 11 2.27002L4 6.27002C3.69626 6.44539 3.44398 6.69754 3.26846 7.00119C3.09294 7.30483 3.00036 7.6493 3 8.00002V16C3.00036 16.3508 3.09294 16.6952 3.26846 16.9989C3.44398 17.3025 3.69626 17.5547 4 17.73L11 21.73C11.304 21.9056 11.6489 21.998 12 21.998C12.3511 21.998 12.696 21.9056 13 21.73L20 17.73C20.3037 17.5547 20.556 17.3025 20.7315 16.9989C20.9071 16.6952 20.9996 16.3508 21 16Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></div><span class="metric-title">Thông số mô hình</span></div>', unsafe_allow_html=True)

        # Metrics độ chính xác
        st.sidebar.markdown("""
        <div class="enhanced-metric-card blue-gradient">
            <div class="metric-header">
                <div class="metric-icon blue-icon-bg">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M2 17L12 22L22 17" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M2 12L12 17L22 12" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                <span class="metric-title">Độ chính xác (R²)</span>
            </div>
            <div class="clean-metric-value blue-value">{accuracy:.4f}</div>
        </div>
        """.format(accuracy=metrics.get('accuracy', 0)), unsafe_allow_html=True)

        st.sidebar.markdown("""<div class="spacer-20"></div>""", unsafe_allow_html=True)

        # Metrics độ lệch chuẩn - RMSE
        st.sidebar.markdown("""
        <div class="enhanced-metric-card purple-gradient">
            <div class="metric-header">
                <div class="metric-icon purple-icon-bg">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M3 9L12 2L21 9V20C21 20.5304 20.7893 21.0391 20.4142 21.4142C20.0391 21.7893 19.5304 22 19 22H5C4.46957 22 3.96086 21.7893 3.58579 21.4142C3.21071 21.0391 3 20.5304 3 20V9Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M9 22V12H15V22" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                <span class="metric-title">RMSE</span>
            </div>
            <div class="clean-metric-value purple-value">{rmse:.4f}</div>
        </div>
        """.format(rmse=metrics.get('rmse', 0)), unsafe_allow_html=True)

        # MARK: - Footer
        st.sidebar.markdown("""<hr class="hr-divider">""", unsafe_allow_html=True)
        st.sidebar.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="info-icon">
                <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M12 16V12" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M12 8H12.01" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <span>Dự đoán giá BĐS Việt Nam</span>
        </div>

        <div class="flex-container">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="info-icon">
                <path d="M21 10C21 17 12 23 12 23C12 23 3 17 3 10C3 7.61305 3.94821 5.32387 5.63604 3.63604C7.32387 1.94821 9.61305 1 12 1C14.3869 1 16.6761 1.94821 18.364 3.63604C20.0518 5.32387 21 7.61305 21 10Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M12 13C13.6569 13 15 11.6569 15 10C15 8.34315 13.6569 7 12 7C10.3431 7 9 8.34315 9 10C9 11.6569 10.3431 13 12 13Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <span>Nguồn: nhadat.cafeland.vn</span>
        </div>
        """, unsafe_allow_html=True)

    def _render_navigation(self) -> None:
        """Hiển thị menu điều hướng"""

        # MARK: - Header

        st.sidebar.markdown("""
        <div class="sidebar-header">
            <img src="https://img.icons8.com/fluency/96/000000/home.png" alt="Logo">
            <h2>BĐS Việt Nam</h2>
            <p>AI Dự Đoán Giá</p>
            <p>Nhóm 05</p>
        </div>
        """, unsafe_allow_html=True)

        # MARK: - Menu điều hướng

        # Lấy danh sách các chế độ ứng dụng
        app_modes = self._viewmodel.app_modes

        # Container cho menu
        menu_container = st.sidebar.container()

        # Tạo các button điều hướng
        for i, mode in enumerate(app_modes):
            if menu_container.button(mode, key=f"nav_{i}",
                                use_container_width=True,
                                on_click=self._handle_navigation,
                                args=(mode,)):
                pass

    def _handle_navigation(self, mode: str) -> None:
        """
        Xử lý điều hướng đến một giao diện khác

        Tham số:
            mode: Chế độ ứng dụng mới
        """
        self._viewmodel.set_app_mode(mode)
