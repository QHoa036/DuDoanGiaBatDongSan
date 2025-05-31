import streamlit as st
from typing import List, Callable

from src.config.app_config import AppConfig

def render_sidebar(app_modes: List[str], on_mode_change: Callable[[str], None]):
    """
    Hiển thị sidebar

    Args:
        app_modes (List[str]): Danh sách các chế độ ứng dụng
        on_mode_change (Callable[[str], None]): Hàm callback khi thay đổi chế độ
    """
    # Lấy thông số mô hình từ session_state nếu có sẵn
    with st.sidebar:
        st.sidebar.markdown("""
        <div class="sidebar-header">
            <img src="https://img.icons8.com/fluency/96/000000/home.png" alt="Logo">
            <h2>BĐS Việt Nam</h2>
            <p>AI Dự Đoán Giá</p>
            <p>Nhóm 05</p>
        </div>
        """, unsafe_allow_html=True)

        if 'app_mode' not in st.session_state:
            st.session_state['app_mode'] = "Dự đoán giá"

        # Phương thức để cập nhật app_mode
        def set_app_mode(mode: str):
            st.session_state['app_mode'] = mode
            on_mode_change(mode)

        # Danh sách các chế độ ứng dụng
        app_modes = AppConfig.APP_MODES

        # Container cho menu
        menu_container = st.sidebar.container()

        # Tạo các button
        for i, mode in enumerate(app_modes):
            if menu_container.button(mode, key=f"nav_{i}",
                                use_container_width=True,
                                on_click=set_app_mode,
                                args=(mode,)):
                pass

        # Hiển thị thông tin mô hình trong nhóm
        st.sidebar.markdown('<div class="model-stats-container"><div class="metric-header"><div class="metric-icon"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M21 16V8.00002C20.9996 7.6493 20.9071 7.30483 20.7315 7.00119C20.556 6.69754 20.3037 6.44539 20 6.27002L13 2.27002C12.696 2.09449 12.3511 2.00208 12 2.00208C11.6489 2.00208 11.304 2.09449 11 2.27002L4 6.27002C3.69626 6.44539 3.44398 6.69754 3.26846 7.00119C3.09294 7.30483 3.00036 7.6493 3 8.00002V16C3.00036 16.3508 3.09294 16.6952 3.26846 16.9989C3.44398 17.3025 3.69626 17.5547 4 17.73L11 21.73C11.304 21.9056 11.6489 21.998 12 21.998C12.3511 21.998 12.696 21.9056 13 21.73L20 17.73C20.3037 17.5547 20.556 17.3025 20.7315 16.9989C20.9071 16.6952 20.9996 16.3508 21 16Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></div><span class="metric-title">Thông số mô hình</span></div>', unsafe_allow_html=True)

        # Lấy thông số hiệu suất mô hình từ session_state
        r2_score = st.session_state.get('model_r2_score', 0.0)

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
                <span class="metric-title">R² Score</span>
            </div>
            <div class="clean-metric-value blue-value">{r2_score:.4f}</div>
        </div>
        """.format(r2_score=r2_score), unsafe_allow_html=True)

        # Thêm khoảng cách giữa hai card thông số mô hình
        st.sidebar.markdown("""<div class="spacer-20"></div>""", unsafe_allow_html=True)

        # Lấy giá trị RMSE từ session_state
        rmse = st.session_state.get('model_rmse', 0.0)

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
        """.format(rmse=rmse), unsafe_allow_html=True)

        # Footer
        st.sidebar.markdown("""<hr class="hr-divider">""", unsafe_allow_html=True)
        st.sidebar.markdown("""
        <div class="flex-container">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="info-icon">
                <path d="M21 10C21 17 12 23 12 23C12 23 3 17 3 10C3 7.61305 3.94821 5.32387 5.63604 3.63604C7.32387 1.94821 9.61305 1 12 1C14.3869 1 16.6761 1.94821 18.364 3.63604C20.0518 5.32387 21 7.61305 21 10Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M12 13C13.6569 13 15 11.6569 15 10C15 8.34315 13.6569 7 12 7C10.3431 7 9 8.34315 9 10C9 11.6569 10.3431 13 12 13Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <span>nhadat.cafeland.vn</span>
        </div>
        """, unsafe_allow_html=True)
        st.sidebar.markdown("</div>", unsafe_allow_html=True)
