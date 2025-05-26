#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI Utilities - Hàm tiện ích cho việc tạo kiểu và cấu hình giao diện người dùng
"""

# MARK: - Thư viện

import os
import streamlit as st
from .logger_utils import get_logger

# MARK: - Cấu hình

logger = get_logger(__name__)

# MARK: - CSS mặc định

def load_css(css_file):
    """
    Tải CSS từ một tập tin và áp dụng vào ứng dụng Streamlit
    """
    try:
        with open(css_file, 'r', encoding='utf-8') as f:
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
        return True

    except Exception:
        return False

def apply_default_styles():
    """
    Áp dụng kiểu CSS mặc định nếu không thể tải CSS từ bên ngoài
    """
    st.markdown("""
    <style>
    .sidebar-header {background: linear-gradient(to right, #2c5282, #1a365d); padding: 1.5rem 1rem; text-align: center; margin-bottom: 1.6rem; border-bottom: 1px solid rgba(255,255,255,0.1); border-radius: 0.8rem;}
    .sidebar-header h2 {color: white; margin: 0; font-size: 1.3rem;}
    .sidebar-header p {color: rgba(255,255,255,0.7); margin: 0; font-size: 0.8rem;}
    .sidebar-header img {max-width: 40px; margin-bottom: 0.5rem;}
    .enhanced-metric-card {border-radius: 10px; padding: 0.75rem; margin: 0.5rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.3s ease;}
    .blue-gradient {background: linear-gradient(145deg, rgba(51,97,255,0.3), rgba(29,55,147,0.5)); border-color: rgba(100,149,237,0.3);}
    .purple-gradient {background: linear-gradient(145deg, rgba(139,92,246,0.3), rgba(76,29,149,0.5)); border-color: rgba(167,139,250,0.3);}
    .green-gradient {background: linear-gradient(145deg, rgba(44,130,96,0.5), rgba(26,93,59,0.7)); border-color: rgba(76,255,154,0.3);}

    .metric-header {display: flex; align-items: center; margin-bottom: 0.5rem;}
    .metric-icon {display: flex; align-items: center; justify-content: center; width: 24px; height: 24px; border-radius: 50%; margin-right: 0.5rem;}
    .blue-icon-bg {background: rgba(51,97,255,0.2);}
    .purple-icon-bg {background: rgba(139,92,246,0.2);}
    .metric-title {font-size: 0.85rem; opacity: 0.8;}
    .clean-metric-value {font-size: 1.5rem; font-weight: 600; margin-top: 0.5rem;}
    .blue-value {color: #3361ff;}
    .purple-value {color: #8b5cf6;}

    .spacer-20 {height: 20px;}
    .hr-divider {margin: 1.5rem 0; opacity: 0.2;}

    .prediction-result-card {background: linear-gradient(145deg, rgba(44,130,96,0.5), rgba(26,93,59,0.7)); padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 1.5rem;}
    .prediction-result-card h3 {margin-top: 0; font-size: 1.2rem; opacity: 0.9;}
    .prediction-value {font-size: 2rem; font-weight: 700; margin: 1rem 0;}
    .prediction-details {font-size: 1.2rem; opacity: 0.9;}

    .about-card {background: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 1.5rem; margin-bottom: 2rem; border: 1px solid rgba(230, 230, 230, 0.1);}
    .about-card-title {display: flex; align-items: center; margin-bottom: 1rem;}
    .about-card-icon {margin-right: 0.75rem; opacity: 0.8;}
    .about-card-title h2 {margin: 0; font-size: 1.2rem;}
    .about-card-content {font-size: 0.95rem; opacity: 0.9;}
    .about-card-content p {margin-bottom: 0.75rem;}
    .about-card-content li {margin-bottom: 0.5rem;}
    .about-card-content strong {color: #4a5568;}
    </style>
    """, unsafe_allow_html=True)

# MARK: - Khởi tạo giao diện

def initialize_styles():
    """
    Khởi tạo kiểu cho ứng dụng
    Cố gắng tải CSS từ tập tin, sử dụng kiểu mặc định nếu cần thiết
    """
    # Điều hướng đến thư mục styles trong mvvm
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    css_path = os.path.join(current_dir, 'styles', 'main.css')

    # Áp dụng kiểu CSS (từ tập tin hoặc mặc định)
    if not load_css(css_path):
        apply_default_styles()
