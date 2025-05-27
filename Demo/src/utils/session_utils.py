#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tiện ích quản lý session state trong Streamlit
Giúp lưu trữ và khôi phục dữ liệu giữa các lần chạy lại
"""

import streamlit as st
from typing import Dict, Any, Optional

# MARK: - Quản lý Metrics

def save_model_metrics(r2: float, rmse: float) -> None:
    """
    Lưu các metrics của mô hình vào session state để duy trì qua các views khác nhau
    """
    st.session_state.model_metrics = {
        "r2": r2,
        "rmse": rmse,
        "timestamp": st.session_state.get("run_id", 0)
    }

def get_model_metrics() -> Dict[str, float]:
    """
    Lấy các metrics của mô hình từ session state
    """
    if "model_metrics" not in st.session_state:
        return {"r2": 0.0, "rmse": 0.0, "timestamp": 0}

    return st.session_state.model_metrics

def metrics_exist() -> bool:
    """
    Kiểm tra xem metrics đã được lưu trong session state chưa
    """
    return "model_metrics" in st.session_state

# MARK: - Quản lý Dữ liệu Dự đoán

def save_prediction_result(prediction_data: Dict[str, Any]) -> None:
    """
    Lưu kết quả dự đoán vào session state
    """
    st.session_state.prediction_results = prediction_data

def get_prediction_result() -> Optional[Dict[str, Any]]:
    """
    Lấy kết quả dự đoán từ session state
    """
    if "prediction_results" not in st.session_state:
        return None

    return st.session_state.prediction_results

# MARK: - Quản lý Session State

def initialize_session() -> None:
    """
    Khởi tạo các biến session state cần thiết nếu chưa tồn tại
    """
    if "run_id" not in st.session_state:
        st.session_state.run_id = 0
    else:
        st.session_state.run_id += 1

    # Đảm bảo các biến session state khác tồn tại
    if "model_metrics" not in st.session_state:
        st.session_state.model_metrics = {"r2": 0.0, "rmse": 0.0, "timestamp": 0}

    if "prediction_results" not in st.session_state:
        st.session_state.prediction_results = None
