#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tiện ích kết nối ngrok để tạo URL public cho ứng dụng Streamlit

Mô-đun này cung cấp các tiện ích để:
- Thiết lập kết nối Ngrok để tạo URL công khai cho ứng dụng Streamlit
- Kiểm tra trạng thái kết nối Ngrok
- Quản lý các tunnel đang hoạt động
"""

# MARK: - Thư viện

import os
import sys
import logging
import streamlit as st
from typing import List, Dict, Optional
from pyngrok import ngrok, conf
from .logger_utils import get_logger, log_execution_time

# MARK: - Cấu hình Ghi nhật ký

def configure_ngrok_logging(log_level=logging.CRITICAL + 1):
    """
    Cấu hình logger của ngrok để hoàn toàn tắt tất cả các tin nhắn log.
    Đặt mức độ log là CRITICAL + 1 để tắt tất cả các tin nhắn log, kể cả các tin nhắn CRITICAL.
    """
    # Đặt biến môi trường để kiểm soát ngrok CLI
    os.environ['NGROK_LOG_LEVEL'] = 'critical'
    os.environ['NGROK_SILENT'] = 'true'
    os.environ['NGROK_NO_LOGS'] = 'true'  # Thêm biến môi trường mới

    # Tắt tất cả các logger liên quan
    logging.getLogger("pyngrok").disabled = True
    logging.getLogger("pyngrok.process").disabled = True
    logging.getLogger("pyngrok.ngrok").disabled = True

    # Cấu hình tất cả các logger liên quan đến ngrok và http
    for logger_name in ["ngrok", "ngrok.client", "ngrok.tunnel", "urllib3", "requests", "http.client", "pyngrok", "pyngrok.process", "pyngrok.ngrok"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        logger.propagate = False  # Ngăn chặn lan truyền logs
        # Xóa tất cả các handler hiện có
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Thêm một handler rỗng (null handler) để chặn tất cả logs
        logger.addHandler(logging.NullHandler())

    # Ghi đè các hàm ghi log của ngrok để hoàn toàn vô hiệu hóa
    class SilentFilter(logging.Filter):
        def filter(self, record):
            # Loại bỏ tất cả mọi tin nhắn log
            return False

    # Áp dụng bộ lọc vô hiệu hóa cho logger cấp cao nhất
    root_logger = logging.getLogger()
    ngrok_filter = SilentFilter()
    root_logger.addFilter(ngrok_filter)

# MARK: - Kết nối Ngrok

@log_execution_time
def run_ngrok() -> Optional[str]:
    """
    Kết nối ứng dụng Streamlit với ngrok để tạo URL công khai
    """
    # Sử dụng logger với mức độ CRITICAL+1 để tắt hoàn toàn các log
    logger = get_logger(__name__, level=logging.CRITICAL+1)

    try:
        # Kiểm tra xem token Ngrok đã được cấu hình chưa
        ngrok_token = os.environ.get("NGROK_TOKEN")

        if not ngrok_token:
            logger.warning("Không tìm thấy NGROK_TOKEN trong biến môi trường")
            st.warning("""
            ### ⚠️ Không tìm thấy NGROK_TOKEN

            Để tạo URL công khai, hãy thêm NGROK_TOKEN vào file .env:
            ```
            NGROK_TOKEN=your_ngrok_token
            ```
            Lấy token từ [ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken)
            """)
            return None

        # Cấu hình Ngrok với token
        conf.get_default().auth_token = ngrok_token

        # Đăng ký bộ lọc log để chặn hoàn toàn log của ngrok
        class NgrokLogFilter(logging.Filter):
            def filter(self, record):
                # Trả về False để loại bỏ tất cả các log của ngrok
                return False

        # Áp dụng bộ lọc cho tất cả các logger liên quan đến ngrok
        ngrok_filter = NgrokLogFilter()
        logging.getLogger('ngrok').addFilter(ngrok_filter)
        logging.getLogger('pyngrok').addFilter(ngrok_filter)

        # Tắt hoàn toàn các log của ngrok và pyngrok
        logging.getLogger('ngrok').setLevel(logging.CRITICAL + 1)
        logging.getLogger('pyngrok').setLevel(logging.CRITICAL + 1)

        # Tăng thời gian timeout để tránh lỗi
        from pyngrok import process
        process._default_pyngrok_config.startup_timeout = 10

        # Chặn stdout/stderr khi khởi tạo ngrok
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        devnull = open(os.devnull, 'w')
        sys.stdout = devnull
        sys.stderr = devnull

        # Tắt log của process module
        logging.getLogger('pyngrok.process').propagate = False

        # Kiểm tra và đóng các tunnel hiện có
        existing_tunnels = ngrok.get_tunnels()
        if existing_tunnels:
            for tunnel in existing_tunnels:
                ngrok.disconnect(tunnel.public_url)
        else:
            logger.debug("Không tìm thấy tunnel nào đang hoạt động")

        # Chặn tất cả đầu ra khi kết nối Ngrok
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        dev_null = open(os.devnull, 'w')
        sys.stdout = dev_null
        sys.stderr = dev_null

        try:
            # Xác định cổng Streamlit
            streamlit_port = int(os.environ.get("STREAMLIT_PORT", 8501))

            # Sử dụng host_header để đảm bảo Streamlit hoạt động bình thường
            ngrok_tunnel = ngrok.connect(
                addr=streamlit_port,
                bind_tls=True,
                options={
                    "bind_tls": True,
                    "host_header": f"localhost:{streamlit_port}"
                }
            )
        finally:
            # Khôi phục đầu ra gốc bất kể có lỗi hay không
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            dev_null.close()

        # Lấy URL công khai mà không ghi log
        public_url = ngrok_tunnel.public_url

        # Hiển thị URL công khai cho người dùng
        st.success(f"""
        ### 🌐 URL Công Khai Đã Sẵn Sàng!

        Ứng dụng của bạn có thể truy cập công khai tại:

        **[{public_url}]({public_url})**

        URL này sẽ hoạt động cho đến khi ứng dụng Streamlit dừng lại.
        """)

        # Không hiển thị thông tin URL trong console, chỉ hiển thị trong Streamlit UI

        return public_url

    except Exception as e:
        logger.error(f"Lỗi khi tạo tunnel Ngrok: {str(e)}", exc_info=True)
        # Hiển thị thông báo lỗi cho người dùng
        st.error(f"""
        ### ❌ Lỗi khi thiết lập kết nối Ngrok

        Chi tiết lỗi: {str(e)}

        Vui lòng kiểm tra:
        1. Token Ngrok có hợp lệ không
        2. Kết nối mạng của bạn
        3. Cổng 8501 đã được sử dụng bởi Streamlit
        """)
        return None


# MARK: - Bộ nhớ đệm

# Cài đặt bộ nhớ đệm để tránh gọi API liên tục
@st.cache_data(ttl=5)  # Cache trong 5 giây
def _cached_get_tunnels():
    return ngrok.get_tunnels()

# MARK: - Kiểm tra trạng thái

@log_execution_time
def check_ngrok_status() -> List[Dict[str, str]]:
    """
    Kiểm tra trạng thái của các kết nối Ngrok hiện tại
    """
    # Khởi tạo logger
    logger = get_logger(__name__)

    try:
        # Sử dụng hàm được cache để lấy tunnels
        tunnels = _cached_get_tunnels()

        # Sử dụng list comprehension để tạo danh sách thông tin nhanh hơn
        tunnels_info = [{
            "public_url": tunnel.public_url,
            "local_url": f"localhost:{tunnel.config.get('addr', 'unknown')}",
            "proto": tunnel.proto,
            "name": tunnel.name
        } for tunnel in tunnels]

        # Chỉ hiển thị thông tin trong giao diện nếu có tunnel
        if tunnels_info:
            # Sử dụng container để tối ưu hiển thị
            with st.container():
                st.success(f"**Đang có {len(tunnels_info)} kết nối Ngrok hoạt động**")
                # Sử dụng columns để hiển thị nhiều URL trong cùng một hàng
                if len(tunnels_info) > 1:
                    cols = st.columns(min(len(tunnels_info), 2))
                    for i, info in enumerate(tunnels_info):
                        with cols[i % 2]:
                            st.write(f"URL: [{info['public_url']}]({info['public_url']}) -> {info['local_url']}")
                else:
                    for info in tunnels_info:
                        st.write(f"URL: [{info['public_url']}]({info['public_url']}) -> {info['local_url']}")

        return tunnels_info

    except Exception as e:
        logger.exception(f"Lỗi khi kiểm tra trạng thái Ngrok")
        st.error(f"Lỗi khi kiểm tra trạng thái Ngrok")
        return []
