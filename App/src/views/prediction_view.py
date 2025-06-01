import streamlit as st
import pandas as pd
import time
from typing import Dict, Any, List, Optional

from src.viewmodels.prediction_viewmodel import PredictionViewModel
from src.utils.logger_util import get_logger

# Khởi tạo logger
logger = get_logger()

def render_prediction_view(viewmodel: PredictionViewModel):
    """
    Hiển thị giao diện dự đoán giá bất động sản

    Args:
        viewmodel (PredictionViewModel): ViewModel xử lý logic dự đoán
    """
    # Tiêu đề trang
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

    # Lấy dữ liệu từ viewmodel
    data = viewmodel._get_data()
    if data is None:
        st.error("Không thể tải dữ liệu. Vui lòng kiểm tra lại.")
        return

    # Tạo layout với 2 cột
    col1, col2 = st.columns([1, 1])

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
        """, unsafe_allow_html=True)

        # Chọn tỉnh/thành phố
        city_options = sorted(data["city_province"].unique())
        city = st.selectbox("Tỉnh/Thành phố", city_options, key='city')

        # Lọc quận/huyện dựa trên tỉnh/thành phố đã chọn
        district_options = sorted(data[data["city_province"] == city]["district"].unique())
        district = st.selectbox("Quận/Huyện", district_options, key='district')

        st.markdown('</div>', unsafe_allow_html=True)

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
        """, unsafe_allow_html=True)

        # Một hàng 2 cột cho thông tin cơ bản
        bc1, bc2 = st.columns(2)
        with bc1:
            area = st.number_input("Diện tích (m²)", min_value=10.0, max_value=1000.0, value=80.0, step=10.0, key='area')
        with bc2:
            category_options = sorted(data["category"].unique())
            category = st.selectbox("Loại BĐS", category_options, key='category')

        # Hàng tiếp theo
        bc3, bc4 = st.columns(2)
        with bc3:
            direction_options = sorted(data["direction"].unique())
            direction = st.selectbox("Hướng nhà", direction_options, key='direction')
        with bc4:
            liability_options = sorted(data["liability"].unique())
            liability = st.selectbox("Tình trạng pháp lý", liability_options, key='liability')

        st.markdown('</div>', unsafe_allow_html=True)


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
        """, unsafe_allow_html=True)

        # Hàng 1
        rc1, rc2 = st.columns(2)
        with rc1:
            bedroom_num = st.number_input("Số phòng ngủ", min_value=0, max_value=10, value=2, step=1, key='bedroom')
        with rc2:
            toilet_num = st.number_input("Số phòng vệ sinh", min_value=0, max_value=10, value=2, step=1, key='toilet')

        # Hàng 2
        rc3, rc4 = st.columns(2)
        with rc3:
            livingroom_num = st.number_input("Số phòng khách", min_value=0, max_value=10, value=1, step=1, key='livingroom')
        with rc4:
            floor_num = st.number_input("Số tầng", min_value=0, max_value=50, value=2, step=1, key='floor')

        st.markdown('</div>', unsafe_allow_html=True)

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
        """, unsafe_allow_html=True)

        # Thông tin chiều rộng đường
        street_width = st.number_input("Chiều rộng đường (m)",
                                    min_value=0.0, max_value=50.0, value=8.0, step=0.5, key='street')

        st.markdown('</div>', unsafe_allow_html=True)

    # Sử dụng cách tiếp cận khác cho nút dự đoán
    st.markdown('<div style="padding: 10px 0 20px 0;"></div>', unsafe_allow_html=True)

    # Các nút được định dạng từ file CSS riêng
    st.markdown('<div class="prediction-button-wrapper"></div>', unsafe_allow_html=True)

    # Nút dự đoán
    if st.button("Dự đoán giá", use_container_width=True, type="tertiary"):
        # Chuẩn bị dữ liệu đầu vào
        input_data = {
            "area (m2)": area,
            "bedroom_num": bedroom_num,
            "floor_num": floor_num,
            "toilet_num": toilet_num,
            "livingroom_num": livingroom_num,
            "street (m)": street_width,
            "city_province": city,
            "district": district,
            "category": category,
            "direction": direction,
            "liability": liability,
            # Các trường cần thiết cho mô hình
            "price_per_m2": 0,  # Giá trị này sẽ bị bỏ qua trong dự đoán
            "price_log": 0      # Giá trị này sẽ bị bỏ qua trong dự đoán
        }

        # Dự đoán giá
        with st.spinner("Đang dự đoán giá..."):
            try:
                # Thêm hiệu ứng chờ để cải thiện UX
                progress_bar = st.progress(0)
                for percent_complete in range(0, 101, 20):
                    time.sleep(0.1)  # Tạo độ trễ giả để hiệu ứng đẹp hơn
                    progress_bar.progress(percent_complete)
                progress_bar.empty()  # Xóa thanh tiến trình sau khi hoàn thành

                # Cập nhật input data vào viewmodel
                for key, value in input_data.items():
                    viewmodel.update_input(key, value)

                # Thực hiện dự đoán
                prediction_result = viewmodel.predict()

                # Kiểm tra kết quả dự đoán và lỗi
                if prediction_result is None:
                    st.error("Không thể dự đoán giá. Vui lòng kiểm tra lại dữ liệu đầu vào.")
                    return
                
                # Kiểm tra thông báo lỗi
                if hasattr(prediction_result, 'error_message') and prediction_result.error_message:
                    st.error(f"Lỗi khi dự đoán: {prediction_result.error_message}")
                    st.warning("Vui lòng huấn luyện mô hình trước khi dự đoán. Bạn có thể huấn luyện mô hình trong phần Phân tích.")
                    return
                    
                # Kiểm tra giá trị dự đoán
                if not hasattr(prediction_result, 'predicted_price') or prediction_result.predicted_price <= 0:
                    st.error("Không thể tính toán giá trị dự đoán hợp lệ. Vui lòng kiểm tra lại dữ liệu đầu vào.")
                    return
                else:
                    # Lấy giá trị dự đoán
                    total_price = prediction_result.predicted_price
                    # Tính giá trỉ mỗi m2 (nếu cần)
                    predicted_price_per_m2 = int(round(total_price / area)) if area > 0 else 0
                    total_price_billion = total_price / 1_000_000_000

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

                    # Định dạng giá tổng
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
                                    <div class="price-value">{predicted_price_per_m2:,.0f} VND</div>
                                </div>
                                <div class="price-card">
                                    <div class="price-label">Tổng giá dự đoán</div>
                                    <div class="price-value">{formatted_total_price}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)

                # Lấy các bất động sản tương tự sử dụng phương thức đã cải tiến từ viewmodel
                # Cập nhật input trước khi lấy bất động sản tương tự
                viewmodel.update_input('area', area)
                viewmodel.update_input('city_province', city)
                viewmodel.update_input('district', district)

                # Gọi phương thức get_similar_properties đã được cải tiến trong viewmodel
                similar_properties = viewmodel.get_similar_properties(area)

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

                st.markdown('<div class="similar-data-wrapper">', unsafe_allow_html=True)
                if similar_properties is not None and not similar_properties.empty:
                    # Xác định các cột cần hiển thị dựa trên các cột có sẵn trong dataframe
                    display_columns = []
                    column_mapping = {}

                    # Kiểm tra và thêm các cột vào danh sách hiển thị
                    if 'area_m2' in similar_properties.columns:
                        display_columns.append('area_m2')
                        column_mapping['area_m2'] = 'Diện tích (m²)'
                    elif 'area' in similar_properties.columns:
                        display_columns.append('area')
                        column_mapping['area'] = 'Diện tích (m²)'

                    if 'price_per_m2' in similar_properties.columns:
                        display_columns.append('price_per_m2')
                        column_mapping['price_per_m2'] = 'Giá/m² (VND)'

                    if 'district' in similar_properties.columns:
                        display_columns.append('district')
                        column_mapping['district'] = 'Quận/Huyện'

                    if 'city_province' in similar_properties.columns:
                        display_columns.append('city_province')
                        column_mapping['city_province'] = 'Tỉnh/Thành phố'

                    # Thêm các cột khác nếu có
                    additional_columns = {
                        'bedroom_num': 'Số phòng ngủ',
                        'floor_num': 'Số tầng',
                        'category': 'Loại BĐS',
                        'bedrooms': 'Số phòng ngủ',
                        'bathrooms': 'Số phòng tắm',
                        'property_type': 'Loại BĐS',
                        'legal_status': 'Tình trạng pháp lý',
                        'year_built': 'Năm xây dựng',
                        'direction': 'Hướng nhà',
                    }

                    for eng_col, vie_col in additional_columns.items():
                        if eng_col in similar_properties.columns:
                            display_columns.append(eng_col)
                            column_mapping[eng_col] = vie_col

                    # Chọn tối đa 5 bất động sản để hiển thị
                    similar_df = similar_properties[display_columns].head(5).reset_index(drop=True)

                    # Đổi tên các cột sang tiếng Việt
                    similar_df.rename(columns=column_mapping, inplace=True)

                    # Format giá trị trong dataframe để hiển thị tốt hơn
                    # Xử lý an toàn tất cả các cột trong dataframe
                    for col in similar_df.columns:
                        # Chuyển các giá trị -1 thành N/A cho tất cả các cột
                        def format_value(x):
                            # Nếu giá trị là -1 hoặc '-1', trả về N/A
                            if x == -1 or x == '-1':
                                return "N/A"

                            # Định dạng các giá trị số
                            try:
                                if col == 'Giá/m² (VND)' and float(x) > 0:
                                    return f"{float(x):,.0f}"
                                elif col == 'Diện tích (m²)' and float(x) > 0:
                                    return f"{float(x):.1f}"
                                return str(x)
                            except (ValueError, TypeError):
                                # Nếu không thể chuyển thành số, giữ nguyên giá trị
                                return str(x)

                        # Áp dụng hàm format cho từng cột
                        similar_df[col] = similar_df[col].apply(format_value)

                    st.dataframe(similar_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Không tìm thấy bất động sản tương tự.")
                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                logger.error(f"Lỗi khi dự đoán trong view: {e}")
                st.error("Xảy ra lỗi khi dự đoán giá. Vui lòng huấn luyện mô hình trước trong phần Phân tích và thử lại.")
                st.info("Chi tiết lỗi: " + str(e))
