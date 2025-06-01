#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vietnam Real Estate Price Prediction
About view for displaying project information
"""

import streamlit as st

def render_about_view():
    """
    Hiển thị giao diện thông tin về dự án
    """
    st.markdown("""
    <div class="about-header">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Flag_of_Vietnam.svg/1200px-Flag_of_Vietnam.svg.png" width="100">
        <div class="about-header-text">
            <h1>Dự đoán giá BĐS Việt Nam</h1>
            <p>Hệ thống dự đoán giá bất động sản dựa trên học máy và Apache Spark</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Giới thiệu tổng quan
    st.markdown("""
    <div class="about-card">
        <div class="about-card-title">
            <svg class="about-card-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M19 9.3V4h-3v2.6L12 3L2 12h3v8h6v-6h2v6h6v-8h3L19 9.3zM17 18h-2v-6H9v6H7v-7.81l5-4.5 5 4.5V18z" fill="currentColor"/>
            </svg>
            <h2>Giới thiệu dự án</h2>
        </div>
        <div class="about-card-content">
            <p>Đây là một ứng dụng <strong>demo</strong> cho mô hình dự đoán giá bất động sản tại Việt Nam sử dụng học máy.</p>
            <p>Ứng dụng là một phần của <strong>dự án nghiên cứu</strong> nhằm khai thác dữ liệu lớn trong phân tích thị trường bất động sản.</p>
            <p>Mục tiêu dự án:</p>
            <ul>
                <li>Xây dựng mô hình dự đoán chính xác giá bất động sản tại Việt Nam</li>
                <li>Tìm hiểu các yếu tố ảnh hưởng đến giá bất động sản</li>
                <li>Tạo nền tảng thu thập và phân tích dữ liệu thị trường BDS tự động</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Công nghệ sử dụng
    st.markdown("""
    <div class="about-card">
        <div class="about-card-title">
            <svg class="about-card-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M22.7 19l-9.1-9.1c.9-2.3.4-5-1.5-6.9-2-2-5-2.4-7.4-1.3L9 6 6 9 1.6 4.7C.4 7.1.9 10.1 2.9 12.1c1.9 1.9 4.6 2.4 6.9 1.5l9.1 9.1c.4.4 1 .4 1.4 0l2.3-2.3c.5-.4.5-1.1.1-1.4z" fill="currentColor"/>
            </svg>
            <h2>Công nghệ sử dụng</h2>
        </div>
        <div class="about-card-content">
            <p>Dự án sử dụng các công nghệ hiện đại để xử lý dữ liệu lớn và học máy:</p>
            <div style="margin-top: 15px;">
                <span class="tech-tag">Selenium</span>
                <span class="tech-tag">Apache Spark</span>
                <span class="tech-tag">PySpark</span>
                <span class="tech-tag">Gradient Boosted Trees</span>
                <span class="tech-tag">Random Forest</span>
                <span class="tech-tag">Linear Regression</span>
                <span class="tech-tag">Streamlit</span>
                <span class="tech-tag">Ngrok</span>
                <span class="tech-tag">Python</span>
            </div>
            <p style="margin-top: 15px;">Từ giải pháp thu thập dữ liệu, đến xem xét dữ liệu lớn, xây dựng mô hình và cung cấp giao diện người dùng, dự án được phát triển với các công nghệ tốt nhất trong lĩnh vực.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Thêm thành viên nhóm
    st.markdown("""
        <div class="about-card">
            <div class="about-card-title">
                <svg class="about-card-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5c-1.66 0-3 1.34-3 3s1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5C6.34 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z" fill="currentColor"/>
                </svg>
                <h2>Thành viên nhóm</h2>
            </div>
            <div class="about-card-content">
                <ul style="margin-top: 10px; list-style-type: none; padding-left: 0;">
                    <li style="margin-bottom: 20px;">
                        <div style="display: flex; align-items: center;">
                            <img src="https://github.com/lcg1908.png" style="width: 50px; height: 50px; border-radius: 50%; margin-right: 15px; border: 2px solid #4c9aff;">
                            <div>
                                <strong>Lê Thị Cẩm Giang</strong>
                                <p style="margin: 0;"><a href="https://github.com/lcg1908" style="color: #4c9aff; text-decoration: none;">github.com/lcg1908</a></p>
                            </div>
                        </div>
                    </li>
                    <li style="margin-bottom: 20px;">
                        <div style="display: flex; align-items: center;">
                            <img src="https://github.com/Blink713.png" style="width: 50px; height: 50px; border-radius: 50%; margin-right: 15px; border: 2px solid #4c9aff;">
                            <div>
                                <strong>Trần Hoàng Nghĩa</strong>
                                <p style="margin: 0;"><a href="https://github.com/Blink713" style="color: #4c9aff; text-decoration: none;">github.com/Blink713</a></p>
                            </div>
                        </div>
                    </li>
                    <li style="margin-bottom: 20px;">
                        <div style="display: flex; align-items: center;">
                            <img src="https://github.com/Quynanhng25.png" style="width: 50px; height: 50px; border-radius: 50%; margin-right: 15px; border: 2px solid #4c9aff;">
                            <div>
                                <strong>Nguyễn Quỳnh Anh</strong>
                                <p style="margin: 0;"><a href="https://github.com/Quynanhng25" style="color: #4c9aff; text-decoration: none;">github.com/Quynanhng25</a></p>
                            </div>
                        </div>
                    </li>
                    <li style="margin-bottom: 20px;">
                        <div style="display: flex; align-items: center;">
                            <img src="https://github.com/CaoHoaiDuyen.png" style="width: 50px; height: 50px; border-radius: 50%; margin-right: 15px; border: 2px solid #4c9aff;">
                            <div>
                                <strong>Nguyễn Cao Hoài Duyên</strong>
                                <p style="margin: 0;"><a href="https://github.com/CaoHoaiDuyen" style="color: #4c9aff; text-decoration: none;">github.com/CaoHoaiDuyen</a></p>
                            </div>
                        </div>
                    </li>
                    <li style="margin-bottom: 20px;">
                        <div style="display: flex; align-items: center;">
                            <img src="https://github.com/QHoa036.png" style="width: 50px; height: 50px; border-radius: 50%; margin-right: 15px; border: 2px solid #4c9aff;">
                            <div>
                                <strong>Đinh Trương Ngọc Quỳnh Hoa</strong>
                                <p style="margin: 0;"><a href="https://github.com/QHoa036" style="color: #4c9aff; text-decoration: none;">github.com/QHoa036</a></p>
                            </div>
                        </div>
                    </li>
                    <li style="margin-bottom: 20px;">
                        <div style="display: flex; align-items: center;">
                            <img src="https://github.com/thaonguyenbi.png" style="width: 50px; height: 50px; border-radius: 50%; margin-right: 15px; border: 2px solid #4c9aff;">
                            <div>
                                <strong>Nguyễn Phương Thảo</strong>
                                <p style="margin: 0;"><a href="https://github.com/thaonguyenbi" style="color: #4c9aff; text-decoration: none;">github.com/thaonguyenbi</a></p>
                            </div>
                        </div>
                    </li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="about-card">
        <div class="about-card-title">
            <svg class="about-card-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M19 3h-4.18C14.4 1.84 13.3 1 12 1c-1.3 0-2.4.84-2.82 2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 0c.55 0 1 .45 1 1s-.45 1-1 1-1-.45-1-1 .45-1 1-1zm-2 14l-4-4 1.41-1.41L10 14.17l6.59-6.59L18 9l-8 8z" fill="currentColor"/>
            </svg>
            <h2>Quy trình xử lý dữ liệu</h2>
        </div>
        <div class="about-card-content">
            <ol style="padding-left: 1.5rem;">
                <li>
                    <strong>Thu thập dữ liệu</strong>:
                    <p>Web scraping từ các trang bất động sản sử dụng Selenium và BeautifulSoup</p>
                </li>
                <li>
                    <strong>Làm sạch dữ liệu</strong>:
                    <p>Loại bỏ giá trị thiếu, chuẩn hóa định dạng, xử lý ngoại lệ để đảm bảo dữ liệu chất lượng cao</p>
                </li>
                <li>
                    <strong>Tạo đặc trưng</strong>:
                    <p>Feature engineering & encoding để biến đổi dữ liệu thô thành các đặc trưng hữu ích cho mô hình</p>
                </li>
                <li>
                    <strong>Huấn luyện mô hình</strong>:
                    <p>Sử dụng Gradient Boosted Trees (GBT) và các thuật toán học máy tiên tiến<br>
                </li>
            </ol>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hướng dẫn sử dụng
    st.markdown(f"""
    <div class="about-card">
        <div class="about-card-title">
            <svg class="about-card-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M21 5c-1.11-.35-2.33-.5-3.5-.5-1.95 0-4.05.4-5.5 1.5-1.45-1.1-3.55-1.5-5.5-1.5S2.45 4.9 1 6v14.65c0 .25.25.5.5.5.1 0 .15-.05.25-.05C3.1 20.45 5.05 20 6.5 20c1.95 0 4.05.4 5.5 1.5 1.35-.85 3.8-1.5 5.5-1.5 1.65 0 3.35.3 4.75 1.05.1.05.15.05.25.05.25 0 .5-.25.5-.5V6c-.6-.45-1.25-.75-2-1zm0 13.5c-1.1-.35-2.3-.5-3.5-.5-1.7 0-4.15.65-5.5 1.5V8c1.35-.85 3.8-1.5 5.5-1.5 1.2 0 2.4.15 3.5.5v11.5z" fill="currentColor"/>
                <path d="M17.5 10.5c.88 0 1.73.09 2.5.26V9.24c-.79-.15-1.64-.24-2.5-.24-1.7 0-3.24.29-4.5.83v1.66c1.13-.64 2.7-.99 4.5-.99zM13 12.49v1.66c1.13-.64 2.7-.99 4.5-.99.88 0 1.73.09 2.5.26V11.9c-.79-.15-1.64-.24-2.5-.24-1.7 0-3.24.3-4.5.83zm4.5 1.84c-1.7 0-3.24.29-4.5.83v1.66c1.13-.64 2.7-.99 4.5-.99.88 0 1.73.09 2.5.26v-1.52c-.79-.16-1.64-.24-2.5-.24z" fill="currentColor"/>
            </svg>
            <h2>Hướng dẫn sử dụng</h2>
        </div>
        <div class="about-card-content">
            <p>Ứng dụng có giao diện trực quan và dễ sử dụng:</p>
            <ul style="margin-top: 10px;">
                <li>
                    <strong>Dự đoán giá:</strong>
                    <p>Chọn tab "Dự đoán giá" ở thanh bên trái, nhập thông tin và nhấn nút dự đoán để xem kết quả.</p>
                </li>
                <li>
                    <strong>Phân tích dữ liệu:</strong>
                    <p>Chọn tab "Phân tích dữ liệu" để khám phá các biểu đồ và xu hướng thị trường bất động sản.</p>
                </li>
                <li>
                    <strong>Chia sẻ ứng dụng:</strong>
                    <p>Sử dụng Ngrok để tạo URL public và chia sẻ ứng dụng với người khác.</p>
                </li>
            </ul>
            <div style="margin-top: 15px; padding: 10px; background: rgba(255, 193, 7, 0.15); border-left: 3px solid #FFC107; border-radius: 4px;">
                <strong style="color: #FFC107; margin-right:2px">Lưu ý:</strong> Để có kết quả dự đoán chính xác, hãy nhập đầy đủ các thông tin chi tiết về bất động sản.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
