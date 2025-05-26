# 🎯 Dự Đoán Giá Bất Động Sản Việt Nam

## 1. Kiến trúc ứng dụng

* Ứng dụng được xây dựng theo mô hình **MVVM (Model-View-ViewModel)**:

  * **Model**: Xử lý dữ liệu và triển khai logic nghiệp vụ
  * **View**: Giao diện người dùng với Streamlit
  * **ViewModel**: Kết nối Model và View, xử lý tương tác
* Hệ thống nhật ký đa cấp độ tích hợp:

  * Các cấp độ: DEBUG, INFO, WARNING, ERROR, CRITICAL
  * Ghi vào cả console và tập tin
  * Giao diện xem và quản lý nhật ký

## 2. Tham số đầu vào

* Mô hình yêu cầu các thông tin sau:

  * **Vị trí**: tọa độ địa lý hoặc địa chỉ chuẩn hóa
  * **Diện tích**: kích thước tính bằng mét vuông
  * **Số phòng**: giá trị nguyên
  * **Tình trạng pháp lý**, **Hướng nhà**, **Năm xây dựng**: giá trị phân loại hoặc số
* Xác thực đầu vào:

  * Tất cả các trường bắt buộc phải được cung cấp để dự đoán chính xác
  * Dữ liệu phải tuân thủ các phạm vi và định dạng mong đợi

## 3. Yêu cầu xử lý dữ liệu

* Dữ liệu đầu vào nên được cấu trúc dưới dạng **JSON** để xử lý
* Xác thực dữ liệu bao gồm:
  * Diện tích phải > 0
  * Năm xây dựng phải nằm trong khoảng hợp lý
  * Ghi nhật ký các lỗi xác thực với cấp độ WARNING hoặc ERROR

## 4. Xử lý mô hình

* Hệ thống dự đoán:

  * Xử lý dữ liệu đầu vào đã cấu trúc
  * Chạy **mô hình dự đoán giá bất động sản đã được huấn luyện trước**
  * Tạo kết quả dự đoán:

    * **Giá trên mỗi mét vuông** hoặc
    * **Tổng giá trị bất động sản**

* Quá trình xử lý được ghi nhật ký chi tiết:

  * Ghi lại thời gian thực thi của các bước quan trọng
  * Xử lý lỗi một cách thanh lịch với thông báo lỗi phù hợp

## 5. Định dạng đầu ra

* Kết quả dự đoán bao gồm:

  * Tính toán **giá trên mỗi mét vuông** và **tổng giá**
  * **Khoảng tin cậy** của dự đoán

* Phân tích so sánh:

  * Giá trung bình cho các khu vực xung quanh
  * Thông tin thống kê bao gồm giá tối thiểu, trung bình và tối đa trong khu vực
  * Biểu đồ trực quan hóa so sánh

## 6. Yêu cầu kỹ thuật

* **Mô hình học máy**:

  * Mô hình được huấn luyện trước và có thể được tải theo yêu cầu
  * Không yêu cầu huấn luyện lại trong quá trình dự đoán

* **Đa nền tảng**:

  * Chạy trên Windows, macOS và Linux
  * Tích hợp Ngrok để tạo URL công khai

* **Tiêu chuẩn bảo mật tối thiểu**:

  * Thông tin người dùng nhạy cảm (như địa chỉ chi tiết) không được ghi nhật ký
  * Hệ thống nhật ký có cơ chế lọc thông tin nhạy cảm trước khi ghi
  * Các thông tin cấu hình phải được lưu trữ an toàn và loại trừ khỏi kiểm soát phiên bản

## 7. Yêu cầu môi trường

* **Hỗ trợ đa nền tảng**:

  * Mô hình có thể chạy trên macOS, Linux và Windows
  * Có thể cần các phụ thuộc đặc thù theo hệ điều hành
  * Khuyến nghị sử dụng môi trường ảo Python để quản lý phụ thuộc
* **Cấu hình môi trường**:

  * Có thể sử dụng biến môi trường cho cấu hình
  * Cấu hình nhạy cảm nên được lưu trữ trong các tập tin riêng không đưa vào kiểm soát phiên bản
* **Kiểm soát phiên bản**:

  * Các mục sau nên được loại trừ khỏi kiểm soát phiên bản:
    * Tập tin cấu hình nhạy cảm
    * Bộ dữ liệu
    * Tập tin đặc thù theo môi trường
    * Các tài nguyên được tạo ra

---

## Tính năng nâng cao

* Thông báo thông minh:
  * Thông báo trạng thái để chỉ rõ việc dự đoán thành công
* Xác thực đầu vào thông minh:

  * Cảnh báo "Diện tích quá nhỏ" (ví dụ: dưới 10m²)
  * Cảnh báo "Năm xây dựng bất thường" (ví dụ: năm > năm hiện tại)
* Trực quan hóa dữ liệu:

  * Hiển thị kết quả rõ ràng với các yếu tố hình ảnh phù hợp
