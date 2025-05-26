## 🎯 Dự Đoán Giá Bất Động Sản Việt Nam

### 1. Kiến trúc ứng dụng

* Ứng dụng được xây dựng theo mô hình **MVVM (Model-View-ViewModel)**:

  * **Model**: Xử lý dữ liệu và triển khai logic nghiệp vụ
  * **View**: Giao diện người dùng với Streamlit
  * **ViewModel**: Kết nối Model và View, xử lý tương tác
* Hệ thống nhật ký đa cấp độ tích hợp:

  * Các cấp độ: DEBUG, INFO, WARNING, ERROR, CRITICAL
  * Ghi vào cả console và tập tin
  * Giao diện xem và quản lý nhật ký

### 2. Tham số đầu vào

* Mô hình yêu cầu các thông tin sau:

  * **Vị trí**: tọa độ địa lý hoặc địa chỉ chuẩn hóa
  * **Diện tích**: kích thước tính bằng mét vuông
  * **Số phòng**: giá trị nguyên
  * **Tình trạng pháp lý**, **Hướng nhà**, **Năm xây dựng**: giá trị phân loại hoặc số
* Xác thực đầu vào:

  * Tất cả các trường bắt buộc phải được cung cấp để dự đoán chính xác
  * Dữ liệu phải tuân thủ các phạm vi và định dạng mong đợi

### 3. Yêu cầu xử lý dữ liệu

* Dữ liệu đầu vào nên được cấu trúc dưới dạng **JSON** để xử lý
* Xác thực dữ liệu bao gồm:
  * Diện tích phải > 0
  * Năm xây dựng phải nằm trong khoảng hợp lý
  * Ghi nhật ký các lỗi xác thực với cấp độ WARNING hoặc ERROR

### 4. Xử lý mô hình

* Hệ thống dự đoán:

  * Xử lý dữ liệu đầu vào đã cấu trúc
  * Chạy **mô hình dự đoán giá bất động sản đã được huấn luyện trước**
  * Tạo kết quả dự đoán:

    * **Giá trên mỗi mét vuông** hoặc
    * **Tổng giá trị bất động sản**

* Quá trình xử lý được ghi nhật ký chi tiết:

  * Ghi lại thời gian thực thi của các bước quan trọng
  * Xử lý lỗi một cách thanh lịch với thông báo lỗi phù hợp

### 5. Định dạng đầu ra

* Kết quả dự đoán bao gồm:

  * Tính toán **giá trên mỗi mét vuông** và **tổng giá**
  * **Khoảng tin cậy** của dự đoán

* Phân tích so sánh:

  * Giá trung bình cho các khu vực xung quanh
  * Thông tin thống kê bao gồm giá tối thiểu, trung bình và tối đa trong khu vực
  * Biểu đồ trực quan hóa so sánh

### 6. Yêu cầu kỹ thuật

* **Mô hình học máy**:

  * Mô hình được huấn luyện trước và có thể được tải theo yêu cầu
  * Không yêu cầu huấn luyện lại trong quá trình dự đoán

* **Đa nền tảng**:

  * Chạy trên Windows, macOS và Linux
  * Tích hợp Ngrok để tạo URL công khai

* **Tiêu chuẩn bảo mật tối thiểu**:

  * Thông tin người dùng nhạy cảm (như địa chỉ chi tiết) không được ghi nhật ký
  * Hệ thống nhật ký có cơ chế lọc thông tin nhạy cảm trước khi ghi
  * Configuration details should be stored securely and excluded from version control.

### 6. Environment Requirements

* **Cross-Platform Support**:

  * The model can run on macOS, Linux, and Windows environments.
  * OS-specific dependencies may be required.
  * Python virtual environments are recommended for dependency management.
* **Environment Configuration**:

  * Environment variables may be used for configuration.
  * Sensitive configuration should be stored in separate files not included in version control.
* **Version Control**:

  * The following should be excluded from version control:
    * Sensitive configuration files
    * Datasets
    * Environment-specific files
    * Generated artifacts

---

## ✨ Enhancements

* Informative messaging:
  * Status messages to indicate successful prediction.
* Smart input validation:

  * "Area too small" warnings (e.g., under 10m²)
  * "Unusual construction year" warnings (e.g., year > current year)
* Data visualization:

  * Clear presentation of results with appropriate visual elements.
