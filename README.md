# MÔN: DỮ LIỆU LỚN VÀ ỨNG DỤNG

## Ứng Dụng Dự Đoán Giá Bất Động Sản Việt Nam

## Giới thiệu

Ứng dụng dự đoán giá bất động sản Việt Nam được xây dựng trên nền tảng Streamlit, PySpark và mô hình học máy. Ứng dụng cung cấp khả năng dự đoán giá bất động sản dựa trên các đặc điểm của tài sản và phân tích dữ liệu thị trường bất động sản.

## 👥 Nhóm phát triển

- **Lê Thị Cẩm Giang** - Tác giả  <https://github.com/lcg1908>
- **Nguyễn Quỳnh Anh** - Đồng tác giả  <https://github.com/Quynanhng25>
- **Nguyễn Cao Hoài Duyên** - Đồng tác giả <https://github.com/CaoHoaiDuyen>
- **Đinh Trương Ngọc Quỳnh Hoa** - Đồng tác giả <https://github.com/QHoa036>
- **Trần Hoàng Nghĩa** - Đồng tác giả <https://github.com/Blink713>
- **Nguyễn Phương Thảo** - Đồng tác giả <https://github.com/thaonguyenbi>

## Cấu trúc kiến trúc MVVM

Ứng dụng được thiết kế theo mô hình kiến trúc MVVM (Model-View-ViewModel) để tạo sự phân tách rõ ràng giữa các thành phần và dễ dàng bảo trì, mở rộng.

### Cấu trúc thư mục

```txt
Vietnam_Real_Estate_Price_Prediction/
│── App/                                   # Ứng dụng chính
│   └── src/                               # Mã nguồn chính của ứng dụng
│       │── data/                          # Dữ liệu mẫu cho phiên bản demo
│       │── logs/                          # Thư mục nhật ký
│       │── styles/                        # CSS và các tài nguyên giao diện
│       │── utils/                         # Các tiện ích
│       │   │── spark_utils.py             # Tiện ích PySpark
│   └── vn_real_estate_app.py              # Điểm vào chính của ứng dụng
│── References/                            # Tài liệu tham khảo
│── .env.example                           # Mẫu cấu hình biến môi trường
│── run_app.sh                             # Tập lệnh chạy ứng dụng chính (đa nền tảng)
```

## Mô tả các thành phần

### App - Ứng dụng chính

Ứng dụng chính là phiên bản đơn giản hóa, tối ưu cho môi trường sản xuất, hội tụ tất cả các tính năng chính trong một tập tin duy nhất:

- **vn_real_estate_app.py**: Tập tin ứng dụng độc lập hội tụ tất cả các chức năng:
  - Xử lý dữ liệu và huấn luyện mô hình
  - Giao diện dự đoán giá bất động sản
  - Trực quan hóa và phân tích dữ liệu
  - Tích hợp Ngrok để tạo URL công khai
  - Hỗ trợ đa nền tảng (Windows, macOS, Linux)

### Demo - Phiên bản cấu trúc module hóa theo MVVM

Phiên bản demo được xây dựng theo mô hình MVVM với cấu trúc rõ ràng, dễ bảo trì và mở rộng:

#### 1. Model

- **property_model.py**: Định nghĩa các lớp mô hình dữ liệu bất động sản và kết quả dự đoán
- **data_service.py**: Cung cấp dịch vụ xử lý dữ liệu, tải dữ liệu và huấn luyện mô hình

#### 2. ViewModel

- **app_viewmodel.py**: Quản lý trạng thái và logic chung của ứng dụng
- **prediction_viewmodel.py**: Xử lý logic dự đoán giá bất động sản
- **analytics_viewmodel.py**: Xử lý logic phân tích dữ liệu và biểu đồ

#### 3. View

- **app_view.py**: Định nghĩa giao diện chính và điều hướng
- **prediction_view.py**: Giao diện dự đoán giá bất động sản
- **analytics_view.py**: Hiển thị biểu đồ phân tích dữ liệu
- **about_view.py**: Hiển thị thông tin về ứng dụng

#### 4. Utils

- **logger_utils.py**: Hệ thống nhật ký đa cấp độ tích hợp
- **ui_utils.py**: Tiện ích giao diện người dùng
- **spark_utils.py**: Hỗ trợ cấu hình PySpark
- **session_utils.py**: Quản lý session và lưu trữ metrics
- **ngrok_utils.py**: Tiện ích tạo URL công khai với Ngrok

### Tài liệu và hướng dẫn

- **Docs/**: Chứa các tài liệu hướng dẫn sử dụng và phạm vi ứng dụng
- **References/**: Chứa tài liệu tham khảo và nghiên cứu liên quan

## 🚀 Chạy ứng dụng

Ứng dụng được thiết kế để chạy trên nhiều nền tảng khác nhau (Windows, macOS, Linux) chỉ với một lệnh duy nhất:

```bash
./run_app.sh
```

Tập lệnh `run_app.sh` sẽ tự động thực hiện:

1. Phát hiện hệ điều hành và thiết lập môi trường
2. Cài đặt các phụ thuộc cần thiết
3. Tạo và kích hoạt môi trường ảo Python
4. Hỏi người dùng có muốn sử dụng Ngrok không
5. Khởi chạy ứng dụng Streamlit

### Sử dụng với Ngrok

Để tạo URL công khai và chia sẻ ứng dụng:

1. Tạo tài khoản tại [ngrok.com](https://ngrok.com)
2. Lấy token xác thực từ bảng điều khiển
3. Chọn 'y' khi được hỏi về Ngrok
4. Nhập token (sẽ được lưu cho lần sau)

## 📊 Hệ thống nhật ký

Ứng dụng tích hợp hệ thống nhật ký toàn diện:

- Nhiều cấp độ nhật ký (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Ghi vào cả console và tập tin
- Giao diện xem nhật ký trong Streamlit
- Lọc và tải xuống nhật ký
- Ghi thời gian thực thi của các hàm quan trọng

## 🌟 Tính năng chính

1. **Dự đoán giá bất động sản**
   - Nhập các đặc điểm bất động sản và nhận dự đoán giá trị
   - Form tương tác với kết quả tức thì
   - Hiển thị khoảng tin cậy dự đoán

2. **Phân tích dữ liệu**
   - Biểu đồ phân phối giá theo khu vực
   - Trực quan hóa mối quan hệ giữa các đặc điểm và giá
   - Bản đồ xu hướng giá theo địa lý

3. **Thống kê thị trường**
   - So sánh giá theo loại bất động sản và vị trí
   - Phân tích xu hướng giá
   - Xác định yếu tố ảnh hưởng đến giá

4. **Giao diện người dùng hiện đại**
   - Thiết kế responsive cho máy tính và di động
   - Điều hướng trực quan
   - Trực quan hóa tương tác

## 💻 Yêu cầu hệ thống

- **Python 3.8+**
- **Java Runtime Environment (JRE)** cho PySpark
- **Git Bash** (khuyến nghị cho Windows)
- **Các thư viện Python**: streamlit, pyspark, pandas, numpy, plotly, matplotlib, seaborn, pyngrok, python-dotenv

## Lời cảm ơn

Nhóm chúng em xin gửi lời cảm ơn sâu sắc đến thầy Nguyễn Mạnh Tuấn, giảng viên bộ môn Dữ liệu lớn và ứng dụng tại Đại học UEH, vì đã tận tình hướng dẫn, truyền đạt kiến thức và kinh nghiệm quý báu giúp chúng em không chỉ nắm vững lý thuyết mà còn áp dụng vào thực tế. Xin chân thành cảm ơn thầy vì tâm huyết và sự nhiệt tình giúp nhóm hoàn thiện đề tài tốt nhất.