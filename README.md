# Dự Đoán Giá Bất Động Sản Việt Nam

Ứng dụng web dựa trên Python để dự đoán giá bất động sản tại Việt Nam sử dụng bộ dữ liệu tự thu thập, Apache Spark, PySpark và Streamlit.

![Cờ Việt Nam](https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Flag_of_Vietnam.svg/1200px-Flag_of_Vietnam.svg.png)

## Tổng Quan Dự Án

Dự án này triển khai một pipeline hoàn chỉnh về học máy để dự đoán giá bất động sản tại Việt Nam:

- **Thu Thập Dữ Liệu**: Web scraping từ nhadat.cafeland.vn sử dụng Selenium
- **Xử Lý Dữ Liệu**: Xử lý dữ liệu lớn với Apache Spark (PySpark)
- **Học Máy**: Huấn luyện mô hình cây tăng cường độ dốc (Gradient Boosted Trees)
- **Ứng Dụng Web**: Giao diện người dùng tương tác xây dựng bằng Streamlit
- **Triển Khai Đa Nền Tảng**: Hỗ trợ macOS, Linux và Windows
- **Công Khai Ứng Dụng**: Tích hợp ngrok để tạo URL public để chia sẻ

## Công Nghệ Sử Dụng

- **Python** - Ngôn ngữ lập trình chính
- **PySpark** - Xử lý dữ liệu lớn và huấn luyện mô hình ML
- **Streamlit** - Framework giao diện người dùng đồ họa
- **Ngrok** - Công cụ tạo URL public thông qua tunnel an toàn
- **CSS & HTML** - Giao diện hiện đại và responsive
- **Shell Scripting** - Tự động hóa quy trình triển khai

## Cấu Trúc Dự Án

```
Vietnam_Real_Estate_Price_Prediction/
├── App/                         # Thư mục chứa ứng dụng chính
│   ├── 1_fetch_real_estate.py   # Thu thập dữ liệu với Selenium
│   ├── 2_property_details.py    # Trích xuất thông tin chi tiết bất động sản
│   ├── 3_preprocess_data.py     # Làm sạch và tiền xử lý dữ liệu với PySpark
│   ├── 4_HDFS_storage.py        # Tích hợp HDFS cho lưu trữ dữ liệu lớn
│   ├── 5_model_training.py      # Huấn luyện mô hình ML với PySpark ML
│   ├── 6_streamlit_app.py       # Ứng dụng web Streamlit
│   └── 7_visualize_data.py      # Các thành phần trực quan hóa dữ liệu
├── Demo/                         # Thư mục chứa phiên bản demo
│   ├── data/                    # Dữ liệu mẫu cho phiên bản demo
│   ├── styles/                  # CSS và các tài nguyên giao diện
│   ├── utils/                   # Các tiện ích hỗ trợ (Spark, logging)
│   └── vn_real_estate_app.py    # Ứng dụng Streamlit demo
├── References/                   # Tài liệu tham khảo
├── Docs/                         # Các file tài liệu
├── manage_token.py               # Tiện ích quản lý token ngrok
├── run_demo.sh                   # Script chạy phiên bản demo
├── run_app.sh                    # Script chạy ứng dụng đầy đủ
└── run_with_ngrok.py             # Công cụ tạo URL public với ngrok
```

## 🔄 Quy Trình Xử Lý Dữ Liệu

Dự án triển khai quy trình xử lý dữ liệu từ đầu đến cuối:

### 1. Thu Thập Dữ Liệu

- Thu thập dữ liệu sử dụng Selenium từ trang bất động sản nhadat.cafeland.vn
- Trích xuất thông tin chi tiết bất động sản bao gồm vị trí, giá, diện tích, tính năng, v.v.
- Lưu trữ dưới định dạng CSV để xử lý tiếp theo

```python
# Ví dụ mã từ App/1_fetch_real_estate.py
def fetch_real_estate_listings(base_url, num_pages=5):
    """
    Lấy danh sách bất động sản từ nhadat.cafeland.vn
    """
    # Chi tiết triển khai về scraping bất động sản
```

### 2. Tiền Xử Lý Dữ Liệu

- Làm sạch và chuyển đổi dữ liệu sử dụng Apache Spark (PySpark)
- Kỹ thuật đặc trưng để tạo ra các thuộc tính có ý nghĩa
- Xử lý giá trị thiếu và ngoại lai
- Lưu trữ trong HDFS để xử lý hiệu quả

```python
# Ví dụ mã từ App/3_preprocess_data.py
def preprocess_data(input_file, output_path):
    """Hàm chính để tiền xử lý dữ liệu bất động sản."""
    # Triển khai pipeline tiền xử lý
```

### 3. Huấn Luyện Mô Hình

- Lựa chọn và chuẩn bị đặc trưng
- Huấn luyện sử dụng các mô hình Gradient Boosted Trees, Random Forest, và Linear Regression
- Tinh chỉnh siêu tham số để đạt hiệu suất tối ưu
- Đánh giá và lựa chọn mô hình dựa trên các chỉ số như R², RMSE, và MAE

```python
# Ví dụ mã từ App/5_model_training.py
def train_real_estate_model(data_path, output_dir="model"):
    """Hàm chính để huấn luyện mô hình dự đoán giá bất động sản."""
    # Triển khai pipeline huấn luyện mô hình
```

### 4. Ứng Dụng Web

- Ứng dụng Streamlit tương tác với giao diện người dùng hiện đại
- Dự đoán giá thời gian thực dựa trên đầu vào của người dùng
- Khả năng trực quan hóa và khám phá dữ liệu
- Triển khai với ngrok để truy cập trên đám mây

```python
# Ví dụ mã từ App/6_streamlit_app.py
def predict_price(pipeline_model, regression_model, input_data):
    """Dự đoán giá sử dụng các mô hình đã được huấn luyện."""
    # Triển khai chức năng dự đoán giá
```

## 🚀 Bắt Đầu

### Yêu Cầu Tiên Quyết

- Python 3.8+
- Java 8+ (cho Apache Spark)
- Apache Spark 3.0+
- PySpark
- Selenium WebDriver
- Streamlit

### Cài Đặt

1. Clone repository:
   ```bash
   git clone https://github.com/QHoa036/Vietnam_Real_Estate_Price_Prediction.git
   cd Vietnam_Real_Estate_Price_Prediction
   ```

2. Cài đặt các gói cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

3. Thiết lập môi trường Spark:
   ```bash
   # Đảm bảo các biến môi trường JAVA_HOME và SPARK_HOME được thiết lập
   export JAVA_HOME=/path/to/java
   export SPARK_HOME=/path/to/spark
   export PYSPARK_PYTHON=python3
   ```

### Chạy Ứng Dụng

1. Thu thập dữ liệu (tùy chọn nếu bạn đã có bộ dữ liệu):
   ```bash
   python App/1_fetch_real_estate.py
   ```

2. Tiền xử lý dữ liệu:
   ```bash
   python App/3_preprocess_data.py
   ```

3. Huấn luyện mô hình:
   ```bash
   python App/5_model_training.py
   ```

4. Khởi chạy ứng dụng Streamlit:
   ```bash
   streamlit run App/6_streamlit_app.py
   ```

5. Để triển khai đám mây với ngrok:
   ```bash
   streamlit run Demo/vn_real_estate_app.py
   ```

## 🌟 Tính Năng

### Dự Đoán Giá

- Ước tính giá bất động sản dựa trên vị trí, diện tích, và các đặc điểm khác
- Form tương tác để nhập thông tin bất động sản
- Dự đoán giá tức thì với khoảng tin cậy

### Khám Phá Dữ Liệu

- Trực quan hóa phân phối giá trên các khu vực khác nhau
- Phân tích mối quan hệ giữa các đặc điểm bất động sản và giá
- Bản đồ tương tác hiển thị xu hướng giá theo địa lý

### Phân Tích Thị Trường

- So sánh giá theo loại bất động sản, vị trí, và các tính năng
- Theo dõi xu hướng giá theo thời gian
- Xác định các yếu tố chính ảnh hưởng đến giá bất động sản

## 📊 Hiệu Suất Mô Hình

Mô hình Gradient Boosted Trees đạt được:
- **R²**: 0.85+ (Hệ số xác định)
- **RMSE**: ~15% giá trung bình
- **MAE**: ~12% giá trung bình

Các yếu tố chính ảnh hưởng đến giá bất động sản:
- Vị trí (quận/huyện và thành phố/tỉnh)
- Diện tích bất động sản
- Số phòng ngủ và phòng tắm
- Tình trạng pháp lý
- Độ rộng đường

## 📱 Giao Diện Người Dùng

Ứng dụng Streamlit có các tính năng:
- Thiết kế responsive cho máy tính và di động
- Giao diện hiện đại với điều hướng trực quan
- Trực quan hóa tương tác
- Dự đoán giá thời gian thực

Giao diện được chia thành bốn phần chính:
1. **Trang Chủ** - Tổng quan và thống kê nhanh
2. **Dự Đoán Giá** - Form nhập thông tin bất động sản
3. **Khám Phá Dữ Liệu** - Trực quan hóa tương tác và phân tích
4. **Giới Thiệu** - Thông tin dự án và chi tiết kỹ thuật

## 🔺 Cải Tiến Mới Nhất

### Tích Hợp Ngrok Đa Nền Tảng

Hệ thống đã được cải tiến để hỗ trợ đa nền tảng và quản lý ngrok tốt hơn:

- **Quản Lý Token An Toàn**: Tạo module `manage_token.py` để quản lý token ngrok an toàn thông qua file `.env.local`
- **Tách Lọc Riêng**: Module `run_ngrok.py` riêng biệt cho tích hợp ngrok, giúp dễ bảo trì và cập nhật
- **Tương Thích Đa Nền Tảng**: Các script được cập nhật để hoạt động trên macOS, Linux và Windows (qua Git Bash)
- **Phát Hiện HĐH Tự Động**: Tự động phát hiện hệ điều hành và sử dụng đường dẫn thư mục chính xác
- **Xử Lý Lỗi Nâng Cao**: Cải thiện xử lý tín hiệu và dọn dẹp tài nguyên khi kết thúc

### Tối Ưu Hóa PySpark

Đã triển khai các cải tiến để giảm thiểu cảnh báo không mong muốn và cải thiện trải nghiệm người dùng:

- **Module Tiện Ích Spark**: Tạo module `utils/spark_utils.py` để tập trung các cấu hình và giảm logging
- **Giảm Cảnh Báo Hadoop**: Đặc biệt giải quyết vấn đề "Unable to load native-hadoop library"
- **Tối Ưu Hóa Bộ Nhớ**: Cấu hình bộ nhớ tốt hơn cho PySpark
- **Các Tùy Chọn Bổ Sung**: Thêm cấu hình chi tiết cho SparkSession để tăng hiệu suất

### Cải Tiến Đa Nền Tảng

Các script chạy hỗ trợ tốt trên nhiều hệ điều hành:

- **macOS & Linux**: Sử dụng đường dẫn `venv/bin` cho các công cụ Python
- **Windows**: Sử dụng đường dẫn `venv/Scripts` được phát hiện tự động
- **Cài đặt Thông Minh**: Tự động cài đặt các phụ thuộc thích hợp cho từng nền tảng

## 🌐 Triển Khai

Ứng dụng có thể được triển khai:
- **Cục bộ** - Sử dụng máy chủ tích hợp của Streamlit
- **Đám mây** - Sử dụng ngrok cho tunneling bảo mật

Đối với triển khai đám mây:
```python
# Từ Demo/vn_real_estate_app.py
def run_ngrok():
    """Kết nối ứng dụng Streamlit với ngrok để tạo URL public."""
    # Triển khai ngrok
```

## 🔮 Cải Tiến Trong Tương Lai

- **Phân Tích Chuỗi Thời Gian**: Tích hợp dữ liệu giá lịch sử
- **Mô Hình ML Nâng Cao**: Thử nghiệm các phương pháp học sâu
- **Nguồn Dữ Liệu Bổ Sung**: Tích hợp nhiều nền tảng bất động sản hơn
- **Trực Quan Hóa Nâng Cao**: Thêm nhiều công cụ trực quan hóa tương tác
- **Xác Thực Người Dùng**: Thêm hồ sơ người dùng cho tìm kiếm đã lưu và tùy chọn

## 👥 Nhóm Phát Triển

- MSSV: 1234567
- MSSV: 1234568
- MSSV: 1234569

## 📄 Giấy Phép

Dự án này được cấp phép theo Giấy phép MIT - xem file LICENSE để biết chi tiết.

## 🙏 Lời Cảm Ơn

- Nguồn dữ liệu: nhadat.cafeland.vn
- Cộng đồng Apache Spark
- Đội phát triển Streamlit
