# ỨNG DỤNG DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN VIỆT NAM

> **MÔN HỌC**: DỮ LIỆU LỚN VÀ ỨNG DỤNG

## Nhóm phát triển

- **Lê Thị Cẩm Giang** - Tác giả  <https://github.com/lcg1908>
- **Nguyễn Quỳnh Anh** - Đồng tác giả  <https://github.com/Quynanhng25>
- **Nguyễn Cao Hoài Duyên** - Đồng tác giả <https://github.com/CaoHoaiDuyen>
- **Đinh Trương Ngọc Quỳnh Hoa** - Đồng tác giả <https://github.com/QHoa036>
- **Trần Hoàng Nghĩa** - Đồng tác giả <https://github.com/Blink713>
- **Nguyễn Phương Thảo** - Đồng tác giả <https://github.com/thaonguyenbi>

## Tổng quan

Ứng dụng Dự đoán giá Bất động sản Việt Nam là một nền tảng hiện đại kết hợp công nghệ PySpark, học máy và Streamlit để cung cấp:

- **Dự đoán giá bất động sản** chính xác dựa trên các đặc điểm của tài sản
- **Phân tích thị trường** với giao diện trực quan, hiện đại và tương tác
- **Xu hướng giá** theo khu vực, thời gian và các yếu tố ảnh hưởng
- **Trải nghiệm người dùng** đa nền tảng và dễ sử dụng

## Cấu trúc dự án

```bash
Vietnam_Real_Estate_Price_Prediction/
├── App/                            # Ứng dụng chính
│   ├── src/                        # Mã nguồn
│   │   ├── data/                   # Dữ liệu mẫu
│   │   ├── logs/                   # Nhật ký hệ thống
│   │   ├── styles/                 # CSS và giao diện
│   │   └── utils/                  # Tiện ích
│   └── vn_real_estate_app.py       # Ứng dụng chính
├── References/                     # Tài liệu tham khảo
├── .env.example                    # Mẫu cấu hình biến môi trường
├── requirements.txt                # Danh sách thư viện
└── run_app.sh                      # Script chạy ứng dụng (đa nền tảng)
```

## Hướng dẫn cài đặt và sử dụng

### Yêu cầu hệ thống

- **Python 3.8+**
- **Java Runtime Environment (JRE)** (cho PySpark)
- **Git Bash** (khuyến nghị cho Windows)

### Cài đặt và chạy ứng dụng

Ứng dụng hỗ trợ nhiều nền tảng (Windows, macOS, Linux) với một lệnh duy nhất:

```bash
./run_app.sh
```

Script này sẽ tự động:

1. Phát hiện hệ điều hành và thiết lập môi trường phù hợp
2. Cài đặt các dependency cần thiết
3. Tạo và kích hoạt môi trường ảo Python
4. Hỏi người dùng có muốn sử dụng Ngrok để tạo URL công khai
5. Khởi chạy ứng dụng Streamlit

### Sử dụng với Ngrok

Để chia sẻ ứng dụng qua URL công khai:

1. Đăng ký tài khoản tại [ngrok.com](https://ngrok.com)
2. Lấy authtoken từ dashboard
3. Nhập authtoken vào file env.local
4. Chọn 'y' khi được hỏi về việc sử dụng Ngrok


## Thư viện chính

- **PySpark**: Xử lý dữ liệu lớn và xây dựng mô hình ML
- **Streamlit**: Xây dựng giao diện web tương tác
- **Pandas & NumPy**: Xử lý và phân tích dữ liệu
- **Plotly & Matplotlib**: Trực quan hóa dữ liệu
- **Ngrok**: Tạo URL công khai để chia sẻ ứng dụng

## Lời cảm ơn

Nhóm chúng em xin gửi lời cảm ơn sâu sắc đến thầy Nguyễn Mạnh Tuấn, giảng viên bộ môn Dữ liệu lớn và ứng dụng tại Đại học UEH, vì đã tận tình hướng dẫn, truyền đạt kiến thức và kinh nghiệm quý báu giúp chúng em không chỉ nắm vững lý thuyết mà còn áp dụng vào thực tế. Xin chân thành cảm ơn thầy vì tâm huyết và sự nhiệt tình giúp nhóm hoàn thiện đề tài tốt nhất.