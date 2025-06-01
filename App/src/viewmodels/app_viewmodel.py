# MARK: - Thư viện

import streamlit as st
from typing import List, Any, Optional

from src.config.app_config import AppConfig
from src.utils.logger_util import get_logger

# Khởi tạo logger
logger = get_logger()

# MARK: - View Model Chính

class AppViewModel:
    """
    ViewModel chính của ứng dụng, quản lý trạng thái và logic của toàn bộ ứng dụng
    """

    # MARK: - Khởi tạo

    def __init__(self, _data_service=None, _model_service=None):
        """
        Khởi tạo AppViewModel

        Args:
            _data_service: Data service (tham số có dấu _ để tránh cache)
            _model_service: Model service (tham số có dấu _ để tránh cache)
        """
        # Tham số có dấu gạch dưới (_) để Streamlit không hash
        if _data_service is None or _model_service is None:
            raise ValueError("Cần cung cấp cả _data_service và _model_service")

        self._data_service = _data_service
        self._model_service = _model_service

        # Khởi tạo session state nếu cần
        self._initialize_session_state()

    # MARK: - Cài đặt session state

    def _initialize_session_state(self):
        """
        Khởi tạo các giá trị trong session state nếu chưa tồn tại
        """
        # Khởi tạo chế độ ứng dụng
        if 'app_mode' not in st.session_state:
            st.session_state.app_mode = AppConfig.APP_MODES[0]

        # Khởi tạo các trạng thái khác
        if 'data' not in st.session_state:
            st.session_state.data = None

        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False

        if 'using_fallback' not in st.session_state:
            st.session_state.using_fallback = False

        if 'progress_bar' not in st.session_state:
            st.session_state.progress_bar = None

    # MARK: - Quản lý chế độ ứng dụng

    def set_app_mode(self, mode: str):
        """
        Thiết lập chế độ ứng dụng

        Args:
            mode (str): Chế độ ứng dụng
        """
        if mode in AppConfig.APP_MODES:
            st.session_state.app_mode = mode

    def handle_mode_change(self, mode: str):
        """
        Xử lý sự kiện thay đổi chế độ ứng dụng

        Args:
            mode (str): Chế độ ứng dụng mới
        """
        self.set_app_mode(mode)
        logger.info(f"Đã thay đổi chế độ ứng dụng thành {mode}")

    def get_app_mode(self) -> str:
        """
        Lấy chế độ ứng dụng hiện tại

        Returns:
            str: Chế độ ứng dụng hiện tại
        """
        return st.session_state.app_mode

    # MARK: - Quản lý dữ liệu

    def load_data(self) -> bool:
        """
        Tải dữ liệu và lưu vào session state

        Returns:
            bool: True nếu tải dữ liệu thành công, False nếu thất bại
        """
        try:
            # Kiểm tra xem dữ liệu đã được tải chưa
            if 'data' not in st.session_state or st.session_state.data is None:
                # Tải dữ liệu từ file
                data = self._data_service.load_data()

                # Kiểm tra xem dữ liệu có rỗng không
                if data.empty:
                    logger.error("Không thể tải dữ liệu")
                    return False

                # Tiền xử lý dữ liệu
                processed_data = self._data_service.preprocess_data(data)

                # Lưu dữ liệu đã xử lý vào session state
                st.session_state.data = processed_data

                logger.info(f"Đã tải và xử lý {len(processed_data)} bản ghi dữ liệu")
                return True

            # Dữ liệu đã có sẵn
            return True

        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu: {e}")
            return False

    def get_data(self) -> Optional[Any]:
        """
        Lấy dữ liệu từ session state

        Returns:
            Optional[Any]: Dữ liệu nếu có, None nếu không có
        """
        if 'data' in st.session_state and st.session_state.data is not None:
            return st.session_state.data

        # Tự động tải dữ liệu nếu chưa có
        if self.load_data():
            return st.session_state.data

        return None

    # MARK: - Huấn luyện mô hình

    def train_model_if_needed(self) -> bool:
        """
        Huấn luyện mô hình nếu cần thiết

        Returns:
            bool: True nếu mô hình đã được huấn luyện, False nếu không
        """
        # Kiểm tra xem mô hình đã được huấn luyện chưa
        if 'model_trained' in st.session_state and st.session_state.model_trained:
            logger.info("Mô hình đã được huấn luyện, bỏ qua bước huấn luyện")
            logger.info(f"Các chỉ số hiện tại trong session state: R2={st.session_state.get('model_r2_score', 0.0)}, RMSE={st.session_state.get('model_rmse', 0.0)}")
            return True

        logger.info("Mô hình cần được huấn luyện, bắt đầu quá trình huấn luyện")

        # Lấy dữ liệu
        data = self.get_data()
        if data is None or data.empty:
            logger.error("Không có dữ liệu để huấn luyện mô hình")
            return False

        try:
            # Hiển thị thanh tiến trình
            progress_bar = st.progress(0)
            st.session_state.progress_bar = progress_bar

            # Cập nhật trạng thái
            progress_bar.progress(10, text="Đang chuẩn bị dữ liệu...")

            # Huấn luyện mô hình
            progress_bar.progress(30, text="Đang huấn luyện mô hình...")
            model, metrics = self._model_service.train_model(data)

            # Lưu mô hình vào session state
            if model is not None:
                st.session_state.model = model
                st.session_state.metrics = metrics
                st.session_state.model_trained = True

                # Lưu các chỉ số r2 và rmse vào session_state để hiển thị trong sidebar
                if metrics:
                    logger.info(f"Đối tượng metrics nhận được từ quá trình huấn luyện mô hình: {metrics.__dict__ if hasattr(metrics, '__dict__') else 'Không có thuộc tính __dict__'}")
                    logger.info(f"Thuộc tính của metrics - r2: {hasattr(metrics, 'r2')}, rmse: {hasattr(metrics, 'rmse')}")

                    if hasattr(metrics, 'r2'):
                        logger.info(f"Giá trị gốc của metrics.r2: {metrics.r2}")
                        st.session_state.model_r2_score = metrics.r2
                    else:
                        logger.warning("Đối tượng metrics không có thuộc tính r2, sử dụng giá trị mặc định 0.0")
                        st.session_state.model_r2_score = 0.0

                    if hasattr(metrics, 'rmse'):
                        logger.info(f"Giá trị gốc của metrics.rmse: {metrics.rmse}")
                        st.session_state.model_rmse = metrics.rmse
                    else:
                        logger.warning("Đối tượng metrics không có thuộc tính rmse, sử dụng giá trị mặc định 0.0")
                        st.session_state.model_rmse = 0.0

                    logger.info(f"Đã lưu các chỉ số mô hình vào session state: R2 = {st.session_state.model_r2_score}, RMSE = {st.session_state.model_rmse}")
                else:
                    logger.error("Không nhận được đối tượng metrics từ quá trình huấn luyện mô hình!")
                    st.session_state.model_r2_score = 0.0
                    st.session_state.model_rmse = 0.0

                logger.info("Đã huấn luyện mô hình thành công")
                return True
            else:
                logger.error("Không thể huấn luyện mô hình")
                progress_bar.progress(100, text="Không thể huấn luyện mô hình!")
                return False

        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện mô hình: {e}")

            # Cập nhật trạng thái
            if 'progress_bar' in st.session_state and st.session_state.progress_bar:
                st.session_state.progress_bar.progress(100, text="Lỗi khi huấn luyện mô hình!")

            return False

    # MARK: - Truy vấn dữ liệu

    def get_unique_values(self, column: str) -> List[Any]:
        """
        Lấy danh sách các giá trị duy nhất của một cột

        Args:
            column (str): Tên cột cần lấy giá trị duy nhất

        Returns:
            List[Any]: Danh sách các giá trị duy nhất
        """
        data = self.get_data()
        if data is None or data.empty:
            return []

        return self._data_service.get_unique_values(data, column)

    # MARK: - Giao diện

    def load_css(self) -> str:
        """
        Tải file CSS

        Returns:
            str: Nội dung CSS
        """
        try:
            css_path = AppConfig.get_css_path()
            with open(css_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Lỗi khi tải file CSS: {e}")
            return ""
