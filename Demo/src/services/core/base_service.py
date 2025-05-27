#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BaseService - Lớp cơ sở cho tất cả các dịch vụ
"""

import os
import pickle
import json
from abc import ABC
from typing import Any, Dict, Optional
from datetime import datetime

from ...utils.logger_utils import get_logger

class BaseService(ABC):
    """
    Lớp cơ sở cho tất cả các dịch vụ
    Cung cấp các tính năng chung cho tất cả các dịch vụ như:
    - Logging
    - Lưu trữ/tải trạng thái
    - Xử lý lỗi
    """

    def __init__(self, service_name: Optional[str] = None):
        """
        Khởi tạo dịch vụ cơ sở
        """
        # Tên dịch vụ, mặc định là tên lớp
        self._service_name = service_name or self.__class__.__name__

        # Khởi tạo logger
        self._logger = get_logger(self._service_name)

        # Thư mục gốc cho logs
        self._logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
        os.makedirs(self._logs_dir, exist_ok=True)

        # Thư mục cho việc lưu trữ mô hình
        self._model_dir = os.path.join(self._logs_dir, 'models')
        os.makedirs(self._model_dir, exist_ok=True)

        # Thư mục cho việc lưu trữ trạng thái
        self._state_dir = os.path.join(self._logs_dir, 'states')
        os.makedirs(self._state_dir, exist_ok=True)

        # Khởi tạo timestamp
        self._created_at = datetime.now()
        self._last_updated = self._created_at

        self._logger.info(f"Dịch vụ {self._service_name} đã được khởi tạo")

    def save_state(self, state_name: str, state_data: Any) -> bool:
        """
        Lưu trạng thái dịch vụ
        """
        try:
            # Tạo tên file với timestamp để tránh xung đột và dễ theo dõi lịch sử
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            state_path = os.path.join(self._state_dir, f"{state_name}_{timestamp}.pkl")

            with open(state_path, 'wb') as f:
                pickle.dump(state_data, f)

            # Tạo file metadata cho trạng thái
            metadata = {
                "state_name": state_name,
                "created_at": timestamp,
                "service": self._service_name,
                "path": state_path
            }

            metadata_path = os.path.join(self._state_dir, f"{state_name}_{timestamp}.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            self._logger.info(f"Đã lưu trạng thái {state_name} tại {state_path}")
            self._last_updated = datetime.now()
            return True

        except Exception as e:
            self._logger.error(f"Lỗi khi lưu trạng thái {state_name}: {str(e)}")
            return False

    def load_state(self, state_name: str, default: Any = None) -> Any:
        """
        Tải trạng thái dịch vụ
        """
        try:
            # Tìm tất cả các file trạng thái phù hợp với tên
            files = [f for f in os.listdir(self._state_dir)
                    if f.startswith(f"{state_name}_") and f.endswith(".pkl")]

            if not files:
                # Thử tìm kiếm theo cách cũ (cho tương thích ngược)
                legacy_path = os.path.join(self._state_dir, f"{state_name}.pkl")
                if os.path.exists(legacy_path):
                    with open(legacy_path, 'rb') as f:
                        state_data = pickle.load(f)
                    self._logger.info(f"Đã tải trạng thái {state_name} từ vị trí cũ")
                    return state_data
                else:
                    # Kiểm tra theo thứ tự ưu tiên
                    locations_to_check = [
                        # 1. Thư mục logs (cấp cao hơn states)
                        self._logs_dir,
                        # 2. Thư mục state cũ (ở cùng cấp với thư mục logs)
                        os.path.join(os.path.dirname(self._logs_dir), 'state'),
                        # 3. Thư mục models cũ (trường hợp có thể lưu nhầm vào models)
                        os.path.join(os.path.dirname(self._logs_dir), 'models')
                    ]

                    # Kiểm tra từng vị trí
                    for location in locations_to_check:
                        if os.path.exists(location):
                            old_path = os.path.join(location, f"{state_name}.pkl")
                            if os.path.exists(old_path):
                                with open(old_path, 'rb') as f:
                                    state_data = pickle.load(f)
                                self._logger.info(f"Đã tải trạng thái {state_name} từ vị trí: {old_path}")

                                # Sao chép vào vị trí mới nếu đã tìm thấy ở vị trí cũ
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                new_path = os.path.join(self._state_dir, f"{state_name}_{timestamp}.pkl")
                                try:
                                    with open(new_path, 'wb') as f:
                                        pickle.dump(state_data, f)
                                    self._logger.info(f"Đã sao chép trạng thái từ vị trí cũ sang vị trí mới: {new_path}")
                                except Exception as copy_error:
                                    self._logger.warning(f"Không thể sao chép trạng thái sang vị trí mới: {str(copy_error)}")

                                return state_data

                self._logger.info(f"Không tìm thấy trạng thái {state_name}, sử dụng giá trị mặc định")
                return default

            # Sắp xếp theo thời gian tạo (giảm dần) để lấy phiên bản mới nhất
            files.sort(reverse=True)
            state_path = os.path.join(self._state_dir, files[0])

            with open(state_path, 'rb') as f:
                state_data = pickle.load(f)

            self._logger.info(f"Đã tải trạng thái {state_name} từ {state_path}")
            return state_data
        except Exception as e:
            self._logger.error(f"Lỗi khi tải trạng thái {state_name}: {str(e)}")
            return default

    def save_model(self, model: Any, model_name: str) -> str:
        """
        Lưu mô hình
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self._model_dir, f"{model_name}_{timestamp}.pkl")

            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            # Lưu metadata của mô hình
            metadata = {
                "model_name": model_name,
                "created_at": timestamp,
                "service": self._service_name,
                "path": model_path
            }

            metadata_path = os.path.join(self._model_dir, f"{model_name}_{timestamp}.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            self._logger.info(f"Đã lưu mô hình {model_name} tại {model_path}")
            self._last_updated = datetime.now()
            return model_path

        except Exception as e:
            self._logger.error(f"Lỗi khi lưu mô hình {model_name}: {str(e)}")
            return ""

    def load_model(self, model_name: str, specific_timestamp: Optional[str] = None) -> Any:
        """
        Tải mô hình
        """
        try:
            model_path = None
            # Tìm file mô hình mới nhất hoặc theo timestamp cụ thể
            if specific_timestamp:
                model_path = os.path.join(self._model_dir, f"{model_name}_{specific_timestamp}.pkl")
                if not os.path.exists(model_path):
                    # Kiểm tra trong thư mục models cũ
                    old_model_dir = os.path.join(os.path.dirname(self._logs_dir), 'models')
                    if os.path.exists(old_model_dir):
                        old_path = os.path.join(old_model_dir, f"{model_name}_{specific_timestamp}.pkl")
                        if os.path.exists(old_path):
                            model_path = old_path
                        else:
                            self._logger.error(f"Không tìm thấy mô hình {model_name} với timestamp {specific_timestamp}")
                            return None
                    else:
                        self._logger.error(f"Không tìm thấy mô hình {model_name} với timestamp {specific_timestamp}")
                        return None
            else:
                # Tìm mô hình mới nhất
                files = [f for f in os.listdir(self._model_dir)
                        if f.startswith(f"{model_name}_") and f.endswith(".pkl")]

                # Kiểm tra thư mục models cũ nếu cần
                old_files = []
                old_model_dir = os.path.join(os.path.dirname(self._logs_dir), 'models')
                if os.path.exists(old_model_dir):
                    old_files = [f for f in os.listdir(old_model_dir)
                            if f.startswith(f"{model_name}_") and f.endswith(".pkl")]

                # Kết hợp các file với đường dẫn đầy đủ
                file_paths = []
                # Thêm đường dẫn đầy đủ cho các file trong thư mục mới
                file_paths.extend([(os.path.join(self._model_dir, f), f) for f in files])
                # Thêm đường dẫn đầy đủ cho các file trong thư mục cũ
                file_paths.extend([(os.path.join(old_model_dir, f), f) for f in old_files])

                if not file_paths:
                    # Kiểm tra file đơn giản trong cả hai thư mục
                    simple_file = f"{model_name}.pkl"

                    simple_path = os.path.join(self._model_dir, simple_file)
                    if os.path.exists(simple_path):
                        model_path = simple_path
                    else:
                        simple_path_old = os.path.join(old_model_dir, simple_file)
                        if os.path.exists(simple_path_old):
                            model_path = simple_path_old
                        else:
                            self._logger.info(f"Không tìm thấy mô hình {model_name}")
                            return None
                else:
                    # Sắp xếp theo tên file (thời gian tạo sẽ ở trong tên file)
                    file_paths.sort(key=lambda x: x[1], reverse=True)
                    model_path = file_paths[0][0]

                    # Tạo bản sao ở vị trí mới nếu đã tìm thấy ở vị trí cũ
                    if old_model_dir in model_path:
                        file_name = os.path.basename(model_path)
                        new_path = os.path.join(self._model_dir, file_name)
                        if not os.path.exists(new_path):
                            try:
                                import shutil
                                shutil.copy2(model_path, new_path)
                                self._logger.info(f"Sao chép mô hình từ vị trí cũ sang vị trí mới: {new_path}")
                            except Exception as copy_error:
                                self._logger.warning(f"Không thể sao chép mô hình sang vị trí mới: {str(copy_error)}")

            if not model_path:
                self._logger.error(f"Không tìm thấy đường dẫn mô hình cho {model_name}")
                return None

            # Tải mô hình
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            self._logger.info(f"Đã tải mô hình {model_name} từ {model_path}")
            return model

        except Exception as e:
            self._logger.error(f"Lỗi khi tải mô hình {model_name}: {str(e)}")
            return None

    def get_service_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin về dịch vụ
        """
        return {
            "service_name": self._service_name,
            "created_at": self._created_at.isoformat(),
            "last_updated": self._last_updated.isoformat(),
            "class": self.__class__.__name__
        }

    @property
    def service_name(self) -> str:
        """
        Tên dịch vụ
        """
        return self._service_name

    @property
    def logger(self):
        """
        Logger của dịch vụ
        """
        return self._logger

    def safe_execute(self, func, *args, default=None, **kwargs):
        """
        Thực thi một hàm an toàn với xử lý lỗi
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self._logger.error(f"Lỗi khi thực thi {func.__name__}: {str(e)}")
            return default
