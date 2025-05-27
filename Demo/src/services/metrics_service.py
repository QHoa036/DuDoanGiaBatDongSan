#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dịch vụ quản lý metrics của mô hình - Theo dõi và lưu trữ các thông số hiệu suất
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime

from .core.base_service import BaseService
from .interfaces.metrics_interface import IMetricsService

class MetricsService(BaseService, IMetricsService):
    """
    Dịch vụ quản lý metrics của các mô hình dự đoán
    Triển khai interface IMetricsService
    """

    def __init__(self):
        """
        Khởi tạo dịch vụ metrics
        """
        # Gọi khởi tạo lớp cơ sở
        super().__init__(service_name="MetricsService")

        # Dictionary lưu trữ metrics của các mô hình
        self._metrics = {}

        # Dictionary lưu trữ lịch sử metrics
        self._metrics_history = {}

        # Tải metrics từ trạng thái đã lưu (nếu có)
        stored_metrics = self.load_state("model_metrics")
        if stored_metrics is not None:
            self._metrics = stored_metrics
            self._logger.info(f"Đã tải metrics cho {len(self._metrics)} mô hình từ bộ nhớ cache")

        # Tải lịch sử metrics từ trạng thái đã lưu (nếu có)
        stored_history = self.load_state("metrics_history")
        if stored_history is not None:
            self._metrics_history = stored_history
            self._logger.info(f"Đã tải lịch sử metrics từ bộ nhớ cache")

    @property
    def all_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Lấy tất cả các metrics đã lưu trữ
        """
        return self._metrics

    def add_metrics(self, model_name: str, metrics: Dict[str, float]) -> bool:
        """
        Thêm metrics mới cho một mô hình
        """
        try:
            # Thêm timestamp cho metrics
            timestamp = datetime.now().isoformat()
            metrics_with_timestamp = {
                **metrics,
                "_timestamp": timestamp
            }

            # Lưu metrics cho mô hình
            self._metrics[model_name] = metrics_with_timestamp

            # Cập nhật lịch sử metrics
            if model_name not in self._metrics_history:
                self._metrics_history[model_name] = {}

            for metric_name, value in metrics.items():
                if metric_name not in self._metrics_history[model_name]:
                    self._metrics_history[model_name][metric_name] = []

                # Thêm giá trị mới vào lịch sử
                self._metrics_history[model_name][metric_name].append({
                    "value": value,
                    "timestamp": timestamp
                })

            # Lưu trạng thái mới
            self.save_state("model_metrics", self._metrics)
            self.save_state("metrics_history", self._metrics_history)

            self._logger.info(f"Đã thêm metrics cho mô hình {model_name}: {metrics}")
            return True

        except Exception as e:
            self._logger.error(f"Lỗi khi thêm metrics cho mô hình {model_name}: {str(e)}")
            return False

    def get_metrics(self, model_name: str) -> Optional[Dict[str, float]]:
        """
        Lấy metrics của một mô hình cụ thể
        """
        return self._metrics.get(model_name)

    def update_metric(self, model_name: str, metric_name: str, value: float) -> bool:
        """
        Cập nhật giá trị của một metric cụ thể
        """
        try:
            # Kiểm tra xem mô hình có tồn tại không
            if model_name not in self._metrics:
                self._metrics[model_name] = {}

            # Cập nhật giá trị metric
            timestamp = datetime.now().isoformat()
            self._metrics[model_name][metric_name] = value
            self._metrics[model_name]["_timestamp"] = timestamp

            # Cập nhật lịch sử metrics
            if model_name not in self._metrics_history:
                self._metrics_history[model_name] = {}

            if metric_name not in self._metrics_history[model_name]:
                self._metrics_history[model_name][metric_name] = []

            # Thêm giá trị mới vào lịch sử
            self._metrics_history[model_name][metric_name].append({
                "value": value,
                "timestamp": timestamp
            })

            # Lưu trạng thái mới
            self.save_state("model_metrics", self._metrics)
            self.save_state("metrics_history", self._metrics_history)

            self._logger.info(f"Đã cập nhật metric {metric_name} cho mô hình {model_name}: {value}")
            return True

        except Exception as e:
            self._logger.error(f"Lỗi khi cập nhật metric {metric_name} cho mô hình {model_name}: {str(e)}")
            return False

    def get_metric_history(self, model_name: str, metric_name: str) -> List[float]:
        """
        Lấy lịch sử giá trị của một metric
        """
        try:
            if (model_name in self._metrics_history and
                metric_name in self._metrics_history[model_name]):
                # Trả về danh sách các giá trị từ lịch sử
                return [entry["value"] for entry in self._metrics_history[model_name][metric_name]]
            return []
        except Exception as e:
            self._logger.error(f"Lỗi khi lấy lịch sử metric {metric_name} cho mô hình {model_name}: {str(e)}")
            return []

    def get_best_model(self, metric_name: str, higher_is_better: bool = True) -> str:
        """
        Lấy tên của mô hình tốt nhất dựa trên một metric
        """
        try:
            best_model = None
            best_value = float('-inf') if higher_is_better else float('inf')

            for model_name, metrics in self._metrics.items():
                if metric_name in metrics:
                    value = metrics[metric_name]

                    if higher_is_better:
                        if value > best_value:
                            best_value = value
                            best_model = model_name
                    else:
                        if value < best_value:
                            best_value = value
                            best_model = model_name

            if best_model is None:
                self._logger.warning(f"Không tìm thấy mô hình nào có metric {metric_name}")
                return ""

            self._logger.info(f"Mô hình tốt nhất theo {metric_name}: {best_model} ({best_value})")
            return best_model

        except Exception as e:
            self._logger.error(f"Lỗi khi tìm mô hình tốt nhất theo metric {metric_name}: {str(e)}")
            return ""

    def export_metrics_report(self, output_path: Optional[str] = None) -> str:
        """
        Xuất báo cáo metrics ra file JSON
        """
        try:
            if output_path is None:
                output_path = self._state_dir

            os.makedirs(output_path, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"metrics_report_{timestamp}.json"
            report_path = os.path.join(output_path, report_filename)

            report_data = {
                "metrics": self._metrics,
                "history": self._metrics_history,
                "generated_at": timestamp,
                "service_info": self.get_service_info()
            }

            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)

            self._logger.info(f"Đã xuất báo cáo metrics ra {report_path}")
            return report_path

        except Exception as e:
            self._logger.error(f"Lỗi khi xuất báo cáo metrics: {str(e)}")
            return ""

    def import_metrics_from_file(self, file_path: str) -> bool:
        """
        Nhập metrics từ file JSON
        """
        try:
            if not os.path.exists(file_path):
                self._logger.error(f"Không tìm thấy file {file_path}")
                return False

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if "metrics" in data:
                self._metrics = data["metrics"]

            if "history" in data:
                self._metrics_history = data["history"]

            # Lưu trạng thái mới
            self.save_state("model_metrics", self._metrics)
            self.save_state("metrics_history", self._metrics_history)

            self._logger.info(f"Đã nhập metrics từ {file_path}")
            return True

        except Exception as e:
            self._logger.error(f"Lỗi khi nhập metrics từ file {file_path}: {str(e)}")
            return False
