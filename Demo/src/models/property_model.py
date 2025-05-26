#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mô hình Bất động sản - Đại diện cho cấu trúc dữ liệu cốt lõi của các bất động sản
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

# MARK: - Mô hình bất động sản

@dataclass
class Property:
    """
    Lớp mô hình đại diện cho một bất động sản
    Chứa tất cả các thuộc tính cốt lõi của bất động sản
    """
    area: float
    location: str
    num_rooms: int
    year_built: int
    legal_status: str
    house_direction: str
    price: Optional[float] = None
    price_per_sqm: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Property':
        """
        Tạo một thể hiện Property từ một từ điển
        """
        return cls(
            area=float(data.get('area', 0)),
            location=data.get('location', ''),
            num_rooms=int(data.get('num_rooms', 0)),
            year_built=int(data.get('year_built', 2000)),
            legal_status=data.get('legal_status', ''),
            house_direction=data.get('house_direction', ''),
            price=float(data.get('price', 0)) if 'price' in data else None,
            price_per_sqm=float(data.get('price_per_sqm', 0)) if 'price_per_sqm' in data else None
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi đối tượng Property thành một từ điển
        """
        result = {
            'area': self.area,
            'location': self.location,
            'num_rooms': self.num_rooms,
            'year_built': self.year_built,
            'legal_status': self.legal_status,
            'house_direction': self.house_direction
        }

        if self.price is not None:
            result['price'] = self.price

        if self.price_per_sqm is not None:
            result['price_per_sqm'] = self.price_per_sqm

        return result


# MARK: - Kết quả dự đoán

@dataclass
class PredictionResult:
    """
    Lớp mô hình đại diện cho kết quả dự đoán giá bất động sản
    """
    predicted_price: float
    predicted_price_per_sqm: float
    property_details: Dict[str, Any]
    comparison_data: Optional[Dict[str, Any]] = None

    @classmethod
    def create(cls,
            predicted_price: float,
            predicted_price_per_sqm: float,
            property_details: Dict[str, Any],
            comparison_data: Optional[Dict[str, Any]] = None) -> 'PredictionResult':
        """
        Tạo một thể hiện PredictionResult mới
        """
        return cls(
            predicted_price=predicted_price,
            predicted_price_per_sqm=predicted_price_per_sqm,
            property_details=property_details,
            comparison_data=comparison_data or {}
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi đối tượng PredictionResult thành một từ điển
        """
        result = {
            'predicted_price': self.predicted_price,
            'predicted_price_per_sqm': self.predicted_price_per_sqm,
            'property_details': self.property_details
        }

        if self.comparison_data:
            result['comparison_data'] = self.comparison_data

        return result
