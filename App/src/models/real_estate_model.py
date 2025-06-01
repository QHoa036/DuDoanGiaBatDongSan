import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Union, Optional

@dataclass
class RealEstateFeatures:
    """
    Lớp mô hình đại diện cho các đặc trưng của bất động sản
    """
    category: str = ""
    district: str = ""
    city_province: str = ""
    area: float = 0.0
    price: float = 0.0
    price_per_m2: float = 0.0
    bedroom_num: int = 0
    toilet_num: int = 0
    floor_num: int = 0
    livingroom_num: int = 0
    direction: str = ""
    built_year: int = 0
    legal_status: str = ""
    street: float = 0.0
    longitude: float = 0.0
    latitude: float = 0.0
    post_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Union[str, float, int]]:
        """
        Chuyển đổi đối tượng thành từ điển

        Returns:
            Dict: Từ điển chứa các thuộc tính của đối tượng
        """
        return {
            "category": self.category,
            "district": self.district,
            "city_province": self.city_province,
            "area": self.area,
            "price": self.price,
            "price_per_m2": self.price_per_m2,
            "bedroom_num": self.bedroom_num,
            "toilet_num": self.toilet_num,
            "floor_num": self.floor_num,
            "livingroom_num": self.livingroom_num,
            "direction": self.direction,
            "built_year": self.built_year,
            "legal_status": self.legal_status,
            "street": self.street,
            "longitude": self.longitude,
            "latitude": self.latitude,
            "post_date": self.post_date
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Chuyển đổi đối tượng thành DataFrame

        Returns:
            pd.DataFrame: DataFrame chứa dữ liệu của đối tượng
        """
        return pd.DataFrame([self.to_dict()])

    @classmethod
    def from_dict(cls, data: Dict[str, Union[str, float, int]]) -> 'RealEstateFeatures':
        """
        Tạo đối tượng từ từ điển

        Args:
            data: Từ điển chứa dữ liệu đầu vào

        Returns:
            RealEstateFeatures: Đối tượng được tạo từ từ điển
        """
        return cls(
            category=data.get("category", ""),
            district=data.get("district", ""),
            city_province=data.get("city_province", ""),
            area=float(data.get("area", 0.0)),
            price=float(data.get("price", 0.0)),
            price_per_m2=float(data.get("price_per_m2", 0.0)),
            bedroom_num=int(data.get("bedroom_num", 0)),
            toilet_num=int(data.get("toilet_num", 0)),
            floor_num=int(data.get("floor_num", 0)),
            livingroom_num=int(data.get("livingroom_num", 0)),
            direction=data.get("direction", ""),
            built_year=int(data.get("built_year", 0)),
            legal_status=data.get("legal_status", ""),
            street=float(data.get("street", 0.0)),
            longitude=float(data.get("longitude", 0.0)),
            latitude=float(data.get("latitude", 0.0)),
            post_date=data.get("post_date", None)
        )

@dataclass
class PredictionResult:
    """
    Lớp mô hình đại diện cho kết quả dự đoán
    """
    predicted_price: float
    confidence_level: float = 0.0
    price_range_low: Optional[float] = None
    price_range_high: Optional[float] = None
    similar_properties: Optional[List[Dict[str, Union[str, float, int]]]] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        """
        Phương thức được gọi sau khi khởi tạo để thiết lập giá trị mặc định
        """
        if not self.price_range_low and self.predicted_price:
            # Mặc định khoảng tin cậy 10% cho giới hạn dưới
            self.price_range_low = self.predicted_price * 0.9

        if not self.price_range_high and self.predicted_price:
            # Mặc định khoảng tin cậy 10% cho giới hạn trên
            self.price_range_high = self.predicted_price * 1.1

@dataclass
class ModelMetrics:
    """
    Lớp mô hình đại diện cho các chỉ số của mô hình
    """
    r2: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """
        Chuyển đổi đối tượng thành từ điển

        Returns:
            Dict: Từ điển chứa các thuộc tính của đối tượng
        """
        return {
            "r2": self.r2,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape
        }
