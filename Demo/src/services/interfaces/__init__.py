#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface cho các dịch vụ trong ứng dụng Vietnam Real Estate Price Prediction
"""

from .base_interface import IBaseService
from .train_model_interface import ITrainModelService
from .prediction_interface import IPredictionService
from .progress_data_interface import IProgressDataService
from .metrics_interface import IMetricsService
