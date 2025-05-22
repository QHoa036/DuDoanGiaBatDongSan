#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tiện ích cấu hình và quản lý PySpark
Cung cấp các hàm để khởi tạo SparkSession với cấu hình tối ưu
và giảm thiểu các cảnh báo không cần thiết
"""

import os
import sys
import logging
from pyspark.sql import SparkSession

# Thiết lập mức độ log toàn cục cho PySpark
def configure_spark_logging():
    """Cấu hình mức độ log cho PySpark để giảm thiểu cảnh báo không cần thiết"""
    # Thiết lập biến môi trường TRUOC KHI IMPORT SPARK
    os.environ['PYSPARK_LOG_LEVEL'] = 'ERROR'

    # Tắt cảnh báo Java và Hadoop native
    os.environ["JAVA_TOOL_OPTIONS"] = "-Dlog4j.configurationFile=log4j2.properties"
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

    # Ẩn cảnh báo native-hadoop
    # Lựa chọn 1: Đặt biến môi trường để không tìm thư viện Hadoop native
    os.environ["HADOOP_HOME"] = ""

    # Giảm mức độ log của các logger cụ thể
    logging.getLogger("py4j").setLevel(logging.ERROR)
    logging.getLogger("org").setLevel(logging.ERROR)
    logging.getLogger("akka").setLevel(logging.ERROR)

    # Tắt toàn bộ cảnh báo Python
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    # Ghi đè hàm print cho py4j để ngăn chặn output
    class NullLogger:
        def write(self, *args, **kwargs):
            pass
        def flush(self):
            pass

    # Cách tiếp cận cực đoan: chuyển hướng stdout của Java trong quá trình khởi tạo Spark
    # Chỉ sử dụng nếu các cách trên không hiệu quả
    # sys.stdout = NullLogger()

def get_spark_session(app_name="Vietnam Real Estate Price Prediction", enable_hive=False):
    """
    Khởi tạo và trả về một SparkSession với cấu hình tối ưu để giảm thiểu cảnh báo

    Tham số:
        app_name (str): Tên ứng dụng hiển thị trong Spark UI
        enable_hive (bool): Bật tích hợp Hive nếu cần

    Trả về:
        SparkSession: Phiên làm việc Spark đã cấu hình
    """
    # Cấu hình logging trước (phải gọi trước khi tạo SparkSession)
    configure_spark_logging()

    # Thêm cấu hình log4j (cách khác để giảm cảnh báo)
    log4j_properties = """
    log4j.rootCategory=ERROR, console
    log4j.appender.console=org.apache.log4j.ConsoleAppender
    log4j.appender.console.target=System.err
    log4j.appender.console.layout=org.apache.log4j.PatternLayout
    log4j.appender.console.layout.ConversionPattern=%d{yy/MM/dd HH:mm:ss} %p %c{1}: %m%n
    log4j.logger.org.apache.spark=ERROR
    log4j.logger.org.apache.hadoop=ERROR
    log4j.logger.org.spark_project=ERROR
    """

    # Tạo builder với các cấu hình chi tiết để giảm cảnh báo
    builder = SparkSession.builder \
        .appName(app_name) \
        .config("spark.ui.showConsoleProgress", "false") \
        .config("spark.executor.logs.rolling.enabled", "true") \
        .config("spark.executor.logs.rolling.maxSize", "10000000") \
        .config("spark.executor.logs.rolling.maxRetainedFiles", "5") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.logConf", "false") \
        .config("spark.driver.extraJavaOptions", "-Dlog4j.logLevel=ERROR") \
        .config("spark.executor.extraJavaOptions", "-Dlog4j.logLevel=ERROR") \
        .config("spark.master.ui.allowFrameAncestors", "true") \
        .config("spark.hadoop.fs.defaultFS", "file:///") \
        .config("spark.sql.session.timeZone", "Asia/Ho_Chi_Minh") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1")

    # Tắt SparkUI để giảm bớt logging
    builder = builder.config("spark.ui.enabled", "false")

    # Đặt mức độ log tối thiểu
    builder = builder.config("spark.sql.hive.thriftServer.singleSession", "true")

    # Bật hỗ trợ Hive nếu cần
    if enable_hive:
        builder = builder.enableHiveSupport()

    # Tạo SparkSession
    spark = builder.getOrCreate()

    # Thiết lập mức độ log cho SparkContext
    spark.sparkContext.setLogLevel("ERROR")

    # Tắt cảnh báo Apache Spark
    spark._jvm.org.apache.log4j.LogManager.getRootLogger().setLevel(spark._jvm.org.apache.log4j.Level.ERROR)

    return spark
