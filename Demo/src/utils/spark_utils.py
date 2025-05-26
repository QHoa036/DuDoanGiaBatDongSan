#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tiện ích cấu hình và quản lý PySpark
Cung cấp các hàm để khởi tạo SparkSession với cấu hình tối ưu
và giảm thiểu các cảnh báo không cần thiết
"""

# MARK: - Thư viện

import os
import sys
import logging
from pyspark.sql import SparkSession

# MARK: - Cấu hình

def configure_spark_logging():
    """
    Cấu hình mức độ log cho PySpark để giảm thiểu cảnh báo không cần thiết
    """
    # Đặt biến môi trường để kiểm soát ngrok CLI
    os.environ['NGROK_LOG_LEVEL'] = 'critical'

    # Đặt biến môi trường để tắt hoàn toàn các thông báo Spark và Ivy
    os.environ['SPARK_SUBMIT_OPTS'] = '-Dlog4j.rootCategory=FATAL -Dorg.apache.ivy.util.Message.level=FATAL -Dorg.apache.ivy.core.settings.IvySettings.level=FATAL -Dorg.apache.ivy.core.report.ResolveReport.level=FATAL'

    # Các biến môi trường bổ sung để kiểm soát log
    os.environ['SPARK_SILENT'] = 'true'
    os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
    os.environ['SPARK_LOG_LEVEL'] = 'FATAL'
    os.environ['PYSPARK_PYTHON_LOG_LEVEL'] = 'FATAL'
    os.environ['PYSPARK_DRIVER_PYTHON_LOG_LEVEL'] = 'FATAL'

    # Xác định đường dẫn tuyệt đối đến tập tin log4j2.properties trong thư mục config
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log4j_file_path = os.path.join(src_dir, 'config', 'log4j2.properties')

    if os.path.exists(log4j_file_path):
        # Tắt cảnh báo Java và Hadoop native sử dụng file cấu hình log4j
        # Sử dụng biến môi trường thông qua spark.driver.extraJavaOptions thay vì JAVA_TOOL_OPTIONS
        # để tránh thông báo "Picked up JAVA_TOOL_OPTIONS"
        os.environ["SPARK_DRIVER_OPTS"] = f"-Dlog4j.configurationFile=file:{log4j_file_path} -Dlog4j.rootCategory=ERROR"
        os.environ["SPARK_EXECUTOR_OPTS"] = f"-Dlog4j.configurationFile=file:{log4j_file_path} -Dlog4j.rootCategory=ERROR"

        # Xóa JAVA_TOOL_OPTIONS nếu đã được đặt trước đó
        if "JAVA_TOOL_OPTIONS" in os.environ:
            del os.environ["JAVA_TOOL_OPTIONS"]
    else:
        # Nếu không tìm thấy file, hủy biến môi trường này để tránh lỗi
        if "JAVA_TOOL_OPTIONS" in os.environ:
            del os.environ["JAVA_TOOL_OPTIONS"]

    # Ẩn cảnh báo native-hadoop
    # Thiết lập các biến môi trường để ẩn cảnh báo
    os.environ["HADOOP_HOME"] = ""
    os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages org.apache.hadoop:hadoop-aws:3.3.1 pyspark-shell"
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    # Biến môi trường quan trọng để tắt cảnh báo native-hadoop
    os.environ["HADOOP_USER_NAME"] = "hadoop"
    os.environ["ARROW_PRE_0_15_IPC_FORMAT"] = "1"

    # Tắt hiển thị cảnh báo Hadoop
    os.environ["HADOOP_CONF_DIR"] = ""
    os.environ["HADOOP_OPTS"] = "-Djava.awt.headless=true -Dlog4j.logger.org.apache.hadoop=ERROR"

    # Giảm mức độ log của các logger cụ thể
    logging.getLogger("py4j").setLevel(logging.CRITICAL)
    logging.getLogger("org").setLevel(logging.CRITICAL)
    logging.getLogger("akka").setLevel(logging.CRITICAL)
    logging.getLogger("ivy").setLevel(logging.CRITICAL)

    # Tắt cảnh báo ivy (dùng cho dependency resolution)
    logging.getLogger("org.apache.spark.util.DependencyUtils").setLevel(logging.CRITICAL)
    logging.getLogger("org.apache.ivy").setLevel(logging.CRITICAL)

    # Tạo bộ lọc để loại bỏ các thông báo Ivy và dependency resolution
    class SparkFilter(logging.Filter):
        def filter(self, record):
            message = record.getMessage().lower()
            filtered_patterns = [
                'ivy', 'dependency', 'dependencies', 'artifact', 'resolve', 'download',
                'jar', 'hadoop', 'listening on', 'classpath', 'default cache', 'retrieving',
                'spark context', 'initializing', 'executor', 'native', 'jvm', 'compiler',
                'warehouse', 'session', 'executor', 'task', 'dataset', 'worker', 'cluster',
                'deploy', 'startup', 'starting', 'created', 'bind', 'bound', 'connect',
                'scheduler', 'property', 'loading', 'loaded', 'catalog', 'container', 'auth',
                'memory', 'allocate', 'security', 'driver', 'resource', 'timestamp'
            ]
            return not any(pattern in message for pattern in filtered_patterns)

    # Áp dụng bộ lọc cho root logger để ảnh hưởng tất cả các logger
    root_logger = logging.getLogger()
    root_logger.addFilter(SparkFilter())

    # Tắt toàn bộ cảnh báo Python
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    # Tắt cảnh báo cụ thể của Ivy và Spark
    warnings.filterwarnings("ignore", category=UserWarning, module="ivy")
    warnings.filterwarnings("ignore", category=UserWarning, module="pyspark")
    warnings.filterwarnings("ignore", message=".*NativeCodeLoader.*")

    # MARK: - Empty Logger

    class NullLogger:
        def __init__(self, original_stream=None):
            self.original_stream = original_stream

        def write(self, message):
            # Bỏ qua các thông báo liên quan đến JAVA_TOOL_OPTIONS
            if self.original_stream and not 'Picked up JAVA_TOOL_OPTIONS' in message:
                self.original_stream.write(message)

        def flush(self):
            if self.original_stream:
                self.original_stream.flush()

    # Cách tiếp cận cực đoan: chuyển hướng stdout của Java trong quá trình khởi tạo Spark
    # Được kích hoạt để lọc các thông báo không mong muốn
    sys.stdout = NullLogger(sys.stdout)
    sys.stderr = NullLogger(sys.stderr)

# MARK: - Hide Ivy and Spark Logs

# Tạo lớp để ẩn tất cả đầu ra của Ivy và Spark trong quá trình khởi tạo
def silence_ivy_and_spark_logs(func):
    """
    Decorator để ẩn tất cả các thông báo Ivy và Spark khi khởi tạo
    """
    def wrapper(*args, **kwargs):
        # Lưu trữ stdout và stderr ban đầu
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Tạo file null để chuyển hướng đầu ra
        null_file = open(os.devnull, 'w')

        # Tạo bộ lọc để chuyển hướng đầu ra tạm thời
        try:
            # Chuyển hướng đầu ra vào /dev/null để ẩn các thông báo
            sys.stdout = null_file
            sys.stderr = null_file

            # Đặt mức log cấp cao nhất để tắt các thông báo
            old_level = logging.root.level
            logging.root.setLevel(logging.CRITICAL + 1)

            # Gọi hàm gốc
            result = func(*args, **kwargs)

            return result
        finally:
            # Khôi phục stdout và stderr, dù có gặp ngoại lệ hay không
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            null_file.close()

            # Khôi phục mức log
            logging.root.setLevel(old_level)

    return wrapper

# MARK: - Khởi tạo Spark

@silence_ivy_and_spark_logs
def get_spark_session(app_name="Vietnam Real Estate Price Prediction", enable_hive=False, silence_warnings=True):
    """
    Khởi tạo và trả về một SparkSession với cấu hình tối ưu để giảm thiểu cảnh báo
    """
    # Cấu hình logging trước (phải gọi trước khi tạo SparkSession)
    configure_spark_logging()

    # Nếu bật chế độ im lặng hoàn toàn, chuyển hướng stdout/stderr
    if silence_warnings:
        # Tạm thời lưu lại đối tượng stdout/stderr gốc
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        null_output = open(os.devnull, 'w')

        # Chuyển hướng stdout/stderr trong quá trình khởi tạo Spark để ẩn cảnh báo
        sys.stdout = null_output
        sys.stderr = null_output

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

    # Lưu cấu hình log4j vào biến môi trường để sử dụng
    os.environ['SPARK_LOG4J_PROPS'] = log4j_properties

    # Tạo builder với các cấu hình chi tiết để giảm cảnh báo
    builder = SparkSession.builder \
        .appName(app_name) \
        .config("spark.ui.showConsoleProgress", False) \
        .config("spark.executor.logs.rolling.enabled", True) \
        .config("spark.executor.logs.rolling.maxSize", "10000000") \
        .config("spark.executor.logs.rolling.maxRetainedFiles", "5") \
        .config("spark.sql.adaptive.enabled", True) \
        .config("spark.logConf", False) \
        .config("spark.driver.extraJavaOptions", "-Dlog4j.logLevel=ERROR") \
        .config("spark.executor.extraJavaOptions", "-Dlog4j.logLevel=ERROR") \
        .config("spark.master.ui.allowFrameAncestors", True) \
        .config("spark.hadoop.fs.defaultFS", "file:///") \
        .config("spark.sql.session.timeZone", "Asia/Ho_Chi_Minh") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1")

    # Tắt SparkUI để giảm bớt logging
    builder = builder.config("spark.ui.enabled", False)

    # Đặt mức độ log tối thiểu
    builder = builder.config("spark.sql.hive.thriftServer.singleSession", True)

    # Bật hỗ trợ Hive nếu cần
    if enable_hive:
        builder = builder.enableHiveSupport()

    # Tạo SparkSession
    spark = builder.getOrCreate()

    # Thiết lập mức độ log cho SparkContext
    spark.sparkContext.setLogLevel("ERROR")

    # Tắt cảnh báo Apache Spark
    spark._jvm.org.apache.log4j.LogManager.getRootLogger().setLevel(spark._jvm.org.apache.log4j.Level.ERROR)

    # Thiết lập cấu hình để ẩn cảnh báo NativeCodeLoader
    conf = spark._jsc.hadoopConfiguration()
    conf.set("mapreduce.app-submission.cross-platform", "true")

    # Khôi phục stdout/stderr gốc nếu đã chuyển hướng
    if silence_warnings:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        null_output.close()

    return spark
