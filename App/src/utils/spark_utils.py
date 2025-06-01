#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tiện ích cấu hình và quản lý PySpark
Cung cấp các hàm để khởi tạo SparkSession với cấu hình tối ưu,
giảm thiểu các cảnh báo không cần thiết, và hỗ trợ xử lý lỗi.
"""

# MARK: - Thư viện

import os
import sys
import logging
import tempfile
from typing import Dict, Optional, Any
from pyspark.sql import SparkSession
from pyspark import SparkConf
from src.utils.logger_util import get_logger

# Khởi tạo logger
logger = get_logger(__name__)

# MARK: - SparkUtils Class

class SparkUtils:
    """
    Class cung cấp các tiện ích để làm việc với PySpark
    """

    @staticmethod
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

        logger.info("Đã cấu hình Spark logging thành công")

    @staticmethod
    def stop_spark_session(spark=None):
        """
        Dừng SparkSession hiện tại một cách an toàn nếu nó tồn tại

        Args:
            spark (SparkSession, optional): Phiên làm việc Spark cần dừng. Nếu không cung cấp, sẽ dừng phiên hiện tại.
        """
        try:
            # Nếu cung cấp một SparkSession cụ thể
            if spark is not None:
                try:
                    spark.stop()
                    logger.info("Đã dừng SparkSession cụ thể")
                    return True
                except Exception as e:
                    logger.error(f"Lỗi khi dừng SparkSession cụ thể: {e}")
                    return False

            # Nếu không cung cấp SparkSession, dừng session hiện tại
            # Kiểm tra nếu Spark đã được khởi tạo
            if SparkSession._instantiatedSession is not None:
                # Lấy phiên làm việc hiện tại
                current_session = SparkSession.getActiveSession() or SparkSession._instantiatedSession
                if current_session is not None:
                    # Lấy SparkContext từ session
                    sc = current_session.sparkContext
                    # Dừng SparkContext một cách an toàn
                    sc.stop()
                    # Xóa các tham chiếu đến session
                    SparkSession._instantiatedSession = None
                    SparkSession._activeSession = None
                    logger.info("Đã dừng phiên làm việc Spark hiện tại")
                    return True
            return False
        except Exception as e:
            logger.warning(f"Lỗi khi dừng SparkSession: {e}")
            return False

    @staticmethod
    def create_spark_session(app_name: str = "VNRealEstatePricePrediction",
                            config: Optional[Dict[str, Any]] = None,
                            enable_hive: bool = True,
                            memory: str = "2g") -> Optional[SparkSession]:
        """
        Khởi tạo và trả về SparkSession với cấu hình tối ưu

        Args:
            app_name (str): Tên ứng dụng Spark
            config (Dict): Cấu hình bổ sung cho Spark
            enable_hive (bool): Bật hỗ trợ Hive
            memory (str): Bộ nhớ cấp cho Spark

        Returns:
            SparkSession: phiên làm việc Spark hoặc None nếu có lỗi
        """
        try:
            # Dừng phiên Spark hiện tại để tránh lỗi LiveListenerBus
            SparkUtils.stop_spark_session()

            # Đảm bảo log đã được cấu hình
            SparkUtils.configure_spark_logging()

            # Thiết lập cấu hình cơ bản
            spark_conf = SparkConf()
            spark_conf.set("spark.driver.memory", memory)
            spark_conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
            spark_conf.set("spark.sql.repl.eagerEval.enabled", "true")
            spark_conf.set("spark.sql.adaptive.enabled", "true")

            # Cấu hình để tránh lỗi LiveListenerBus
            spark_conf.set("spark.driver.allowMultipleContexts", "true")
            spark_conf.set("spark.dynamicAllocation.enabled", "false")

            # Thiết lập cấu hình tạm
            temp_dir = tempfile.gettempdir()
            spark_conf.set("spark.local.dir", temp_dir)

            # Áp dụng cấu hình từ tham số
            if config:
                for key, value in config.items():
                    spark_conf.set(key, value)

            # Bắt đầu tạo SparkSession builder
            builder = SparkSession.builder \
                .appName(app_name) \
                .config(conf=spark_conf)

            # Thêm hỗ trợ Hive nếu yêu cầu
            if enable_hive:
                builder = builder.enableHiveSupport()

            # Khởi tạo phiên làm việc
            spark = builder.getOrCreate()

            # Thiết lập log level
            spark.sparkContext.setLogLevel("ERROR")

            logger.info(f"Đã khởi tạo SparkSession '{app_name}' thành công")
            return spark

        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo SparkSession: {e}")
            return None

    # Đã có phương thức stop_spark_session ở trên

    @staticmethod
    def is_spark_available() -> bool:
        """
        Kiểm tra xem PySpark có sẵn sàng để sử dụng không

        Returns:
            bool: True nếu Spark khả dụng, False nếu không
        """
        try:
            # Thử tạo một phiên làm việc Spark đơn giản
            spark = SparkUtils.create_spark_session(app_name="SparkAvailabilityTest", memory="1g")
            available = spark is not None

            # Đóng phiên làm việc thử nghiệm nếu đã tạo
            if available:
                SparkUtils.stop_spark_session(spark=spark)

            return available
        except Exception:
            return False

    @staticmethod
    def get_spark_version() -> str:
        """
        Lấy phiên bản của Spark đang sử dụng

        Returns:
            str: Phiên bản Spark hoặc "Không khả dụng" nếu không thể xác định
        """
        try:
            from pyspark.version import __version__
            return __version__
        except ImportError:
            return "Không khả dụng"

    @staticmethod
    def diagnose_spark() -> Dict[str, str]:
        """
        Chuẩn đoán trạng thái của Spark và trả về thông tin chi tiết

        Returns:
            Dict[str, str]: Thông tin chẩn đoán Spark
        """
        diagnosis = {}

        # Kiểm tra xem Spark có được cài đặt không
        try:
            import pyspark
            diagnosis["spark_installed"] = "Có"
            diagnosis["spark_version"] = SparkUtils.get_spark_version()
        except ImportError:
            diagnosis["spark_installed"] = "Không"
            diagnosis["spark_version"] = "Không cài đặt"
            return diagnosis

        # Kiểm tra xem Spark có khả dụng không
        if SparkUtils.is_spark_available():
            diagnosis["spark_available"] = "Có"
        else:
            diagnosis["spark_available"] = "Không"

        # Thông tin về Java
        try:
            java_home = os.environ.get("JAVA_HOME", "Không thiết lập")
            diagnosis["java_home"] = java_home

            # Kiểm tra phiên bản Java
            if java_home != "Không thiết lập":
                import subprocess
                try:
                    result = subprocess.run(["java", "-version"],
                                          capture_output=True,
                                          text=True,
                                          stderr=subprocess.STDOUT)
                    diagnosis["java_version"] = result.stdout.split('\n')[0]
                except Exception:
                    diagnosis["java_version"] = "Không thể xác định"
            else:
                diagnosis["java_version"] = "Không thể xác định"
        except Exception as e:
            diagnosis["java_error"] = str(e)

        return diagnosis

    @staticmethod
    def set_python_env():
        """
        Đặt biến môi trường Python cho Spark
        """
        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

        # Thiết lập các biến môi trường quan trọng khác
        os.environ["HADOOP_USER_NAME"] = "hadoop"
        os.environ["ARROW_PRE_0_15_IPC_FORMAT"] = "1"
        os.environ["HADOOP_CONF_DIR"] = ""
        os.environ["HADOOP_OPTS"] = "-Djava.awt.headless=true -Dlog4j.logger.org.apache.hadoop=ERROR"

        # Tắt các cảnh báo Python
        import warnings
        warnings.filterwarnings("ignore")

        logger.info("Biến môi trường Python đã được thiết lập cho Spark")

    @staticmethod
    def configure_loggers():
        """
        Cấu hình các logger để giảm thiểu các thông báo không cần thiết
        """
        # Giảm mức độ log của các logger cụ thể
        logging.getLogger("py4j").setLevel(logging.CRITICAL)
        logging.getLogger("org").setLevel(logging.CRITICAL)
        logging.getLogger("akka").setLevel(logging.CRITICAL)
        logging.getLogger("ivy").setLevel(logging.CRITICAL)

        # Tắt cảnh báo ivy (dùng cho dependency resolution)
        logging.getLogger("org.apache.spark.util.DependencyUtils").setLevel(logging.CRITICAL)
        logging.getLogger("org.apache.ivy").setLevel(logging.CRITICAL)

        # Tắt cảnh báo cụ thể của Ivy và Spark
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="ivy")
        warnings.filterwarnings("ignore", category=UserWarning, module="pyspark")
        warnings.filterwarnings("ignore", message=".*NativeCodeLoader.*")

        logger.info("Các logger đã được cấu hình để giảm thiểu cảnh báo")

    @staticmethod
    def create_spark_filter():
        """
        Tạo bộ lọc để loại bỏ các thông báo Ivy và dependency resolution
        """
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
        filter_instance = SparkFilter()
        root_logger = logging.getLogger()
        root_logger.addFilter(filter_instance)

        return filter_instance

    @staticmethod
    def silence_outputs():
        """
        Chuyển hướng và lọc đầu ra stdout/stderr để giảm cảnh báo

        Returns:
            Tuple: (original_stdout, original_stderr, null_logger) để khôi phục sau này
        """
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

        # Lưu trữ đầu ra gốc
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Tạo logger mới
        null_logger_out = NullLogger(sys.stdout)
        null_logger_err = NullLogger(sys.stderr)

        # Chuyển hướng stdout và stderr
        sys.stdout = null_logger_out
        sys.stderr = null_logger_err

        return original_stdout, original_stderr, (null_logger_out, null_logger_err)

    @staticmethod
    def restore_outputs(original_stdout, original_stderr):
        """
        Khôi phục stdout/stderr gốc

        Args:
            original_stdout: Đầu ra stdout gốc
            original_stderr: Đầu ra stderr gốc
        """
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    @staticmethod
    def silence_ivy_and_spark_logs(func):
        """
        Decorator để ẩn tất cả các thông báo Ivy và Spark khi khởi tạo

        Args:
            func: Hàm cần bọc decorator

        Returns:
            Hàm wrapper đã được áp dụng decorator
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

# MARK: - Các hàm tương thích ngược

@SparkUtils.silence_ivy_and_spark_logs
def get_spark_session(app_name="Vietnam Real Estate Price Prediction", enable_hive=False, silence_warnings=True):
    """
    Hàm tương thích ngược để khởi tạo và trả về một SparkSession

    Args:
        app_name (str): Tên ứng dụng Spark
        enable_hive (bool): Bật hỗ trợ Hive
        silence_warnings (bool): Tắt cảnh báo không cần thiết

    Returns:
        SparkSession: Session Spark đã được cấu hình
    """
    # Sử dụng phương thức tĩnh của lớp SparkUtils
    # Removed silence_warnings as it's not a valid parameter for create_spark_session
    return SparkUtils.create_spark_session(app_name=app_name,
                                        enable_hive=enable_hive)


def configure_spark_logging():
    """
    Hàm tương thích ngược để cấu hình Spark logging
    """
    return SparkUtils.configure_spark_logging()


def is_spark_available():
    """
    Hàm tương thích ngược để kiểm tra xem Spark có khả dụng hay không

    Returns:
        bool: True nếu Spark khả dụng, False nếu không
    """
    return SparkUtils.is_spark_available()


def get_spark_version():
    """
    Hàm tương thích ngược để lấy phiên bản Spark

    Returns:
        str: Phiên bản Spark hoặc "Không khả dụng" nếu không thể xác định
    """
    return SparkUtils.get_spark_version()


def stop_spark_session(spark=None):
    """
    Hàm tương thích ngược để dừng SparkSession

    Args:
        spark (SparkSession, optional): Phiên làm việc Spark cần dừng. Nếu không cung cấp, sẽ dừng phiên hiện tại.

    Returns:
        bool: True nếu dừng thành công, False nếu có lỗi
    """
    return SparkUtils.stop_spark_session(spark=spark)


def diagnose_spark():
    """
    Hàm tương thích ngược để chẩn đoán thiết lập Spark

    Returns:
        Dict[str, str]: Thông tin chẩn đoán Spark
    """
    return SparkUtils.diagnose_spark()


def convert_pandas_to_spark(spark, df):
    """
    Hàm tương thích ngược để chuyển đổi DataFrame Pandas sang Spark

    Args:
        spark (SparkSession): Phiên làm việc Spark hiện có
        df (pd.DataFrame): DataFrame cần chuyển đổi

    Returns:
        SparkSession.DataFrame: Spark DataFrame hoặc None nếu có lỗi
    """
    return SparkUtils.convert_pandas_to_spark(spark, df)


def convert_spark_to_pandas(spark_df):
    """
    Hàm tương thích ngược để chuyển đổi DataFrame Spark sang Pandas

    Args:
        spark_df: Spark DataFrame cần chuyển đổi

    Returns:
        pd.DataFrame: Pandas DataFrame hoặc None nếu có lỗi
    """
    return SparkUtils.convert_spark_to_pandas(spark_df)
