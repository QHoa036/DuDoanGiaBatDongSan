#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for configuring and managing PySpark.
Provides functions to initialize a SparkSession with optimal configuration
and minimize unnecessary warnings.
"""

# MARK: - Libraries

import os
import sys
import logging
from pyspark.sql import SparkSession

# MARK: - Configuration

def configure_spark_logging():
    """
    Configures the logging level for PySpark to minimize unnecessary warnings.
    """
    # Set environment variable to control the ngrok CLI
    os.environ['NGROK_LOG_LEVEL'] = 'critical'

    # Set environment variables to completely disable Spark and Ivy messages
    os.environ['SPARK_SUBMIT_OPTS'] = '-Dlog4j.rootCategory=FATAL -Dorg.apache.ivy.util.Message.level=FATAL -Dorg.apache.ivy.core.settings.IvySettings.level=FATAL -Dorg.apache.ivy.core.report.ResolveReport.level=FATAL'

    # Additional environment variables to control logging
    os.environ['SPARK_SILENT'] = 'true'
    os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
    os.environ['SPARK_LOG_LEVEL'] = 'FATAL'
    os.environ['PYSPARK_PYTHON_LOG_LEVEL'] = 'FATAL'
    os.environ['PYSPARK_DRIVER_PYTHON_LOG_LEVEL'] = 'FATAL'

    # Determine the absolute path to the log4j2.properties file in the config directory
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log4j_file_path = os.path.join(src_dir, 'config', 'log4j2.properties')

    if os.path.exists(log4j_file_path):
        # Disable Java and native Hadoop warnings using the log4j configuration file
        # Use environment variables via spark.driver.extraJavaOptions instead of JAVA_TOOL_OPTIONS
        # to avoid the "Picked up JAVA_TOOL_OPTIONS" message
        os.environ["SPARK_DRIVER_OPTS"] = f"-Dlog4j.configurationFile=file:{log4j_file_path} -Dlog4j.rootCategory=ERROR"
        os.environ["SPARK_EXECUTOR_OPTS"] = f"-Dlog4j.configurationFile=file:{log4j_file_path} -Dlog4j.rootCategory=ERROR"

        # Delete JAVA_TOOL_OPTIONS if it was previously set
        if "JAVA_TOOL_OPTIONS" in os.environ:
            del os.environ["JAVA_TOOL_OPTIONS"]
    else:
        # If the file is not found, unset this environment variable to avoid errors
        if "JAVA_TOOL_OPTIONS" in os.environ:
            del os.environ["JAVA_TOOL_OPTIONS"]

    # Hide native-hadoop warnings
    # Set environment variables to hide warnings
    os.environ["HADOOP_HOME"] = ""
    os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages org.apache.hadoop:hadoop-aws:3.3.1 pyspark-shell"
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    # Important environment variable to disable native-hadoop warnings
    os.environ["HADOOP_USER_NAME"] = "hadoop"
    os.environ["ARROW_PRE_0_15_IPC_FORMAT"] = "1"

    # Disable display of Hadoop warnings
    os.environ["HADOOP_CONF_DIR"] = ""
    os.environ["HADOOP_OPTS"] = "-Djava.awt.headless=true -Dlog4j.logger.org.apache.hadoop=ERROR"

    # Reduce the logging level of specific loggers
    logging.getLogger("py4j").setLevel(logging.CRITICAL)
    logging.getLogger("org").setLevel(logging.CRITICAL)
    logging.getLogger("akka").setLevel(logging.CRITICAL)
    logging.getLogger("ivy").setLevel(logging.CRITICAL)

    # Disable ivy warnings (used for dependency resolution)
    logging.getLogger("org.apache.spark.util.DependencyUtils").setLevel(logging.CRITICAL)
    logging.getLogger("org.apache.ivy").setLevel(logging.CRITICAL)

    # Create a filter to remove Ivy and dependency resolution messages
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

    # Apply the filter to the root logger to affect all loggers
    root_logger = logging.getLogger()
    root_logger.addFilter(SparkFilter())

    # Disable all Python warnings
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    # Disable specific Ivy and Spark warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="ivy")
    warnings.filterwarnings("ignore", category=UserWarning, module="pyspark")
    warnings.filterwarnings("ignore", message=".*NativeCodeLoader.*")

    # MARK: - Null Logger

    class NullLogger:
        def __init__(self, original_stream=None):
            self.original_stream = original_stream

        def write(self, message):
            # Ignore messages related to JAVA_TOOL_OPTIONS
            if self.original_stream and not 'Picked up JAVA_TOOL_OPTIONS' in message:
                self.original_stream.write(message)

        def flush(self):
            if self.original_stream:
                self.original_stream.flush()

    # Extreme approach: redirect Java's stdout during Spark initialization
    # Activated to filter out unwanted messages
    sys.stdout = NullLogger(sys.stdout)
    sys.stderr = NullLogger(sys.stderr)

# MARK: - Suppress Ivy and Spark Logs

# Create a decorator to suppress all Ivy and Spark output during initialization
def silence_ivy_and_spark_logs(func):
    """
    Decorator to suppress all Ivy and Spark messages during initialization.
    """
    def wrapper(*args, **kwargs):
        # Store the original stdout and stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Create a null file to redirect output
        null_file = open(os.devnull, 'w')

        # Create a filter to temporarily redirect output
        try:
            # Redirect output to /dev/null to hide messages
            sys.stdout = null_file
            sys.stderr = null_file

            # Set the highest logging level to disable messages
            old_level = logging.root.level
            logging.root.setLevel(logging.CRITICAL + 1)

            # Call the original function
            result = func(*args, **kwargs)

            return result
        finally:
            # Restore stdout and stderr, regardless of whether an exception occurred
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            null_file.close()

            # Restore the logging level
            logging.root.setLevel(old_level)

    return wrapper

# MARK: - Initialize Spark

@silence_ivy_and_spark_logs
def get_spark_session(app_name="Vietnam Real Estate Price Prediction", enable_hive=False, silence_warnings=True):
    """
    Initializes and returns a SparkSession with an optimal configuration to minimize warnings.
    """
    # Configure logging first (must be called before creating a SparkSession)
    configure_spark_logging()

    # If full silent mode is enabled, redirect stdout/stderr
    if silence_warnings:
        # Temporarily save the original stdout/stderr objects
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        null_output = open(os.devnull, 'w')

        # Redirect stdout/stderr during Spark initialization to hide warnings
        sys.stdout = null_output
        sys.stderr = null_output

    # Add log4j configuration (another way to reduce warnings)
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

    # Save the log4j configuration to an environment variable for use
    os.environ['SPARK_LOG4J_PROPS'] = log4j_properties

    # Create a builder with detailed configurations to reduce warnings
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

    # Disable SparkUI to reduce logging
    builder = builder.config("spark.ui.enabled", False)

    # Set the minimum logging level
    builder = builder.config("spark.sql.hive.thriftServer.singleSession", True)

    # Enable Hive support if needed
    if enable_hive:
        builder = builder.enableHiveSupport()

    # Create SparkSession
    spark = builder.getOrCreate()

    # Set the logging level for the SparkContext
    spark.sparkContext.setLogLevel("ERROR")

    # Disable Apache Spark warnings
    spark._jvm.org.apache.log4j.LogManager.getRootLogger().setLevel(spark._jvm.org.apache.log4j.Level.ERROR)

    # Set configuration to hide NativeCodeLoader warnings
    conf = spark._jsc.hadoopConfiguration()
    conf.set("mapreduce.app-submission.cross-platform", "true")

    # Restore original stdout/stderr if they were redirected
    if silence_warnings:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        null_output.close()

    return spark
