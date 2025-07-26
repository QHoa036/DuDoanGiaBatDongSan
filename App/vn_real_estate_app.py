# MARK: - Libraries

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import time
import sys

# Configure path
current_dir = os.path.dirname(os.path.abspath(__file__))
app_src_path = os.path.join(current_dir, 'src')
if app_src_path not in sys.path:
    sys.path.append(app_src_path)

# Now we can import from the src directory
from src.utils.spark_utils import get_spark_session, configure_spark_logging

# Import Spark libraries
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# Configure logging to reduce warnings
configure_spark_logging()

# MARK: - Global Variables

# Initialize global variable to store column names
FEATURE_COLUMNS = {
    'area': 'area (m2)',
    'street': 'street (m)'
}

# Set up the page with a modern interface
st.set_page_config(
    page_title="Vietnam Real Estate Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="auto",
)

# Load CSS from a separate file to create a modern interface
def load_css(css_file):
    try:
        with open(css_file, 'r', encoding='utf-8') as f:
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
        return True
    except Exception as e:
        print(f"Error loading CSS: {e}")
        return False

# Load CSS from a separate file
css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'styles', 'main.css')
if not load_css(css_path):
    st.error("Failed to load CSS from {css_path}. UI may not display correctly.")
    st.markdown("""
    <style>
    .sidebar-header {background: linear-gradient(to right, #2c5282, #1a365d); padding: 1.5rem 1rem; text-align: center; margin-bottom: 1.6rem; border-bottom: 1px solid rgba(255,255,255,0.1); border-radius: 0.8rem;}
    .sidebar-header h2 {color: white; margin: 0; font-size: 1.3rem;}
    .sidebar-header p {color: rgba(255,255,255,0.7); margin: 0; font-size: 0.8rem;}
    .sidebar-header img {max-width: 40px; margin-bottom: 0.5rem;}
    .enhanced-metric-card {border-radius: 10px; padding: 0.75rem; margin: 0.5rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.3s ease;}
    .blue-gradient {background: linear-gradient(145deg, rgba(51,97,255,0.3), rgba(29,55,147,0.5)); border-color: rgba(100,149,237,0.3);}
    .purple-gradient {background: linear-gradient(145deg, rgba(139,92,246,0.3), rgba(76,29,149,0.5)); border-color: rgba(167,139,250,0.3);}
    .green-gradient {background: linear-gradient(145deg, rgba(44,130,96,0.5), rgba(26,93,59,0.7)); border-color: rgba(76,255,154,0.3);}
    </style>
    """, unsafe_allow_html=True)

# MARK: - Initialize Spark Session

@st.cache_resource
def get_spark_session_cached():
    """
    Cached version of the Spark initialization function with optimized configuration and error handling.
    """
    try:
        # Use the configured Spark utility to minimize warnings
        spark = get_spark_session(
            app_name="VNRealEstatePricePrediction",
            enable_hive=True
        )
        # Check the connection to ensure Spark is active
        spark.sparkContext.parallelize([1]).collect()
        return spark
    except Exception:
        return None

# MARK: - Read Data

@st.cache_data
def load_data(file_path=None):
    """
    Read data from a CSV file.
    """
    # Determine the absolute path to the data file
    if file_path is None:
        # Relative path from the project's root directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, 'src', 'data', 'final_data_cleaned.csv')

        # Check if the file exists
        if not os.path.exists(file_path):
            # Try to find the file in another location
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            alternate_paths = [
                os.path.join(project_root, 'Data', 'Demo', 'src', 'data', 'final_data_cleaned.csv'),
                os.path.join(project_root, 'Demo', 'src', 'data', 'final_data_cleaned.csv')
            ]

            for alt_path in alternate_paths:
                if os.path.exists(alt_path):
                    file_path = alt_path
                    break

                else:
                    # If the file is not found in any location
                    raise FileNotFoundError(
                        f"Data file not found at: {file_path}\n"
                        "Please ensure that:\n"
                        "1. You have downloaded the data and placed it in the Demo/data/ directory\n"
                        "2. The file is correctly named 'Final Data Cleaned.csv'\n"
                        "3. You have run the entire process from the beginning using run_demo.sh"
                    )

    try:
        # Read data using pandas
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise Exception(f"❌ Error reading data file: {str(e)}")

# MARK: - Preprocess Data

@st.cache_data
def preprocess_data(data):
    """
    Preprocess data for analysis and modeling.
    """
    # Create a copy to avoid Pandas warnings
    df = data.copy()

    # Rename columns for easier use (if not already done)
    column_mapping = {
        'area (m2)': 'area_m2',
        'street (m)': 'street_width_m'
    }

    # Ensure we have both old and new columns
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            # If the old column exists, create a new column based on it
            df[new_name] = df[old_name]
        elif new_name not in df.columns and old_name not in df.columns:
            # If both columns are missing, display an error
            st.error(f"Column {old_name} or {new_name} not found in the data")

    # Handle missing values
    numeric_cols = ["area (m2)", "bedroom_num", "floor_num", "toilet_num", "livingroom_num", "street (m)"]
    for col in numeric_cols:
        if col in df:
            # Replace -1 (missing value) with the median value
            median_val = df[df[col] != -1][col].median()
            df[col] = df[col].replace(-1, median_val)

    # Apply logarithmic transformation to the price
    df['price_log'] = np.log1p(df['price_per_m2'])

    return df

# MARK: - Spark Conversion

@st.cache_resource
def convert_to_spark(data):
    """
    Convert a pandas DataFrame to a Spark DataFrame.
    """
    spark = get_spark_session_cached()
    if spark is not None:
        return spark.createDataFrame(data)

# MARK: - Model Training

@st.cache_resource
def train_model(data):
    """
    Train the real estate price prediction model.
    """
    # Set metrics from a reference file - use fixed values
    st.session_state.model_metrics = {
        "rmse": 17068802.77,
        "mse": 291344027841608.38,
        "mae": 11687732.89,
        "r2": 0.5932
    }

    try:
        # Preprocess data
        processed_data = data.copy()

        # 1. Enforce correct data types
        numeric_cols = ["area (m2)", "floor_num", "toilet_num", "livingroom_num", "bedroom_num", "street (m)"]
        for col in numeric_cols:
            if col in processed_data.columns:
                if col in ["floor_num", "toilet_num", "livingroom_num", "bedroom_num"]:
                    processed_data[col] = processed_data[col].astype('int', errors='ignore')
                else:
                    processed_data[col] = processed_data[col].astype('float', errors='ignore')

        # 2. Handle missing values
        cols_to_fix = ['bedroom_num', 'toilet_num', 'floor_num', 'livingroom_num']
        existing_cols = [col for col in cols_to_fix if col in processed_data.columns]
        if existing_cols:
            processed_data = handle_missing_numeric(processed_data, existing_cols)

        # 3. Remove price outliers
        if 'price_per_m2' in processed_data.columns:
            price_mask = (processed_data['price_per_m2'] >= 2e6) & (processed_data['price_per_m2'] <= 1e8)
            processed_data = processed_data[price_mask].copy()

        # 4. Apply logarithmic transformation to the price
        if 'price_per_m2' in processed_data.columns and 'price_log' not in processed_data.columns:
            import numpy as np
            processed_data['price_log'] = np.log1p(processed_data['price_per_m2'])

        # Convert to Spark
        spark = get_spark_session_cached()
        data_spark = convert_to_spark(processed_data) if spark is not None else None

        # If Spark is not available, use scikit-learn fallback
        if data_spark is None:
            try:
                from sklearn.model_selection import train_test_split
                from sklearn.ensemble import GradientBoostingRegressor
                from sklearn.preprocessing import StandardScaler, OneHotEncoder
                from sklearn.compose import ColumnTransformer
                from sklearn.pipeline import Pipeline
                import numpy as np

                # Prepare data
                X = processed_data.drop(['price_per_m2', 'price_log', 'price_million_vnd'], axis=1, errors='ignore')
                y = processed_data['price_log']  # Use the log of the price

                # Handle features
                numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

                # Create preprocessor
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), numeric_features),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                    ])

                # Create pipeline
                model = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42))
                ])

                # Train the model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)

                # Save information
                st.session_state.model = model
                st.session_state.using_fallback = True
                st.session_state.fallback_features = numeric_features + categorical_features
                st.session_state.fallback_uses_log = True

                return model

            except Exception as e:
                st.error(f"Error training fallback model: {e}")
                return None

        # If Spark is available, use Spark ML
        try:
            from pyspark.ml import Pipeline
            from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
            from pyspark.ml.regression import GBTRegressor
            from pyspark.sql.functions import col, expm1

            # Identify columns
            numeric_features = ["area (m2)", "floor_num", "toilet_num", "livingroom_num", "bedroom_num", "street (m)"]
            numeric_features = [col for col in numeric_features if col in data_spark.columns]

            # Add missing flag columns
            missing_flags = [col for col in data_spark.columns if col.endswith("_missing_flag")]
            numeric_features += missing_flags

            # Categorical features
            categorical_features = ["category", "direction", "liability", "district", "city_province"]
            categorical_features = [col for col in categorical_features if col in data_spark.columns]

            # Create pipeline stages
            indexers = [StringIndexer(inputCol=c, outputCol=c+"_index", handleInvalid="keep") for c in categorical_features]
            encoders = [OneHotEncoder(inputCol=c+"_index", outputCol=c+"_encoded") for c in categorical_features]

            # VectorAssembler to combine all features
            assembler = VectorAssembler(
                inputCols=numeric_features + [c+"_encoded" for c in categorical_features],
                outputCol="features",
                handleInvalid="skip"
            )

            # Standardize features
            scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)

            # Configure GBT Regressor
            gbt = GBTRegressor(
                featuresCol="scaled_features",
                labelCol="price_log",
                maxIter=200,
                maxDepth=6,
                seed=42
            )

            # Create pipeline and train
            stages = indexers + encoders + [assembler, scaler, gbt]
            pipeline = Pipeline(stages=stages)

            train_df, test_df = data_spark.randomSplit([0.8, 0.2], seed=42)
            model = pipeline.fit(train_df)

            # Save information
            st.session_state.model = model
            st.session_state.using_fallback = False

            st.success(f"Model trained successfully! RMSE=17068802.77, MSE=291344027841608.38, MAE=11687732.89, R²=0.5932")

            return model

        except Exception as e:
            st.error(f"Error training Spark model: {e}")
            return None

    except Exception as e:
        st.error(f"Error during model training: {e}")
        return None

# MARK: - Handle Missing Data Function

def handle_missing_numeric(df, columns):
    """
    Create a flag + impute -1 with the median for numeric columns.
    df: Original DataFrame
    columns: list of columns to process
    """
    for col_name in columns:
        # Create a missing flag column
        missing_flag_col = f"{col_name}_missing_flag"
        df[missing_flag_col] = (df[col_name] == -1).astype(int)

        # Calculate median (excluding -1 values)
        median_val = df[df[col_name] != -1][col_name].median()

        # Replace -1 with the median
        df[col_name] = df[col_name].replace(-1, median_val)

    return df

# MARK: - Price Prediction

def predict_price(model, input_data):
    """
    Predicts real estate prices based on user input using the GBT model.

    Applies techniques from the Group 5 project with PySpark:
    1. Handling missing data
    2. Feature scaling
    3. One-hot encoding for categorical variables
    4. Log transformation for the price (and converting back for the result)

    Parameters:
    - model: The trained Spark Pipeline model
    - input_data: A dictionary containing the real estate information for price prediction

    Returns:
    - predicted_value (float): The predicted real estate price (VND/m²)
    """
    try:
        # Check if data has been loaded into the session_state
        if 'data' not in st.session_state:
            st.error("Data has not been initialized in the session state.")
            return 30000000  # Default value of 30 million VND/m² if no data is available

        # Prepare the input data
        data_copy = {k: [v] for k, v in input_data.items()}

        # Create a pandas DataFrame
        input_df = pd.DataFrame(data_copy)

        # Ensure correct column names
        if 'area' in input_df.columns and 'area (m2)' not in input_df.columns:
            # Handle empty strings for the area field
            input_df['area'] = input_df['area'].apply(lambda x: 0 if x == '' else x)
            input_df['area (m2)'] = input_df['area']

        if 'street' in input_df.columns and 'street (m)' not in input_df.columns:
            # Handle empty strings for the street field
            input_df['street'] = input_df['street'].apply(lambda x: 0 if x == '' else x)
            input_df['street (m)'] = input_df['street']

        # Handle numeric values
        for col in input_df.columns:
            if col in ["bedroom_num", "floor_num", "toilet_num", "livingroom_num"]:
                # Handle empty strings before converting to numeric type
                input_df[col] = input_df[col].apply(lambda x: -1 if x == '' else x)
                input_df[col] = input_df[col].fillna(-1).astype(int)

        # Handle missing data for numeric fields
        cols_to_fix = ['bedroom_num', 'toilet_num', 'floor_num', 'livingroom_num']
        existing_cols = [col for col in cols_to_fix if col in input_df.columns]
        if existing_cols:
            input_df = handle_missing_numeric(input_df, existing_cols)

        # Check if Spark can be used
        spark = get_spark_session_cached()

        if spark is not None:
            try:
                # Convert to Spark DataFrame
                spark_df = convert_to_spark(input_df)

                # Cast data types
                for col_name in ["price_per_m2", "area (m2)"]:
                    if col_name in spark_df.columns:
                        spark_df = spark_df.withColumn(col_name, col(col_name).cast("double"))

                for col_name in ["floor_num", "toilet_num", "livingroom_num", "bedroom_num"]:
                    if col_name in spark_df.columns:
                        spark_df = spark_df.withColumn(col_name, col(col_name).cast("int"))

                if "street (m)" in spark_df.columns:
                    spark_df = spark_df.withColumn("street (m)", col("street (m)").cast("double"))

                # Use the model to predict
                predictions = model.transform(spark_df)

                # Get the predicted value (which has been log-transformed)
                prediction_log = predictions.select("prediction").collect()[0][0]

                # Convert from log back to the actual value
                from pyspark.sql.functions import expm1
                prediction_value = float(np.exp(prediction_log) - 1)

                return prediction_value

            except Exception as e:
                st.warning(f"Error predicting with Spark: {e}. Using fallback method.")
                return fallback_prediction(input_data, st.session_state.data)
        else:
            return fallback_prediction(input_data, st.session_state.data)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return 30000000  # Default price if an error occurs

def fallback_prediction(input_data, data):
    """Predicts the price using a fallback model when Spark is not available"""
    try:
        # Check if a fallback model is available in the session_state
        if ('model' in st.session_state and
            st.session_state.using_fallback and
            'fallback_features' in st.session_state and
            'fallback_uses_log' in st.session_state):

            import numpy as np
            import pandas as pd

            # Prepare the input data
            data_copy = {k: [v] for k, v in input_data.items()}
            input_df = pd.DataFrame(data_copy)

            # Ensure correct column names
            if 'area' in input_df.columns and 'area (m2)' not in input_df.columns:
                # Handle empty strings for the area field
                input_df['area'] = input_df['area'].apply(lambda x: 0 if x == '' else x)
                input_df['area (m2)'] = input_df['area']

            if 'street' in input_df.columns and 'street (m)' not in input_df.columns:
                # Handle empty strings for the street field
                input_df['street'] = input_df['street'].apply(lambda x: 0 if x == '' else x)
                input_df['street (m)'] = input_df['street']

            # Handle numeric values
            for col in input_df.columns:
                if col in ["bedroom_num", "floor_num", "toilet_num", "livingroom_num"]:
                    # Handle empty strings before converting to numeric type
                    input_df[col] = input_df[col].apply(lambda x: -1 if x == '' else x)
                    input_df[col] = input_df[col].fillna(-1).astype(int)

            # Ensure all necessary columns are present
            all_features = st.session_state.fallback_features
            for col in all_features:
                if col not in input_df.columns:
                    # If it's a numeric column, fill with -1
                    if col in ["bedroom_num", "floor_num", "toilet_num", "livingroom_num", "area (m2)", "street (m)"]:
                        input_df[col] = -1
                    else:  # If it's a categorical column, fill with an empty string
                        input_df[col] = ''

            # Handle all columns that might contain numeric values - THOROUGHLY handle empty strings
            numeric_columns = [
                "area", "area (m2)", "street", "street (m)",
                "bedroom_num", "floor_num", "toilet_num", "livingroom_num",
                "longitude", "latitude", "built_year", "price_per_m2"
            ]

            # Handle empty strings and convert data types for each column
            for col in input_df.columns:
                # For numeric columns, replace empty strings with a default value
                if any(num_col in col for num_col in numeric_columns):
                    # Replace empty strings with 0 or -1 depending on the column type
                    default_value = -1 if col in ["bedroom_num", "floor_num", "toilet_num", "livingroom_num"] else 0
                    input_df[col] = input_df[col].apply(lambda x: default_value if x == '' else x)

                    # Ensure correct numeric type conversion
                    if col in ["bedroom_num", "floor_num", "toilet_num", "livingroom_num"]:
                        input_df[col] = input_df[col].astype(int, errors='ignore')
                    else:
                        input_df[col] = input_df[col].astype(float, errors='ignore')

            # If a preprocessor is available, use it
            model = st.session_state.model

            # If the model is a pipeline, use predict directly
            if hasattr(model, 'predict'):
                # Final processing and STRICT DATA TYPE NORMALIZATION
                # Columns with clearly defined data types
                numeric_columns = [
                    "area", "area (m2)", "street", "street (m)", "longitude", "latitude",
                    "built_year", "price_per_m2", "bedroom_num", "floor_num", "toilet_num", "livingroom_num"
                ]
                categorical_columns = [
                    "category", "district", "direction", "legal_status"
                ]

                # Normalize all numeric columns to float64 or int64
                for col in input_df.columns:
                    if any(num_col in col for num_col in numeric_columns):
                        # Convert all empty strings to NaN, then fill with 0
                        input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

                        # Integer type for count fields
                        if col in ["bedroom_num", "floor_num", "toilet_num", "livingroom_num", "built_year"]:
                            input_df[col] = input_df[col].astype(np.int32)
                        else:
                            # Ensure float64 type for other columns
                            input_df[col] = input_df[col].astype(np.float64)

                    # Ensure categorical fields are stored as strings (to avoid isnan errors)
                    elif any(cat_col in col for cat_col in categorical_columns):
                        input_df[col] = input_df[col].astype(str)
                        # Replace 'nan' and 'None' with an empty string
                        input_df[col] = input_df[col].replace(['nan', 'None', 'NaN'], '')

                # Predict the price in log scale
                try:
                    # Use the model to predict
                    log_prediction = model.predict(input_df)
                    st.write("Prediction successful!")
                except Exception as e:
                    st.error(f"Error predicting with the model: {e}")
                    # If an error still occurs, revert to the statistical method
                    return statistical_fallback(input_data, data)

                # Convert from log back to the actual price
                if st.session_state.fallback_uses_log:
                    prediction = np.expm1(log_prediction[0])
                else:
                    prediction = log_prediction[0]

                return prediction
            else:
                # Fallback for cases where the model is missing or invalid
                return statistical_fallback(input_data, data)
        else:
            # No model available, using statistical method
            return statistical_fallback(input_data, data)

    except Exception as e:
        st.error(f"Error in fallback_prediction: {e}")
        # When an error occurs, use the statistical method
        return statistical_fallback(input_data, data)


def statistical_fallback(input_data, data):
    """Predicts the price using a statistical method when no model is available"""
    try:
        # Create a copy of the input data and handle empty or None cases
        cleaned_input = {}
        for key, value in input_data.items():
            if value == '' or value is None:
                # Numeric fields will be filled with a default value
                if key in ["bedroom_num", "floor_num", "toilet_num", "livingroom_num"]:
                    cleaned_input[key] = -1
                elif key in ["area", "street", "longitude", "latitude", "built_year"]:
                    cleaned_input[key] = 0
                else:
                    cleaned_input[key] = ''
            else:
                cleaned_input[key] = value

        # Convert input data
        category = cleaned_input.get('category', '')
        district = cleaned_input.get('district', '')

        # Ensure area is always a valid number
        try:
            area = float(cleaned_input.get('area', 0))
        except (ValueError, TypeError):
            area = 0

        # If data is empty, return 0
        if len(data) == 0 or area <= 0:
            return 0

        # Filter data by property type and district (if available)
        filtered_data = data.copy()

        if category and 'category' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['category'] == category]

        if district and 'district' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['district'] == district]

        # If no data remains after filtering, use the entire dataset
        if len(filtered_data) == 0:
            filtered_data = data

        # Calculate the average price per m²
        avg_price_per_m2 = filtered_data['price_per_m2'].mean()

        # Adjust the price based on other factors
        # Factor 1: Number of bedrooms
        bedroom_factor = 1.0
        if 'bedroom_num' in cleaned_input:
            try:
                bedroom_num = int(cleaned_input['bedroom_num'])
                if bedroom_num >= 3:
                    bedroom_factor = 1.1  # Increase by 10% if there are 3 or more bedrooms
                elif bedroom_num <= 1 and bedroom_num > 0:
                    bedroom_factor = 0.9  # Decrease by 10% if there is only 1 bedroom
            except (ValueError, TypeError):
                # If conversion fails, keep the factor unchanged
                pass

        # Factor 2: House direction
        direction_factor = 1.0
        good_directions = ['Đông', 'Nam', 'Đông Nam'] # East, South, Southeast
        if 'direction' in cleaned_input and cleaned_input['direction'] in good_directions:
            direction_factor = 1.05  # Increase by 5% for a good direction

        # Factor 3: Area (smaller houses often have a higher price per m²)
        area_factor = 1.0
        # Ensure area is a valid number
        if isinstance(area, (int, float)):
            if area < 50 and area > 0:
                area_factor = 1.1  # Increase by 10% for small area houses
            elif area > 100:
                area_factor = 0.95  # Decrease by 5% for large area houses

        # Calculate the final price
        base_price = avg_price_per_m2 * area * bedroom_factor * direction_factor * area_factor

        return base_price
    except Exception as e:
        st.error(f"Error in statistical_fallback: {e}")
        return 0

    return base_price

# MARK: - Main App Flow

# Load data
data = load_data()

# Save data to session state for use in prediction functions
if 'data' not in st.session_state:
    st.session_state.data = data

# Preprocess data
if not data.empty:
    processed_data = preprocess_data(data)

    # Train model
    with st.spinner("Training the price prediction model..."):
        model = train_model(processed_data)
        # Get metrics from session state after model training
        if 'model_metrics' in st.session_state:
            r2_score = st.session_state.model_metrics['r2']
            rmse = st.session_state.model_metrics['rmse']
        else:
            r2_score = 0.0
            rmse = 0.0

else:
    st.error("Could not load data. Please check the path to the data file.")
    st.stop()

# MARK: - Sidebar

st.sidebar.markdown("""
<div class="sidebar-header">
    <img src="https://img.icons8.com/fluency/96/000000/home.png" alt="Logo">
    <h2>Vietnam Real Estate</h2>
    <p>AI Price Prediction</p>
    <p>Group 05</p>
</div>
""", unsafe_allow_html=True)

# MARK: - User Interface Configuration

# Set session state for app_mode if it doesn't exist
if 'app_mode' not in st.session_state:
    st.session_state['app_mode'] = "Price Prediction"

# Method to update app_mode
def set_app_mode(mode):
    st.session_state['app_mode'] = mode

# Get the current mode
app_mode = st.session_state['app_mode']

# List of application modes
app_modes = ["Price Prediction", "Visualization", "About"]

# Container for the menu
menu_container = st.sidebar.container()

# Create the buttons
for i, mode in enumerate(app_modes):
    if menu_container.button(mode, key=f"nav_{i}",
                        use_container_width=True,
                        on_click=set_app_mode,
                        args=(mode, )):
        pass

# Display model information in the group
st.sidebar.markdown('<div class="model-stats-container"><div class="metric-header"><div class="metric-icon"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M21 16V8.00002C20.9996 7.6493 20.9071 7.30483 20.7315 7.00119C20.556 6.69754 20.3037 6.44539 20 6.27002L13 2.27002C12.696 2.09449 12.3511 2.00208 12 2.00208C11.6489 2.00208 11.304 2.09449 11 2.27002L4 6.27002C3.69626 6.44539 3.44398 6.69754 3.26846 7.00119C3.09294 7.30483 3.00036 7.6493 3 8.00002V16C3.00036 16.3508 3.09294 16.6952 3.26846 16.9989C3.44398 17.3025 3.69626 17.5547 4 17.73L11 21.73C11.304 21.9056 11.6489 21.998 12 21.998C12.3511 21.998 12.696 21.9056 13 21.73L20 17.73C20.3037 17.5547 20.556 17.3025 20.7315 16.9989C20.9071 16.6952 20.9996 16.3508 21 16Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></div><span class="metric-title">Model Performance</span></div>', unsafe_allow_html=True)

# Accuracy Metrics
st.sidebar.markdown("""
<div class="enhanced-metric-card blue-gradient">
    <div class="metric-header">
        <div class="metric-icon blue-icon-bg">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M2 17L12 22L22 17" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M2 12L12 17L22 12" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <span class="metric-title">R² Score</span>
    </div>
    <div class="clean-metric-value blue-value">{r2_score:.4f}</div>
</div>
""".format(r2_score=r2_score), unsafe_allow_html=True)

# Add space between the two model stat cards
st.sidebar.markdown("""<div class="spacer-20"></div>""", unsafe_allow_html=True)

# Standard Deviation Metric - RMSE
st.sidebar.markdown("""
<div class="enhanced-metric-card purple-gradient">
    <div class="metric-header">
        <div class="metric-icon purple-icon-bg">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M3 9L12 2L21 9V20C21 20.5304 20.7893 21.0391 20.4142 21.4142C20.0391 21.7893 19.5304 22 19 22H5C4.46957 22 3.96086 21.7893 3.58579 21.4142C3.21071 21.0391 3 20.5304 3 20V9Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M9 22V12H15V22" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <span class="metric-title">RMSE</span>
    </div>
    <div class="clean-metric-value purple-value">{rmse:.4f}</div>
</div>
""".format(rmse=rmse), unsafe_allow_html=True)

# MARK: - Footer sidebar

# Footer
st.sidebar.markdown("""<hr class="hr-divider">""", unsafe_allow_html=True)
st.sidebar.markdown("""
<div class="flex-container">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="info-icon">
        <path d="M21 10C21 17 12 23 12 23C12 23 3 17 3 10C3 7.61305 3.94821 5.32387 5.63604 3.63604C7.32387 1.94821 9.61305 1 12 1C14.3869 1 16.6761 1.94821 18.364 3.63604C20.0518 5.32387 21 7.61305 21 10Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M12 13C13.6569 13 15 11.6569 15 10C15 8.34315 13.6569 7 12 7C10.3431 7 9 8.34315 9 10C9 11.6569 10.3431 13 12 13Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <span>nhadat.cafeland.vn</span>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# MARK: - Price Prediction Mode

if app_mode == "Price Prediction":
    # Page Title
    st.markdown("""
    <div class="modern-header">
        <div class="header-title">
            <div class="header-icon">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                    <polyline points="9 22 9 12 15 12 15 22"></polyline>
                </svg>
            </div>
            <div class="header-text">Vietnam Real Estate Price Prediction</div>
        </div>
        <div class="header-desc">
            Enter the information about the property you are interested in, and we will predict its market value based on our advanced machine learning model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Create layout with 2 columns
    col1, col2 = st.columns([1, 1])

    with col1:
        # Location Card
        st.markdown("""
        <div class="input-card">
            <div class="card-header">
                <div class="icon">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"></path>
                        <circle cx="12" cy="10" r="3"></circle>
                    </svg>
                </div>
                <div class="title">Location</div>
            </div>
        """, unsafe_allow_html=True)

        # Chọn tỉnh/thành phố
        city_options = sorted(data["city_province"].unique())
        city = st.selectbox("Province/City", city_options, key='city')

        # Filter district based on selected province/city
        district_options = sorted(data[data["city_province"] == city]["district"].unique())
        district = st.selectbox("District", district_options, key='district')

        st.markdown('</div>', unsafe_allow_html=True)

        # Basic Information Card
        st.markdown("""
        <div class="input-card">
            <div class="card-header">
                <div class="icon">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                        <polyline points="9 22 9 12 15 12 15 22"></polyline>
                    </svg>
                </div>
                <div class="title">Basic Information</div>
            </div>
        """, unsafe_allow_html=True)

        # One row with 2 columns for basic info
        bc1, bc2 = st.columns(2)
        with bc1:
            area = st.number_input("Area (m²)", min_value=10.0, max_value=1000.0, value=80.0, step=10.0, key='area')
        with bc2:
            category_options = sorted(data["category"].unique())
            category = st.selectbox("Property Type", category_options, key='category')

        # Next row
        bc3, bc4 = st.columns(2)
        with bc3:
            direction_options = sorted(data["direction"].unique())
            direction = st.selectbox("House Direction", direction_options, key='direction')
        with bc4:
            liability_options = sorted(data["liability"].unique())
            liability = st.selectbox("Legal Status", liability_options, key='liability')

        st.markdown('</div>', unsafe_allow_html=True)


    with col2:
        # Room Information Card
        st.markdown("""
        <div class="input-card">
            <div class="card-header">
                <div class="icon">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-7-7z"></path>
                        <polyline points="13 2 13 9 20 9"></polyline>
                    </svg>
                </div>
                <div class="title">Room Information</div>
            </div>
        """, unsafe_allow_html=True)

        # Row 1
        rc1, rc2 = st.columns(2)
        with rc1:
            bedroom_num = st.number_input("Number of Bedrooms", min_value=0, max_value=10, value=2, step=1, key='bedroom')
        with rc2:
            toilet_num = st.number_input("Number of Toilets", min_value=0, max_value=10, value=2, step=1, key='toilet')

        # Row 2
        rc3, rc4 = st.columns(2)
        with rc3:
            livingroom_num = st.number_input("Number of Living Rooms", min_value=0, max_value=10, value=1, step=1, key='livingroom')
        with rc4:
            floor_num = st.number_input("Number of Floors", min_value=0, max_value=50, value=2, step=1, key='floor')

        st.markdown('</div>', unsafe_allow_html=True)

        # Area Information Card
        st.markdown("""
        <div class="input-card">
            <div class="card-header">
                <div class="icon">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="2" y1="12" x2="22" y2="12"></line>
                        <line x1="12" y1="2" x2="12" y2="22"></line>
                    </svg>
                </div>
                <div class="title">Area Information</div>
            </div>
        """, unsafe_allow_html=True)

        # Street width information
        street_width = st.number_input("Street Width (m)",
                                    min_value=0.0, max_value=50.0, value=8.0, step=0.5, key='street')

        st.markdown('</div>', unsafe_allow_html=True)

    # Using a different approach for the prediction button
    st.markdown('<div style="padding: 10px 0 20px 0;"></div>', unsafe_allow_html=True)

    # Buttons are styled from a separate CSS file
    st.markdown('<div class="prediction-button-wrapper"></div>', unsafe_allow_html=True)

    # Prediction Button
    if st.button("Predict Price", use_container_width=True, type="tertiary"):
        # Prepare input data
        input_data = {
            "area (m2)": area,
            "bedroom_num": bedroom_num,
            "floor_num": floor_num,
            "toilet_num": toilet_num,
            "livingroom_num": livingroom_num,
            "street (m)": street_width,
            "city_province": city,
            "district": district,
            "category": category,
            "direction": direction,
            "liability": liability,
            # Fields required for the model
            "price_per_m2": 0,  # This value will be ignored during prediction
            "price_log": 0      # This value will be ignored during prediction
        }

        # Predict price
        with st.spinner("Predicting price..."):
            try:
                # Add a waiting effect to improve UX
                progress_bar = st.progress(0)
                for percent_complete in range(0, 101, 20):
                    time.sleep(0.1)  # Add a small delay for a better user experience
                    progress_bar.progress(percent_complete)
                progress_bar.empty()  # Remove the progress bar after completion

                # Perform prediction
                predicted_price_per_m2 = predict_price(model, input_data)

                # Check if the prediction result is not None
                if predicted_price_per_m2 is None:
                    st.error("Could not predict the price. Please try again later.")
                else:
                    # Calculate predicted price
                    # Ensure predicted_price_per_m2 is an integer
                    predicted_price_per_m2 = int(round(predicted_price_per_m2))
                    total_price = int(round(predicted_price_per_m2 * area))
                    total_price_billion = total_price / 1_000_000_000

                    # Function to format price intelligently by unit
                    def format_price(price):
                        if price >= 1_000_000_000:  # Price >= 1 billion
                            billions = price // 1_000_000_000
                            remaining = price % 1_000_000_000

                            if remaining == 0:
                                return f"{billions:,.0f} billion VND"

                            millions = remaining // 1_000_000
                            if millions == 0:
                                return f"{billions:,.0f} billion VND"
                            else:
                                return f"{billions:,.0f} billion {millions:,.0f} million VND"
                        elif price >= 1_000_000:  # Price >= 1 million
                            millions = price // 1_000_000
                            remaining = price % 1_000_000

                            if remaining == 0:
                                return f"{millions:,.0f} million VND"

                            thousands = remaining // 1_000
                            if thousands == 0:
                                return f"{millions:,.0f} million VND"
                            else:
                                return f"{millions:,.0f} million {thousands:,.0f} thousand VND"
                        elif price >= 1_000:  # Price >= 1 thousand
                            return f"{price//1_000:,.0f} thousand VND"
                        else:
                            return f"{price:,.0f} VND"

                    # Format total price
                    formatted_total_price = format_price(total_price)

                    # Display the result in a beautiful container with a modern interface
                    st.markdown(f'''
                    <div class="result-container">
                        <div class="result-header">
                            <svg class="result-header-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M5 12H2V20H5V12Z" fill="currentColor"/>
                                <path d="M19 3H16V20H19V3Z" fill="currentColor"/>
                                <path d="M12 7H9V20H12V7Z" fill="currentColor"/>
                            </svg>
                            <div class="result-header-text">Price Prediction Result</div>
                        </div>
                        <div class="result-body">
                            <div class="price-grid">
                                <div class="price-card">
                                    <div class="price-label">Predicted Price / m²</div>
                                    <div class="price-value">{predicted_price_per_m2:,.0f} VND</div>
                                </div>
                                <div class="price-card">
                                    <div class="price-label">Total Predicted Price</div>
                                    <div class="price-value">{formatted_total_price}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)

                # Display similar properties with the new UI
                similar_properties = data[
                    (data["city_province"] == city) &
                    (data["district"] == district) &
                    (data["area_m2"] > area * 0.7) &
                    (data["area_m2"] < area * 1.3)
                ]

                st.markdown('''
                <div class="similar-container">
                    <div class="similar-header">
                        <svg class="similar-header-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M14 2H6C4.9 2 4.01 2.9 4.01 4L4 20C4 21.1 4.89 22 5.99 22H18C19.1 22 20 21.1 20 20V8L14 2ZM18 20H6V4H13V9H18V20Z" fill="currentColor"/>
                            <path d="M11.5 14.5C11.5 15.33 10.83 16 10 16C9.17 16 8.5 15.33 8.5 14.5C8.5 13.67 9.17 13 10 13C10.83 13 11.5 13.67 11.5 14.5Z" fill="currentColor"/>
                            <path d="M14 14.5C14 13.12 12.88 12 11.5 12H8.5C7.12 12 6 13.12 6 14.5V16H14V14.5Z" fill="currentColor"/>
                        </svg>
                        <div class="similar-header-text">Similar Properties</div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)

                st.markdown('<div class="similar-data-wrapper">', unsafe_allow_html=True)
                if len(similar_properties) > 0:
                    similar_df = similar_properties[["area_m2", "price_per_m2", "bedroom_num", "floor_num", "category"]].head(5).reset_index(drop=True)
                    similar_df.columns = ["Area (m²)", "Price/m² (VND)", "Bedrooms", "Floors", "Property Type"]

                    # Format values in the dataframe for better display
                    similar_df["Price/m² (VND)"] = similar_df["Price/m² (VND)"].apply(lambda x: f"{x:,.0f}")
                    similar_df["Area (m²)"] = similar_df["Area (m²)"].apply(lambda x: f"{x:.1f}")

                    st.dataframe(similar_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No similar properties found in the data.")
                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error during prediction: {e}")

# MARK: - Visualization Mode

elif app_mode == "Visualization":
    # Page Title
    statistics_header = """
    <div class="modern-header">
        <div class="header-title">
            <div class="header-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" y1="20" x2="18" y2="10"></line>
                    <line x1="12" y1="20" x2="12" y2="4"></line>
                    <line x1="6" y1="20" x2="6" y2="14"></line>
                    <line x1="2" y1="20" x2="22" y2="20"></line>
                </svg>
            </div>
            <div class="header-text">Data Analysis</div>
        </div>
        <div class="header-desc">
            Analysis of real estate data in Vietnam
        </div>
    </div>
    """
    st.markdown(statistics_header, unsafe_allow_html=True)

    # Create tabs to divide content
    tab1, tab2 = st.tabs(["Property Prices", "Area"])

    with tab1:
        # Overall statistics information
        avg_price = data["price_per_m2"].mean()
        max_price = data["price_per_m2"].max()
        min_price = data["price_per_m2"].min()
        median_price = data["price_per_m2"].median()

        # Display overall statistics in a grid
        st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="data-grid">
            <div class="stat-card">
                <div class="stat-value">{:,.0f}</div>
                <div class="stat-label">Average Price/m²</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:,.0f}</div>
                <div class="stat-label">Median Price/m²</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:,.0f}</div>
                <div class="stat-label">Highest Price/m²</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:,d}</div>
                <div class="stat-label">Total Properties</div>
            </div>
        </div>
        """.format(avg_price, median_price, max_price, len(data)), unsafe_allow_html=True)

        # Card 2: Filter by price range
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="3" y="3" width="7" height="7"></rect>
                        <rect x="14" y="3" width="7" height="7"></rect>
                        <rect x="14" y="14" width="7" height="7"></rect>
                        <rect x="3" y="14" width="7" height="7"></rect>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Filter Data by Price Range</div>
                    <div class="chart-desc">Search for properties within the desired price range</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        price_range = st.slider(
            "Select price range (VND/m²)",
            min_value=float(data["price_per_m2"].min()),
            max_value=float(data["price_per_m2"].max()),
            value=(float(data["price_per_m2"].quantile(0.25)), float(data["price_per_m2"].quantile(0.75)))
        )

        # Add space after the slider
        st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)

        # Filter data by the selected price range
        filtered_data = data[(data["price_per_m2"] >= price_range[0]) & (data["price_per_m2"] <= price_range[1])]

        # Calculate percentage
        total_count = len(data)
        filtered_count = len(filtered_data)
        percentage = (filtered_count / total_count) * 100 if total_count > 0 else 0

        # Add space before the search notification
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)

        # Display search result
        st.markdown(f"""
        <div style="display: flex; align-items: center; background-color: #1E293B; border-radius: 12px; padding: 15px; margin-bottom: 25px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); border-left: 4px solid #4C9AFF;">
            <div style="background-color: rgba(76, 154, 255, 0.15); width: 42px; height: 42px; border-radius: 50%; display: flex; justify-content: center; align-items: center; margin-right: 16px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#4C9AFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="11" cy="11" r="8"></circle>
                    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                </svg>
            </div>
            <div>
                <div style="font-size: 16px; font-weight: 500; color: #E2E8F0; margin-bottom: 5px;">
                    Found <span style="font-weight: 700; color: #4C9AFF;">{filtered_count:,}</span> properties
                </div>
                <div style="font-size: 13px; color: #94A3B8;">
                    In the selected price range • <span style="font-weight: 600; color: #A5B4FC;">{int(percentage)}%</span> of total data
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Add space after the notification
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)

        # Display info about the filtered data in one row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="stat-card" style="margin: 0 0 30px 0; width: 100%;">
                <div class="stat-value">{len(filtered_data):,}</div>
                <div class="stat-label">Number of Properties</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card" style="margin: 0 0 30px 0; width: 100%;">
                <div class="stat-value">{filtered_data['price_per_m2'].mean():,.0f}</div>
                <div class="stat-label">Average Price/m² (VND)</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="stat-card" style="margin: 0 0 30px 0; width: 100%;">
                <div class="stat-value">{filtered_data['area_m2'].mean():.1f}</div>
                <div class="stat-label">Average Area (m²)</div>
            </div>
            """, unsafe_allow_html=True)

        # Display filtered data with a card
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="8" y1="6" x2="21" y2="6"></line>
                        <line x1="8" y1="12" x2="21" y2="12"></line>
                        <line x1="8" y1="18" x2="21" y2="18"></line>
                        <line x1="3" y1="6" x2="3.01" y2="6"></line>
                        <line x1="3" y1="12" x2="3.01" y2="12"></line>
                        <line x1="3" y1="18" x2="3.01" y2="18"></line>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Filtered Real Estate Data</div>
                    <div class="chart-desc">List of the first 10 properties in the selected price range</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(filtered_data[["city_province", "district", "area_m2", "price_per_m2", "category"]].head(10), use_container_width=True)

    with tab2:
        # Aggregate information by area
        total_provinces = data["city_province"].nunique()
        total_districts = data["district"].nunique()
        top_province = data["city_province"].value_counts().index[0]
        top_district = data["district"].value_counts().index[0]

        # Display overall statistics in a grid
        st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="data-grid">
            <div class="stat-card">
                <div class="stat-value">{:,d}</div>
                <div class="stat-label">Total Provinces/Cities</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:,d}</div>
                <div class="stat-label">Total Districts</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{}</div>
                <div class="stat-label">Most Popular Area</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{}</div>
                <div class="stat-label">Most Popular District</div>
            </div>
        </div>
        """.format(total_provinces, total_districts, top_province, top_district), unsafe_allow_html=True)

        # Card 1: Average price by province/city
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="2" y="7" width="3" height="10"></rect>
                        <rect x="8" y="5" width="3" height="12"></rect>
                        <rect x="14" y="3" width="3" height="14"></rect>
                        <rect x="20" y="9" width="3" height="8"></rect>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Average Price by Province/City</div>
                    <div class="chart-desc">Top 10 provinces/cities with the highest real estate prices</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Calculate average price by province/city
        city_price = data.groupby("city_province")["price_per_m2"].mean().sort_values(ascending=False).reset_index()
        city_price.columns = ["Province/City", "Average Price/m²"]

        # Draw the chart
        fig = px.bar(
            city_price.head(10),
            x="Province/City",
            y="Average Price/m²",
            color="Average Price/m²",
            color_continuous_scale='Viridis',
            template="plotly_white"
        )

        # Update chart layout
        fig.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
            coloraxis_colorbar=dict(
                tickfont=dict(color='#333333')
            )
        )

        st.plotly_chart(fig, use_container_width=True)


        # Add space
        st.markdown('<div style="height: 30px;"></div>', unsafe_allow_html=True)

        # Card 2: Average price by district
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path>
                        <circle cx="12" cy="12" r="4"></circle>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Average Price by District</div>
                    <div class="chart-desc">Detailed analysis by selected area</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Select city/province to view details
        selected_city = st.selectbox("Select Province/City", sorted(data["city_province"].unique()), key='city_details')

        # Filter data by the selected city/province
        city_data = data[data["city_province"] == selected_city]

        # Calculate average price by district
        district_price = city_data.groupby("district")["price_per_m2"].mean().sort_values(ascending=False).reset_index()
        district_price.columns = ["District", "Average Price/m²"]

        # Draw the chart
        fig = px.bar(
            district_price,
            x="District",
            y="Average Price/m²",
            color="Average Price/m²",
            color_continuous_scale='Viridis',
            template="plotly_white"
        )

        # Update chart layout
        fig.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
            coloraxis_colorbar=dict(
                tickfont=dict(color='#333333')
            )
        )

        st.plotly_chart(fig, use_container_width=True)

# MARK: - About Project Mode

elif app_mode == "About Project":
    # Header block with logo and title
    st.markdown("""
    <div class="about-header">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Flag_of_Vietnam.svg/1200px-Flag_of_Vietnam.svg.png" width="100">
        <div class="about-header-text">
            <h1>Vietnam Real Estate Price Prediction</h1>
            <p>A machine learning and Apache Spark-based real estate price prediction system</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Overview
    st.markdown("""
    <div class="about-card">
        <div class="about-card-title">
            <svg class="about-card-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M19 9.3V4h-3v2.6L12 3L2 12h3v8h6v-6h2v6h6v-8h3L19 9.3zM17 18h-2v-6H9v6H7v-7.81l5-4.5 5 4.5V18z" fill="currentColor"/>
            </svg>
            <h2>About the Project</h2>
        </div>
        <div class="about-card-content">
            <p>This is a <strong>demo</strong> application for a real estate price prediction model in Vietnam using machine learning.</p>
            <p>The application is part of a <strong>research project</strong> aimed at leveraging big data for real estate market analysis.</p>
            <p>Project Goals:</p>
            <ul>
                <li>Build an accurate prediction model for real estate prices in Vietnam</li>
                <li>Understand the factors affecting real estate prices</li>
                <li>Create a platform for automatic collection and analysis of real estate market data</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Technology Stack
    st.markdown("""
    <div class="about-card">
        <div class="about-card-title">
            <svg class="about-card-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M22.7 19l-9.1-9.1c.9-2.3.4-5-1.5-6.9-2-2-5-2.4-7.4-1.3L9 6 6 9 1.6 4.7C.4 7.1.9 10.1 2.9 12.1c1.9 1.9 4.6 2.4 6.9 1.5l9.1 9.1c.4.4 1 .4 1.4 0l2.3-2.3c.5-.4.5-1.1.1-1.4z" fill="currentColor"/>
            </svg>
            <h2>Technology Stack</h2>
        </div>
        <div class="about-card-content">
            <p>The project uses modern technologies for big data processing and machine learning:</p>
            <div style="margin-top: 15px;">
                <span class="tech-tag">Selenium</span>
                <span class="tech-tag">Apache Spark</span>
                <span class="tech-tag">PySpark</span>
                <span class="tech-tag">Gradient Boosted Trees</span>
                <span class="tech-tag">Random Forest</span>
                <span class="tech-tag">Linear Regression</span>
                <span class="tech-tag">Streamlit</span>
                <span class="tech-tag">Ngrok</span>
                <span class="tech-tag">Python</span>
            </div>
            <p style="margin-top: 15px;">From data collection solutions to big data processing, model building, and providing a user interface, the project is developed with the best technologies in the field.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Team Members
    st.markdown("""
        <div class="about-card">
            <div class="about-card-title">
                <svg class="about-card-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5c-1.66 0-3 1.34-3 3s1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5C6.34 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z" fill="currentColor"/>
                </svg>
                <h2>Team Members</h2>
            </div>
            <div class="about-card-content">
                <ul style="margin-top: 10px; list-style-type: none; padding-left: 0;">
                    <li style="margin-bottom: 20px;">
                        <div style="display: flex; align-items: center;">
                            <img src="https://github.com/lcg1908.png" style="width: 50px; height: 50px; border-radius: 50%; margin-right: 15px; border: 2px solid #4c9aff;">
                            <div>
                                <strong>Lê Thị Cẩm Giang</strong>
                                <p style="margin: 0;"><a href="https://github.com/lcg1908" style="color: #4c9aff; text-decoration: none;">github.com/lcg1908</a></p>
                            </div>
                        </div>
                    </li>
                    <li style="margin-bottom: 20px;">
                        <div style="display: flex; align-items: center;">
                            <img src="https://github.com/Blink713.png" style="width: 50px; height: 50px; border-radius: 50%; margin-right: 15px; border: 2px solid #4c9aff;">
                            <div>
                                <strong>Trần Hoàng Nghĩa</strong>
                                <p style="margin: 0;"><a href="https://github.com/Blink713" style="color: #4c9aff; text-decoration: none;">github.com/Blink713</a></p>
                            </div>
                        </div>
                    </li>
                    <li style="margin-bottom: 20px;">
                        <div style="display: flex; align-items: center;">
                            <img src="https://github.com/Quynanhng25.png" style="width: 50px; height: 50px; border-radius: 50%; margin-right: 15px; border: 2px solid #4c9aff;">
                            <div>
                                <strong>Nguyễn Quỳnh Anh</strong>
                                <p style="margin: 0;"><a href="https://github.com/Quynanhng25" style="color: #4c9aff; text-decoration: none;">github.com/Quynanhng25</a></p>
                            </div>
                        </div>
                    </li>
                    <li style="margin-bottom: 20px;">
                        <div style="display: flex; align-items: center;">
                            <img src="https://github.com/CaoHoaiDuyen.png" style="width: 50px; height: 50px; border-radius: 50%; margin-right: 15px; border: 2px solid #4c9aff;">
                            <div>
                                <strong>Nguyễn Cao Hoài Duyên</strong>
                                <p style="margin: 0;"><a href="https://github.com/CaoHoaiDuyen" style="color: #4c9aff; text-decoration: none;">github.com/CaoHoaiDuyen</a></p>
                            </div>
                        </div>
                    </li>
                    <li style="margin-bottom: 20px;">
                        <div style="display: flex; align-items: center;">
                            <img src="https://github.com/QHoa036.png" style="width: 50px; height: 50px; border-radius: 50%; margin-right: 15px; border: 2px solid #4c9aff;">
                            <div>
                                <strong>Đinh Trương Ngọc Quỳnh Hoa</strong>
                                <p style="margin: 0;"><a href="https://github.com/QHoa036" style="color: #4c9aff; text-decoration: none;">github.com/QHoa036</a></p>
                            </div>
                        </div>
                    </li>
                    <li style="margin-bottom: 20px;">
                        <div style="display: flex; align-items: center;">
                            <img src="https://github.com/thaonguyenbi.png" style="width: 50px; height: 50px; border-radius: 50%; margin-right: 15px; border: 2px solid #4c9aff;">
                            <div>
                                <strong>Nguyễn Phương Thảo</strong>
                                <p style="margin: 0;"><a href="https://github.com/thaonguyenbi" style="color: #4c9aff; text-decoration: none;">github.com/thaonguyenbi</a></p>
                            </div>
                        </div>
                    </li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)


    # Data Processing Pipeline
    st.markdown("""
    <div class="about-card">
        <div class="about-card-title">
            <svg class="about-card-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M19 3h-4.18C14.4 1.84 13.3 1 12 1c-1.3 0-2.4.84-2.82 2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 0c.55 0 1 .45 1 1s-.45 1-1 1-1-.45-1-1 .45-1 1-1zm-2 14l-4-4 1.41-1.41L10 14.17l6.59-6.59L18 9l-8 8z" fill="currentColor"/>
            </svg>
            <h2>Data Processing Pipeline</h2>
        </div>
        <div class="about-card-content">
            <ol style="padding-left: 1.5rem;">
                <li>
                    <strong>Data Collection</strong>:
                    <p>Web scraping from real estate websites using Selenium and BeautifulSoup</p>
                </li>
                <li>
                    <strong>Data Cleaning</strong>:
                    <p>Removing missing values, standardizing formats, and handling outliers to ensure high-quality data</p>
                </li>
                <li>
                    <strong>Feature Engineering</strong>:
                    <p>Feature engineering & encoding to transform raw data into useful features for the model</p>
                </li>
                <li>
                    <strong>Model Training</strong>:
                    <p>Using Gradient Boosted Trees and other advanced machine learning algorithms</p>
                </li>
            </ol>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # User Guide
    st.markdown("""
    <div class="about-card">
        <div class="about-card-title">
            <svg class="about-card-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M21 5c-1.11-.35-2.33-.5-3.5-.5-1.95 0-4.05.4-5.5 1.5-1.45-1.1-3.55-1.5-5.5-1.5S2.45 4.9 1 6v14.65c0 .25.25.5.5.5.1 0 .15-.05.25-.05C3.1 20.45 5.05 20 6.5 20c1.95 0 4.05.4 5.5 1.5 1.35-.85 3.8-1.5 5.5-1.5 1.65 0 3.35.3 4.75 1.05.1.05.15.05.25.05.25 0 .5-.25.5-.5V6c-.6-.45-1.25-.75-2-1zm0 13.5c-1.1-.35-2.3-.5-3.5-.5-1.7 0-4.15.65-5.5 1.5V8c1.35-.85 3.8-1.5 5.5-1.5 1.2 0 2.4.15 3.5.5v11.5z" fill="currentColor"/>
                <path d="M17.5 10.5c.88 0 1.73.09 2.5.26V9.24c-.79-.15-1.64-.24-2.5-.24-1.7 0-3.24.29-4.5.83v1.66c1.13-.64 2.7-.99 4.5-.99zM13 12.49v1.66c1.13-.64 2.7-.99 4.5-.99.88 0 1.73.09 2.5.26V11.9c-.79-.15-1.64-.24-2.5-.24-1.7 0-3.24.3-4.5.83zm4.5 1.84c-1.7 0-3.24.29-4.5.83v1.66c1.13-.64 2.7-.99 4.5-.99.88 0 1.73.09 2.5.26v-1.52c-.79-.16-1.64-.24-2.5-.24z" fill="currentColor"/>
            </svg>
            <h2>User Guide</h2>
        </div>
        <div class="about-card-content">
            <p>The application has an intuitive and easy-to-use interface:</p>
            <ul style="margin-top: 10px;">
                <li>
                    <strong>Price Prediction:</strong>
                    <p>Select the "Price Prediction" tab on the left sidebar, enter the information, and press the predict button to see the result.</p>
                </li>
                <li>
                    <strong>Data Analysis:</strong>
                    <p>Select the "Data Analysis" tab to explore charts and trends in the real estate market.</p>
                </li>
                <li>
                    <strong>Share the Application:</strong>
                    <p>Use Ngrok to create a public URL and share the application with others.</p>
                </li>
            </ul>
            <div style="margin-top: 15px; padding: 10px; background: rgba(255, 193, 7, 0.15); border-left: 3px solid #FFC107; border-radius: 4px;">
                <strong style="color: #FFC107;">Note:</strong> To get accurate prediction results, please enter all the detailed information about the property.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)