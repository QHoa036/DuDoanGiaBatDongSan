# VIETNAM REAL ESTATE PRICE PREDICTION APPLICATION

> **COURSE**: BIG DATA AND APPLICATIONS

## Development Team

- **Lê Thị Cẩm Giang** - Author <https://github.com/lcg1908>
- **Nguyễn Quỳnh Anh** - Co-author <https://github.com/Quynanhng25>
- **Nguyễn Cao Hoài Duyên** - Co-author <https://github.com/CaoHoaiDuyen>
- **Đinh Trương Ngọc Quỳnh Hoa** - Co-author <https://github.com/QHoa036>
- **Trần Hoàng Nghĩa** - Co-author <https://github.com/Blink713>
- **Nguyễn Phương Thảo** - Co-author <https://github.com/thaonguyenbi>

## Overview

The Vietnam Real Estate Price Prediction application is a modern platform that combines PySpark, machine learning, and Streamlit technologies to provide:

- **Accurate real estate price predictions** based on property characteristics
- **Market analysis** with an intuitive, modern, and interactive interface
- **Price trends** by region, time, and influencing factors
- **A cross-platform and user-friendly experience**

## Project Structure

```bash
Vietnam_Real_Estate_Price_Prediction/
├── App/                            # Main Application
│   ├── src/                        # Source Code
│   │   ├── data/                   # Sample Data
│   │   ├── logs/                   # System Logs
│   │   ├── styles/                 # CSS and Interface
│   │   └── utils/                  # Utilities
│   └── vn_real_estate_app.py       # Main application file
├── References/                     # Reference Documents
├── .env.example                    # Environment variable configuration template
├── requirements.txt                # Library list
└── run_app.sh                      # Application run script (cross-platform)
```

## Installation and Usage Guide

### System Requirements

- **Python 3.8+**
- **Java Runtime Environment (JRE)** (for PySpark)
- **Git Bash** (recommended for Windows)

### Installation and Running the Application

The application supports multiple platforms (Windows, macOS, Linux) with a single command:

```bash
./run_app.sh
```

This script will automatically:

1. Detect the operating system and set up the environment accordingly
2. Install the necessary dependencies
3. Create and activate the Python virtual environment
4. Ask the user if they want to use Ngrok to create a public URL
5. Run the Streamlit application

### Using with Ngrok

To share the application via a public URL:

1. Register for an account at [ngrok.com](https://ngrok.com)
2. Get the authtoken from your dashboard
3. Enter the authtoken into the `env.local` file
4. Select 'y' when asked about using Ngrok


## Key Libraries

- **PySpark**: For big data processing and building ML models
- **Streamlit**: For building the interactive web interface
- **Pandas & NumPy**: For data processing and analysis
- **Plotly & Matplotlib**: For data visualization
- **Ngrok**: For creating a public URL to share the application

## Acknowledgements

Our team would like to express our sincere gratitude to Mr. Nguyen Manh Tuan, lecturer for the Big Data and Applications course at UEH University, for his dedicated guidance and for sharing valuable knowledge and experience that helped us not only to master the theory but also to apply it in practice. We sincerely thank him for his dedication and enthusiasm in helping our group complete this project to the best of our ability.