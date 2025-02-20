# Japan-energy-weather-analysis


This project focuses on analyzing energy performance metrics in conjunction with regional weather data. By leveraging machine learning models, we aim to predict energy demand based on various meteorological factors.

## Project Structure

- `data/`: Contains datasets.
  - `raw/`: Original datasets.
  - `processed/`: Preprocessed datasets.

- `models/`: Stores trained models and scalers.
  - `best_gru_model.keras`: Trained GRU model.
  - `feature_scaler.pkl`: Scaler for input features.
  - `target_scaler.pkl`: Scaler for target variable.

- `src/`: Source code.
  - `predictor.py`: `GRUModelPredictor` class for predictions.

- `notebooks/`: Jupyter notebooks for analysis.
  - `energy-weather-analysis.ipynb`: Data exploration and Model performance evaluation.

## Introduction

Understanding the relationship between energy consumption and weather conditions is crucial for efficient energy management and planning. This project integrates energy performance data with weather metrics from multiple regions to build predictive models for energy demand forecasting.

## Dataset

The dataset comprises energy performance metrics and weather data from the following regions:

- Kyoto
- Wakayama
- Toyooka
- Kobe
- Shionomisaki
- Osaka
- Hikone

Key features include temperature, humidity, wind speed, and a holiday indicator.

## Model Performance Metrics

The following metrics evaluate the performance of our GRU model on the validation dataset:

| Metric                               | Value  |
|--------------------------------------|--------|
| **Root Mean Squared Error (RMSE)**   | 18.14  |
| **Mean Absolute Error (MAE)**        | 14.02  |
| **Mean Absolute Percentage Error (MAPE)** | 0.88% |
| **R-squared (R²)**                   | 0.997  |

**Interpretation:**

1. **Root Mean Squared Error (RMSE):** Measures the square root of the average squared differences between predicted and actual values. An RMSE of 18.14 indicates that, on average, the model's predictions deviate from the actual values by approximately 18.14 units. Given the target variable's range (955 to 2827), this deviation is relatively small, suggesting high prediction accuracy.


2. **Mean Absolute Error (MAE):** Calculates the average absolute differences between predicted and actual values. With an MAE of 14.02, the model's predictions are, on average, 14.02 units away from the actual values. This low MAE further confirms the model's precision.


3. **Mean Absolute Percentage Error (MAPE):** Expresses the average absolute error as a percentage of the actual values. A MAPE of 0.88% signifies that the model's predictions are, on average, less than 1% off from the actual values, highlighting exceptional predictive performance.


4. **R-squared (R²):** Represents the proportion of variance in the dependent variable that is predictable from the independent variables. An R² value of 0.997 indicates that 99.7% of the variance in the target variable is explained by the model, demonstrating an excellent fit to the data.

Collectively, these metrics underscore the model's robustness and its capability to deliver precise predictions on the validation dataset.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/energy-performance-weather-analysis.git
    cd energy-performance-weather-analysis
    ```

2. **Create a virtual environment**:

    ```
    python3 -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```
   
3. **Install the required packages**:

    ```
    pip install -r requirements.txt
    ```

## Usage 

### Load the trained model
```python
model_path = 'best_gru_model.keras'
```
### Load scalers
```python
feature_scaler = joblib.load('feature_scaler.pkl')
target_scaler = joblib.load('target_scaler.pkl')
```

### Load Data 
```python
df = pd.read_csv('power_generation_weather_final.csv')
```

### Initialize the Predictor
```python
target_column = 'actual_performance(10000 kW)'
look_back = 24  # Adjust based on your model's configuration

# Initialize the predictor
predictor = GRUModelPredictor(
    model_path=model_path,
    target_column=target_column,
    look_back=look_back,
    feature_scaler=feature_scaler,
    target_scaler=target_scaler
)
```
### Make and display predictions 
```python
predictions = predictor.predict(df_reduced)
predictor.display_prediction(df_reduced, predictions=predictions, bins=10, show=100)
```

### Evaluate Specific Data Points
```python
row_index = 2896  # Replace with the index of the row you want to evaluate
metrics = predictor.evaluate_row(df_reduced, row_index)
predictor.print_metrics(metrics)
```

## License
This project is licensed under the MIT License. See the LICENSE file for details
