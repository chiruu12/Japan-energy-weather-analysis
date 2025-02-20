from predictor import GRUModelPredictor
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


model_path = 'best_gru_model.keras'
target_column = "actual_performance(10000 kW)"
look_back = 24
feature_scaler = joblib.load('feature_scaler.pkl')
target_scaler = joblib.load('target_scaler.pkl')

predictor = GRUModelPredictor(model_path = model_path, target_column = target_column,
                              look_back = look_back,
                              feature_scaler = feature_scaler,
                              target_scaler = target_scaler)

df = pd.read_csv("csv_path.csv")
prediction = predictor.predict(df)

predictor.display_prediction(df, predictions=prediction, bins=10, show=100)

row_index = 2896
metrics = predictor.evaluate_row(df, row_index)
predictor.print_metrics(metrics)