import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jpholiday
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


class GRUModelPredictor:
    def __init__(self,
                feature_scalar, target_scalar,
                model_path: str = 'best_model_gru.keras',
                target_column: int = 1000, look_back: int = 24,
                ):

        self.model_path = model_path
        self.required_columns = self.load_required_columns()
        self.target_column = target_column
        self.look_back = look_back
        self.model = self._load_model()
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

    @staticmethod
    def load_required_columns(self):
        return [
            'actual_performance(10000 kW)', 'kyoto_temperature', 'kyoto_humidity',
            'kyoto_wind_speed', 'wakayama_temperature', 'wakayama_humidity',
            'wakayama_wind_speed', 'toyooka_temperature', 'toyooka_humidity',
            'toyooka_wind_speed', 'kobe_temperature', 'kobe_humidity',
            'kobe_wind_speed', 'shionomisaki_temperature', 'shionomisaki_humidity',
            'shionomisaki_wind_speed', 'osaka_temperature', 'osaka_humidity',
            'osaka_wind_speed', 'hikone_temperature', 'hikone_humidity',
            'hikone_wind_speed', 'is_holiday'
        ]

    def _load_model(self):
        """Load the Keras model from the specified path."""
        model = load_model(self.model_path)
        print(f"Model loaded from {self.model_path}")
        return model

    def _add_is_holiday_and_filter(self, row):
        row_copy = row.copy()
        if 'is_holiday' not in row_copy.index:
            row_copy['is_holiday'] = 1 if jpholiday.is_holiday(row_copy.name) else 0
        filtered_row = row_copy.reindex(self.required_columns)
        return filtered_row

    def _preprocess(self, df):
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"The following required columns are missing from the input data: {missing_cols}")
        df_processed = df.apply(self._add_is_holiday_and_filter, axis=1)
        features_scaled = self.feature_scaler.fit_transform(df_processed)
        target_scaled = self.target_scaler.fit_transform(df[[self.target_column]])

        return features_scaled, target_scaled

    def predict(self, df):
        features_scaled, _ = self._preprocess(df)
        if len(features_scaled) < self.look_back:
            raise ValueError(f"Input data must have at least {self.look_back} rows.")

        X = []
        for i in range(self.look_back, len(features_scaled)):
            X.append(features_scaled[i - self.look_back:i])
        X = np.array(X)
        predictions_scaled = self.model.predict(X)
        predictions = self.target_scaler.inverse_transform(predictions_scaled)

        return predictions

    def evaluate_row(self, df, row_index):
        if row_index < self.look_back:
            raise ValueError("row_index must be at least as large as look_back")

        original_value = df.iloc[row_index][self.target_column]
        sample_df = df.iloc[row_index - self.look_back: row_index].copy()
        features_scaled, _ = self._preprocess(sample_df)
        sample_array_scaled = features_scaled.reshape(1, self.look_back, len(self.required_columns))

        pred_scaled = self.model.predict(sample_array_scaled)
        pred_original = self.target_scaler.inverse_transform(pred_scaled)
        predicted_value = pred_original.flatten()[0]

        abs_error = np.abs(predicted_value - original_value)
        perc_error = (abs_error / np.abs(original_value)) * 100 if original_value != 0 else None
        rmse = np.sqrt(np.mean(abs_error ** 2))
        mae = np.mean(abs_error)

        return {
            'original': original_value,
            'predicted': predicted_value,
            'absolute_error': abs_error,
            'percentage_error': perc_error,
            'rmse': rmse,
            'mae': mae
        }

    def print_metrics(self, metrics):
        print("Evaluation Metrics:")
        print(f"Original Value      : {metrics['original']}")
        print(f"Predicted Value     : {metrics['predicted']}")
        print(f"Absolute Error      : {metrics['absolute_error']}")
        if metrics['percentage_error'] is not None:
            print(f"Percentage Error    : {metrics['percentage_error']:.2f}%")
        else:
            print("Percentage Error    : N/A")
        print(f"RMSE                : {metrics['rmse']}")
        print(f"MAE                 : {metrics['mae']}")

    def display_prediction(self, df, predictions=None, bins=10, show='all'):
        _, target_scaled = self._preprocess(df)
        if len(target_scaled) < self.look_back:
            raise ValueError(f"Input data must have at least {self.look_back} rows.")
        y_test_scaled = target_scaled[self.look_back:]
        y_test = self.target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
        if predictions is None:
            predictions = self.predict(df)
        else:
            predictions = np.array(predictions)
        if predictions.shape[0] != y_test.shape[0]:
            raise ValueError("Number of predictions does not match number of samples in y_test.")
        residuals = y_test - predictions.flatten()
        percentage_error = np.abs(residuals) / (np.abs(y_test) + 1e-6) * 100

        # Plot 1: Actual vs. Predicted values
        plt.figure(figsize=(10, 5))
        if show == 'all':
            plt.plot(y_test, label='Actual (y_test)', marker='o')
            plt.plot(predictions, label='Predicted (y_pred)', marker='x')
        else:
            try:
                show_int = int(show)
            except Exception:
                show_int = len(y_test)
            plt.plot(y_test[:show_int], label='Actual (y_test)', marker='o')
            plt.plot(predictions[:show_int], label='Predicted (y_pred)', marker='x')
        plt.title("Actual vs. Predicted Values")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot 2: Histogram of Residuals
        plt.figure(figsize=(10, 5))
        plt.hist(residuals, bins=bins, edgecolor='black', color='blue')
        plt.title("Frequency Distribution of Residuals (y_test - y_pred)")
        plt.xlabel("Residual Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

        # Plot 3: Histogram of Percentage Errors
        plt.figure(figsize=(10, 5))
        plt.hist(percentage_error, bins=bins, edgecolor='black', color='red')
        plt.title("Frequency Distribution of Percentage Errors")
        plt.xlabel("Percentage Error (%)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()
