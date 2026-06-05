import pandas as pd
import numpy as np
import joblib
from gluonts.dataset.common import ListDataset
from gluonts.model.predictor import Predictor
from pathlib import Path
import torch

class ModelProcessor:

    def __init__(self, processed_path, model_path, scalers_path):
        self.processed_path = processed_path
        self.model_path = model_path
        self.scalers_path = scalers_path

        self.df = None
        self.predictor = None
        self.scalers = None
    
    def load_everything(self):
        self.df = self.load_processed_data()
        self.df = self.add_time_features(self.df)

        print("Model Loaded!")
        # Patch torch.load to force CPU before gluonts calls it internally
        original_torch_load = torch.load
        torch.load = lambda *args, **kwargs: original_torch_load(
            *args, **{**kwargs, 'map_location': torch.device('cpu')}
        )
        self.predictor = Predictor.deserialize(Path(self.model_path))
        torch.load = original_torch_load  # restore after loading

        print("Scalers Loaded!")
        self.scalers = joblib.load(self.scalers_path)

    def load_processed_data(self):
        print("Data Loaded!")

        df = pd.read_csv(self.processed_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df['StockCode'] = df['StockCode'].astype(str)
        df = df.sort_values(['StockCode', 'Date'])

        return df
    
    def add_time_features(self, df):
        print("Added Data Features!")

        df['day_of_week'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month

        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        return df
    
    def create_single_dataset(self, product_id):
        group = self.df[self.df['StockCode'] == product_id].sort_values('Date').copy()
        
        if group.empty:
            raise ValueError(f"Product ID {product_id} not found in dataset.")

        scaler = self.scalers.get(product_id)
        if scaler is None:
            raise ValueError(f"No scaler found for product {product_id}")
            
        vals = group['Quantity'].values.reshape(-1, 1)
        target_values = scaler.transform(vals).flatten().astype(np.float32)

        feature_cols = ['dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'is_weekend']
        past_features = group[feature_cols].values.astype(np.float32)

        last_date = group['Date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7, freq='D')
        
        future_df = pd.DataFrame({'Date': future_dates})
        future_df['day_of_week'] = future_df['Date'].dt.dayofweek
        future_df['month'] = future_df['Date'].dt.month
        future_df['dow_sin'] = np.sin(2 * np.pi * future_df['day_of_week'] / 7)
        future_df['dow_cos'] = np.cos(2 * np.pi * future_df['day_of_week'] / 7)
        future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
        future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)
        future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
        
        future_features = future_df[feature_cols].values.astype(np.float32)
        all_features = np.vstack([past_features, future_features]).T

        dataset = ListDataset([
            {
                'start': pd.Period(group['Date'].iloc[0], freq='D'),
                'target': target_values,
                'feat_dynamic_real': all_features
            }
        ], freq='D')

        return dataset
    
    def inverse_scale(self, product_id, scaled_values):
        scaler = self.scalers.get(product_id)

        if scaler is None:
            return scaled_values
        
        vals = np.array(scaled_values).reshape(-1, 1)
        return scaler.inverse_transform(vals).flatten()
    
    def predict_demand(self, product_id):
        ds = self.create_single_dataset(product_id)

        forecast = list(self.predictor.predict(ds))[0]

        p10_scaled = forecast.quantile(0.1)
        p50_scaled = forecast.quantile(0.5)
        p90_scaled = forecast.quantile(0.9)

        p10 = self.inverse_scale(product_id, p10_scaled)
        p50 = self.inverse_scale(product_id, p50_scaled)
        p90 = self.inverse_scale(product_id, p90_scaled)

        return {
            "p10": p10.tolist(),
            "p50": p50.tolist(),
            "p90": p90.tolist()
        }

if __name__ == "__main__":
    print("ModelProcessor Unit Test!")

    try:
        processor = ModelProcessor(
            processed_path = 'data/processed/demand_timeseries.csv',
            model_path = 'data/processed/DeepAR',
            scalers_path = 'data/processed/scalers.pkl'
        )

        processor.load_everything()

        test_id = "21212"
        result = processor.predict_demand(test_id)

        print(f"\nTest Prediction for {test_id}:")
        print(f"p50 (Median): {result['p50'][0]:.2f} units for day 1")
        print(f"p90 (Upper): {result['p90'][0]:.2f} units for day 1")
        print("\nModelProcessor is working!")
    
    except Exception as e:
        print(f"\nTest Failed! Error: {e}")