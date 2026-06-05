import pandas as pd

class DemandDataProcessor:
    """
    """

    def __init__(self, raw_path, processed_path):
        self.raw_path = raw_path
        self.processed_path = processed_path
    
    def load_raw_data(self):
        print("Data Loaded!")
        return pd.read_excel(self.raw_path, engine = "openpyxl")
    
    def clean_data(self, df):
        print("Data Cleaned!")

        # We want to remove negatives (returns) and zero prices
        df = df[(df['Quantity'] > 0) & (df['Price'] > 0)].copy()

        # Standardize dates
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['Date'] = df['InvoiceDate'].dt.date
        df['StockCode'] = df['StockCode'].astype(str)
        return df
    
    def aggregate_daily(self, df):
        print("Data Aggregate!")

        df = df.groupby([
            'StockCode', 
            'Date'
        ]).agg({
            'Quantity': 'sum'
        }).reset_index()

        return df
    
    def filling_gaps(self, df):
        print("Data fill missing days with zero!")

        all_dates = pd.date_range(start = df['Date'].min(), end = df['Date'].max())
        all_stocks = df['StockCode'].unique()

        # Multi-index to find the missing days
        index = pd.MultiIndex.from_product([all_dates, all_stocks], names = ['Date', 'StockCode'])
        df = df.set_index([
            'StockCode',
            'Date'
        ]).reindex(index, fill_value = 0).reset_index()
        
        return df
    
    def filter_top_products(self, df, top_n = 50):
        print(f"Filtered Top {top_n} products!")

        top_products = df.groupby(
            'StockCode'
        )['Quantity'].sum().nlargest(top_n).index

        return df[df['StockCode'].isin(top_products)]
    
    def clip_outliers(self, df):

        def _apply_clip(group):
            upper_limit = group['Quantity'].quantile(0.95)
            group['Quantity'] = group['Quantity'].clip(upper = upper_limit)
            return group
        
        return df.groupby(
            'StockCode'
        ).apply(_apply_clip).reset_index(drop = True)
    
    def run_pipeline(self, top_n = 50):
        df = self.load_raw_data()
        df = self.clean_data(df)
        df = self.aggregate_daily(df)
        df = self.filling_gaps(df)
        df = self.filter_top_products(df, top_n = top_n)
        df = self.clip_outliers(df)

        df.to_csv(self.processed_path, index = False)
        print(f"Processed data saved to {self.processed_path}!")


if __name__ == "__main__":
    print("DemandDataProcessor Unit Test!")

    try:
        processor = DemandDataProcessor(
            raw_path = 'data/raw/online_retail_II.xlsx',
            processed_path = 'data/processed/demand_timeseries.csv'
        )

        processor.run_pipeline(top_n = 50)

    except Exception as e:
        print("\nTest Failed! Error: {e}")

    