from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from src.models.forecaster import ModelProcessor
from src.optimization.optimizer import suggest_optimal_price

# Initial FastAPI app
app = FastAPI(
    title = "NexusDemand AI API",
    description = "API for Probabilistic Demand Forecasting and Dynamic Pricing Optimization"
)

# Global Variabel (Loaded "ONCE" at startup)
processor = None
product_metadata = None
elasticity_df = None

@app.on_event("startup")
async def startup_event():
    global processor, product_metadata, elasticity_df

    print("Initializing NexusDemand System...")

    # Load Model and Scalers
    processor = ModelProcessor(
        processed_path = 'data/processed/demand_timeseries.csv',
        model_path = 'data/processed/DeepAR',
        scalers_path = 'data/processed/scalers.pkl'
    )

    processor.load_everything()

    # Load Product Metadata (Cost, Inventory, Current Price)
    product_metadata = pd.read_csv('data/processed/product_daily_business.csv')
    product_metadata = product_metadata.sort_values('Date').groupby('StockCode').tail(1)
    product_metadata.set_index('StockCode', inplace = True)

    # Load Elasticity data
    elasticity_df = pd.read_csv('data/processed/elasticity_by_product.csv')
    elasticity_df.set_index('StockCode', inplace = True)

    print('API is ready to be used!')

@app.get("/")
async def root():
    return {"message": "NexusDemand API is Online", "endpoints": ["/forecast/{id}", "/optimize/{id}"]}

@app.get("/forecast/{product_id}")
async def get_forecast(product_id: str):
    
    try:
        # Generate forecast
        forecast = processor.predict_demand(product_id)
        return {
            "product_id": product_id,
            "forecast": forecast,
            "horizon": "7 Days"
        }
    
    except Exception as e:
        raise HTTPException(status_code = 404, detail = f"Product {product_id} not found or forecast failed: {str(e)}")
    
@app.get("/optimize/{product_id}")
async def optimize_price(product_id: str):
    try:
        # Get Forecast
        forecast_data = processor.predict_demand(product_id)
        p50_sum = sum(forecast_data['p50']) # Total demand for the next 7 days

        # Get Product Metadata (Gather from CSV)
        if product_id not in product_metadata.index:
            raise HTTPException(status_code = 404, detail = "Product metadata not found")
        
        meta = product_metadata.loc[product_id]
        current_price = float(meta['AvgPrice'])

        # MOCK COST AND INVENTORY!
        cost_price = current_price * 0.60
        current_inventory = p50_sum * np.random.choice([0.5, 1.0, 2.0]) # Stock Levels Simulation

        # Get Elasticity
        if product_id not in elasticity_df.index:
            elasticity = -1.0 # Fallback elasticity
        else:
            elasticity = float(elasticity_df.loc[product_id, 'ElasticityFinal'])
        
        # Run Optimizer Logic
        recommendation = suggest_optimal_price(
            product_id = product_id,
            predicted_demand = p50_sum,
            current_inventory = current_inventory,
            current_price = current_price,
            cost_price = cost_price,
            elasticity = elasticity
        )

        return recommendation

    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Optimization error: {str(e)}")

@app.get("/products")
async def get_all_products():
    try:
        return {"products": product_metadata.index.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching product list: {str(e)}")