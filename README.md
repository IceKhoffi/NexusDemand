# NexusDemand

> **An end-to-end intelligent pricing system**.
> Probabilistic demand forecasting with DeepAR, paired with a profit-maximizing dynamic pricing optimizer, served via FastAPI and visualized in a real-time Streamlit dashboard.

---

## The Problem

Traditional retail pricing is static. Products are priced based on gut feel or simple rules, ignoring demand signals, inventory levels, and price elasticity. This leads to two costly failure modes:
* **understocking at prices too low**,
* **overstock sitting at prices too high**.

NexusDemand solves this by combining **probabilistic demand forecasting** with a **grid-search profit optimizer** to recommend the exact price that maximizes expected profit given current inventory, cost, and demand elasticity.

---

## System Architecture

```
UCI Online Retail Data
        │
        ▼
┌─────────────────┐
│  Data Pipeline  │  processor.py — clean, aggregate, fill gaps, clip outliers
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  DeepAR Model   │  GluonTS — probabilistic forecasting (p10/p50/p90)
│  (forecaster)   │  Trained on top-50 products, 7-day horizon
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Optimizer    │  optimizer.py — constant-elasticity demand curve + grid search
│                 │  Finds price that maximizes expected profit
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────────┐
│   FastAPI       │────▶│  Streamlit Dashboard  │
│   /forecast     │     │  Metrics + Forecast   │
│   /optimize     │     │  Chart + Diagnostics  │
└─────────────────┘     └──────────────────────┘
```

---

## Why DeepAR? Why Not LSTM?

| | LSTM | DeepAR |
|---|---|---|
| **Output** | Single point estimate | Full probability distribution |
| **Uncertainty** | ❌ No confidence intervals | ✅ p10 / p50 / p90 quantiles |
| **Multi-series** | Train one model per product | Train one model across all products |
| **Cold start** | ❌ Needs retraining | ✅ Generalizes to new products |
| **Inventory decisions** | Risky, no uncertainty info | Safe, plan for worst/best case |
 
A point forecast tells you "sell 50 units." A probabilistic forecast tells you "sell 30–70 units, most likely 50." For inventory and pricing decisions, **the uncertainty is the signal**.
 
---

## Model Performance
 
Evaluated on a held-out test set across the top-50 products:
 
| Metric | Value |
|---|---|
| Average MAE (p50) | 27.95 units |
| Quantile Loss p10 | 5.80 |
| Quantile Loss p50 | 13.98 |
| Quantile Loss p90 | 10.88 |
| **80% Coverage** | **80.6%** ✅ |

The 80% coverage score of **80.6%** means the model's confidence interval is well-calibrated — the true demand falls inside the predicted range exactly as often as it should. This is the most important metric for a probabilistic forecaster.

---

## Dynamic Pricing — Before vs After
 
The optimizer uses a **constant-elasticity demand curve** to simulate how demand changes at different price points, then selects the price with maximum expected profit:
 
```
Profit(p) = (p - cost) × min(demand(p), inventory)
```
 
Example for Product `21212`:
| | Static Pricing | Dynamic Pricing |
|---|---|---|
| Price | Current price | Optimizer recommendation |
| Strategy | Fixed | Elasticity-adjusted |
| Outcome | Baseline profit | +Uplift shown in dashboard |
 
---

## Tech Stack
 
| Layer | Technology |
|---|---|
| Forecasting | GluonTS DeepAR (PyTorch) |
| Optimization | NumPy grid search + elasticity model |
| Backend API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Deployment | Docker + Docker Compose |
| Data | UCI Online Retail II dataset |
 
---

## Project Structure
 
```
NexusDemand/
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI endpoints
│   ├── data_pipeline/
│   │   └── processor.py         # Data cleaning & aggregation
│   ├── models/
│   │   └── forecaster.py        # DeepAR inference wrapper
│   └── optimization/
│       └── optimizer.py         # Profit optimizer
├── notebooks/
│   ├── 01_Exploratory_Data_Analysis.ipynb
│   ├── 02_Statistical_Analysis_and_Baseline.ipynb
│   ├── 03_Probabilistic_Forecasting_DeepAR.ipynb
│   └── 04_Price_Optimization_and_Profit_Analysis.ipynb
├── data/
│   └── processed/               # CSVs + trained model (not in Git)
├── streamlit_app.py
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt
└── requirements-streamlit.txt
```
 
---

## Quickstart
 
### Docker
 
```bash
git clone https://github.com/IceKhoffi/NexusDemand.git
cd NexusDemand
docker-compose up --build
```
 
- API + docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Dashboard: [http://localhost:8501](http://localhost:8501)
> **Note:** The `data/processed/` folder (model weights + CSVs) is not included in the repo due to size. See [Data Setup](#data-setup) below.

---
 
## Data Setup
 
The trained DeepAR model and processed data files are not committed to Git (too large). To reproduce:
 
1. Download the [UCI Online Retail II dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place it at `data/raw/online_retail_II.xlsx`
2. Run the data pipeline:
```bash
   python src/data_pipeline/processor.py
```
3. Train the model by running `notebooks/03_Probabilistic_Forecasting_DeepAR.ipynb`
Or contact me for access to the pre-processed files.
 
---
 
## API Endpoints
 
| Endpoint | Description |
|---|---|
| `GET /` | Health check |
| `GET /products` | List all available product IDs |
| `GET /forecast/{product_id}` | 7-day probabilistic forecast (p10/p50/p90) |
| `GET /optimize/{product_id}` | Price recommendation + profit analysis |
 
Interactive docs available at `/docs` when running.
