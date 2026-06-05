import os
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- CONFIGURATION ---
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="NexusDemand AI", layout="wide")

# --- UI STYLING ---
st.title("NexusDemand: AI Pricing & Demand Dashboard")
st.markdown("### Probabilistic Forecasting & Dynamic Price Optimization")
st.divider()

# --- SIDEBAR ---
st.sidebar.header("Control Panel")

# 1. Fetch the list of available products from the API
try:
    products_response = requests.get(f"{API_URL}/products")
    if products_response.status_code == 200:
        available_products = products_response.json().get("products", [])
    else:
        available_products = ["Error loading products"]
except:
    available_products = ["API Offline"]

# 2. Use a Dropdown (Selectbox) instead of text input
product_id = st.sidebar.selectbox(
    "Select Product ID", 
    options=available_products,
    index=available_products.index("21212") if "21212" in available_products else 0
)

analyze_btn = st.sidebar.button("Run Analysis")

# --- HELPER FUNCTIONS ---
def fetch_forecast(pid):
    try:
        response = requests.get(f"{API_URL}/forecast/{pid}")
        return response.json() if response.status_code == 200 else None
    except:
        return None

def fetch_optimization(pid):
    try:
        response = requests.get(f"{API_URL}/optimize/{pid}")
        return response.json() if response.status_code == 200 else None
    except:
        return None

# --- MAIN LOGIC ---
if analyze_btn:
    with st.spinner(f"Analyzing product {product_id}..."):
        # 1. Fetch Data from FastAPI
        forecast_data = fetch_forecast(product_id)
        opt_data = fetch_optimization(product_id)

        if forecast_data and opt_data:
            # --- SECTION 1: PRICE RECOMMENDATION ---
            st.subheader("Pricing Recommendation")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"${opt_data['CurrentPrice']:.2f}")
            col2.metric("Recommended Price", f"${opt_data['RecommendedPrice']:.2f}", 
                        delta=f"{opt_data['RecommendedPrice'] - opt_data['CurrentPrice']:.2f}")
            col3.metric("Expected Profit Uplift", f"${opt_data['ExpectedProfitUplift']:.2f}")
            col4.metric("Inventory Status", opt_data['InventoryStatus'].upper())

            st.info(f"**Recommended Action:** {opt_data['RecommendedAction'].replace('_', ' ').title()}")

            st.divider()

            # --- SECTION 2: DEMAND FORECAST ---
            st.subheader("7-Day Demand Forecast")
            
            # Prepare dates for the X-axis
            start_date = datetime.now()
            dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
            
            # Get forecast values
            p10 = forecast_data['forecast']['p10']
            p50 = forecast_data['forecast']['p50']
            p90 = forecast_data['forecast']['p90']

            # Create Plotly Figure
            fig = go.Figure()

            # Add P50 (Median)
            fig.add_trace(go.Scatter(x=dates, y=p50, name="Median Forecast (p50)", 
                                     line=dict(color='orange', width=3), mode='lines+markers'))
            
            # Add Shaded Uncertainty Area (p10 to p90)
            fig.add_trace(go.Scatter(
                x=dates + dates[::-1],
                y=p90 + p10[::-1],
                fill='toself',
                fillcolor='rgba(0,176,246,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name="80% Confidence Interval"
            ))

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Units Sold",
                hovermode="x unified",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # --- SECTION 3: DETAILS ---
            with st.expander("View Technical Diagnostics"):
                st.write(f"**Price Elasticity:** {opt_data['Elasticity']:.4f}")
                st.write(f"**Predicted Demand (Sum p50):** {opt_data['PredictedDemand']:.2f} units")
                st.write(f"**Current Inventory:** {opt_data['CurrentInventory']:.0f} units")

        else:
            st.error("Could not fetch data from API. Please check if the FastAPI server is running.")
else:
    st.write("Enter a Product ID in the sidebar and click 'Run Analysis' to begin.")