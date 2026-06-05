import numpy as np
import pandas as pd

def adjusted_demand_constant_elasticity(
    baseline_demand,
    current_price,
    candidate_price,
    elasticity
):
    """
    Adjust demand using a constant-elasticity demand curve.

    Parameters
    ----------
    baseline_demand : float
        Forecasted demand at the current price.
    current_price : float
        Current observed product price.
    candidate_price : float
        Candidate optimized price.
    elasticity : float
        Price elasticity of demand. Usually negative.

    Returns
    -------
    float
        Demand adjusted for the candidate price.
    """
    if baseline_demand < 0:
        baseline_demand = 0

    if current_price <= 0:
        raise ValueError("current_price must be greater than 0.")

    if candidate_price <= 0:
        raise ValueError("candidate_price must be greater than 0.")

    demand = baseline_demand * (candidate_price / current_price) ** elasticity

    return max(0, demand)


def suggest_optimal_price(
    product_id,
    predicted_demand,
    current_inventory,
    current_price,
    cost_price,
    elasticity,
    min_price_factor=0.80,
    max_price_factor=1.20,
    grid_size=81
):
    """
    Suggest the profit-maximizing price for a product using grid search.

    Parameters
    ----------
    product_id : str
        Product identifier.
    predicted_demand : float
        Forecasted demand at the current price.
    current_inventory : float
        Available stock.
    current_price : float
        Current selling price.
    cost_price : float
        Unit cost.
    elasticity : float
        Price elasticity of demand.
    min_price_factor : float
        Lower bound as a fraction of current price.
    max_price_factor : float
        Upper bound as a fraction of current price.
    grid_size : int
        Number of candidate prices to test.

    Returns
    -------
    dict
        Best price recommendation and optimization diagnostics.
    """

    if current_price <= 0:
        raise ValueError("current_price must be greater than 0.")

    if cost_price < 0:
        raise ValueError("cost_price cannot be negative.")

    if current_inventory < 0:
        raise ValueError("current_inventory cannot be negative.")

    lower_price = current_price * min_price_factor
    upper_price = current_price * max_price_factor

    # Avoid recommending prices below cost for profit maximization.
    lower_price = max(lower_price, cost_price * 1.01)

    candidate_prices = np.linspace(lower_price, upper_price, grid_size)

    rows = []

    for candidate_price in candidate_prices:
        adjusted_demand = adjusted_demand_constant_elasticity(
            baseline_demand=predicted_demand,
            current_price=current_price,
            candidate_price=candidate_price,
            elasticity=elasticity
        )

        actual_sales = min(adjusted_demand, current_inventory)

        revenue = candidate_price * actual_sales
        profit = (candidate_price - cost_price) * actual_sales

        rows.append({
            "ProductID": product_id,
            "CandidatePrice": candidate_price,
            "AdjustedDemand": adjusted_demand,
            "ActualSales": actual_sales,
            "Revenue": revenue,
            "Profit": profit
        })

    optimization_table = pd.DataFrame(rows)

    best_row = optimization_table.loc[
        optimization_table["Profit"].idxmax()
    ].to_dict()

    current_adjusted_demand = adjusted_demand_constant_elasticity(
        baseline_demand=predicted_demand,
        current_price=current_price,
        candidate_price=current_price,
        elasticity=elasticity
    )

    current_sales = min(current_adjusted_demand, current_inventory)
    current_profit = (current_price - cost_price) * current_sales

    best_price = best_row["CandidatePrice"]
    best_profit = best_row["Profit"]

    if current_inventory < predicted_demand:
        inventory_status = "understock"
    elif current_inventory > predicted_demand * 2:
        inventory_status = "overstock"
    else:
        inventory_status = "balanced"

    if best_price > current_price:
        action = "raise_price"
    elif best_price < current_price:
        action = "lower_price"
    else:
        action = "keep_price"

    return {
        "ProductID": product_id,
        "CurrentPrice": current_price,
        "RecommendedPrice": best_price,
        "CostPrice": cost_price,
        "Elasticity": elasticity,
        "PredictedDemand": predicted_demand,
        "CurrentInventory": current_inventory,
        "ExpectedSales": best_row["ActualSales"],
        "ExpectedRevenue": best_row["Revenue"],
        "ExpectedProfit": best_profit,
        "CurrentProfit": current_profit,
        "ExpectedProfitUplift": best_profit - current_profit,
        "InventoryStatus": inventory_status,
        "RecommendedAction": action,
        "OptimizationTable": optimization_table
    }