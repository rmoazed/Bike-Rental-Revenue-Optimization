from __future__ import annotations

import numpy as np
import pandas as pd

BASE_PRICE = 5.0
ALPHA = 0.15


def adjusted_demand(
    predicted_demand: float,
    price: float,
    base_price: float = BASE_PRICE,
    alpha: float = ALPHA,
) -> float:
    """Estimate demand after a price change using a simple elasticity assumption."""
    demand = predicted_demand * (1 - alpha * (price - base_price))
    return float(max(demand, 0.0))


def revenue(
    predicted_demand: float,
    price: float,
    base_price: float = BASE_PRICE,
    alpha: float = ALPHA,
) -> float:
    """Compute revenue at a candidate price under the elasticity model."""
    return float(price * adjusted_demand(predicted_demand, price, base_price, alpha))


def find_best_price(
    predicted_demand: float,
    price_min: float = 3.0,
    price_max: float = 8.0,
    step: float = 0.25,
    base_price: float = BASE_PRICE,
    alpha: float = ALPHA,
) -> dict[str, float]:
    """Search across a grid of prices and return the revenue-maximizing option."""
    prices = np.arange(price_min, price_max + step, step)
    best_price = None
    best_revenue = -np.inf
    best_demand = None

    for p in prices:
        adj_d = adjusted_demand(predicted_demand, p, base_price, alpha)
        rev = revenue(predicted_demand, p, base_price, alpha)
        if rev > best_revenue:
            best_price = p
            best_revenue = rev
            best_demand = adj_d

    return {
        "predicted_baseline_demand": float(predicted_demand),
        "optimal_price": float(best_price),
        "expected_adjusted_demand": float(best_demand),
        "expected_revenue": float(best_revenue),
    }


def optimize_prices_for_predictions(
    predictions: np.ndarray | list[float],
    base_price: float = BASE_PRICE,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Run pricing optimization across many predicted demand values."""
    rows = [find_best_price(float(pred), base_price=base_price, alpha=alpha) for pred in predictions]
    results = pd.DataFrame(rows)
    results["baseline_revenue"] = base_price * np.asarray(predictions, dtype=float)
    results["revenue_gain"] = results["expected_revenue"] - results["baseline_revenue"]
    return results


def total_percent_revenue_increase(results: pd.DataFrame) -> float:
    """Compute total percent revenue gain from optimized pricing vs baseline pricing."""
    total_baseline = results["baseline_revenue"].sum()
    total_optimized = results["expected_revenue"].sum()
    return float(((total_optimized - total_baseline) / total_baseline) * 100)
