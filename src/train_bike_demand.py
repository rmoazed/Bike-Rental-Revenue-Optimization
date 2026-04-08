from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from preprocessing import prepare_training_data
from pricing import optimize_prices_for_predictions, total_percent_revenue_increase


RANDOM_STATE = 42


def train_model(csv_path: str) -> tuple[RandomForestRegressor, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X, y = prepare_training_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test


def evaluate_model(
    model: RandomForestRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_rmse = float(np.sqrt(mean_squared_error(y_train, train_preds)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, np.full(len(y_test), y_train.mean()))))

    return {
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "baseline_rmse": baseline_rmse,
    }


def save_artifacts(model: RandomForestRegressor, feature_columns: list[str], output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path / "bike_demand_model.joblib")
    joblib.dump(feature_columns, output_path / "feature_columns.joblib")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the bike demand model and save artifacts.")
    parser.add_argument("csv_path", help="Path to the raw bike-sharing CSV file")
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory where the trained model and feature metadata should be saved",
    )
    args = parser.parse_args()

    model, X_train, y_train, X_test, y_test = train_model(args.csv_path)
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

    print(f"Train RMSE: {metrics['train_rmse']:.2f}")
    print(f"Test RMSE: {metrics['test_rmse']:.2f}")
    print(f"Baseline RMSE: {metrics['baseline_rmse']:.2f}")

    test_preds = model.predict(X_test)
    pricing_results = optimize_prices_for_predictions(test_preds)
    pct_gain = total_percent_revenue_increase(pricing_results)
    print(f"Pricing revenue lift (%): {pct_gain:.2f}")

    save_artifacts(model, list(X_train.columns), args.output_dir)
    print(f"Saved model artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()
