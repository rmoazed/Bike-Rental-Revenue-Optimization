from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from preprocessing import align_feature_columns, build_features
from pricing import find_best_price


def predict_one_row(input_csv: str, model_dir: str = "models") -> dict[str, float]:
    model_path = Path(model_dir) / "bike_demand_model.joblib"
    columns_path = Path(model_dir) / "feature_columns.joblib"

    model = joblib.load(model_path)
    feature_columns = joblib.load(columns_path)

    df = pd.read_csv(input_csv)
    X = build_features(df)
    X = align_feature_columns(X, feature_columns)

    pred = float(model.predict(X)[0])
    return find_best_price(pred)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for one or more rows and return pricing output.")
    parser.add_argument("input_csv", help="Path to CSV containing raw input row(s)")
    parser.add_argument("--model-dir", default="models", help="Directory containing saved model artifacts")
    args = parser.parse_args()

    result = predict_one_row(args.input_csv, args.model_dir)
    print(result)


if __name__ == "__main__":
    main()
