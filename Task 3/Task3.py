#!/usr/bin/env python3
"""
Task3.py - Car price prediction script (robust target detection)

Usage examples:
  python Task3.py -i "car data.csv" -o results    # will auto-detect target from common names
"""

import argparse
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

COMMON_TARGET_ALIASES = [
    "price", "Price", "selling_price", "Selling_Price", "selling price",
    "Selling Price", "target", "y", "label"
]

def find_target_column(df, requested):
    # If user explicitly requested a column and it exists, use it
    if requested:
        # allow case-insensitive match and also match underscores vs spaces
        candidates = {c.lower().replace(" ", "_"): c for c in df.columns}
        key = requested.lower().replace(" ", "_")
        if key in candidates:
            return candidates[key]
        # direct exact match as fallback
        if requested in df.columns:
            return requested
    # Try common aliases
    candidates = {c.lower().replace(" ", "_"): c for c in df.columns}
    for alias in COMMON_TARGET_ALIASES:
        k = alias.lower().replace(" ", "_")
        if k in candidates:
            return candidates[k]
    return None

def basic_preprocess(df, target_col):
    # simple preprocessing: drop rows with no target, basic dummies for categoricals
    df = df.copy()
    df = df.dropna(subset=[target_col])
    # drop obviously non-feature columns if present
    drop_cols = []
    # preserve numeric features and create dummies for categoricals
    for c in df.columns:
        if c == target_col:
            continue
        if df[c].dtype == 'O' or df[c].dtype.name == 'category':
            # if textual but likely identifier (Car_Name), drop it
            if c.lower() in ("car_name", "name", "id"):
                drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=drop_cols)
    # One-hot encode categorical features
    df = pd.get_dummies(df, drop_first=True)
    return df

def main():
    parser = argparse.ArgumentParser(description="Task3: Car price prediction")
    parser.add_argument("-i", "--input", required=True, help="Input CSV file")
    parser.add_argument("-o", "--outdir", required=True, help="Output directory")
    parser.add_argument("--target", default=None, help="Target column name (optional)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print("Loading data...", args.input)
    df = pd.read_csv(args.input)
    print("Rows/cols:", df.shape)

    # find a target column
    target_col = find_target_column(df, args.target)
    if target_col is None:
        print("\nCould not find a target column. Available columns are:\n")
        for c in df.columns:
            print("  -", c)
        print("\nPlease specify --target using one of the above names (case-insensitive).")
        sys.exit(2)

    print("Using target column:", target_col)

    # prepare data
    df_proc = basic_preprocess(df, target_col)
    if target_col not in df_proc.columns:
        # If the preprocessing dropped the original name (e.g., it had spaces),
        # try to locate the numeric target column by similarity
        # but normally target should still be present
        possible = [c for c in df_proc.columns if c.lower().startswith(target_col.lower().replace(" ", "_"))]
        if possible:
            target_col = possible[0]
            print("Adjusted target column after preprocessing to:", target_col)
        else:
            print("Error: target column missing after preprocessing. Columns now:", df_proc.columns.tolist(), file=sys.stderr)
            sys.exit(3)

    # split X/y
    X = df_proc.drop(columns=[target_col])
    y = df_proc[target_col]

    # simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)
    print("Train/test sizes:", X_train.shape[0], X_test.shape[0])

    # simple model (RandomForest)
    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=args.random_state)
    model.fit(X_train, y_train)

    # evaluate
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Test MSE: {mse:.4f}, R2: {r2:.4f}")

    # save outputs
    os.makedirs(args.outdir, exist_ok=True)
    model_path = os.path.join(args.outdir, "rf_model.joblib")
    joblib.dump(model, model_path)
    print("Saved model to:", model_path)

    # save metrics
    metrics_path = os.path.join(args.outdir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"mse: {mse}\nr2: {r2}\n")
    print("Saved metrics to:", metrics_path)

if __name__ == "__main__":
    main()
