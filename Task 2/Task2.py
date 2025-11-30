#!/usr/bin/env python3
"""
Task2.py — single-file unemployment analysis (clean version)

Automatically:
- Detects/Parses date column
- Creates Year/Month
- Detects & cleans unemployment rate
- Builds monthly timeseries
- Saves: preprocessed CSV, monthly CSV, monthly trend plot
- Silently skips decomposition & boxplot if dataset is too short
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
except ImportError:
    seasonal_decompose = None   # fallback if not installed

# --------- Helpers ---------

def clean_rate_series(s):
    """Remove % signs, commas, whitespace and convert to numeric."""
    s = s.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")

def detect_date_column(df, provided=None):
    if provided and provided in df.columns:
        return provided

    candidates = ["Date", "date", "DATE", "Period", "Month"]
    for c in candidates:
        if c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], dayfirst=True, errors="coerce")
                if parsed.notna().sum() > len(df) / 4:
                    return c
            except:
                pass

    # Auto-detect fallback
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        parsed = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
        if parsed.notna().sum() > len(df) / 2:
            return col

    raise ValueError("No usable date column found. Use --date-col DATE.")

def detect_rate_column(df, provided=None):
    if provided and provided in df.columns:
        return provided

    lowered = [c.lower() for c in df.columns]
    for key in ["unemployment", "rate", "estimated unemployment"]:
        for c, lc in zip(df.columns, lowered):
            if key in lc:
                return c

    # fallback: numeric-like column
    for c in df.columns:
        try:
            pd.to_numeric(df[c], errors="ignore")
            return c
        except:
            pass

    raise ValueError("Could not detect rate column. Use --rate-col.")

def prepare_dataframe(path, date_col=None, year_col=None, month_col=None, rate_col=None):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # If Year/Month already exist
    if year_col in df.columns and month_col in df.columns:
        if rate_col and rate_col in df.columns:
            df[rate_col + "_clean"] = clean_rate_series(df[rate_col])
        return df

    # Detect date column
    dcol = detect_date_column(df, date_col)
    df[dcol] = pd.to_datetime(df[dcol], dayfirst=True, errors="coerce")
    df["Year"] = df[dcol].dt.year
    df["Month"] = df[dcol].dt.month

    # Detect rate column
    rcol = detect_rate_column(df, rate_col)
    df[rcol + "_clean"] = clean_rate_series(df[rcol])

    return df

def monthly_timeseries(df, rate_clean_col):
    temp = df.dropna(subset=["Year", "Month", rate_clean_col]).copy()
    temp["day"] = 1
    temp["dt"] = pd.to_datetime(temp[["Year", "Month", "day"]])
    temp = temp.set_index("dt")
    return temp[rate_clean_col].resample("MS").mean()

def plot_monthly(series, outdir):
    plt.figure(figsize=(10,4))
    series.plot(title="Monthly Unemployment Rate (aggregated)")
    plt.xlabel("Date")
    plt.ylabel("Unemployment rate")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "monthly_timeseries.png"))
    plt.close()

# --------- Main ---------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task2 unemployment analysis")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--outdir", required=True)
    parser.add_argument("--date-col", help="Date column name")
    parser.add_argument("--rate-col", help="Unemployment rate column name")
    parser.add_argument("--freq", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # FIXED TYPO — using args.date_col instead of args.date
    df = prepare_dataframe(args.input, args.date_col, None, None, args.rate_col)

    # locate cleaned rate column
    rate_clean_cols = [c for c in df.columns if c.endswith("_clean")]
    rate_clean = rate_clean_cols[0]

    # Save preprocessed CSV
    preproc_path = os.path.join(args.outdir, "preprocessed_input.csv")
    df.to_csv(preproc_path, index=False)
    print("Saved preprocessed CSV to:", preproc_path)

    # Monthly TS
    monthly = monthly_timeseries(df, rate_clean)
    monthly_csv = os.path.join(args.outdir, "monthly_timeseries.csv")
    monthly.to_frame("rate").to_csv(monthly_csv)
    print("Saved monthly timeseries CSV to:", monthly_csv)

    # Plot monthly
    plot_monthly(monthly, args.outdir)

    print("Done.")
