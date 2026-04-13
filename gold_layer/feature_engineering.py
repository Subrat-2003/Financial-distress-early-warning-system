import os
import polars as pl
import pandas as pd
from sklearn.preprocessing import StandardScaler

def apply_feature_engineering(df, features_to_scale):
    # 1. Revenue Growth Rate
    df = df.sort(["cik", "ddate"]).unique(subset=["cik", "ddate"])
    df = df.with_columns(
        ((pl.col("Revenues") / pl.col("Revenues").shift(1).over("cik")) - 1).fill_null(0).alias("revenue_growth_rate")
    )

    # 2. Persistent Distress Flag (Interest Coverage < 1.5)
    df = df.with_columns((pl.col("interest_coverage") < 1.5).cast(pl.Int32).alias("is_distressed"))
    df = df.with_columns(
        pl.col("is_distressed").rolling_sum(window_size=2).over("cik").fill_null(0).alias("persistent_distress_flag")
    )

    # 3. Clean Infinite values and Forward Fill
    all_features = features_to_scale + ["revenue_growth_rate", "persistent_distress_flag"]
    df = df.with_columns([
        pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c) for c in all_features
    ])
    
    # Apply forward fill per CIK to maintain time-series integrity
    df = df.group_by("cik").map_groups(lambda group: group.fill_null(strategy="forward")).fill_null(0)

    # 4. Final Standard Scaling
    pdf = df.to_pandas()
    scaler = StandardScaler()
    pdf[all_features] = scaler.fit_transform(pdf[all_features].astype('float64'))
    
    return pl.from_pandas(pdf)