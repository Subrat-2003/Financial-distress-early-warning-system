import os
import polars as pl

def process_single_folder(folder_path, tags):
    num_path = os.path.join(folder_path, 'num.tsv')
    sub_path = os.path.join(folder_path, 'sub.tsv')
    if not os.path.exists(num_path) or not os.path.exists(sub_path): return None

    try:
        # Filter and pivot raw numeric data
        num_lazy = pl.scan_csv(num_path, separator='\t', ignore_errors=True) \
            .filter(pl.col("tag").is_in(tags)) \
            .filter(pl.col("qtrs").is_in([0, 1, 4])) \
            .select([pl.col("adsh"), pl.col("tag"), pl.col("value").cast(pl.Float32), pl.col("ddate")])

        df_pivoted = num_lazy.collect(streaming=True).pivot(
            values="value", index=["adsh", "ddate"], on="tag", aggregate_function="first"
        )

        # Ensure all required tags exist as columns
        for tag in tags:
            if tag not in df_pivoted.columns:
                df_pivoted = df_pivoted.with_columns(pl.lit(None).cast(pl.Float32).alias(tag))

        # Calculate initial safety ratios
        df_pivoted = df_pivoted.lazy().with_columns([
            (pl.col("AssetsCurrent") / pl.col("LiabilitiesCurrent")).alias("current_ratio"),
            ((pl.col("AssetsCurrent") - pl.col("Inventory")) / pl.col("LiabilitiesCurrent")).alias("quick_ratio"),
            (pl.col("NetIncomeLoss") / pl.col("Assets")).alias("roa"),
            (pl.col("Liabilities") / pl.col("Assets")).alias("debt_to_assets")
        ]).collect()

        sub_df = pl.read_csv(sub_path, separator='\t', ignore_errors=True) \
            .select([pl.col("adsh"), pl.col("cik").cast(pl.Int32), pl.col("name")])

        return df_pivoted.join(sub_df, on="adsh", how="left")
    except Exception as e:
        print(f" Error processing {folder_path}: {e}")
        return None