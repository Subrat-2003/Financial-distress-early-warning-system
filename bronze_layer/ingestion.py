import os
import polars as pl

def discover_raw_data(base_path):
    """Scans for raw SEC submission folders ending in '_notes'."""
    folders = [f for f in os.listdir(base_path) if f.endswith('_notes')]
    years_found = sorted(list(set([f[:4] for f in folders])))
    print(f" Found data for years: {years_found}")
    return folders, years_found

if __name__ == "__main__":
    # Path should point to where your raw .tsv folders live
    BASE_PATH = "/content/drive/My Drive/sec_data"
    raw_folders, years = discover_raw_data(BASE_PATH)