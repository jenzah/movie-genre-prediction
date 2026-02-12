import polars as pl
from pathlib import Path
from typing import List, Dict


def load_all_csv_files(data_folder: str = "./data"):
    """
    Load all CSV files from a folder into a dictionary of DataFrames.
    
    Args:
        data_folder: Path to the folder containing CSV files
        
    Returns:
        Dictionary with filename as key and DataFrame as value
    """
    data_path = Path(data_folder)
    dataframes = {}
    
    if not data_path.exists():
        raise FileNotFoundError(f"Folder {data_folder} does not exist")
    
    csv_files = list(data_path.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_folder}")
        return dataframes
    
    for file_path in csv_files:
        try:
            df = pl.read_csv(file_path)
            dataframes[file_path.stem] = df
            print(f"Loaded {file_path.name}: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
    
    return dataframes

