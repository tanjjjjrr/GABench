import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq



def save_to_parquet(output_dir, train, test):
    # Save combined DataFrames as Parquet files
    train = train.astype(str)
    test = test.astype(str)

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    
    train_table = pa.Table.from_pandas(train)
    test_table = pa.Table.from_pandas(test)
    
    pq.write_table(train_table, train_path)
    pq.write_table(test_table, test_path)
    
    print(f"Saved training set ({len(train)} rows) to {train_path}")
    print(f"Saved test set ({len(test)} rows) to {test_path}")
    
    

def process_graph_datasets(input_dir, output_dir, train_samples=2000, test_samples=100):
    os.makedirs(output_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    train_dfs = []
    test_dfs = []
    tasks = []

    for file_path in csv_files:
        tasks.append(file_path)
        file_name = Path(file_path).stem
        print(f"Processing {file_name}...")
        df = pd.read_csv(file_path)
        
        # Sample data
        if len(df) < train_samples + test_samples:
            print(f"Warning: {file_name} has fewer rows ({len(df)}) than requested samples ({train_samples + test_samples})")
            # Split proportionally if not enough data
            test_actual = 100
            train_actual = len(df) - test_actual
            indices = np.random.choice(len(df), len(df), replace=False)
            train_indices = indices[:train_actual]
            test_indices = indices[train_actual:train_actual+test_actual]
        else:
            # Enough data, sample requested amounts
            indices = np.random.choice(len(df), train_samples + test_samples, replace=False)
            train_indices = indices[:train_samples]
            test_indices = indices[train_samples:train_samples+test_samples]
        
        train_df = df.iloc[train_indices].copy()
        test_df = df.iloc[test_indices].copy()
        
        train_dfs.append(train_df)
        test_dfs.append(test_df)


    # Combine all DataFrames
    combined_train = pd.concat(train_dfs, ignore_index=True)
    combined_test = pd.concat(test_dfs, ignore_index=True)
    
    train_directed = sum(combined_train['direction'] == 'directed')
    test_directed = sum(combined_test['direction'] == 'directed')
    train_stats = {
        "train_samples": len(combined_train),
        "directed_percentages": train_directed / len(combined_train),
    }
    test_stats = {
        "test_samples": len(combined_test),
        "directed_percentage": test_directed / len(combined_test),
    }
    print("Train stats:", train_stats)
    print("Test stats:", test_stats)
    
    train_df = combined_train.sample(frac=1).reset_index(drop=True)
    test_df = combined_test.sample(frac=1).reset_index(drop=True)

    save_to_parquet(output_dir, train_df, test_df)
    print("Total tasks: ", len(tasks))



if __name__ == "__main__":
    np.random.seed(1024)

    INPUT_DIR = "/data2/graph/graph_task_small_raw"
    OUTPUT_DIR = "/data2/graph/graph_task_small"
    
    process_graph_datasets(INPUT_DIR, OUTPUT_DIR, train_samples=2000, test_samples=100)
