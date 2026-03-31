import os
import json
import subprocess
import itertools
import pandas as pd
import re
import time

def generate_tuning_grid():
    # Grid Search Parameters
    tuning_params = {
        "adv_weight": [0.05, 0.1, 0.2],
        "mask_ratio": [0.15, 0.25, 0.40],
        "trend_weight": [0.1, 0.5, 0.8],
        "contrastive_weight": [0.05, 0.1, 0.5],
        "decay": [0.90, 0.95, 0.99]
    }
    
    keys = tuning_params.keys()
    values = (tuning_params[k] for k in keys)
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Generated {len(combinations)} total configurations to sweep.")
    return combinations

def update_config(base_path, temp_path, params, model_id=999):
    with open(base_path, 'r') as f:
        config = json.load(f)
        
    for key, value in params.items():
        config[key] = value
        
    with open(temp_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    return temp_path

def execute_run(model_id, data="MSL"):
    print(f"\n--- Starting Subprocess: Training Model ID {model_id} on {data} ---")
    
    command = [
        "python", "-u", "run.py",
        "--mode", "train",
        "--configs_path", "./configs/",
        "--save_path", "./test_results/",
        "--root_path", "./dataset",
        "--data", data,
        "--data_origin", "DADA",
        "--gpu", "0",
        "--id", str(model_id)
    ]
    
    start_time = time.time()
    
    result = subprocess.run(command, capture_output=True, text=True)
    execution_time = time.time() - start_time
    
    # Simple regex parsing for typical evaluation metrics in standard TSAD outputs
    f1_match = re.search(r'F1:\s*([\d.]+)', result.stdout)
    auc_match = re.search(r'AUC:\s*([\d.]+)', result.stdout)
    ap_match = re.search(r'AP:\s*([\d.]+)', result.stdout)
    
    f1 = float(f1_match.group(1)) if f1_match else None
    auc = float(auc_match.group(1)) if auc_match else None
    ap = float(ap_match.group(1)) if ap_match else None
    
    # If the metrics aren't printed directly as F1: 0.95 or similar, 
    # the user can modify the regex above to match `ts_ad_evaluation` print blocks.
    
    if result.returncode != 0:
        print(f"Run {model_id} failed!")
        print(result.stderr[:500]) # Only print first 500 characters of error
        
    return {
        "execution_time_sec": round(execution_time, 2),
        "f1": f1,
        "auc": auc,
        "ap": ap,
        "return_code": result.returncode
    }

def main():
    base_config = "./configs/MSL/model_configs_2.json"
    temp_config = "./configs/MSL/model_configs_999.json"
    tuning_log = "./test_results/tuning_log.csv"
    
    combinations = generate_tuning_grid()
    results_list = []
    
    for idx, params in enumerate(combinations):
        print(f"\n======================================")
        print(f"Sweeping Combination {idx + 1} / {len(combinations)}")
        print(f"Params: {params}")
        
        # 1. Update duplicate config
        update_config(base_config, temp_config, params, model_id=999)
        
        # 2. Execute PyTorch Training & Eval script
        run_metrics = execute_run(model_id=999, data="MSL")
        
        # 3. Aggregate
        combined_dict = {**params, **run_metrics}
        results_list.append(combined_dict)
        print(f"Results: {run_metrics}")
        
        # Save incrementally in case script crashes midway
        df = pd.DataFrame(results_list)
        df.to_csv(tuning_log, index=False)
        
    print(f"\n======================================")
    print(f"Tuning Sweep Complete! All logs saved to: {tuning_log}")
    
    # Print the very best result based on AUC or F1 if available
    try:
        best_run = df.sort_values(by="f1", ascending=False).iloc[0]
        print("\n\t*** BEST CONFIGURATION ***")
        for k, v in best_run.items():
            print(f"\t{k}: {v}")
    except:
        print("Could not sort Best F1 correctly (Regex may need adjustment based on custom stdout).")

if __name__ == "__main__":
    main()
