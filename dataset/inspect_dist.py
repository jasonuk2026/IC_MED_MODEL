import pandas as pd
from build_task_data import TASK_DESCRIPTIONS

tasks = [k for k, v in TASK_DESCRIPTIONS.items() if 'new_' in k]
# tasks = ["new_pancan"]

for task in tasks:
    f = f"EHRSHOT_ASSETS/llm_data_v6/{task}/train.parquet"
    df = pd.read_parquet(f)

    num_pos = len(df[df["label"] == "True"])
    num_neg = len(df[df["label"] == "False"])

    print(f"We have {num_pos} pos samples, {num_neg} neg samples.")