import pandas as pd
df = pd.read_parquet("./data/embedding_inputs/sharded_m500_ntrain50000_nval2000_ntest50000/val.parquet")
print(df["split"].unique())