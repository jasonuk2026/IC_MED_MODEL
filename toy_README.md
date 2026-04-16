# Frist
Get the EHRSHOT dataset and unzip it here.

# Second
To run embedding experiments, use `extract_embedding_data.py` to extract data parquets first. Potentially using
```
python extract_embedding_data.py --num_workers 16 --event_sample_prob 0.5 --event_sample_target 500 --train_target 50000 --val_target 2000 --test_target 5000
```

# Third
Marge and shard the previous generated parquets for different ranks to use.
```
python shard_embedding_data.py \
--input_files data/embedding_inputs/new_diagnosis/*m500_ntrain50000_nval2000_ntest5000.parquet \
--output_dir data/embedding_inputs/sharded_m500_ntrain50000_nval2000_ntest50000 \
--n_shards 16
```