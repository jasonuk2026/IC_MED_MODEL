#!/bin/bash
# my_list=("new_hypertension")
my_list=("new_hypertension" "new_hyperlipidemia" "new_pancan" "new_celiac" "new_lupus" "new_acutemi")

for item in "${my_list[@]}"; do
    echo "Processing: $item"
    
    # python 02_collect_train_data/build_task_data_patched.py --task "$item" --max_events 1000 --num_workers 32 --splits train --epochs 1 --output_dir data/02_outputs --embed_dir data/01_outputs/01_outputs_biolinkbert_embeddings
    python 02_collect_train_data/build_task_data.py --task "$item" --max_events 1000 --num_workers 32 --splits train --epochs 1 --output_dir data/02_baseline_sampled --embed_dir data/baseline_embs
done

# for item in "${my_list[@]}"; do
#     echo "Processing: $item"
    
#     python dataset/build_eval_task_data.py --task "$item" --num_events 1000 --num_workers 32 --splits val test 
# done