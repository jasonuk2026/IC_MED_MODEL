#!/bin/bash
# my_list=("new_hypertension")
my_list=("new_hypertension" "new_hyperlipidemia" "new_pancan" "new_celiac" "new_lupus" "new_acutemi")

for item in "${my_list[@]}"; do
    echo "Processing: $item"
    
    python dataset/build_task_data.py --task "$item" --max_events 1000 --num_workers 32 --splits train --epochs 1 --output_dir 02_collect_train_data_results --embed_dir 01_results
done

# for item in "${my_list[@]}"; do
#     echo "Processing: $item"
    
#     python dataset/build_eval_task_data.py --task "$item" --num_events 1000 --num_workers 32 --splits val test 
# done