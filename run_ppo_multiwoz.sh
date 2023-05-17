gpu=0,1,2

# multiwoz with hint
CUDA_VISIBLE_DEVICES=$gpu python scripts/training/train_text_generation.py \
    --base_path_to_store_results ./rl4lms_exps \
    --project_name multiwoz_with_hint \
    --experiment_name flan-t5-large_nlpo_on_supervised_2.0_80 \
    --config_path scripts/training/task_configs/multiwoz_with_hint/flan-t5_nlpo_on_supervised_2.0_80.yml