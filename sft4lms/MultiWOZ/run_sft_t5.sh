gpu=0

# training
CUDA_VISIBLE_DEVICES=$gpu python -m run_sft_t5 --n_train 250 \
                                            --n_val 500 \
                                            --n_test 500 \
                                            --dataset multiwoz \
                                            --task da \
                                            --model flan-t5-large \
                                            --train_batch_size 8 \
                                            --eval_batch_size 24 \
                                            --learning_rate 2e-5 \
                                            --epochs 5 \
                                            --logging_steps 200 \
                                            --save_total_limit 3 \
                                            --early_stopping_patience 3 \
                                            --do_train \
                                            --do_inference \
                                            --push_to_hub

# evaluation
# CUDA_VISIBLE_DEVICES=$gpu python run_sft_t5.py --fs_ratio 0.2 \
#                                             --extraction_mode textrank \
#                                             --model t5-base \
#                                             --eval_batch_size 32 \
#                                             --do_inference \
#                                             --output_dir ../ckpt/cnndm_fs0.1/textrank/t5-base/checkpoint-3600/