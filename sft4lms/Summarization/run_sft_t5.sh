gpu=4,5

# training
CUDA_VISIBLE_DEVICES=$gpu python run_sft_t5.py --task extraction \
                                                --n_train 4000 \
                                                --n_val 500 \
                                                --n_test 500 \
                                                --extraction_mode textrank \
                                                --extraction_source all \
                                                --model flan-t5-large \
                                                --train_batch_size 8 \
                                                --eval_batch_size 24 \
                                                --learning_rate 2e-5 \
                                                --epochs 20 \
                                                --logging_steps 200 \
                                                --save_total_limit 3 \
                                                --early_stopping_patience 5 \
                                                --do_train \
                                                --do_inference \
                                                --push_to_hub