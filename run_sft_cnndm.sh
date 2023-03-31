gpu=4,5,6,7

for n_train in 1000 2000 4000
do
# training
CUDA_VISIBLE_DEVICES=$gpu python -m sft4lms.Summarization.run_sft_t5 --task extraction \
                                                                        --n_train $n_train \
                                                                        --n_val 13368 \
                                                                        --n_test 500 \
                                                                        --load_strategy load_initial \
                                                                        --extraction_mode textrank \
                                                                        --extraction_source all \
                                                                        --model flan-t5-large \
                                                                        --train_batch_size 8 \
                                                                        --eval_batch_size 24 \
                                                                        --learning_rate 2e-5 \
                                                                        --epochs 10 \
                                                                        --logging_steps 200 \
                                                                        --save_total_limit 3 \
                                                                        --early_stopping_patience 3 \
                                                                        --do_inference \
                                                                        --do_train \
                                                                        --push_to_hub
done