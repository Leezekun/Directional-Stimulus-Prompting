gpu=0,1,2

for n_train in 80 800
do
for dataset_version in 2.0 2.1
do
    # training
    CUDA_VISIBLE_DEVICES=$gpu python -m sft4lms.MultiWOZ.run_sft_t5 \
                                                --n_train $n_train \
                                                --n_val 1000 \
                                                --n_test 1000 \
                                                --dataset multiwoz \
                                                --dataset_version $dataset_version \
                                                --task da \
                                                --load_strategy load_initial \
                                                --model flan-t5-large \
                                                --train_batch_size 8 \
                                                --eval_batch_size 24 \
                                                --learning_rate 2e-5 \
                                                --epochs 25 \
                                                --logging_steps 200 \
                                                --save_total_limit 3 \
                                                --early_stopping_patience 5 \
                                                --do_inference \
                                                --do_train \
                                                # --push_to_hub                                           
done
done