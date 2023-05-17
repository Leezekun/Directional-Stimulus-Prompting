import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import numpy as np
from sklearn.metrics import *
import nltk

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import EarlyStoppingCallback
from datasets import load_dataset, load_metric, Dataset

from sft4lms.Summarization.data_loader import get_data_split, EXTRACTION_PREFIX, SUMMARIZATION_PREFIX, SPLIT
from rl4lms.envs.text_generation.gpt3_utils import GPT3, avoid_keywords


def load_model(output_dir, model_path, strategy):
    if output_dir and os.path.exists(output_dir):
        if "checkpoint" in output_dir:
            model = AutoModelForSeq2SeqLM.from_pretrained(output_dir)
            tokenizer = AutoTokenizer.from_pretrained(output_dir)
        else:
            if strategy == 'load_last':
                latest_checkpoint_idx = 0
                dir_list = os.listdir(output_dir) # find the latest checkpoint
                for d in dir_list:
                    if "checkpoint" in d and "best" not in d:
                        checkpoint_idx = int(d.split("-")[-1])
                        if checkpoint_idx > latest_checkpoint_idx:
                            latest_checkpoint_idx = checkpoint_idx
                if latest_checkpoint_idx > 0 and os.path.exists(os.path.join(output_dir, f"checkpoint-{latest_checkpoint_idx}")):
                    ft_model_path = os.path.join(output_dir, f"checkpoint-{latest_checkpoint_idx}")
                    model = AutoModelForSeq2SeqLM.from_pretrained(ft_model_path)
                    tokenizer = AutoTokenizer.from_pretrained(ft_model_path)
                    return model, tokenizer
            elif strategy == 'load_best':
                ft_model_path = os.path.join(output_dir, f"best_checkpoint")
                if os.path.exists(ft_model_path):
                    model = AutoModelForSeq2SeqLM.from_pretrained(ft_model_path)
                    tokenizer = AutoTokenizer.from_pretrained(ft_model_path)
                    return model, tokenizer
            elif strategy == 'load_initial':
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                return model, tokenizer

    # load pretrained model for hf
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def fine_tune_hf(
    task,
    model_name,
    dataset_name,
    n_train,
    push_to_hub,
    model,
    tokenizer,
    extraction_source,
    max_ctx_len,
    max_tgt_len,
    output_dir,
    train_data,
    val_data,
    test_data,
    epochs,
    train_batch_size,
    eval_batch_size,
    logging_steps,
    save_total_limit,
    early_stopping_patience,
    learning_rate,
    seed,
    do_train,
    do_inference
):  
    
    def preprocess_function_for_summarization(batch):
        inputs = [SUMMARIZATION_PREFIX + doc for doc in batch["article"]]
        targets = [doc for doc in batch["summary"]]
        model_inputs = tokenizer(inputs, max_length=max_ctx_len, truncation=True)
        labels = tokenizer(targets, max_length=max_tgt_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_function_for_extraction(batch):
        inputs = [EXTRACTION_PREFIX + doc for doc in batch["article"]]
        targets = [doc for doc in batch["target"]]
        model_inputs = tokenizer(inputs, max_length=max_ctx_len, truncation=True)
        labels = tokenizer(targets, max_length=max_tgt_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def convert_label_to_key(labels):
        return " ".join([str(i) for i in labels])
    
    def dataset_summary_mapping(dataset):
        mapping_dict = {}
        for d in dataset:
            label = d['labels']
            label_key = convert_label_to_key(label)
            mapping_dict[label_key] = d['summary']
        return mapping_dict

    # training the model with Huggingface 🤗 trainer
    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)
    test_dataset = Dataset.from_dict(test_data)
    train_dataset = train_dataset.map(remove_columns=["phrases"])
    val_dataset = val_dataset.map(remove_columns=["phrases"])
    test_dataset = test_dataset.map(remove_columns=["phrases"])
    # tokenize the dataset
    if task == 'summarization':
        train_dataset = train_dataset.map(preprocess_function_for_summarization, batched=True, remove_columns=["target", "article", "summary"])
        val_dataset = val_dataset.map(preprocess_function_for_summarization, batched=True, remove_columns=["target", "article"])
        test_dataset = test_dataset.map(preprocess_function_for_summarization, batched=True, remove_columns=["target", "article"])
    elif task == 'extraction':
        train_dataset = train_dataset.map(preprocess_function_for_extraction, batched=True, remove_columns=["target", "article", "summary"])
        val_dataset = val_dataset.map(preprocess_function_for_extraction, batched=True, remove_columns=["target", "article"])
        test_dataset = test_dataset.map(preprocess_function_for_extraction, batched=True, remove_columns=["target", "article"])
    
    # mapping from label to summaries
    val_summary_mapping = dataset_summary_mapping(val_dataset)
    test_summary_mapping = dataset_summary_mapping(test_dataset)
    val_test_summary_mapping = {**val_summary_mapping, **test_summary_mapping}
    # remove unused columns summary
    val_dataset = val_dataset.map(remove_columns=["summary"])
    test_dataset = test_dataset.map(remove_columns=["summary"])

    # customized metrics
    metric = load_metric("rouge")
    def compute_rouge_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}
    
    def compute_hit_metrics(eval_pred):

        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        hint_precisions, hint_hit_nums, hint_nums = [], [], []
        assert len(labels) == len(decoded_preds)
        for i  in range(len(decoded_preds)):
            pred = decoded_preds[i].strip().lower()
            pred = pred[:-1] if pred[-1] == "." else pred
            pred = pred.split(SPLIT.strip())
            # remove the repeated words
            pred = sorted(pred, key=lambda x: len(x), reverse=True)
            # label -> summary
            label = labels[i]
            label = label[label!=-100]
            label_key = convert_label_to_key(label)
            label = val_test_summary_mapping[label_key].lower()

            hit_pred = []
            for p in pred:
                p = p.strip()
                if p not in " ".join(hit_pred) and p in label and p not in avoid_keywords:
                    hit_pred.append(p)

            # calculate hit score and precision
            n = len(pred)
            hit_num = len(hit_pred)
            hit_precision = hit_num / n if n > 0 else 0

            # store results    
            hint_precisions.append(hit_precision)
            hint_hit_nums.append(hit_num)
            hint_nums.append(n)

        # Extract a few results
        result = {"hint_hit_num": np.mean(hint_hit_nums), "hint_precision": np.mean(hint_precisions), "num": np.mean(hint_nums)}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}
    
    # tokenize the dataset
    if task == 'summarization':
        compute_metrics = compute_rouge_metrics
        best_metric = "rouge1"
    elif task == 'extraction':
        compute_metrics = compute_hit_metrics
        best_metric = "loss"

    # arguments
    training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir, # output directory
    # output_dir=output_dir if not push_to_hub else hf_path, # output directory
    num_train_epochs=epochs, # total number of training epochs
    per_device_train_batch_size=train_batch_size,  # batch size per device during training
    per_device_eval_batch_size=eval_batch_size,   # batch size for evaluation
    evaluation_strategy='steps',
    learning_rate=learning_rate, # 2e-5
    weight_decay=0.01,
    # fp16=True,
    logging_steps=logging_steps, # the same as eval_step
    eval_steps=logging_steps,
    save_steps=logging_steps, # doesn't work if load_best_model_at_end=True, will save every eval_steps (logging_steps)
    logging_dir=os.path.join(output_dir, "runs/"),
    save_total_limit=save_total_limit,
    seed=seed,
    push_to_hub=push_to_hub,
    predict_with_generate=True, # for evaluation metrics
    # load_best_model_at_end=True,
    # metric_for_best_model=best_metric,
    remove_unused_columns=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,model=model)

    trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    # callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]    
    )
    
    if do_train:
        train_results = trainer.train()
        print(train_results)
        eval_results = trainer.evaluate()
        print(eval_results)
        # save model locally and push it to the hub
        trainer.save_model(os.path.join(output_dir, "best_checkpoint/"))
        print(f'Save best model in {os.path.join(output_dir, "best_checkpoint/")}')
        if push_to_hub:
            trainer.push_to_hub()
            
    # inference on the test set
    if do_inference:
        test_results = trainer.predict(test_dataset)
        print(test_results)
    
    return model


def main():

    parser = argparse.ArgumentParser()
    # arguments for dataset
    parser.add_argument('--dataset', type=str, default='cnndm', choices=['cnndm']) #

    parser.add_argument('--n_train', type=int, default=2000) #
    parser.add_argument('--n_val', type=int, default=500) #
    parser.add_argument('--n_test', type=int, default=500) #
    parser.add_argument('--extraction_mode', type=str, default='textrank', choices=['textrank', 'patternrank', 'keybert', 'yake', 'prompt'])
    parser.add_argument('--extraction_source', type=str, default='all', choices=['all', 'article', 'summary'])
    parser.add_argument('--max_ctx_len', type=int, default=512)
    parser.add_argument('--max_tgt_len', type=int, default=64)

    # arguments for huggingface training
    parser.add_argument('--model', type=str, default='t5-base') #
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--load_strategy', type=str, default='load_initial', choices=['load_initial', 'load_best', 'load_last']) #

    parser.add_argument('--seed', type=int, default=1799)
    parser.add_argument('--save_total_limit', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--logging_steps', type=int, default=500)
    parser.add_argument('--save_steps', type=int, default=2000)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--early_stopping_patience', type=int, default=5)

    # training task
    parser.add_argument('--task', type=str, default='extraction', choices=['extraction', 'summarization'])

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_inference', action='store_true')
    parser.add_argument('--push_to_hub', action='store_true')
    
    args, unknown = parser.parse_known_args()

    dataset = args.dataset
    task = args.task
    load_strategy = args.load_strategy
    n_train, n_val, n_test = args.n_train, args.n_val, args.n_test
    extraction_mode = args.extraction_mode
    extraction_source = args.extraction_source
    max_ctx_len = args.max_ctx_len
    max_tgt_len = args.max_tgt_len
    assert max_ctx_len <= 1024 # for T5

    if args.model in ['t5-base', 't5-large', 't5-3b']:
        model_path = args.model
    elif args.model in ['flan-t5-large', 'flan-t5-base', 'flan-t5-small']:
        model_path = f"google/{args.model}"

    """prepare for training""" 
    if args.output_dir is None:
        if task == 'summarization':
            output_dir = f"./sft4lms/ckpt/{dataset}_{n_train}/summarization/{args.model}/"
        elif task == 'extraction':
            output_dir = f"./sft4lms/ckpt/{dataset}_{n_train}/{extraction_mode}-{extraction_source}/{args.model}-ep{args.epochs}"
    else:
        output_dir = args.output_dir

    # LOAD DATA
    Ptrain, Pval, Ptest = get_data_split(dataset, n_train, n_val, n_test, extraction_mode, extraction_source)
    # LOAD MODEL
    model, tokenizer = load_model(output_dir, model_path, load_strategy)

    fine_tune_hf(
    task=args.task,
    model_name=args.model,
    dataset_name=args.dataset,
    n_train=n_train,
    push_to_hub=args.push_to_hub,
    model=model,
    tokenizer=tokenizer,
    extraction_source=extraction_source,
    max_ctx_len=max_ctx_len,
    max_tgt_len=max_tgt_len,
    output_dir=output_dir,
    train_data=Ptrain,
    val_data=Pval,
    test_data=Ptest,
    epochs=args.epochs,
    train_batch_size=args.train_batch_size,
    eval_batch_size=args.eval_batch_size,
    logging_steps=args.logging_steps,
    save_total_limit=args.save_total_limit,
    early_stopping_patience=args.early_stopping_patience,
    learning_rate=args.learning_rate,
    seed=args.seed,
    do_train=args.do_train,
    do_inference=args.do_inference
    )


if __name__ == "__main__":
    main()

