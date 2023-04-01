import os
import numpy as np
import pandas as pd
import json

from tqdm import trange
from tqdm import tqdm

import datasets
from transformers import AutoTokenizer
import random
from .eval import MultiWozEvaluator
import copy

"""
How to construct the multiwoz data
"""
DA_PREFIX = "translate dialogue to dialogue action: "
BS_PREFIX = "translate dialogue to belief state: "
NLG_PREFIX = "translate dialogue to system response: "
SPLIT = "; "

def dictlist2df(dict_list):
    dl = {}
    for d in dict_list:
        for k, v in d.items():
            if k in dl:
                dl[k].append(v)
            else:
                dl[k] = [v]
    df = pd.DataFrame.from_dict(dl)
    return df


def dictlist2dict(dict_list):
    dl = {}
    for d in dict_list:
        for k, v in d.items():
            if k in dl:
                dl[k].append(v)
            else:
                dl[k] = [v]
    return dl


def get_multiwoz_data(data, n_dialogs):

    # Start
    processed_data = []
    processed_dialogs = 0
    for k in data:
        turns = data[k]
        history_users, history_resps, history_acts = [], [], []
        for turn_id, turn in enumerate(turns):
            user = turn["user"]
            resp = turn["resp"]
            aspn = turn["aspn"]
            da_input = turn["da_input"]
            da_output = turn["da_output"]
            bs_input = turn["bs_input"]
            bs_output = turn["bs_output"]
            nlg_input = turn["nlg_input"]
            nlg_output = turn["nlg_output"]

            # for multiwoz evaluation
            eval_turn = copy.deepcopy(turn)
            eval_turn["bspn_gen"] = bs_output
            eval_turn["aspn_gen"] = da_output
            eval_turn["resp_gen"] = nlg_output # need to fill

            # save data
            processed_data.append({
                                    "turn_id": turn_id,
                                    "user": user, "resp": resp, "aspn": aspn,
                                    "da_input": da_input, "da_output": da_output,
                                    "bs_input": bs_input, "bs_output": bs_output,
                                    "nlg_input": nlg_input, "nlg_output": nlg_output,
                                    "history_users": history_users, "history_resps": history_resps, "history_acts": history_acts, # history
                                    "eval_turn": eval_turn
                                    })

            history_users.append(user)
            history_resps.append(resp)
            history_acts.append(aspn)

        processed_dialogs += 1
        if processed_dialogs >= n_dialogs:
            break
        
    return processed_data


def get_data_split(dataset, dataset_version, n_train, n_test, n_val, return_dict=True):

    if dataset_version == "2.0":
        rawdata_path = "multi-woz-2.0-rawdata"
    elif dataset_version == "2.1":
        rawdata_path = "multi-woz-2.1-rawdata"
    elif dataset_version == "2.3":
        rawdata_path = "multi-woz-2.3-rawdata"
    else:
        raise NotImplementedError 

    if dataset == 'multiwoz':
        with open(f"./sft4lms/data/multiwoz/data/{rawdata_path}/train_raw_dials.json", 'r') as file:
            train_data = json.load(file)
            train_data = get_multiwoz_data(train_data, n_train)

        with open(f"./sft4lms/data/multiwoz/data/{rawdata_path}/dev_raw_dials.json", 'r') as file:
            val_data = json.load(file)
            val_data = get_multiwoz_data(val_data, n_val)
        
        with open(f"./sft4lms/data/multiwoz/data/{rawdata_path}/test_raw_dials.json", 'r') as file:
            test_data = json.load(file)
            test_data = get_multiwoz_data(test_data, n_test)
    else:
        raise NotImplementedError

    if return_dict:
        return dictlist2dict(train_data), dictlist2dict(val_data), dictlist2dict(test_data)
    else:
        return train_data, val_data, test_data


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    for dataset in ['multiwoz']:
        for dataset_version in ["2.1"]:
            for n_train in [10000]:
                for n_test in [1000]:
                    train_data, val_data, test_data = get_data_split(dataset=dataset, dataset_version=dataset_version, n_train=n_train, n_val=1000, n_test=n_test)
                
                    from datasets import Dataset
                    train_dataset = Dataset.from_dict(train_data)
                    val_dataset = Dataset.from_dict(val_data)
                    test_dataset = Dataset.from_dict(test_data)
                    print(len(train_dataset))
                    print(len(val_dataset))
                    print(len(test_dataset))
                    
                    # eval_turns = []
                    # evaluator = MultiWozEvaluator(dataset_version=dataset_version, tokenizer_path="google/flan-t5-large")
                    # for i in range(len(test_dataset)):
                    #     eval_turn = test_dataset[i]["eval_turn"]
                    #     eval_turns.append(eval_turn)

                    # dev_bleu, dev_success, dev_match, total_successes, total_matches, dial_nums = evaluator.validation_metric(eval_turns)
                    # dev_score = 0.5 * (dev_success + dev_match) + dev_bleu
                    # print(dev_bleu, dev_success, dev_match)
                    # print(dev_score)
                    # print(total_successes, total_matches, dial_nums)

                    # input_lens = []
                    # output_lens = []
                    # resp_lens = []
                    # num = 0
                    # tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
                    # for i in range(len(train_dataset)):
                    #     da_input, da_output, nlg_output = train_dataset[i]["da_input"], train_dataset[i]["da_output"], train_dataset[i]["nlg_output"]
                    #     input_len = len(tokenizer(da_input)["input_ids"])
                    #     output_len = len(tokenizer(da_output)["input_ids"])
                    #     resp_len = len(tokenizer(nlg_output)["input_ids"])
                            
                    #     input_lens.append(input_len)
                    #     output_lens.append(output_len)
                    #     resp_lens.append(resp_len)

                    # input_lens = np.array(input_lens)
                    # output_lens = np.array(output_lens)
                    # resp_lens = np.array(resp_lens)
                    
                    # len_min = np.min(input_lens)
                    # len_mean = np.mean(input_lens)
                    # len_median = np.median(input_lens)
                    # len_max = np.max(input_lens)
                    # print("mean of input len:{}, median of input len:{}, max of input len:{}, min of input len {}".format(len_mean, len_median, len_max, len_min))

                    # len_min = np.min(output_lens)
                    # len_mean = np.mean(output_lens)
                    # len_median = np.median(output_lens)
                    # len_max = np.max(output_lens)
                    # print("mean of output len:{}, median of output len:{}, max of output len:{}, min of output len {}".format(len_mean, len_median, len_max, len_min))

                    # len_min = np.min(resp_lens)
                    # len_mean = np.mean(resp_lens)
                    # len_median = np.median(resp_lens)
                    # len_max = np.max(resp_lens)
                    # print("mean of resp len:{}, median of resp len:{}, max of resp len:{}, min of resp len {}".format(len_mean, len_median, len_max, len_min))

                    # evaluator = MultiWozEvaluator()
                    # for i in range(100):
                    #     eval_turn = val_dataset[i]["eval_turn"]
                    #     dev_bleu, dev_success, dev_match = evaluator.validation_metric([eval_turn])
                    #     dev_score = 0.5 * (dev_success + dev_match) + dev_bleu
                    #     print(dev_bleu, dev_success, dev_match)
                    #     print(dev_score)

                    # # DEBUG
                    for i in range(len(test_dataset)):
                        # print("dialog:")
                        # print(test_dataset[i]['prev_context'])
                        # print("da_input:")
                        # print(test_dataset[i]['da_input'])
                        # print("da_output:")
                        # print(test_dataset[i]['da_output'])
                        if test_dataset[i]['resp'] == 'there are [value_choice] [value_food] restaurant -s in [value_area] what price range do you want ?':
                            print(test_dataset[i])






        
        


