import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import json

from tqdm import trange
from tqdm import tqdm

import spacy
import datasets
from keybert import KeyBERT
import yake
import pytextrank
from transformers import AutoTokenizer

# from sft4lms.Summarization.utils import *
from rl4lms.envs.text_generation.gpt3_utils import GPT3, avoid_keywords

EXTRACTION_PREFIX = "Extract the keywords: "
SUMMARIZATION_PREFIX = "Summarize: "
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


def filter_keywords(phrases, summary, article):

    # sort according to lengths
    phrases = sorted(phrases, key=lambda x: len(x))[::-1]

    phrase_indices = []
    selected_phrases = []
    avoid_phrases = ["one", "two", "three", "1", "2", "3", "a", "he", "she", "i", "we", "you", "it", "this", 
    "that", "the", "those", "these", "they", "me", "them", "what", "him", "her", "my", "which", "who", "why", 
    "your", "my", "his", "her", "ours", "our", "could", "with", "whom", "whose", " ", ",", ".", "?", "!", ";"]

    for p in phrases:
        
        if p.lower() in avoid_phrases: ## avoid phrases
            continue
        if p.lower() in SPLIT.join(selected_phrases).lower(): ## has already selected
            continue
        if p.lower() not in summary.lower(): ## not in summary
            continue
        if p.lower() not in article.lower(): ## not in article
            continue

        # place in the order of summary
        p_index = summary.lower().index(p.lower())

        selected_phrases.append(p)
        phrase_indices.append(p_index)

    # sort
    selected_phrases = list(zip(selected_phrases, phrase_indices))
    selected_phrases = sorted(selected_phrases, key=lambda x: x[1])
    selected_phrases = [p[0] for p in selected_phrases]
    return selected_phrases


def get_extraction_data(dataset, data, extraction_mode, extraction_source):
    
    # Init
    if extraction_mode == 'textrank':
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("textrank")
    elif extraction_mode == 'yake':
        kw_extractor = yake.KeywordExtractor()
    elif extraction_mode == 'prompt':
        gpt3 = GPT3(2.0) # sleep 5s before each call
        keyword_prompt_path = f"../prompts/{dataset}_keyword_article_fs.txt"
        f = open(keyword_prompt_path, 'r') 
        keyword_prompt = f.read().strip()
        stop_words = ["\n", "Sentences:"]

    # Start
    processed_data = []
    phrase_num = []
    for d in trange(len(data)):

        article = data[d]['article']
        summary = data[d]['highlights']
        id = data[d]['id']
        selected_phrases = []

        if not article or not summary:
            continue

        # extract from summary or article
        if extraction_source == "article":
            source = article
        elif extraction_source == "summary":
            source = summary
        elif extraction_source == "all":
            source = article + "\n" + summary
        else:
            raise NotImplementedError

        """
        Step 1: extraction
        """
        # Extraction with textrank
        if extraction_mode == 'textrank':
            doc = nlp(source)
            for phrase in doc._.phrases:
                selected_phrases.append(phrase.text)

        # Extraction with yake
        elif extraction_mode == 'yake':
            phrases = kw_extractor.extract_keywords(source)
            for phrase in phrases:
                selected_phrases.append(phrase[0])
    
        # Extraction with GPT3 Prompt
        elif extraction_mode == 'prompt':
            input = keyword_prompt.replace("[[QUESTION]]", source)
            candidates = gpt3.call(prompt=input,
                                temperature=0.7,
                                max_tokens=96,
                                n=1,
                                top_p=1.0,
                                stop=stop_words
                                )
            for candidate in candidates:
                for keyword in candidate[:-1].split(SPLIT):
                    selected_phrases.append(keyword.strip())
            selected_phrases = list(set(selected_phrases))

        # Not Implemented Yet...
        else:
            raise NotImplementedError()

        """
        Step 2: selection and tokenization
        """
        # sort phrases according to appearances in the article
        selected_phrases = filter_keywords(selected_phrases, summary, article)
        target = SPLIT.join(selected_phrases) + "." if len(selected_phrases) > 0 else ""
        phrase_num.append(len(selected_phrases))

        # save data
        processed_data.append({"article": article, "summary": summary, "id": id, "phrases": selected_phrases, "target": target})
        
    # Statistics
    len_min = np.min(phrase_num)
    len_mean = np.mean(phrase_num)
    len_median = np.median(phrase_num)
    len_max = np.max(phrase_num)
    print("mean of phrase num:{}, median of phrase num:{}, max of phrase num:{}, min of phrase num {}".format(len_mean, len_median, len_max, len_min))

    return processed_data


def get_data_split(dataset, n_train, n_val, n_test, extraction_mode, extraction_source, keyword_num_range=(1, 20), return_dict=True):

    # load existing data
    data_path = f"./sft4lms/data/{dataset}/{extraction_mode}-{extraction_source}/"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    train_data_path = data_path + "train.npy"
    val_data_path = data_path + "val.npy"
    test_data_path = data_path + "test.npy"

    # training data
    if os.path.exists(train_data_path):
        train_data = np.load(train_data_path, allow_pickle=True)
    else:
        train_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="train")
        # get extraction data
        train_data = get_extraction_data(dataset, selected_data, extraction_mode, extraction_source)
        random.shuffle(train_data)
        np.save(train_data_path, train_data)

    # test data, select 500 subset
    if os.path.exists(test_data_path):
        test_data = np.load(test_data_path, allow_pickle=True)
    else:
        test_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="test")
        # select 500 data
        selected_data = []
        with open(f"./sft4lms/data/{dataset}_test500.json", 'r') as file:
            test_data_subset = json.load(file)
        for idx, d in test_data_subset.items():
            for d_ in test_data:
                if d_['id'] == d['id']:
                    selected_data.append(d_)
                    break  
        print(len(selected_data))
        # get extraction data
        test_data = get_extraction_data(dataset, selected_data, extraction_mode, extraction_source)
        np.save(test_data_path, test_data)

    # validation data
    if os.path.exists(val_data_path):
        val_data = np.load(val_data_path, allow_pickle=True)
    else:
        val_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="validation")
        # select 500 data
        selected_data = []
        with open(f"./sft4lms/data/{dataset}_val500.json", 'r') as file:
            val_data_subset = json.load(file)
        for idx, d in val_data_subset.items():
            for d_ in val_data:
                if d_['id'] == d['id']:
                    selected_data.append(d_)
                    break  
        print(len(selected_data))
        # get extraction data
        val_data = get_extraction_data(dataset, selected_data, extraction_mode, extraction_source)
        np.save(val_data_path, val_data)

    # select n samples
    train_data = train_data[:min(n_train, len(train_data))]
    val_data = val_data[:min(n_val, len(val_data))]
    test_data = test_data[:min(n_test, len(test_data))]

    if return_dict:
        return dictlist2dict(train_data), dictlist2dict(val_data), dictlist2dict(test_data)
    else:
        return train_data, val_data, test_data


if __name__ == "__main__":

    for dataset in ['cnndm']:
        for extraction_mode in ['textrank']:
            for extraction_source in ['all']:
                for n_train in [1000, 2000, 4000]:
                    for n_val in [500]:
                        for n_test in [500]:
                            train_data, val_data, test_data = get_data_split(dataset=dataset, n_train=n_train, n_val=n_val, n_test=n_test,
                                                                            extraction_mode=extraction_mode, extraction_source=extraction_source, 
                                                                            keyword_num_range=(1, 20))

                            from datasets import Dataset
                            train_dataset = Dataset.from_dict(train_data)
                            val_dataset = Dataset.from_dict(val_data)
                            test_dataset = Dataset.from_dict(test_data)
                            tokenizer = AutoTokenizer.from_pretrained("gpt2")

                            phrase_nums = []
                            article_lens = []
                            summary_lens = []
                            target_lens = []
                            analyzed_dataset = train_dataset
                            
                            for i in range(len(analyzed_dataset)):
                            # for i in random.sample(range(len(dataset)), 100):
                                phrase_nums.append(len(analyzed_dataset[i]['phrases']))
                                article = analyzed_dataset[i]['article']
                                summary = analyzed_dataset[i]['summary']
                                target = analyzed_dataset[i]['target']
                                article_lens.append(len(tokenizer(article).input_ids))
                                summary_lens.append(len(tokenizer(summary).input_ids))
                                target_lens.append(len(tokenizer(target).input_ids))

                                # # # DEBUG
                                # print("Index:")
                                # print(i)

                                # print("Keywords:")
                                # print(analyzed_dataset[i]['phrases'])

                                # print("Target:")
                                # print(analyzed_dataset[i]['target'])

                                # _ = input("continue.........")


