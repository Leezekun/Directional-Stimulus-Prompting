import os
import spacy
import numpy as np
import pandas as pd

from tqdm import trange
from tqdm import tqdm

import datasets
from keybert import KeyBERT
import yake
import pytextrank
# from keyphrase_vectorizers import KeyphraseCountVectorizer
# from sentence_transformers import SentenceTransformer

# from sft4lms.Summarization.utils import *
from rl4lms.envs.text_generation.gpt3_utils import GPT3, avoid_keywords

CONTEXT = "[[ARTICLE]] Keywords: "
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


def select_keywords(phrases, summary, article):

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

        # place in the order of summary
        p_index = summary.lower().index(p.lower())

        selected_phrases.append(p)
        phrase_indices.append(p_index)

    # sort
    selected_phrases = list(zip(selected_phrases, phrase_indices))
    selected_phrases = sorted(selected_phrases, key=lambda x: x[1])
    selected_phrases = [p[0] for p in selected_phrases]
    return selected_phrases


def filter_data(data, min_amount=1):
    assert min_amount >= 1
    filtered_data = []
    for d in data:
        if len(d['phrases']) >= min_amount: # at least one token
            filtered_data.append(d)
    return filtered_data


def get_extraction_data(dataset, data, extraction_mode, extraction_source):
    
    # Init
    if extraction_mode == 'textrank':
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("textrank")
    elif extraction_mode == 'yake':
        kw_extractor = yake.KeywordExtractor()
    # elif extraction_mode == 'patternrank':
    #     vectorizer = KeyphraseCountVectorizer()
    # elif extraction_mode == 'keybert':
    #     sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device='cuda:0')
    #     kw_model = KeyBERT(model=sentence_model)    
    elif extraction_mode == 'prompt':
        gpt3 = GPT3(2.0) # sleep 5s before each call
        keyword_prompt_path = f"../prompts/{dataset}_keyword_article_fs.txt"
        f = open(keyword_prompt_path, 'r') 
        keyword_prompt = f.read().strip()
        stop_words = ["\n", "Sentences:"]

    # Start
    processed_data = []
    phrase_len = []
    for d in trange(len(data)):

        selected_phrases = []
        if "article" in data[d] and "highlights" in data[d]:
            article = data[d]['article']
            summary = data[d]['highlights']
        elif "document" in data[d] and "summary" in data[d]:
            article = data[d]['document']
            summary = data[d]['summary']
        elif "text" in data[d] and "summary" in data[d]:
            article = data[d]['text']
            summary = data[d]['summary']
        elif "maintext" in data[d] and "description" in data[d]:
            article = data[d]['maintext']
            summary = data[d]['description']
        else:
            raise NotImplementedError

        if not article or not summary:
            continue

        context = CONTEXT.replace("[[ARTICLE]]", article)

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

        # # Extraction with patternrank (keyphrase_vectorizer)
        # elif extraction_mode == 'patternrank':
        #     vectorizer.fit([source])
        #     selected_phrases = vectorizer.get_feature_names_out()

        # # Extraction with keybert
        # elif extraction_mode == 'keybert':
        #     phrases = kw_model.extract_keywords(source, vectorizer=KeyphraseCountVectorizer())
        #     for phrase in phrases:
        #         selected_phrases.append(phrase[0])

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
        selected_phrases = select_keywords(selected_phrases, summary, article)
        target = SPLIT.join(selected_phrases) + "." if len(selected_phrases) > 0 else ""
        phrase_len.append(len(selected_phrases))

        # save data
        processed_data.append({"article": article, "summary": summary, "phrases": selected_phrases, "context": context, "target": target})
        
    # Statistics
    len_min = np.min(phrase_len)
    len_mean = np.mean(phrase_len)
    len_median = np.median(phrase_len)
    len_max = np.max(phrase_len)
    print("mean of phrase num:{}, median of phrase num:{}, max of phrase num:{}, min of phrase num {}".format(len_mean, len_median, len_max, len_min))

    return processed_data


def get_data_split(dataset, n_train, n_val, n_test, extraction_mode, extraction_source, min_keywords=1, return_dict=True):

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
        if dataset == 'cnndm':
            train_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="train")
        elif dataset == 'xsum':
            train_data = datasets.load_dataset("xsum", split=f"train")

        # get extraction
        train_data = get_extraction_data(dataset, train_data, extraction_mode, extraction_source)
        np.save(train_data_path, train_data)

    # validation data
    if os.path.exists(val_data_path):
        val_data = np.load(val_data_path, allow_pickle=True)
    else:
        if dataset == 'cnndm':
            val_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split=f"validation")
        elif dataset == 'xsum':
            val_data = datasets.load_dataset("xsum", split=f"validation")

        # add extraction labels
        val_data = get_extraction_data(dataset, val_data, extraction_mode, extraction_source)
        np.save(val_data_path, val_data)

    # test data
    if os.path.exists(test_data_path):
        test_data = np.load(test_data_path, allow_pickle=True)
    else:
        if dataset == 'cnndm':
            test_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split=f"test")
        elif dataset == 'xsum':
            test_data = datasets.load_dataset("xsum", split=f"test")

        # add extraction labels
        test_data = get_extraction_data(dataset, test_data, extraction_mode, extraction_source)
        np.save(test_data_path, test_data)

    # filter the data with too few phrases
    print(f"Before filtering phrases less than {min_keywords}", len(train_data), len(val_data), len(test_data))
    train_data = filter_data(train_data, min_amount=min_keywords)
    val_data = filter_data(val_data, min_amount=min_keywords)
    test_data = filter_data(test_data, min_amount=min_keywords)
    print(f"After filtering phrases less than {min_keywords}", len(train_data), len(val_data), len(test_data))

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
                    train_data, val_data, test_data = get_data_split(dataset=dataset, n_train=n_train, n_val=500, n_test=500, extraction_mode=extraction_mode, extraction_source=extraction_source, min_keywords=1)
                
                    from datasets import Dataset
                    train_dataset = Dataset.from_dict(train_data)
                    phrase_lens = []

                    # DEBUG
                    for i in range(len(train_dataset)):

                        phrase_lens.append(len(train_dataset[i]['phrases']))

                        print(">>>>>>>>>>>summary:")
                        print(train_dataset[i]['article'])

                        print(">>>>>>>>>>>summary:")
                        print(train_dataset[i]['summary'])

                        print(">>>>>>>>>>>keywords:")
                        print(train_dataset[i]['phrases'])

                        _ = input("continue.........")

                    print(np.array(phrase_lens).mean())







    
    


