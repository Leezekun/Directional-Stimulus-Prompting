from rl4lms.data_pools.text_generation_pool import TextGenPool, Sample
from rl4lms.data_pools.custom_text_generation_pools import CommonGen
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import numpy as np
import random
from sft4lms.Summarization.data_loader import get_data_split as summarization_get_data_split
from sft4lms.MultiWOZ.data_loader import get_data_split as multiwoz_get_data_split


class TestTextGenPool(TextGenPool):
    @classmethod
    def prepare(cls, split: str, prompt: str, n_samples=100):
        samples = [Sample(id=ix,
                          prompt_or_input_text=prompt,  # a dummy prompt
                          references=[]
                          ) for ix in range(n_samples)]
        pool_instance = cls(samples)
        return pool_instance


class CNNDailyMailHint(TextGenPool):
    @classmethod
    def prepare(cls,
                split: str,
                dataset: str,
                n_train: int,
                n_val: int = 500,
                n_test: int = 500,
                extraction_mode: str = "textrank",
                extraction_source: str = "all",
                prompt_suffix: str = "",
                prompt_prefix: str = "",
                truncate_article: int = None,
                max_size: int = None):

        train_data, val_data, test_data = summarization_get_data_split(dataset=dataset, n_train=n_train, n_val=n_val, n_test=n_test, 
                                                                       extraction_mode=extraction_mode, extraction_source=extraction_source, 
                                                                       return_dict=False)
        if split == "train":
            random.shuffle(train_data)
            dataset = train_data
        elif split == "val":
            dataset = val_data
        elif split == "test":
            dataset = test_data

        samples = []
        for ix, item in tqdm(enumerate(dataset),
                             desc="Tokenizing dataset",
                             total=len(dataset)):

            if truncate_article is not None:
                tokens = word_tokenize(item["article"])
                tokens = tokens[:truncate_article]
                item["article"] = " ".join(tokens)

            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text=prompt_prefix+item["article"]+prompt_suffix,
                            references=[item["summary"]],
                            meta_data={"phrases": item["phrases"], "target": item["target"]}
                            )
            samples.append(sample)

            if max_size is not None and ix == (max_size-1):
                break

        pool_instance = cls(samples)
        return pool_instance


class MultiWOZHint(TextGenPool):
    @classmethod
    def prepare(cls,
                split: str,
                version: str,
                n_train: int,
                n_val: int = 1000,
                n_test: int = 1000,
                ):

        train_data, val_data, test_data = multiwoz_get_data_split(dataset="multiwoz", dataset_version=version, 
                                                                  n_train=n_train, n_val=n_val, n_test=n_test, 
                                                                  return_dict=False)
        
        if split == 'train':
            random.shuffle(train_data)
            dataset = train_data
        elif split == 'val': 
            dataset = val_data
        elif split == 'test':
            dataset = test_data

        samples = []
        utterance_id = 0
        for item in dataset:
            sample = Sample(id=utterance_id,
                            prompt_or_input_text=item["da_input"],
                            references=[item["resp"]],
                            meta_data={"da_output": item["da_output"], 
                                       "user": item["user"], "resp": item["resp"],
                                       "turn_id": item["turn_id"],
                                       "history_users": item["history_users"],
                                       "history_resps": item["history_resps"],
                                       "history_acts": item["history_acts"],
                                       "eval_turn": item["eval_turn"]
                                    }
                            )
            samples.append(sample)
            utterance_id += 1

        dp_instance = cls(samples)
        return dp_instance

