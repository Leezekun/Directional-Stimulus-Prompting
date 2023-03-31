import os
import requests
import json
import time
from typing import List
import random

import json
import random
import re
import copy
import os
import numpy as np
import logging

from .ontology import *

num2word = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten"}

all_domain = [
    "[taxi]","[police]","[hospital]","[hotel]","[attraction]","[train]","[restaurant]"
]

requestable_slots = {
    "taxi": ["car", "phone"],
    "police": ["postcode", "address", "phone"],
    "hospital": ["address", "phone", "postcode"],
    "hotel": ["address", "postcode", "internet", "phone", "parking", "type", "pricerange", "stars", "area", "reference"],
    "attraction": ["price", "type", "address", "postcode", "phone", "area", "reference"],
    "train": ["time", "leave", "price", "arrive", "id", "reference"],
    "restaurant": ["phone", "postcode", "address", "pricerange", "food", "area", "reference"]
}
all_reqslot = ["car", "address", "postcode", "phone", "internet",  "parking", "type", "pricerange", "food",
                      "stars", "area", "reference", "time", "leave", "price", "arrive", "id"]
# count: 17

informable_slots = {
    "taxi": ["leave", "destination", "departure", "arrive"],
    "police": [],
    "hospital": ["department"],
    "hotel": ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name"],
    "attraction": ["area", "type", "name"],
    "train": ["destination", "day", "arrive", "departure", "people", "leave"],
    "restaurant": ["food", "pricerange", "area", "name", "time", "day", "people"]
}
all_infslot = ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name",
                     "leave", "destination", "departure", "arrive", "department", "food", "time"]
# count: 17

all_slots = all_reqslot + all_infslot + ["stay", "day", "people", "name", "destination", "departure", "department"]
all_slots = set(all_slots)

dialog_acts = {
    'restaurant': ['inform', 'request', 'nooffer', 'recommend', 'select', 'offerbook', 'offerbooked', 'nobook'],
    'hotel': ['inform', 'request', 'nooffer', 'recommend', 'select', 'offerbook', 'offerbooked', 'nobook'],
    'attraction': ['inform', 'request', 'nooffer', 'recommend', 'select'],
    'train': ['inform', 'request', 'nooffer', 'offerbook', 'offerbooked', 'select'],
    'taxi': ['inform', 'request'],
    'police': ['inform', 'request'],
    'hospital': ['inform', 'request'],
    # 'booking': ['book', 'inform', 'nobook', 'request'],
    'general': ['bye', 'greet', 'reqmore', 'welcome'],
}
all_acts = []
for acts in dialog_acts.values():
    for act in acts:
        act = f"[{act}]"
        if act not in all_acts:
            all_acts.append(act)

# all_acts = ['[inform]', '[request]', '[nooffer]', '[recommend]', '[select]', '[offerbook]', '[offerbooked]', '[nobook]', '[bye]', '[greet]', '[reqmore]', '[welcome]']


GENERAL_TYPO = {
        # type
        "guesthouse":"guest house", "guesthouses":"guest house", "guest":"guest house", "mutiple sports":"multiple sports",
        "sports":"multiple sports", "mutliple sports":"multiple sports","swimmingpool":"swimming pool", "concerthall":"concert hall",
        "concert":"concert hall", "pool":"swimming pool", "night club":"nightclub", "mus":"museum", "ol":"architecture",
        "colleges":"college", "coll":"college", "architectural":"architecture", "musuem":"museum", "churches":"church",
        # area
        "center":"centre", "center of town":"centre", "near city center":"centre", "in the north":"north", "cen":"centre", "east side":"east",
        "east area":"east", "west part of town":"west", "ce":"centre",  "town center":"centre", "centre of cambridge":"centre",
        "city center":"centre", "the south":"south", "scentre":"centre", "town centre":"centre", "in town":"centre", "north part of town":"north",
        "centre of town":"centre", "cb30aq": "none",
        # price
        "mode":"moderate", "moderate -ly": "moderate", "mo":"moderate",
        # day
        "next friday":"friday", "monda": "monday",
        # parking
        "free parking":"free",
        # internet
        "free internet":"yes",
        # star
        "4 star":"4", "4 stars":"4", "0 star rarting":"none",
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
        # others
        "y":"yes", "any":"dontcare", "n":"no", "does not care":"dontcare", "not men":"none", "not":"none", "not mentioned":"none",
        '':"none", "not mendtioned":"none", "3 .":"3", "does not":"no", "fun":"none", "art":"none",
        }

class Vocab(object):
    def __init__(self, vocab_size=0):
        self.vocab_size = vocab_size
        self.vocab_size_oov = 0   
        self._idx2word = {}   
        self._word2idx = {}  
        self._freq_dict = {} 
        for w in ['<pad>', '<go_r>', '<unk>', '<go_b>', '<go_a>','<eos_u>', '<eos_r>',
                      '<eos_b>', '<eos_a>', '<go_d>','<eos_d>']:
            self._absolute_add_word(w)

    def _absolute_add_word(self, w):
        idx = len(self._idx2word)
        self._idx2word[idx] = w
        self._word2idx[w] = idx

    def add_word(self, word):
        if word not in self._freq_dict:
            self._freq_dict[word] = 0
        self._freq_dict[word] += 1

    def has_word(self, word):
        return self._freq_dict.get(word)

    def _add_to_vocab(self, word):
        if word not in self._word2idx:
            idx = len(self._idx2word)
            self._idx2word[idx] = word
            self._word2idx[word] = idx

    def construct(self):
        l = sorted(self._freq_dict.keys(), key=lambda x: -self._freq_dict[x])
        print('Vocabulary size including oov: %d' % (len(l) + len(self._idx2word)))
        if len(l) + len(self._idx2word) < self.vocab_size:
            logging.warning('actual label set smaller than that configured: {}/{}'
                            .format(len(l) + len(self._idx2word), self.vocab_size))
        for word in all_domains + ['general']:
            word = '[' + word + ']'
            self._add_to_vocab(word)
        for word in all_acts:
            word = '[' + word + ']'
            self._add_to_vocab(word)
        for word in all_slots:
            self._add_to_vocab(word)
        for word in l:
            if word.startswith('[value_') and word.endswith(']'):
                self._add_to_vocab(word)
        for word in l:
            self._add_to_vocab(word)
        self.vocab_size_oov = len(self._idx2word)

    def load_vocab(self, vocab_path):
        self._freq_dict = json.loads(open(vocab_path+'.freq.json', 'r').read())
        self._word2idx = json.loads(open(vocab_path+'.word2idx.json', 'r').read())
        self._idx2word = {}
        for w, idx in self._word2idx.items():
            self._idx2word[idx] = w
        self.vocab_size_oov = len(self._idx2word)
        print('vocab file loaded from "'+vocab_path+'"')
        print('Vocabulary size including oov: %d' % (self.vocab_size_oov))

    def save_vocab(self, vocab_path):
        _freq_dict = OrderedDict(sorted(self._freq_dict.items(), key=lambda kv:kv[1], reverse=True))
        write_dict(vocab_path+'.word2idx.json', self._word2idx)
        write_dict(vocab_path+'.freq.json', _freq_dict)


    def encode(self, word, include_oov=True):
        if include_oov:
            if self._word2idx.get(word, None) is None:
                raise ValueError('Unknown word: %s. Vocabulary should include oovs here.'%word)
            return self._word2idx[word]
        else:
            word = '<unk>' if word not in self._word2idx else word
            return self._word2idx[word]

    def sentence_encode(self, word_list):
        return [self.encode(_) for _ in word_list]

    def oov_idx_map(self, idx):
        return 2 if idx > self.vocab_size else idx

    def sentence_oov_map(self, index_list):
        return [self.oov_idx_map(_) for _ in index_list]


    def decode(self, idx, indicate_oov=False):
        if not self._idx2word.get(idx):
            raise ValueError('Error idx: %d. Vocabulary should include oovs here.'%idx)
        if not indicate_oov or idx<self.vocab_size:
            return self._idx2word[idx]
        else:
            return self._idx2word[idx]+'(o)'

    def sentence_decode(self, index_list, eos=None, indicate_oov=False):
        l = [self.decode(_, indicate_oov) for _ in index_list]
        if not eos or eos not in l:
            return ' '.join(l)
        else:
            idx = l.index(eos)
            return ' '.join(l[:idx])

    def nl_decode(self, l, eos=None):
        return [self.sentence_decode(_, eos) + '\n' for _ in l]

def normalize_domain_slot(schema):
    normalized_schema = []
    for service in schema:
        if service['service_name'] == 'bus':
            service['service_name'] = 'taxi'

        slots = service['slots']
        normalized_slots = []

        for slot in slots: # split domain-slots to domains and slots
            domain_slot = slot['name']
            domain, slot_name = domain_slot.split('-')
            if domain == 'bus':
                domain = 'taxi'
            if slot_name == 'bookstay':
                slot_name = 'stay'
            if slot_name == 'bookday':
                slot_name = 'day'
            if slot_name == 'bookpeople':
                slot_name = 'people'
            if slot_name == 'booktime':
                slot_name = 'time'
            if slot_name == 'arriveby':
                slot_name = 'arrive'
            if slot_name == 'leaveat':
                slot_name = 'leave'
            domain_slot = "-".join([domain, slot_name])
            slot['name'] = domain_slot
            normalized_slots.append(slot)

        service['slots'] = normalized_slots
        normalized_schema.append(service)

    return normalized_schema
         
def paser_bs_to_list(sent):
    """Convert compacted bs span to triple list
        Ex:  
    """
    sent = sent.split()
    belief_state = []
    domain_idx = [idx for idx,token in enumerate(sent) if token in all_domain]
    for i,d_idx in enumerate(domain_idx):
        next_d_idx = len(sent) if i+1 == len(domain_idx) else domain_idx[i+1]
        domain = sent[d_idx]
        sub_span = sent[d_idx+1:next_d_idx]
        sub_s_idx = [idx for idx,token in enumerate(sub_span) if token in all_slots]
        for j,s_idx in enumerate(sub_s_idx):
            next_s_idx = len(sub_span) if j == len(sub_s_idx) - 1 else sub_s_idx[j+1]
            slot = sub_span[s_idx]
            value = ' '.join(sub_span[s_idx+1:next_s_idx])
            bs = " ".join([domain,slot,value])
            belief_state.append(bs)
    return list(set(belief_state))

def paser_bs_to_dict(sent):
    """Convert compacted bs span to triple list
        Ex:  
    """
    sent = sent.split()
    belief_state = {}
    domain_idx = [idx for idx,token in enumerate(sent) if token in all_domain]
    for i,d_idx in enumerate(domain_idx):
        next_d_idx = len(sent) if i+1 == len(domain_idx) else domain_idx[i+1]
        domain = sent[d_idx]
        if domain in belief_state:
            domain_bs = belief_state[domain]
        else:
            domain_bs = {}
        sub_span = sent[d_idx+1:next_d_idx]
        sub_s_idx = [idx for idx,token in enumerate(sub_span) if token in all_slots]
        for j,s_idx in enumerate(sub_s_idx):
            next_s_idx = len(sub_span) if j+1 == len(sub_s_idx) else sub_s_idx[j+1]
            slot = sub_span[s_idx]
            value = " ".join(sub_span[s_idx+1:next_s_idx])
            bs = " ".join([domain,slot,value])
            domain_bs[slot] = value
        belief_state[domain] = domain_bs
    return belief_state

def paser_bs_reform_to_dict(sent):
    """Convert compacted bs span to triple list
        Ex:  
    """
    all_domain = ["[taxi]","[police]","[hospital]","[hotel]","[attraction]","[train]","[restaurant]","[general]"]
    sent = sent.split()
    belief_state = {}
    domain_idx = [idx for idx,token in enumerate(sent) if token in all_domain] 
    for i,d_idx in enumerate(domain_idx):
        next_d_idx = len(sent) if i+1 == len(domain_idx) else domain_idx[i+1]
        domain = sent[d_idx]
        if domain in belief_state:
            domain_bs = belief_state[domain]
        else:
            domain_bs = {}
        sub_span = " ".join(sent[d_idx+1:next_d_idx])
        for bs in sub_span.split(","):
            if bs and len(bs.split(" is ")) == 2:
                slot_name, slot_value = bs.split(" is ")
                slot_name = slot_name.strip()
                slot_value = slot_value.strip()
                if slot_name and slot_value:
                    domain_bs[slot_name] = slot_value
        belief_state[domain] = domain_bs
    return belief_state

def paser_bs_from_dict_to_list(bs):
        """
        Convert compacted bs span to triple list
        Ex:  
        """
        belief_state = []
        for domain, domain_bs in bs.items():
            if domain_bs:
                for slot_name, slot_value in domain_bs.items():
                    belief_state.append(" ".join([domain, slot_name]))
        return list(set(belief_state))

def paser_dict_to_bs(goal, ignore_none_bs=True):
    bs_text = []            
    for domain, bs in goal.items(): # reverse the dict to align with the original pptod fotmat 
        if not ignore_none_bs:
            bs_text.append(domain)
        if bs:
            if ignore_none_bs:
                bs_text.append(domain)
            for slot_name, slot_value in bs.items():
                bs_text.append(slot_name)
                bs_text.append(slot_value)

    bs_text = " ".join(bs_text)
    return bs_text

def paser_dict_to_bs_reform(goal, ignore_none_bs=True):
    bs_reform_text = []            
    for domain, bs in goal.items(): # reverse the dict to align with the original pptod fotmat 
        if not ignore_none_bs:
            bs_reform_text.append(domain)
        if bs:
            if ignore_none_bs:
                bs_reform_text.append(domain)
            domain_text = []
            for slot_name, slot_value in bs.items():
                domain_text.append(f"{slot_name} is {slot_value}")
            domain_text = " , ".join(domain_text)
            bs_reform_text.append(domain_text)

    bs_reform_text = " ".join(bs_reform_text)
    return bs_reform_text

def paser_dict_to_bsdx(goal, ignore_none_bs=True):
    bsdx_text = []            
    for domain, bs in goal.items(): # reverse the dict to align with the original pptod fotmat 
        if not ignore_none_bs:
            bsdx_text.append(domain)
        if bs:
            if ignore_none_bs:
                bsdx_text.append(domain)
            for slot_name, slot_value in bs.items():
                bsdx_text.append(slot_name)

    bsdx_text = " ".join(bsdx_text)
    return bsdx_text

def paser_dict_to_bsdx_reform(goal, ignore_none_bs=True):
    bsdx_reform_text = []            
    for domain, bs in goal.items(): # reverse the dict to align with the original pptod fotmat 
        if not ignore_none_bs:
            bsdx_reform_text.append(domain)
        if bs:
            if ignore_none_bs:
                bsdx_reform_text.append(domain)
            bsdx_domain_text = []
            for slot_name, slot_value in bs.items():
                bsdx_domain_text.append(slot_name)
            bsdx_domain_text = " , ".join(bsdx_domain_text)
            bsdx_reform_text.append(bsdx_domain_text)

    bsdx_reform_text = " ".join(bsdx_reform_text)
    return bsdx_reform_text

def paser_aspn_to_dict(sent):
    sent = sent.split()
    dialog_act = {}
    domain_idx = [idx for idx,token in enumerate(sent) if token in all_domain+["[general]"]]
    for i,d_idx in enumerate(domain_idx):
        next_d_idx = len(sent) if i+1 == len(domain_idx) else domain_idx[i+1]
        domain = sent[d_idx]
        if domain in dialog_act:
            domain_da = dialog_act[domain]
        else:
            domain_da = {}
        sub_span = sent[d_idx+1:next_d_idx]
        sub_a_idx = [idx for idx,token in enumerate(sub_span) if token in all_acts]
        for j,a_idx in enumerate(sub_a_idx):
            next_a_idx = len(sub_span) if j+1 == len(sub_a_idx) else sub_a_idx[j+1]
            act = sub_span[a_idx]
            act_slots = sub_span[a_idx+1:next_a_idx]
            domain_da[act] = act_slots
        dialog_act[domain] = domain_da
    return dialog_act
    
def paser_dict_to_list(goal, level):
    if level == 1:
        return list(goal.keys())
    elif level == 2:
        belief_state = []
        for domain, domain_bs in goal.items():
            for slot_name, slot_value in domain_bs.items():
                belief_state.append(" ".join([domain, slot_name]))
        return list(set(belief_state))
    elif level == 3:
        belief_state = []
        for domain, domain_bs in goal.items():
            for slot_name, slot_value in domain_bs.items():
                if isinstance(slot_value, str):
                    belief_state.append(" ".join([domain, slot_name, slot_value]))
                elif isinstance(slot_value, List):
                    if slot_value:
                        for slot_value_ in slot_value:
                            belief_state.append(" ".join([domain, slot_name, slot_value_]))
                    else:
                        belief_state.append(" ".join([domain, slot_name]))
        return list(set(belief_state))

def compare_dict(old_dict, new_dict):
    differ = {}
    for domain, slot in new_dict.items():
        if domain not in old_dict:
            differ[domain] = slot
        else:
            old_slot = old_dict[domain]
            for slot_name, slot_value in slot.items():
                if slot_name not in old_slot:
                    if domain not in differ:
                        differ[domain] = {}
                    differ[domain][slot_name] = slot_value
                elif old_slot[slot_name] != slot_value:
                    if domain not in differ:
                        differ[domain] = {}
                    differ[domain][slot_name] = slot_value
    return differ

def dict_jaccard_similarity(old_dict, new_dict, levels=[3]):
    def jaccard(list1, list2):
        intersection = list(set(list1) & set(list2))
        unionset = list(set(list1).union(set(list2)))
        if unionset:
            return float(len(intersection) / len(unionset))
        else:
            return 0.0
    similarity = 0.
    for level in levels:
        old_list = paser_dict_to_list(old_dict, level=level)
        new_list = paser_dict_to_list(new_dict, level=level)
        similarity += jaccard(old_list, new_list)
    similarity /= len(levels)
    return similarity


if __name__ == '__main__':
    # sent = "[hotel] people 2 stay 3"
    # sent = paser_bs_to_dict(sent)
    # print(sent)
    sent1 = "[hotel] people 2 stay 3"
    sent2 = "[hotel] people 1 stay 3 [restaurant] people 2"
    dict1, dict2 = paser_bs_to_dict(sent1),  paser_bs_to_dict(sent2)
    print(dict1)
    print(dict2)
    dict_similarity = dict_jaccard_similarity(dict1, dict2, levels=[3])
    print(dict_similarity)

    # sent = "[hotel] [request] stay people "
    # sent = paser_aspn_to_dict(sent)
    # print(sent)
    sent1 = "[general] [welcome] [restaurant] [request] day people food"
    sent2 = "[general] [greet] [restaurant] [request] food"
    dict1, dict2 = paser_aspn_to_dict(sent1),  paser_aspn_to_dict(sent2)
    print(dict1)
    print(dict2)
    dict_similarity = dict_jaccard_similarity(dict1, dict2, levels=[3])
    print(dict_similarity)

