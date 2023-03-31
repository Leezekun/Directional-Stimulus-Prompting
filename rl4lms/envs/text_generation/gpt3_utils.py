from typing import Any, Dict, List
import os
import time
import openai
import logging
from transformers import GPT2TokenizerFast

avoid_keywords = ["one", "two", "three", "1", "2", "3", "a", "he", "she", "i", "we", "you", "it", "this", 
        "that", "the", "those", "these", "they", "me", "them", "what", "him", "her", "my", "which", "who", "why", 
        "your", "my", "his", "her", "ours", "our", "could", "with", "whom", "whose"]

class GPT3():
    def __init__(self, model="gpt-3.5-turbo", interval=0.5, timeout=10.0, exp=2, patience=10, max_interval=4, max_prompt_length=4096):
        self.model = model
        self.interval = interval
        self.timeout = timeout
        self.base_timeout = timeout
        self.patience = patience
        self.exp = exp
        self.max_prompt_length = max_prompt_length
        self.max_interval = max_interval
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def call(
        self, prompt, temperature=1.0, top_p=1.0, max_tokens=64, n=1, 
        frequency_penalty=0, presence_penalty=0, stop="Q:", rstrip=False,
        **kwargs):

        openai.api_key = os.environ.get('OPENAI_ACCESS_KEY', None)

        # check if exceeding len limit
        input_len = len(self.tokenizer(prompt).input_ids)
        if input_len + max_tokens >= self.max_prompt_length:
            logging.warning("OpenAI length limit error.")
            return [""] * n

        # stop words
        if isinstance(stop, List):
            pass
        elif isinstance(stop, str):
            stop = [stop]

        if rstrip:
            prompt = prompt.rstrip()

        retry_interval_exp = 1 
        t1 = time.time()

        while True and retry_interval_exp <= self.patience:
            try:
                if self.model == "gpt-3.5-turbo": # chat completion
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                    response = openai.ChatCompletion.create(model=self.model,
                                                        messages=messages,
                                                        temperature=temperature,
                                                        max_tokens=max_tokens,
                                                        n=n,
                                                        top_p=top_p,
                                                        frequency_penalty=frequency_penalty,
                                                        presence_penalty=presence_penalty,
                                                        stop=stop,
                                                        request_timeout=self.timeout # timeout!
                                                        )  
                    candidates = response["choices"]
                    candidates = [candidate["message"]["content"] for candidate in candidates]

                else: # text completion
                    response = openai.Completion.create(model=self.model,
                                                        prompt=prompt,
                                                        temperature=temperature,
                                                        max_tokens=max_tokens,
                                                        n=n,
                                                        top_p=top_p,
                                                        frequency_penalty=frequency_penalty,
                                                        presence_penalty=presence_penalty,
                                                        stop=stop,
                                                        request_timeout=self.timeout # timeout!
                                                        )    
                    candidates = response["choices"]
                    candidates = [candidate["text"] for candidate in candidates]
                
                t2 = time.time()
                logging.info(f"{input_len} tokens, {t2-t1} secs")  

                return candidates

            # except openai.error.RateLimitError as e:
            except Exception as e:
                # logging.warning("OpenAI rate limit error. Retry")
                logging.warning(e)
                # Exponential backoff
                time.sleep(max(self.max_interval, self.interval * (self.exp ** retry_interval_exp)))
                retry_interval_exp += 1
        
        return None