import re
from typing import Any, Dict, List
import numpy as np

from rl4lms.envs.text_generation.observation import Observation
from rl4lms.envs.text_generation.reward import RewardFunction
from rl4lms.envs.text_generation.gpt3_utils import GPT3, avoid_keywords


class RewardIncreasingNumbers(RewardFunction):
    def __init__(self,
                 min_tokens: int) -> None:
        super().__init__()
        self.min_tokens = min_tokens

    @staticmethod
    def is_number(text):
        try:
            float(text)
            return True
        except ValueError:
            return False

    @staticmethod
    def reward_increasing_numbers_in_text(gen_text: str,
                                          min_tokens: int):
        gen_tokens = gen_text.split()
        number_tokens = [float(token)
                         for token in gen_tokens if RewardIncreasingNumbers.is_number(token)]
        if len(number_tokens) > 0:
            # then we check how many numbers are in the sorted order
            sorted_count = 1
            previous_token = number_tokens[0]
            for token in number_tokens[1:]:
                if token > previous_token:
                    sorted_count += 1
                    previous_token = token
                else:
                    break
            return ((sorted_count)/max(len(gen_tokens), (min_tokens/2)))
        return 0

    def __call__(self, prev_observation: Observation,
                 action: int,
                 current_observation: Observation,
                 done: bool,
                 meta_info: Dict[str, Any] = None) -> float:
        if done:
            gen_text = current_observation.context_text
            reward = RewardIncreasingNumbers.reward_increasing_numbers_in_text(
                gen_text, self.min_tokens)
            return reward
        return 0


class RewardSentencesWithDates:

    def date_in_text(text: str):
        match = re.search(r'\d{4}-\d{2}-\d{2}',
                          text)
        if match is not None:
            return 1
        else:
            return 0

    def __call__(self, prev_observation: Observation,
                 action: int,
                 current_observation: Observation,
                 done: bool,
                 meta_info: Dict[str, Any] = None) -> float:
        if done:
            return RewardSentencesWithDates.date_in_text(current_observation.context_text)
        return 0


class RewardSummarizationWithHint(RewardFunction):
    def __init__(self,
                 gpt3_model: str,
                 interval: float,
                 timeout: float,
                 exp: float,
                 patience: int,
                 temperature: float,
                 max_tokens: int,
                 num_seqs: int,
                 selection_strategy: str,
                 top_p: float,
                 stop_words: List[str],
                 prompt_prefix: str,
                 prompt_path: str,
                 hint_prompt_path: str,
                 gpt3_metric: str,
                 gpt3_coef: float,
                 use_baseline: bool,
                 t5_metric: str,
                 t5_coef: float,
                 t5_pos_coef: float,
                 t5_neg_coef: float,
                 step_reward_coef: float,
                 split_token: str = ";",
                 split_token_id: int = 117 # token id of ";"
                ) -> None:
        super().__init__()
        self.gpt3 = GPT3(model=gpt3_model, interval=interval, timeout=timeout, exp=exp, patience=patience)
        # arguments for t5 reward
        self.split_token = split_token
        self.split_token_id = split_token_id
        self.t5_coef = t5_coef
        self.step_reward_coef = step_reward_coef
        self.t5_pos_coef = t5_pos_coef
        self.t5_neg_coef = t5_neg_coef
        # arguments for gpt3 inference
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_seqs = num_seqs
        self.top_p = top_p
        self.stop_words = stop_words
        self.prompt_prefix = prompt_prefix
        # arguments for gpt3 reward
        self.gpt3_coef = gpt3_coef
        self.use_baseline = use_baseline
        self.selection_strategy = selection_strategy
        # prompt for gpt3
        f = open(prompt_path, 'r') 
        self.prompt = f.read().strip()
        f = open(hint_prompt_path, 'r') 
        self.hint_prompt = f.read().strip()
        # metric for gpt3 and t5
        from rl4lms.envs.text_generation.registry import MetricRegistry
        # reward for gpt3:
        args = {}
        if gpt3_metric in ["rouge1", "rouge2", "rougeL"]:
            self.gpt3_score_keys = [f"lexical/rouge_{gpt3_metric}"] 
            gpt3_metric = "rouge"
        elif gpt3_metric == "rouge-avg":
            self.gpt3_score_keys = ["lexical/rouge_rouge1", "lexical/rouge_rouge2", "lexical/rouge_rougeL"] 
            gpt3_metric = "rouge"
        elif gpt3_metric == 'meteor':
            self.gpt3_score_keys = ["lexical/meteor"]
        elif gpt3_metric == 'bleu':
            self.gpt3_score_keys = ["lexical/bleu"]
        else:
            raise NotImplementedError
        self.gpt3_metric = MetricRegistry.get(gpt3_metric, args)
        # rewards for t5:
        args = {}
        if t5_metric == "rouge":
            self.t5_score_keys = ["lexical/rouge_rouge1", "lexical/rouge_rouge2", "lexical/rouge_rougeL"]
        elif t5_metric == "hint_hit":
            self.t5_score_keys = ["keyword/hint_hit"]
        else:
            raise NotImplementedError
        self.t5_metric = MetricRegistry.get(t5_metric, {})
        self.t5_metric_type = t5_metric

    @staticmethod
    def clean_generation(text, stop_words):
        text = text.strip()
        end_idx = len(text)
        for end_word in stop_words:
            if end_word in text:
                end_idx = min(end_idx, text.find(end_word))
        text = text[:end_idx]
        return text

    @staticmethod
    def gpt3_hint_generation(gpt3,
                             input: str, 
                             temperature: float,
                             max_tokens: int,
                             num_seqs: int,
                             top_p: float,
                             stop_words: List[str]):
            candidates = gpt3.call(prompt=input,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    n=num_seqs,
                                    top_p=top_p,
                                    stop=stop_words
                                    )
            candidates = [RewardSummarizationWithHint.clean_generation(candidate, stop_words+["\n\n", "\n"]) for candidate in candidates]
            return candidates
    
    @staticmethod
    def generation_selection(strategy: str,
                             candidates: List[str]) -> List[str]:
        if strategy == 'lcs':
            from string2string.edit_distance import EditDistAlgs
            algs_unit = EditDistAlgs()
            n = len(candidates)
            matrix = np.zeros((n,n))
            for j1, cand1 in enumerate(candidates):
                cand1_split = cand1.split(' ')
                for j2, cand2 in enumerate(candidates):
                    cand2_split = cand2.split(' ')
                    max_length = max(len(cand1_split), len(cand2_split))
                    dist, _ = algs_unit.longest_common_subsequence(
                        cand1_split,
                        cand2_split,
                        printBacktrack=False,
                        boolListOfList=True
                        )
                    score = dist / max_length
                    matrix[j1][j2] = score
            matrix = np.mean(matrix, axis=1)
            index = np.argmax(matrix)
            return [candidates[index]]
        
        elif strategy == 'choose_first':
            return [candidates[0]]
        
        elif strategy == 'choose_all':
            return candidates
        
        elif strategy == 'random':
            index = np.random.randint(0, len(candidates))
            return [candidates[index]]
        
        return candidates


    def __call__(self, prev_observation: Observation,
                 action: int,
                 current_observation: Observation,
                 done: bool,
                 meta_info: Dict[str, Any] = None) -> float:

        if done:
            references = [current_observation.target_or_reference_texts]
            meta_data = current_observation.meta_info
            phrases, target = meta_data['phrases'], [meta_data['target']]
            t5_gen_text = current_observation.context_text
            t5_input_text = current_observation.prompt_or_input_text.replace(self.prompt_prefix, "") # remove the prefix for t5 to get the original article
            print(t5_gen_text)
            
            # reward for t5
            if self.t5_coef != 0:
                if self.t5_metric_type == "rouge":
                    metric_dict = self.t5_metric.compute(None, [t5_gen_text], [references])
                    t5_reward = [metric_dict[k][1] for k in self.t5_score_keys]
                    t5_reward = np.mean(t5_reward)
                elif self.t5_metric_type == "hint_hit":
                    metric_dict = self.t5_metric.compute(None, [t5_gen_text], [references])
                    t5_reward = metric_dict["keyword/hint_hit"][1] - self.t5_neg_coef*metric_dict["keyword/hint_not_hit"][1]
                else:
                    raise NotImplementedError
            else: # avoid calculation if not used
                t5_reward = 0.
            
            # reward for gpt3
            if self.gpt3_coef != 0:
                # generate multiple outputs with hint
                gpt3_input_text = self.hint_prompt.replace("[[QUESTION]]", t5_input_text)
                gpt3_input_text = gpt3_input_text.replace("[[HINT]]", t5_gen_text)
                gpt3_hint_gen_texts = RewardSummarizationWithHint.gpt3_hint_generation(
                    self.gpt3, gpt3_input_text, self.temperature, self.max_tokens, self.num_seqs, self.top_p, self.stop_words)
                gpt3_hint_gen_texts = RewardSummarizationWithHint.generation_selection(self.selection_strategy, gpt3_hint_gen_texts)
                # reward for gpt3
                gpt3_rewards = []
                for i, gpt3_hint_gen_text in enumerate(gpt3_hint_gen_texts):
                    metric_dict = self.gpt3_metric.compute(None, [gpt3_hint_gen_text], [references])
                    gpt3_reward = [metric_dict[k][1] for k in self.gpt3_score_keys]
                    gpt3_reward = np.mean(gpt3_reward)
                    gpt3_rewards.append(gpt3_reward)
                gpt3_reward = np.mean(gpt3_rewards)
                
                if self.use_baseline:
                    # gpt3 generation
                    gpt3_input_text = self.prompt.replace("[[QUESTION]]", t5_input_text)
                    # only generate one without hint as baseline
                    gpt3_gen_texts = RewardSummarizationWithHint.gpt3_hint_generation(
                        self.gpt3, gpt3_input_text, self.temperature, self.max_tokens, self.num_seqs, self.top_p, self.stop_words)
                    gpt3_gen_texts = RewardSummarizationWithHint.generation_selection(self.selection_strategy, gpt3_gen_texts)
                    # baseline reward for gpt3
                    baseline_gpt3_rewards = []
                    for i, gpt3_gen_text in enumerate(gpt3_gen_texts):
                        metric_dict = self.gpt3_metric.compute(None, [gpt3_gen_text], [references])
                        baseline_gpt3_reward = [metric_dict[k][1] for k in self.gpt3_score_keys]
                        baseline_gpt3_reward = np.mean(baseline_gpt3_reward)
                        baseline_gpt3_rewards.append(baseline_gpt3_reward)
                    baseline_gpt3_reward = np.mean(baseline_gpt3_rewards)
                    # the improvement over baseline as reward
                    gpt3_reward = 10*(gpt3_reward - baseline_gpt3_reward)
            else:
                gpt3_reward = 0.

            # combine reward for gpt3 and t5
            reward = self.gpt3_coef*gpt3_reward + self.t5_coef*t5_reward
            print(f"gpt3: {gpt3_reward}, t5: {t5_reward}, total: {reward}")
            return reward

        # use step-wise reward
        elif self.step_reward_coef > 0. and action == self.split_token_id:
            t5_gen_text = current_observation.context_text.lower()
            reference = current_observation.target_or_reference_texts[0].lower()
            references = current_observation.target_or_reference_texts
            t5_gen_hints = t5_gen_text.split(self.split_token)[:-1]
            t5_gen_hint = t5_gen_hints[-1].strip()
            history_gen_hints = self.split_token.join(t5_gen_hints[:-1]) if len(t5_gen_hints) >= 2 else ""
            if self.t5_metric_type == 'hint_hit':
                # none generation
                if t5_gen_hint == "":
                    t5_reward = self.t5_neg_coef
                # meaningless generation
                elif t5_gen_hint in avoid_keywords:
                    t5_reward = self.t5_neg_coef
                # repeated generation
                elif t5_gen_hint in history_gen_hints:
                    t5_reward = self.t5_neg_coef
                # evaluate generation
                elif self.t5_metric_type == "hint_hit":
                    if t5_gen_hint in reference:
                        t5_reward = self.t5_pos_coef
                    else:
                        t5_reward = self.t5_neg_coef
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            t5_reward = self.step_reward_coef*t5_reward     
            return t5_reward
        
        return 0


class RewardMultiWOZWithHint(RewardFunction):
    def __init__(self,
                 gpt3_model: str,
                 interval: float,
                 timeout: float,
                 exp: float,
                 patience: int,
                 temperature: float,
                 max_tokens: int,
                 num_seqs: int,
                 selection_strategy: str,
                 top_p: float,
                 stop_words: List[str],
                 gpt3_metric: str,
                 gpt3_coef: float,
                 use_baseline: bool,
                 prompt_path: str,
                 hint_prompt_path: str,
                 t5_metric: str,
                 t5_coef: float,
                 split_token: str = ";",
                 split_token_id: int = 117,
                 user_prefix: str = "User: ",
                 system_prefix: str = "Assistant: ",
                 system_hint_prefix: str = "Assistant([[HINT]]): "
            ) -> None:
        super().__init__()
        self.gpt3 = GPT3(model=gpt3_model, interval=interval, timeout=timeout, exp=exp, patience=patience)
        # arguments for t5 reward
        self.t5_coef = t5_coef
        self.gpt3_coef = gpt3_coef
        # arguments for gpt3 inference
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_seqs = num_seqs
        self.top_p = top_p
        self.stop_words = stop_words
        self.split_token = split_token
        self.split_token_id = split_token_id
        self.user_prefix = user_prefix
        self.system_prefix = system_prefix
        self.system_hint_prefix = system_hint_prefix
        # arguments for gpt3 reward
        self.use_baseline = use_baseline
        self.selection_strategy = selection_strategy
        # prompt for gpt3
        f = open(prompt_path, 'r') 
        self.prompt = f.read().strip()
        f = open(hint_prompt_path, 'r') 
        self.hint_prompt = f.read().strip()
        # metric for gpt3 and t5
        from rl4lms.envs.text_generation.registry import MetricRegistry
        # reward for gpt3:
        if gpt3_metric == "rouge":
            self.gpt3_score_keys = ["lexical/rouge_rouge1", "lexical/rouge_rouge2", "lexical/rouge_rougeL", "lexical/rouge_rougeLsum"]  
        elif gpt3_metric == "google_bleu":
            self.gpt3_score_keys = ["lexical/google_bleu"] 
        elif gpt3_metric == "sacre_bleu":
            self.gpt3_score_keys = ["lexical/sacrebleu"]  
        elif gpt3_metric == "meteor":
            self.gpt3_score_keys = ["lexical/meteor"]  
        else:
            raise NotImplementedError
        self.gpt3_metric = MetricRegistry.get(gpt3_metric, {})
        # rewards for t5:
        if t5_metric == "dialog_act_accuracy":
            self.t5_score_keys = ["act/accuracy"]
        else:
            raise NotImplementedError
        self.t5_metric = MetricRegistry.get(t5_metric, {})
        self.t5_metric_type = t5_metric
        
    def __call__(self, prev_observation: Observation,
                 action: int,
                 current_observation: Observation,
                 done: bool,
                 meta_info: Dict[str, Any] = None) -> float:

        if done:
            # references = [current_observation.target_or_reference_texts]
            references = current_observation.target_or_reference_texts
            meta_data = current_observation.meta_info
            da_output = meta_data['da_output']
            current_user, current_resp = meta_data['user'], meta_data['resp']
            turn_id = meta_data['turn_id']
            history_users, history_resps, history_acts = meta_data['history_users'], meta_data['history_resps'], meta_data['history_acts']
            t5_gen_text = current_observation.context_text
            t5_gen_text = t5_gen_text.split(self.split_token)[0].strip()
            print(f"t5 gen: {t5_gen_text}, target: {da_output}")

            # construct the prompt for GPT-3
            dialog, dialog_with_hint = "", ""
            for user, resp, da in zip(history_users[:turn_id], history_resps[:turn_id], history_acts[:turn_id]):
                dialog += self.user_prefix + " " + user + "\n" + self.system_prefix + " " + resp + "\n"
                dialog_with_hint += self.user_prefix + " " + user + "\n" + self.system_hint_prefix.replace("[[HINT]]", da) + " " + resp + "\n"
            # current turn, given the generated intent/emotion
            dialog += self.user_prefix + " " + current_user + "\n" + self.system_prefix
            dialog_with_hint += self.user_prefix + " " + current_user + "\n" + self.system_hint_prefix.replace("[[HINT]]", t5_gen_text)

            # reward for t5
            if self.t5_coef != 0:
                if self.t5_metric_type == "dialog_act_accuracy":
                    metric_dict = self.t5_metric.compute(None, [t5_gen_text], [[da_output]])
                    t5_reward = [metric_dict[k][1] for k in self.t5_score_keys]
                    t5_reward = np.mean(t5_reward)
                else:
                    raise NotImplementedError
            else: # avoid calculation if not used
                t5_reward = 0.
            
            if self.gpt3_coef != 0:
                # generate multiple outputs with hint
                gpt3_input_text = self.hint_prompt.replace("[[DIALOG]]", dialog_with_hint)
                gpt3_hint_gen_texts = RewardSummarizationWithHint.gpt3_hint_generation(
                    self.gpt3, gpt3_input_text, self.temperature, self.max_tokens, self.num_seqs, self.top_p, self.stop_words)
                gpt3_hint_gen_texts = RewardSummarizationWithHint.generation_selection(self.selection_strategy, gpt3_hint_gen_texts)
                # reward for gpt3
                gpt3_rewards = []
                for i, gpt3_hint_gen_text in enumerate(gpt3_hint_gen_texts):
                    metric_dict = self.gpt3_metric.compute(None, [gpt3_hint_gen_text], [references])
                    gpt3_reward = [metric_dict[k][1] for k in self.gpt3_score_keys]
                    gpt3_reward = np.mean(gpt3_reward)
                    gpt3_rewards.append(gpt3_reward)
                gpt3_reward = np.mean(gpt3_rewards)
                
                if self.use_baseline:
                    # gpt3 generation
                    gpt3_input_text = self.prompt.replace("[[DIALOG]]", dialog)
                    # only generate one without hint as baseline
                    gpt3_gen_texts = RewardSummarizationWithHint.gpt3_hint_generation(
                        self.gpt3, gpt3_input_text, self.temperature, self.max_tokens, self.num_seqs, self.top_p, self.stop_words)
                    gpt3_gen_texts = RewardSummarizationWithHint.generation_selection(self.selection_strategy, gpt3_gen_texts)
                    # baseline reward for gpt3
                    baseline_gpt3_rewards = []
                    for i, gpt3_gen_text in enumerate(gpt3_gen_texts):
                        metric_dict = self.gpt3_metric.compute(None, [gpt3_gen_text], [references])
                        baseline_gpt3_reward = [metric_dict[k][1] for k in self.gpt3_score_keys]
                        baseline_gpt3_reward = np.mean(baseline_gpt3_reward)
                        baseline_gpt3_rewards.append(baseline_gpt3_reward)
                    baseline_gpt3_reward = np.mean(baseline_gpt3_rewards)
                    # the improvement over baseline as reward
                    gpt3_reward = 10*(gpt3_reward - baseline_gpt3_reward)
            else:
                gpt3_reward = 0.
            
            # add reward for gpt3 and t5
            reward = self.gpt3_coef*gpt3_reward + self.t5_coef*t5_reward
            print(f"gpt3: {gpt3_reward}, t5: {t5_reward}, total: {reward}")
            return reward

        return 0