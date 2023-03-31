
from typing import Any, Dict, List

import numpy as np
from rl4lms.envs.text_generation.metric import BaseMetric
from rl4lms.envs.text_generation.test_reward import (RewardIncreasingNumbers,
                                                     RewardSentencesWithDates,
                                                     RewardSummarizationWithHint,
                                                     GPT3)
from rl4lms.envs.text_generation.metric import MultiWOZMetric
from transformers import PreTrainedModel


class IncreasingNumbersinText(BaseMetric):
    def __init__(self, min_tokens: int) -> None:
        super().__init__()
        self._min_tokens = min_tokens

    def compute(self, prompt_texts: List[str],
                generated_texts: List[str],
                reference_texts: List[List[str]],
                meta_infos: List[Dict[str, Any]] = None,
                model: PreTrainedModel = None,
                split_name: str = None) -> Dict[str, float]:

        all_rewards = []
        for gen_text in generated_texts:
            reward = RewardIncreasingNumbers.reward_increasing_numbers_in_text(
                gen_text, self._min_tokens)
            all_rewards.append(reward)

        metric_dict = {
            "synthetic/increasing_numbers_in_text": (all_rewards, np.mean(all_rewards))
        }
        return metric_dict


class DateInText(BaseMetric):
    def compute(self, prompt_texts: List[str],
                generated_texts: List[str],
                reference_texts: List[List[str]],
                meta_infos: List[Dict[str, Any]] = None,
                model: PreTrainedModel = None,
                split_name: str = None) -> Dict[str, float]:

        all_rewards = []
        for gen_text in generated_texts:
            reward = RewardSentencesWithDates.date_in_text(
                gen_text)
            all_rewards.append(reward)

            
        metric_dict = {
            "synthetic/dates_in_text": (all_rewards, np.mean(all_rewards))
        }
        return metric_dict


class SummarizationWithHintMetric(BaseMetric):
    def __init__(self,
                 gpt3_model: str,
                 interval: float,
                 timeout: float,
                 exp: float,
                 patience: int, 
                 split_token: str,
                 split_token_id: int,
                 temperature: float,
                 max_tokens: int,
                 num_seqs: int,
                 selection_strategy: str,
                 top_p: float,
                 stop_words: List[str],
                 prompt_prefix: str,
                 prompt_path: str,
                 hint_prompt_path: str,
                 gpt3_metrics: List[dict],
                 t5_metrics: List[dict],
                 use_upper_baseline: bool = False,
                 use_lower_baseline: bool = False,
                 ) -> None:
        super().__init__()
        self.gpt3 = GPT3(model=gpt3_model, interval=interval, timeout=timeout, exp=exp, patience=patience)
        self.use_upper_baseline = use_upper_baseline
        self.use_lower_baseline = use_lower_baseline
        self.prompt_prefix = prompt_prefix
        self.selection_strategy = selection_strategy
        # generation arguments for gpt3
        self.temperature = temperature
        self.split_token = split_token
        self.split_token_id = split_token_id
        self.max_tokens = max_tokens
        self.num_seqs = num_seqs
        self.top_p = top_p
        self.stop_words = stop_words
        # prompt for gpt3
        f = open(prompt_path, 'r') 
        self.prompt = f.read().strip()
        f = open(hint_prompt_path, 'r') 
        self.hint_prompt = f.read().strip()

        # metric for gpt3 and t5
        from rl4lms.envs.text_generation.registry import MetricRegistry

        # Multiple metrics for gpt3
        self.gpt3_metrics, self.gpt3_metric_types, self.gpt3_score_keys = [], [], []
        for gpt3_metric in gpt3_metrics:
            gpt3_metric_type, gpt3_metric_args = gpt3_metric['id'], gpt3_metric['args']
            if gpt3_metric_type == 'rouge':
                gpt3_score_keys = ["lexical/rouge_rouge1", "lexical/rouge_rouge2", "lexical/rouge_rougeL", "lexical/rouge_rougeLsum"]
            elif gpt3_metric_type == 'meteor':
                gpt3_score_keys = ["lexical/meteor"]
            elif gpt3_metric_type == 'bleu':
                gpt3_score_keys = ["lexical/bleu"]
            elif gpt3_metric_type == 'bert_score':
                gpt3_score_keys = ['semantic/bert_score']
            elif gpt3_metric_type == 'summaCZS':
                gpt3_score_keys = ['consistency/summaczs']
            else:
                raise NotImplementedError
            self.gpt3_metrics.append(MetricRegistry.get(gpt3_metric_type, gpt3_metric_args))
            self.gpt3_metric_types.append(gpt3_metric_type)
            self.gpt3_score_keys.append(gpt3_score_keys)

        # Multiple metrics for t5
        self.t5_metrics, self.t5_metric_types, self.t5_score_keys = [], [], []
        for t5_metric in t5_metrics:
            t5_metric_type, t5_metric_args = t5_metric['id'], t5_metric['args']
            if t5_metric_type == 'rouge':
                t5_score_keys = ["lexical/rouge_rouge1", "lexical/rouge_rouge2"]
            elif t5_metric_type == 'hint_hit':
                t5_score_keys = ["keyword/hint_hit", "keyword/hint_num"]
            elif t5_metric_type == 'hint_bleu':
                t5_score_keys = ["keyword/hint_bleu"]
            else:
                raise NotImplementedError
            self.t5_metrics.append(MetricRegistry.get(t5_metric_type, t5_metric_args))
            self.t5_metric_types.append(t5_metric_type)
            self.t5_score_keys.append(t5_score_keys)


    def compute(self, prompt_texts: List[str],
                generated_texts: List[str],
                reference_texts: List[List[str]],
                meta_infos: List[Dict[str, Any]] = None,
                model: PreTrainedModel = None,
                split_name: str = None) -> Dict[str, float]:

        gpt3_generated_texts, upper_gpt3_generated_texts, lower_gpt3_generated_texts = [], [], []
        gpt3_rewards = [[[] for _ in range(len(gpt3_score_keys))] for gpt3_score_keys in self.gpt3_score_keys]
        lower_baseline_gpt3_rewards = [[[] for _ in range(len(gpt3_score_keys))] for gpt3_score_keys in self.gpt3_score_keys]
        upper_baseline_gpt3_rewards = [[[] for _ in range(len(gpt3_score_keys))] for gpt3_score_keys in self.gpt3_score_keys]
        t5_rewards = [[[] for _ in range(len(t5_score_keys))] for t5_score_keys in self.t5_score_keys]
        
        for i, t5_gen_text in enumerate(generated_texts):
            prompt_text = prompt_texts[i] # str
            reference_text = reference_texts[i] # List
            meta_info = meta_infos[i]
            phrases, target = meta_info['phrases'], meta_info['target']
            t5_input_text = prompt_text.replace(self.prompt_prefix, "") # remove the prefix for t5

            # rewards for t5
            for j, t5_metric in enumerate(self.t5_metrics):
                t5_metric_type = self.t5_metric_types[j]
                t5_score_keys = self.t5_score_keys[j]
                metric_results = t5_metric.compute(None, [t5_gen_text], [reference_text])
                for k, score_key in enumerate(t5_score_keys):
                    score = metric_results[score_key][1]
                    score = 0 if score == 'n/a' else score
                    t5_rewards[j][k].append(score)
                    print(f"{score_key}: {score}")

            # gpt3 generation
            gpt3_input_text = self.hint_prompt.replace("[[QUESTION]]", t5_input_text)
            gpt3_input_text = gpt3_input_text.replace("[[HINT]]", t5_gen_text)
            gpt3_gen_texts = RewardSummarizationWithHint.gpt3_hint_generation(
                self.gpt3, gpt3_input_text, self.temperature, self.max_tokens, self.num_seqs, self.top_p, self.stop_words)
            gpt3_gen_texts = RewardSummarizationWithHint.generation_selection(self.selection_strategy, gpt3_gen_texts)
            gpt3_generated_texts.append(gpt3_gen_texts)

            # reward for gpt3
            for j, gpt3_metric in enumerate(self.gpt3_metrics):
                gpt3_score_keys = self.gpt3_score_keys[j]
                gpt3_scores = [[] for _ in range(len(gpt3_score_keys))]
                for g, gpt3_gen_text in enumerate(gpt3_gen_texts):
                    metric_results = gpt3_metric.compute([t5_input_text], [gpt3_gen_text], [reference_text])
                    for k, score_key in enumerate(gpt3_score_keys):
                        score = metric_results[score_key][1]
                        score = 0 if score == 'n/a' else score
                        gpt3_scores[k].append(score)
                # average score
                for k, score_key in enumerate(gpt3_score_keys):
                    avg_score = np.mean(gpt3_scores[k])
                    gpt3_rewards[j][k].append(avg_score)
                    print(f"{score_key}: {avg_score}")

            if self.use_lower_baseline:
                gpt3_input_text = self.prompt.replace("[[QUESTION]]", t5_input_text)
                gpt3_gen_texts = RewardSummarizationWithHint.gpt3_hint_generation(
                    self.gpt3, gpt3_input_text, self.temperature, self.max_tokens, self.num_seqs, self.top_p, self.stop_words)
                gpt3_gen_texts = RewardSummarizationWithHint.generation_selection(self.selection_strategy, gpt3_gen_texts)
                lower_gpt3_generated_texts.append(gpt3_gen_texts)

                # reward for gpt3
                for j, gpt3_metric in enumerate(self.gpt3_metrics):
                    gpt3_score_keys = self.gpt3_score_keys[j]
                    gpt3_scores = [[] for _ in range(len(gpt3_score_keys))]
                    for g, gpt3_gen_text in enumerate(gpt3_gen_texts):
                        metric_results = gpt3_metric.compute([t5_input_text], [gpt3_gen_text], [reference_text])
                        for k, score_key in enumerate(gpt3_score_keys):
                            score = metric_results[score_key][1]
                            score = 0 if score == 'n/a' else score
                            gpt3_scores[k].append(score)
                    # average score
                    for k, score_key in enumerate(gpt3_score_keys):
                        avg_score = np.mean(gpt3_scores[k])
                        lower_baseline_gpt3_rewards[j][k].append(avg_score)
                        print(f"{score_key}: {avg_score}")
            
            if self.use_upper_baseline:
                gpt3_input_text = self.hint_prompt.replace("[[QUESTION]]", t5_input_text)
                gpt3_input_text = gpt3_input_text.replace("[[HINT]]", target)
                gpt3_gen_texts = RewardSummarizationWithHint.gpt3_hint_generation(
                    self.gpt3, gpt3_input_text, self.temperature, self.max_tokens, self.num_seqs, self.top_p, self.stop_words)
                gpt3_gen_texts = RewardSummarizationWithHint.generation_selection(self.selection_strategy, gpt3_gen_texts)
                upper_gpt3_generated_texts.append(gpt3_gen_texts)

                # reward for gpt3
                for j, gpt3_metric in enumerate(self.gpt3_metrics):
                    gpt3_score_keys = self.gpt3_score_keys[j]
                    gpt3_scores = [[] for _ in range(len(gpt3_score_keys))]
                    for g, gpt3_gen_text in enumerate(gpt3_gen_texts):
                        metric_results = gpt3_metric.compute([t5_input_text], [gpt3_gen_text], [reference_text])
                        for k, score_key in enumerate(gpt3_score_keys):
                            score = metric_results[score_key][1]
                            score = 0 if score == 'n/a' else score
                            gpt3_scores[k].append(score)
                    # average score
                    for k, score_key in enumerate(gpt3_score_keys):
                        avg_score = np.mean(gpt3_scores[k])
                        upper_baseline_gpt3_rewards[j][k].append(avg_score)
                        print(f"{score_key}: {avg_score}")

        metric_dict = {}
        # metrics for t5
        for i, score_keys in enumerate(self.t5_score_keys):
            for j, score_key in enumerate(score_keys):
                metric_dict[f"t5/{score_key}"] = (t5_rewards[i][j], np.mean(t5_rewards[i][j]))
        
        # metric for gpt3
        for i, score_keys in enumerate(self.gpt3_score_keys):
            for j, score_key in enumerate(score_keys):
                metric_dict[f"gpt3/{score_key}"] = (gpt3_rewards[i][j], np.mean(gpt3_rewards[i][j]))
        metric_dict["gpt3_generated_text"] = (gpt3_generated_texts, 0.)

        # metric for baseline gpt3
        if self.use_lower_baseline:
            for i, score_keys in enumerate(self.gpt3_score_keys):
                for j, score_key in enumerate(score_keys):
                    metric_dict[f"lower_baseline_gpt3/{score_key}"] = (lower_baseline_gpt3_rewards[i][j], np.mean(lower_baseline_gpt3_rewards[i][j]))
            metric_dict["lower_gpt3_generated_text"] = (lower_gpt3_generated_texts, 0.)

        if self.use_upper_baseline:
            for i, score_keys in enumerate(self.gpt3_score_keys):
                for j, score_key in enumerate(score_keys):
                    metric_dict[f"upper_baseline_gpt3/{score_key}"] = (upper_baseline_gpt3_rewards[i][j], np.mean(upper_baseline_gpt3_rewards[i][j]))
            metric_dict["upper_gpt3_generated_text"] = (upper_gpt3_generated_texts, 0.)

        return metric_dict


class MultiWOZWithHintMetric(BaseMetric):
    def __init__(self,
                 gpt3_model: str,
                 interval: float,
                 timeout: float,
                 exp: float,
                 patience: int,
                 temperature: float,
                 split_token: str,
                 split_token_id: int,
                 max_tokens: int,
                 num_seqs: int,
                 selection_strategy: str,
                 top_p: float,
                 stop_words: List[str],
                 prompt_path: str,
                 hint_prompt_path: str,
                 gpt3_metric: str,
                 multiwoz_version: str,
                 t5_metrics: List[str],
                 use_lower_baseline: bool = False,
                 use_upper_baseline: bool = False,
                 user_prefix: str = "User: ",
                 system_prefix: str = "Assistant: ",
                 system_hint_prefix: str = "Assistant([[HINT]]): "
                 ) -> None:
        super().__init__()
        self.gpt3 = GPT3(model=gpt3_model, interval=interval, timeout=timeout, exp=exp, patience=patience)
        self.use_lower_baseline = use_lower_baseline
        self.use_upper_baseline = use_upper_baseline
        self.selection_strategy = selection_strategy
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
        # prompt for gpt3
        f = open(prompt_path, 'r') 
        self.prompt = f.read().strip()
        f = open(hint_prompt_path, 'r') 
        self.hint_prompt = f.read().strip()

        # metric for gpt3 and t5
        from rl4lms.envs.text_generation.registry import MetricRegistry

        # metric for gpt3:
        if gpt3_metric == "multiwoz":
            self.gpt3_score_keys = ["multiwoz/bleu", "multiwoz/success", "multiwoz/inform", "multiwoz/combined_score"]
            self.gpt3_metric = MultiWOZMetric(dataset_version=multiwoz_version)
        else:
            raise NotImplementedError
        
        # Multiple metric for t5
        self.t5_metrics, self.t5_metric_types, self.t5_score_keys = [], [], []
        for t5_metric in t5_metrics:
            if t5_metric == "dialog_act_accuracy":
                t5_score_keys = ["act/accuracy"]
            else:
                raise NotImplementedError
            self.t5_metrics.append(MetricRegistry.get(t5_metric, {}))
            self.t5_metric_types.append(t5_metric)
            self.t5_score_keys.append(t5_score_keys)

    def compute(self, prompt_texts: List[str],
                generated_texts: List[str],
                reference_texts: List[List[str]],
                meta_infos: List[Dict[str, Any]] = None,
                model: PreTrainedModel = None,
                split_name: str = None) -> Dict[str, float]:
        
        gpt3_generated_texts, upper_gpt3_generated_texts, lower_gpt3_generated_texts = [], [], []
        gpt3_rewards = [[] for _ in range(len(self.gpt3_score_keys))]
        lower_baseline_gpt3_rewards = [[] for _ in range(len(self.gpt3_score_keys))]
        upper_baseline_gpt3_rewards = [[] for _ in range(len(self.gpt3_score_keys))]
        t5_rewards = [[[] for _ in range(len(t5_score_keys))] for t5_score_keys in self.t5_score_keys]
        for i, t5_gen_text in enumerate(generated_texts):
            t5_gen_text = t5_gen_text.split(self.split_token)[0].strip()
            prompt_text = prompt_texts[i]
            reference_text = reference_texts[i]
            meta_data = meta_infos[i]
            da_output = meta_data['da_output']
            current_user, current_resp = meta_data['user'], meta_data['resp']
            turn_id = meta_data['turn_id']
            history_users, history_resps, history_acts = meta_data['history_users'], meta_data['history_resps'], meta_data['history_acts']

            # construct the prompt for GPT-3
            dialog, dialog_with_hint = "", ""
            for user, resp, da in zip(history_users[:turn_id], history_resps[:turn_id], history_acts[:turn_id]):
                dialog += self.user_prefix + " " + user + "\n" + self.system_prefix + " " + resp + "\n"
                dialog_with_hint += self.user_prefix + " " + user + "\n" + self.system_hint_prefix.replace("[[HINT]]", da) + " " + resp + "\n"
            # current turn, given the generated intent/emotion
            dialog += self.user_prefix + " " + current_user + "\n" + self.system_prefix
            dialog_with_reference_hint = dialog_with_hint + self.user_prefix + " " + current_user + "\n" + self.system_hint_prefix.replace("[[HINT]]", da_output)
            dialog_with_predicted_hint = dialog_with_hint + self.user_prefix + " " + current_user + "\n" + self.system_hint_prefix.replace("[[HINT]]", t5_gen_text)

            # rewards for t5
            for j, t5_metric in enumerate(self.t5_metrics):
                t5_metric_type = self.t5_metric_types[j]
                t5_score_keys = self.t5_score_keys[j]
                if t5_metric_type == "dialog_act_accuracy":
                    metric_results = t5_metric.compute(None, [t5_gen_text], [da_output])
                else:
                    raise NotImplementedError
                for k, score_key in enumerate(t5_score_keys):
                    score = metric_results[score_key][1]
                    t5_rewards[j][k].append(score)
                    print(f"{score_key}: {score}")

            # gpt3 generation
            gpt3_input_text = self.hint_prompt.replace("[[DIALOG]]", dialog_with_predicted_hint)
            gpt3_gen_texts = RewardSummarizationWithHint.gpt3_hint_generation(
                self.gpt3, gpt3_input_text, self.temperature, self.max_tokens, self.num_seqs, self.top_p, self.stop_words)
            gpt3_gen_texts = RewardSummarizationWithHint.generation_selection(self.selection_strategy, gpt3_gen_texts)
            gpt3_generated_texts.append(gpt3_gen_texts)

            # lower bound for gpt3, not using hints
            if self.use_lower_baseline:
                gpt3_input_text = self.prompt.replace("[[DIALOG]]", dialog)
                gpt3_gen_texts = RewardSummarizationWithHint.gpt3_hint_generation(
                    self.gpt3, gpt3_input_text, self.temperature, self.max_tokens, self.num_seqs, self.top_p, self.stop_words)
                gpt3_gen_texts = RewardSummarizationWithHint.generation_selection(self.selection_strategy, gpt3_gen_texts)
                lower_gpt3_generated_texts.append(gpt3_gen_texts)
            
            # upper bound for gpt3, using reference hints
            if self.use_upper_baseline:
                gpt3_input_text = self.hint_prompt.replace("[[DIALOG]]", dialog_with_reference_hint)
                gpt3_gen_texts = RewardSummarizationWithHint.gpt3_hint_generation(
                    self.gpt3, gpt3_input_text, self.temperature, self.max_tokens, self.num_seqs, self.top_p, self.stop_words)
                gpt3_gen_texts = RewardSummarizationWithHint.generation_selection(self.selection_strategy, gpt3_gen_texts)
                upper_gpt3_generated_texts.append(gpt3_gen_texts)

        # evaluate on the corpus level
        for g in range(len(gpt3_generated_texts[0])):
            gpt3_generated_text = [gen_texts[g] for gen_texts in gpt3_generated_texts]
            metric_results = self.gpt3_metric.compute(None, gpt3_generated_text, None, meta_infos)
            for j, score_key in enumerate(self.gpt3_score_keys):
                gpt3_rewards[j].append(metric_results[score_key][1])
        # average over multiple inferences
        for j, score_key in enumerate(self.gpt3_score_keys):
            gpt3_rewards[j] = np.mean(gpt3_rewards[j])

        if self.use_lower_baseline:
            for g in range(len(lower_gpt3_generated_texts[0])):
                lower_gpt3_generated_text = [gen_texts[g] for gen_texts in lower_gpt3_generated_texts]
                metric_results = self.gpt3_metric.compute(None, lower_gpt3_generated_text, None, meta_infos)
                for j, score_key in enumerate(self.gpt3_score_keys):
                    lower_baseline_gpt3_rewards[j].append(metric_results[score_key][1])
            # average over multiple inferences
            for j, score_key in enumerate(self.gpt3_score_keys):
                lower_baseline_gpt3_rewards[j] = np.mean(lower_baseline_gpt3_rewards[j])

        if self.use_upper_baseline:
            for g in range(len(upper_gpt3_generated_texts[0])):
                upper_gpt3_generated_text = [gen_texts[g] for gen_texts in upper_gpt3_generated_texts]
                metric_results = self.gpt3_metric.compute(None, upper_gpt3_generated_text, None, meta_infos)
                for j, score_key in enumerate(self.gpt3_score_keys):
                    upper_baseline_gpt3_rewards[j].append(metric_results[score_key][1])
            # average over multiple inferences
            for j, score_key in enumerate(self.gpt3_score_keys):
                upper_baseline_gpt3_rewards[j] = np.mean(upper_baseline_gpt3_rewards[j])

        metric_dict = {}
        # metrics for t5
        for i, score_keys in enumerate(self.t5_score_keys):
            for j, score_key in enumerate(score_keys):
                metric_dict[f"t5_{score_key}"] = (t5_rewards[i][j], np.mean(t5_rewards[i][j]))

        # metric for gpt3
        for i, score_key in enumerate(self.gpt3_score_keys):
            metric_dict[f"gpt3_{score_key}"] = (None, gpt3_rewards[i])
        metric_dict["gpt3_generated_text"] = (gpt3_generated_texts, 0.)

        # metric for lower and upper baseline gpt3
        if self.use_lower_baseline:
            for i, score_key in enumerate(self.gpt3_score_keys):
                metric_dict[f"lower_baseline_gpt3_{score_key}"] = (None, lower_baseline_gpt3_rewards[i])
            metric_dict["lower_gpt3_generated_text"] = (lower_gpt3_generated_texts, 0.)

        if self.use_upper_baseline:
            for i, score_key in enumerate(self.gpt3_score_keys):
                metric_dict[f"upper_baseline_gpt3_{score_key}"] = (None, upper_baseline_gpt3_rewards[i])
            metric_dict["upper_gpt3_generated_text"] = (upper_gpt3_generated_texts, 0.)

        return metric_dict

