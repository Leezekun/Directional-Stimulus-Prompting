import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from huggingface_hub import create_repo

model_path = '../../rl4lms_exps/summarization_with_hint/'
project_name = 'flan-t5-large_nlpo_on_supervised_4000'
model_type = "google/flan-t5-large"

'''
load policy, value, and trainer state from checkpoints
'''
for idx in range(10):
    ckpt_path = os.path.join(model_path, project_name, "checkpoints", f"checkpoint_{idx}")
    state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
    policy_state_dict = state_dict["policy_state"]
    alg_state_dict = state_dict["alg_state"]
    trainer_state = state_dict["trainer_state"]
    model = AutoModelForSeq2SeqLM.from_pretrained(model_type)
    model.load_state_dict(policy_state_dict["policy_model"])

    # saved in local file
    iter = 2*idx+1
    save_path = os.path.join(model_path, project_name, "checkpoints_policy", f"checkpoint_{iter}")
    model.save_pretrained(save_path, from_pt=True)