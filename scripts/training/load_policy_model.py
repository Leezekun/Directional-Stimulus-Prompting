import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from huggingface_hub import create_repo

# load language model and push to the hub
model_name = 't5-base'
tokenizer_name = 't5-base'
dataset_name ="cnndm"
fs_ratio = 0.01

model_path = '$PROJECT_PATH/rl4lms_exps/summarization_with_hint/'
project_name = 'flan-t5_nlpo_on_supervised_2000-final'

'''
load best policy model
'''
# model_path = os.path.join(model_path, project_name, "model")
# print(model_path)
# hub_path = f"{model_name}-extraction-{dataset_name}_fs{fs_ratio}-h-ppo"
# model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
# model.push_to_hub(hub_path)

'''
load policy, value, and trainer state from checkpoints
'''
for iter in [1,3,5,7,9]:
    ckpt_path = os.path.join(model_path, project_name, "checkpoints", f"checkpoint_{iter}")
    state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
    policy_state_dict = state_dict["policy_state"]
    alg_state_dict = state_dict["alg_state"]
    trainer_state = state_dict["trainer_state"]
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    model.load_state_dict(policy_state_dict["policy_model"])

    # saved in local file
    save_path = os.path.join(model_path, project_name, "checkpoints_policy", f"checkpoint_{iter}")
    model.save_pretrained(save_path, from_pt=True)

# push to huggingface hub
# hub_path = f"Zekunli/{project_name}-ppo-ckpt{iter}"
# model.push_to_hub(hub_path)