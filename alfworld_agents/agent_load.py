import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
os.environ['HF_HOME'] = '/scratch/svaidy33/hf_cache'

def load_model(model_name='google/gemma-3-4b-it'):
    
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" # Automatically use GPU if available
        )
        return model, tokenizer
 
 