import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import DEVICE, TARGET_MODEL_NAME, TEMPERATURE, TOP_K, TOP_P
from sampling import sample_token

class BaselineGenerator:
    """
    The 'Control Group'. 
    Generates text one token at a time using only the Target Model.
    """
    def __init__(self):
        print(f" Initializing Baseline (Standard) Generator on {DEVICE}...")
        self.device = DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME)
        
        print(f" Loading Target Model ({TARGET_MODEL_NAME})...")
        self.model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_NAME).to(self.device)
        self.model.eval()
        print(" Baseline Ready.")

    def generate(self, prompt, max_new_tokens=30):
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device).input_ids
        
        start_time = time.time()
        
        # The Slow Loop
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids)
                next_token_logits = outputs.logits[0, -1, :]
                
                next_token_id = sample_token(next_token_logits, TEMPERATURE, TOP_K, TOP_P)
                
                input_tensor = torch.tensor([[next_token_id]], device=self.device)
                input_ids = torch.cat([input_ids, input_tensor], dim=1)
                
                if next_token_id == self.tokenizer.eos_token_id:
                    break
        
        end_time = time.time()
        num_generated = input_ids.shape[1] - len(self.tokenizer(prompt).input_ids)
        speed = num_generated / (end_time - start_time)
        
        return speed, self.tokenizer.decode(input_ids[0], skip_special_tokens=True)