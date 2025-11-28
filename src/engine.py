import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import DEVICE, TARGET_MODEL_NAME, DRAFT_MODEL_NAME, TEMPERATURE, TOP_K, TOP_P
from sampling import sample_token, verify_and_sample

class SpeculativeEngine:
    def __init__(self):
        print(f" Initializing Engine on {DEVICE}...")
        self.device = DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME)
        
        print(f"   Loading Target Model ({TARGET_MODEL_NAME})...")
        self.target_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_NAME).to(self.device)
        self.target_model.eval()

        print(f"   Loading Draft Model ({DRAFT_MODEL_NAME})...")
        self.draft_model = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL_NAME).to(self.device)
        self.draft_model.eval()
        
        # Default metrics
        self.metrics = {
            'total_draft_tokens': 0, 
            'total_accepted_tokens': 0, 
            'acceptance_rate': 0.0
        }
        
        print(" Engine Ready.")

    def get_last_run_metrics(self):
        """Return metrics from the most recent generation."""
        return self.metrics.copy()

    def _generate_draft_tokens(self, input_ids, gamma):
        draft_tokens = []
        draft_logits_list = []
        current_input = input_ids.clone()

        for _ in range(gamma):
            with torch.no_grad():
                outputs = self.draft_model(current_input)
                next_token_logits = outputs.logits[0, -1, :]
                next_token_id = sample_token(next_token_logits, TEMPERATURE, TOP_K, TOP_P)
                
                draft_tokens.append(next_token_id)
                draft_logits_list.append(next_token_logits)
                
                next_token_tensor = torch.tensor([[next_token_id]], device=self.device)
                current_input = torch.cat([current_input, next_token_tensor], dim=1)
        
        return draft_tokens, draft_logits_list

    def _verify_tokens(self, input_ids, draft_tokens, draft_logits_list):
        accepted_tokens = []
        
        draft_tensor = torch.tensor([draft_tokens], device=self.device)
        full_input = torch.cat([input_ids, draft_tensor], dim=1)
        
        with torch.no_grad():
            target_outputs = self.target_model(full_input)
            target_logits_full = target_outputs.logits[0]

        start_pos = input_ids.shape[1] - 1
        
        for i, draft_token_id in enumerate(draft_tokens):
            current_target_logits = target_logits_full[start_pos + i]
            current_draft_logits  = draft_logits_list[i]
            
            accepted, final_token = verify_and_sample(
                current_target_logits, 
                current_draft_logits, 
                draft_token_id, 
                TEMPERATURE
            )
            
            accepted_tokens.append(final_token)
            if not accepted:
                break

        return accepted_tokens

    def generate(self, prompt, max_new_tokens=30, gamma=4):
        # Local metrics
        run_metrics = {'total_draft_tokens': 0, 'total_accepted_tokens': 0, 'acceptance_rate': 0.0}
        
        # 1. Tokenize 
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device).input_ids
        
        # 2. Start Timer 
        start_time = time.time()
        
        num_generated = 0
        original_input_len = input_ids.shape[1]
        
        while num_generated < max_new_tokens:
            draft_tokens, draft_logits = self._generate_draft_tokens(input_ids, gamma)
            accepted_tokens = self._verify_tokens(input_ids, draft_tokens, draft_logits)
            
            run_metrics['total_draft_tokens'] += gamma
            run_metrics['total_accepted_tokens'] += len(accepted_tokens)
            
            accepted_tensor = torch.tensor([accepted_tokens], device=self.device)
            input_ids = torch.cat([input_ids, accepted_tensor], dim=1)
            num_generated += len(accepted_tokens)
            
            if self.tokenizer.eos_token_id in accepted_tokens:
                break

        # 3. Stop Timer
        end_time = time.time()
        
        # Calculate Speed
        actual_tokens_gen = input_ids.shape[1] - original_input_len
        total_time = end_time - start_time
        if total_time == 0: total_time = 0.001
        speed = actual_tokens_gen / total_time
        
        # Update metrics
        if run_metrics['total_draft_tokens'] > 0:
            run_metrics['acceptance_rate'] = run_metrics['total_accepted_tokens'] / run_metrics['total_draft_tokens']
        
        self.metrics = run_metrics
        print(f"📊 Run Stats: Acceptance Rate: {self.metrics['acceptance_rate']:.2%}")
        
        # Return Tuple: (Speed, Text)
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return speed, text