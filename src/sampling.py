import torch
import torch.nn.functional as F

def sample_token(logits, temperature=1.0, top_k=50, top_p=0.9):
    logits = logits / (temperature + 1e-10)

    if top_k > 0:
        top_k_values, _ = torch.topk(logits, top_k)
        min_value = top_k_values[..., -1].unsqueeze(-1)
        logits = torch.where(logits < min_value, torch.tensor(float('-inf')).to(logits.device), logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        
        
        sorted_indices_to_remove[..., 0] = False 
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')

    probs = F.softmax(logits, dim=-1)
    
    if torch.isnan(probs).any():
        probs = torch.ones_like(probs) / probs.shape[-1]
        
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()

def verify_and_sample(target_logits, draft_logits, draft_token, temperature=1.0):
    target_probs = F.softmax(target_logits / temperature, dim=-1)
    draft_probs  = F.softmax(draft_logits / temperature, dim=-1)

    p_target = target_probs[draft_token].item()
    p_draft  = draft_probs[draft_token].item()

    acceptance_prob = min(1.0, p_target / (p_draft + 1e-10))

    if torch.rand(1).item() < acceptance_prob:
        return True, draft_token
    else:
        residual_probs = torch.clamp(target_probs - draft_probs, min=0.0)
        residual_probs = residual_probs / (residual_probs.sum() + 1e-10)
        new_token = torch.multinomial(residual_probs, num_samples=1).item()
        return False, new_token