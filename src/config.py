import torch


# We use a large "Target" model and a small "Draft" model.
# They MUST share the same tokenizer for this to work easily.
TARGET_MODEL_NAME = "facebook/opt-1.3b" 
DRAFT_MODEL_NAME  = "facebook/opt-125m"

# Automatically detect if we have a GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Generation Parameters
TEMPERATURE = 1.0
TOP_K = 50
TOP_P = 0.9