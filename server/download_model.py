import torch
from transformers import AutoProcessor, SeamlessM4Tv2Model

MODEL_ID = "facebook/seamless-m4t-v2-large"
AutoProcessor.from_pretrained(MODEL_ID)
SeamlessM4Tv2Model.from_pretrained(MODEL_ID, torch_dtype=torch.float16)