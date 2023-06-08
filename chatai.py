from transformers import AutoTokenizer, AutoModel
import torch
import config
from typing import List, Tuple

tokenizer = AutoTokenizer.from_pretrained(config.CHATGLM_MODEL_PATH, trust_remote_code=True)

if torch.has_cuda:
    print('cuda')
    model = AutoModel.from_pretrained(config.CHATGLM_MODEL_PATH, trust_remote_code=True).half().cuda()
elif torch.has_mps:
    print('mps')
    model = AutoModel.from_pretrained(config.CHATGLM_MODEL_PATH, trust_remote_code=True).half().to('mps')
else:
    print('cpu')
    model = AutoModel.from_pretrained(config.CHATGLM_MODEL_PATH, trust_remote_code=True).half()

model = model.eval()


def chat(prompt: str, history: List[Tuple[str, str]] = None, max_length=3000, top_p=0.7, temperature=0.95):
    response, history = model.chat(
            tokenizer,
            prompt,
            history=history,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature)
    return response, history


def stream_chat(prompt: str, history: List[Tuple[str, str]] = None, max_length=3000, top_p=0.7, temperature=0.95):
    for response, history in model.stream_chat(
            tokenizer,
            prompt,
            history=history,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature):
        yield response, history
