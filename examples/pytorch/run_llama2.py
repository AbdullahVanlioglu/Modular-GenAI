import torch
import time
import json

from typing import Optional
from pathlib import Path
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from modular_genai.transformer import Llama2Transformer, Llama2Args, BaseTransformer

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p 
    probs_sort[mask] = 0.0 
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token) 
    return next_token


def main():
    seed = torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    prompts = [
    # Scientific explanation
    "Quantum mechanics can best be described as the branch of physics that",

    # Creative storytelling
    "Once upon a time in a small village at the edge of the forest, there was a young boy named Finn who dreamed of",

    # Philosophical thought experiment
    "If artificial intelligence develops a mind of its own, humanity will",

    # Mathematical word problem
    "If a train leaves New York at 8:00 AM traveling at 80 miles per hour, and another train leaves Chicago at 9:00 AM traveling at 90 miles per hour, where will they meet?",

    # Few-shot translation prompt
    """Translate the following English words to Spanish:

    apple => manzana
    dog => perro
    book => libro
    computer =>""",

    # Zero-shot sentiment analysis
    """The movie was breathtaking and left me speechless. The performances were top-notch, and the cinematography was extraordinary.
    Sentiment:""",

    # Code generation task
    """Write a Python function to check if a given string is a palindrome:
    """,

    # Factual knowledge retrieval
    "Who was the first person to step on the Moon?",

    # Dialogue completion
    """Alice: How was your day?
    Bob: It was pretty good! I went hiking in the morning, and later I""",

    # Poetry completion
    """Roses are red,
    Violets are blue,
    I never thought I'd"""
    ]

    model_args = Llama2Args(
        max_seq_len=1024,
        max_batch_size=64,
        device=device,
    )

    tokenizer = SentencePieceProcessor()

    llama = BaseTransformer.build(model_class=Llama2Transformer,
                                  model_args=model_args,
                                  checkpoints_dir='/path/to/your/llama-2-7b/',
                                  tokenizer=tokenizer,
                                  tokenizer_path='/path/to/your/tokenizer.model',
                                  pretrained_model=True,
                                  )

    # Text Completion Task
    prompt_tokens = [llama.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
    batch_size = len(prompt_tokens)
    max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
    total_len = min(llama.args.max_seq_len, llama.args.max_gen_len + max_prompt_len)

    pad_id = llama.tokenizer.pad_id()
    tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)

    for k, t in enumerate(prompt_tokens):
        tokens[k, len(t)] = torch.tensor(t, dtype=torch.long, device=device)

    eos_reached = torch.Tensor([False] * batch_size, device=device)
    prompt_tokens_mask = tokens != pad_id

    for cur_pos in tqdm(range(1, total_len)):
        with torch.no_grad():
            logits = llama.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
        if llama.args.temperature > 0:
            probs = torch.softmax(logits[:, -1] / llama.args.temperature, dim=-1)
            next_token = sample_top_p(probs, llama.args.top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        # Save prompt tokens and replace the padding tokens with predicted tokens
        next_token = next_token.reshape(-1)
        next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        
        # Eos reached
        eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == llama.tokenizer.eos_id())
        if all(eos_reached):
            break

    out_tokens = []
    out_text = []
    for current_prompt_tokens in range(tokens.tolist()):
        # Cut prompt when reach the EOS
        if llama.tokenizer.eos_id() in current_prompt_tokens:
            eos_idx = current_prompt_tokens.index(llama.tokenizer.eos_id())
            current_prompt_tokens = current_prompt_tokens[:eos_idx]
        out_tokens.append(current_prompt_tokens)
        out_text.append(llama.tokenizer.decode(current_prompt_tokens))
    return (out_tokens, out_text)


if __name__ == "__main__":
    main()
    





