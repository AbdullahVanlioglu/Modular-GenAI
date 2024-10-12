import torch
import time
import json

from typing import Optional
from pathlib import Path
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from modular_genai.transformer import Llama2Transformer, Llama2Args, BaseTransformer

def main():
    seed = torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
        # Few shot promt
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        # Zero shot prompt
        """Continue to the song:
        What is love baby dont hurt me, no more
        """
    ]

    model_args = Llama2Args(
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device,
    )

    model = BaseTransformer.build(model_class=Llama2Transformer,
                                  checkpoints_dir='/mnt/c/Users/avanl/OneDrive/Masaüstü/Modular-GenAI/cache_dir/llama-2-7b/',
                                  tokenizer_class=SentencePieceProcessor,
                                  tokenizer_path='/mnt/c/Users/avanl/OneDrive/Masaüstü/Modular-GenAI/cache_dir/tokenizer.model',
                                  pretrained_model=True,
                                  model_args=model_args,
                                  )

if __name__ == "__main__":
    main()
    





