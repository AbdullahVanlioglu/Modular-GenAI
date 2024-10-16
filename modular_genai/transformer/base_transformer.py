import torch
import torch.nn as nn
import json
import logging
import time

from modular_genai.transformer import Llama2Args
from typing import Optional, Type, Any
from pathlib import Path


class BaseTransformer(nn.Module):
    def __init__(self, 
                 model: Optional[nn.Module], 
                 model_args: Optional[Any],
                 tokenizer: Optional[Any], 
                 encoder: Optional[nn.Module] = None,
                 encoder_args: Optional[Any] = None
                 ):
        super().__init__()
        self.model = model
        self.model_args = model_args
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.encoder_args = encoder_args

    @staticmethod
    def build(model_class: Type[nn.Module],
              model_args: Optional[Any],
              checkpoints_dir: str, 
              tokenizer: Optional[Any],
              tokenizer_path: str, 
              pretrained_model: bool, 
              encoder_class: Optional[nn.Module] = None,
              encoder_args: Optional[Any] = None
              ):
        prev_time = time.time()
        if pretrained_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            
            assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"

            ckpt_path = checkpoints[0]
            logging.info(f'Loading checkpoint "{ckpt_path}"')
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            logging.info(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            prev_time = time.time()
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args = Llama2Args(
            **params
        )
        # Inıtialize the tokenizer
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()
        
        if getattr(model_args, 'device', 'cpu') == "cuda":
            torch.set_default_dtype(torch.float16)
            torch.set_default_device(torch.device("cuda"))
        else:
            torch.set_default_dtype(torch.bfloat16)
            torch.set_default_device(torch.device("cpu"))

        # Inıtialize the transformer
        model = model_class(model_args)

        if pretrained_model:
            # Some implementations of LLaMA 2 or fine-tuned versions might handle RoPE differently
            # Deleted to avoid compatibility issues
            del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")

        # Initialize encoder if provided
        encoder = encoder_class(encoder_args) if encoder_class else None        
        return BaseTransformer(model, model_args, tokenizer, encoder, encoder_args)
    
        