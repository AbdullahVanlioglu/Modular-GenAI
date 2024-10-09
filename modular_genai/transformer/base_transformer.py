import torch
import torch.nn as nn
import json
import logging
import time

from transformer import Llama2Transformer, Llama2Args
from typing import Optional, Type
from sentencepiece import SentencePieceProcessor
from pathlib import Path

class BaseTransformer(nn.Module):
    def __init__(self, 
                 model: Optional[Llama2Transformer], 
                 tokenizer: Optional[SentencePieceProcessor], 
                 model_args: Optional[Llama2Args]):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args

    @classmethod
    def build(cls,
              model_class: Type[nn.Module],
              checkpoints_dir: str, 
              tokenizer_path: str, 
              pretrained_model: bool, 
              args: Optional[Llama2Args]):
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

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        args.vocab_size = tokenizer.vocab_size()
        
        if args.device == "cuda":
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = model_class(args)

        if pretrained_model:
            del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")
        
        return cls(model, tokenizer, args)
    


        