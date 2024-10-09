import torch
import torch.nn as nn
import json
import logging
import time

from transformer import Llama2Transformer, Llama2Args
from typing import Optional
from sentencepiece import SentencePieceProcessor
from pathlib import Path


class BaseArgs(nn.Module):
    @classmethod
    def build(cls, 
              args: dict, 
              max_seq_len: int,
              max_batch_size: int,
              device: str,
              **kwargs
              ):
        
        model_args = cls(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **kwargs
        )

        model_args = cls(args).to(device)
        return model_args
    

class BaseTransformer(nn.Module):
    def __init__(self, 
                 model: Optional[Llama2Transformer], 
                 tokenizer: Optional[SentencePieceProcessor], 
                 model_args: Optional[Llama2Args]):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @classmethod
    def build(cls,
              checkpoints_dir: str, 
              tokenizer_path: str, 
              load_model: bool, 
              args: BaseArgs):
        if load_model:
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
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        if load_model:
            del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")
        
        return BaseTransformer(model, tokenizer, args)
    


        