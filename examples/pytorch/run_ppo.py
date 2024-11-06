import torch

from sentencepiece import SentencePieceProcessor
from modular_genai.finetune import PPO, PPOArgs, LMWithValueHead
from modular_genai.transformer import BaseTransformer, Llama2Transformer, Llama2Args

device = "cuda" if torch.cuda.is_available() else "cpu"

backbone_model_args = Llama2Args(
        max_seq_len=1024,
        max_batch_size=64,
        device=device,
    )

tokenizer = SentencePieceProcessor()

backbone_llm_model = BaseTransformer.build(model_class=Llama2Transformer,
                                  model_args=backbone_model_args,
                                  checkpoints_dir='/path/to/your/llama-2-7b/',
                                  tokenizer=tokenizer,
                                  tokenizer_path='/path/to/your/tokenizer.model',
                                  pretrained_model=True,
                                  )

args = PPOArgs()

model = LMWithValueHead(base_model=backbone_llm_model,
                        args=args)
ref_model = LMWithValueHead(base_model=backbone_llm_model,
                        args=args)

