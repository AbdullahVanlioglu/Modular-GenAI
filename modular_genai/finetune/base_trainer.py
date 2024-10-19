import torch
import torch.nn as nn

from typing import Optional, Any, Type
from modular_genai.finetune import PPOArgs


class BaseTrainer(nn.Module):
    def __init__(self, 
                 model: Type[nn.Module], 
                 model_args: Any,
                 tokenizer: Any, 
                 ):
        super().__init__()
        self.model = model
        self.model_args = model_args
        self.tokenizer = tokenizer
        
    @staticmethod
    def build(model_class,
              model_args: Optional[Any],
              tokenizer: Optional[Any],
              ):
        struct = np.dtype([
            ("observation", np.float32, observation_space.shape),
            ("next_observation", np.float32, observation_space.shape),
            ("action", action_space.dtype, (action_dim,)),
            ("reward", np.float32, (1,)),
            ("termination", np.float32, (1,)),
            ("truncation", np.float32, (1,)),
            ("old_log_prob", np.float32, (1,)),
        ])
        buffer = Buffer(struct, args.rollout_len, env.num_envs)
        collector = RolloutCollector(
            env=env,
            buffer=buffer,
            agent=agent,
            store_normalizer_stats=False,
        )
        return BaseTrainer(
            model=model_class,
            collector=collector,
            args=model_args,
        )