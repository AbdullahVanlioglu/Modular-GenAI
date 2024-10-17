import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Tuple

@dataclass
class PPOArgs:
    r"""
    Dataclass for PPO (Proximal Policy Optimization) hyperparameters.

    Args:
        rollout_len (int): 
            Number of steps to collect in each environment per rollout.
        ent_coef (float): 
            Coefficient for entropy regularization to encourage exploration.
        value_coef (float): 
            Coefficient for the value loss, which balances the importance of the value function update.
        gamma (float): 
            Discount factor for future rewards, controlling how much the agent values long-term rewards.
        gae_lambda (float): 
            Smoothing factor for Generalized Advantage Estimation (GAE), balancing bias vs variance.
        epochs (int): 
            Number of times to update the policy using the collected rollout data per PPO iteration.
        seed (int): 
            Random seed.
        lr (float): 
            Learning rate.
        clip_value (float): 
            Clipping parameter for the policy ratio, ensuring updates stay within a safe range.
        batch_size (int): 
            Number of samples used for each policy update.
        max_grad_norm (float): 
            Maximum value for gradient norm clipping to stabilize training and avoid exploding gradients.
        normalize_advantage (bool): 
            If `True`, normalizes advantages to have mean 0 and standard deviation 1.
        log_interval (int): 
            Number of updates between logging performance metrics.
        total_timesteps (int): 
            Total number of timesteps to train the agent.
        Device ('str'):
            Cpu or Cuda   
    """
    rollout_len: int
    ent_coef: float
    value_coef: float
    gamma: float
    gae_lambda: float
    epochs: int
    seed: int
    lr: float
    clip_value: float
    batch_size: int
    max_grad_norm: float
    normalize_advantage: bool
    log_interval: int
    total_timesteps: int
    value_head_dropout_prob: float = 0.1
    value_head_hidden_size: int = 2
    device: str = None


class ValueHead(nn.Module):
    def __init__(self,
                 args: PPOArgs):
        self.hidden_size = args.value_head_hidden_size
        self.dropout_prob = args.value_head_dropout_prob

        self.dropout = nn.Dropout(self.dropout_prob) if self.dropout_prob else nn.Identity()
        self.network = nn.Linear(self.hidden_size, 1)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor):
        output = self.dropout(x)
        output = self.network(output)

        return output


class LMWithValueHead(nn.Module):
    def __init__(self,
                 pretrained_model: Any[nn.Module],
                 args: PPOArgs,
                 ):
        self.value_head = ValueHead(args)


class PPO(nn.Module):
    def __init__(self,
                 args: PPOArgs,
                 model: nn.Module,
                 reward_model: nn.Module,
                 tokenizer,
                 ):
        self.args = args
        self.model = model
        self.reward_model = reward_model
        self.tokenizer = tokenizer

        if not isinstance(self.args, PPOArgs):
            raise ValueError(f"Args must be a PPOArgs, but got {type(args)}")
        
    def generate(self,
                 queries,
                 ):

        for query in range(queries):
            batch_mask = [torch.ones_like(element) for element in query]
            inputs = {"input_ids": query, "attention_mask": batch_mask}

            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_Length=None,
                pad_to_multiple_of=None,
                returns_tensors="pt"
                ).to(self.args.device)

    def train(self):
        self.model.train()
        raise NotImplementedError



