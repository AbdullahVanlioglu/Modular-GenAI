import torch
import logging
import time
import torch.nn as nn

from pathlib import Path
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

class PolicyAndCriticWrapper(nn.Module):
    def __init__(self, policy_model, critic_model):
        super().__init__()
        self.policy_model = policy_model
        self.critic_model = critic_model

    def forward(self,
                x: torch.Tensor,
                **kwargs):
        return self.policy_model(input=x, **kwargs), self.critic_model(input=x, only_last=False, **kwargs)


class PolicyHead(nn.Module):
    def __init__(self,
                 args):
        self.args = args


class LMWithPolicyHead(nn.Module):
    def __init__(self,
                 backbone_model: nn.Module,
                 args):
        self.backbone_model = backbone_model
        self.policy_head = PolicyHead(args)
        self.args = args

    def forward(self,
                input_ids,
                past_key_values,
                attention_mask,
                **kwargs,
                ):
        kwargs["output_hidden_states"] = True
        kwargs["past_key_values"] = past_key_values
        
        backbone_output = self.backbone_model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                                  )

        last_hidden_state = backbone_output.hidden_states[-1]
        lm_logits = backbone_output.logits
        loss = backbone_output.loss

        value = self.v_head(last_hidden_state).squeeze(-1)

        return (lm_logits, loss, value)
        

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
                 backbone_model: nn.Module,
                 args,
                 ):
        self.backbone_model = backbone_model
        self.value_head = ValueHead(args)
        self._init_weights()

    def _init_weights(self):
        self.value_head.network.weight.data.normal_(mean=0.0, std=0.2)
        self.value_head.network.bias.data.zero_()

    def forward(self,
                input_ids,
                past_key_values,
                attention_mask,
                **kwargs,
                ):
        kwargs["output_hidden_states"] = True
        kwargs["past_key_values"] = past_key_values
        
        backbone_output = self.backbone_model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                                  )

        last_hidden_state = backbone_output.hidden_states[-1]
        lm_logits = backbone_output.logits
        loss = backbone_output.loss

        value = self.v_head(last_hidden_state).squeeze(-1)

        return (lm_logits, loss, value)


class PPO(nn.Module):
    def __init__(self,
                 policy_model: nn.Module,
                 critic_model: nn.Module,
                 ref_policy_model: nn.Module,
                 reward_model: nn.Module,
                 args: PPOArgs,
                 tokenizer,
                 ):
        super().__init__()
        self.model = PolicyAndCriticWrapper(policy_model=policy_model, critic_model=critic_model)
        self.ref_policy_model = ref_policy_model
        self.reward_model = reward_model
        self.args = args
        self.tokenizer = tokenizer

    def generate(self,
                 queries: List[torch.Tensor],
                 batch_size: int,
                 length_sampler: Optional[callable] = None,
                 **generation_kwargs,
                 ):
        # self.tokenizer.padding_side = "left"
        padding_side_default = self.tokenizer.padding_side
        batch_size = min(len(queries), batch_size)

        for i in range(0, len(queries), batch_size):
            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()
            end_index = min(len(queries), i + batch_size)

            batch = queries[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {"input_ids": batch, "attention_mask": batch_mask}

            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_Length=None,
                pad_to_multiple_of=None,
                returns_tensors="pt"
                ).to(self.args.device)

            outputs = self.model.forward(**padded_inputs, **generation_kwargs)

            for generation, mask in zip(outputs, padded_inputs["attention_mask"]):
                if padding_side_default == "left":
                    output = generation[(1 - mask).sum():]
                else:
                    output = generation

            return output

    def train(self):
        self.model.train()
        